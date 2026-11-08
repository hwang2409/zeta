"""The account registry: accounts.json, account ids, and the orphan guard."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from ..core.session_files import SessionFileReplaced, child_directory
from ..storage_format import (
    CLOSED_VARIANTS,
    reject_unknown_keys,
    require_closed_variant,
)
from .files import (
    read_optional as _read_optional,
)
from .files import (
    read_optional_bytes as _read_optional_bytes,
)

_PROVIDERS = cast(frozenset[str], CLOSED_VARIANTS["account.provider"])
_MAX_SWEEP_WARNING_PATHS = 5
_SLUG = re.compile(r"^[a-z0-9_-]{1,32}$")


class _Registry(Enum):
    """How accounts.json read; only LOADED may drive a destructive sweep."""

    MISSING = "missing"
    LOADED = "loaded"
    BROKEN = "broken"


_UNSAFE_REGISTRY_WARNINGS = {
    _Registry.MISSING: "accounts.json is missing; kept every account directory and "
    "skipped account cleanup; restore accounts.json, then restart zeta",
    _Registry.BROKEN: "accounts.json is invalid; kept it and every account directory "
    "and skipped account cleanup; fix or restore accounts.json, then restart zeta",
}


def _account_label(value: str) -> str:
    if type(value) is not str:
        raise ValueError("invalid account label")
    label = value.lower()
    if not _SLUG.fullmatch(label):
        raise ValueError("invalid account label")
    return label


def _registry_reference_index(
    value: Mapping[str, Any],
) -> dict[tuple[str, str], AccountId]:
    accounts = value["accounts"]
    labels = value["labels"]
    references: dict[tuple[str, str], AccountId] = {}
    for key, metadata in accounts.items():
        if not isinstance(key, str) or not isinstance(metadata, Mapping):
            raise TypeError
        provider, name = key.split(":", 1)
        require_closed_variant("account.provider", provider)
        account = AccountId(provider, name)
        if key != str(account) or any(
            metadata.get(field) != expected
            for field, expected in {
                "provider": account.provider,
                "name": account.name,
            }.items()
        ):
            raise ValueError
        label = labels.get(key, account.name)
        if _account_label(label) != label:
            raise ValueError
        for reference in {account.name, label}:
            previous = references.setdefault((account.provider, reference), account)
            if previous != account:
                raise ValueError
    if set(labels) - set(accounts):
        raise ValueError
    return references


class AccountError(RuntimeError):
    """A credential-safe account-store error."""


class AccountRegistryMissingError(AccountError):
    """Adding an account would create a registry that orphans existing ones."""

    def __init__(self) -> None:
        super().__init__(
            "accounts.json is missing but other account directories exist; "
            "restore accounts.json before adding an account"
        )


@dataclass(frozen=True, slots=True)
class AccountId:
    provider: str
    name: str

    def __post_init__(self) -> None:
        name = self.name.lower()
        if self.provider not in _PROVIDERS or not _SLUG.fullmatch(name):
            raise ValueError("invalid account id")
        object.__setattr__(self, "name", name)

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


def _publication_record(account: AccountId, credential: bytes) -> dict[str, str]:
    return {
        "provider": account.provider,
        "name": account.name,
        "sha256": hashlib.sha256(credential).hexdigest(),
    }


def _read_publication_pending(directory_fd: int) -> dict[str, str] | None:
    raw = _read_optional(directory_fd, "publish-pending.json")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if isinstance(value, Mapping):
            reject_unknown_keys(value, "account_publication_marker")
        if (
            not isinstance(value, Mapping)
            or require_closed_variant("account.provider", value["provider"])
            != value["provider"]
            or not isinstance(value["name"], str)
            or not _SLUG.fullmatch(value["name"])
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        ):
            raise ValueError
        return dict(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


class _AccountRegistry:
    """Read accounts.json and guard account directories against orphan sweeps."""

    def _list_orphan_directories(
        self, home_fd: int, registered: set[str]
    ) -> list[tuple[str, str, str, set[str]]]:
        orphans: list[tuple[str, str, str, set[str]]] = []
        for root, allowed in (
            ("accounts", {"oauth.json", "usage.json", "publish-pending.json"}),
            ("account-locks", {"lifecycle.lock", "usage.lock"}),
        ):
            try:
                with child_directory(home_fd, root) as root_fd:
                    for provider in _PROVIDERS:
                        try:
                            with child_directory(root_fd, provider) as provider_fd:
                                names = [
                                    (
                                        name,
                                        stat.S_ISDIR(
                                            os.stat(
                                                name,
                                                dir_fd=provider_fd,
                                                follow_symlinks=False,
                                            ).st_mode
                                        ),
                                    )
                                    for name in sorted(os.listdir(provider_fd))
                                ]
                        except FileNotFoundError:
                            continue
                        orphans.extend(
                            (root, provider, name, allowed)
                            for name, is_directory in names
                            if is_directory and f"{provider}:{name}" not in registered
                        )
            except FileNotFoundError:
                continue
        return orphans

    def _refuse_orphaning_registration(self, home_fd: int, account: AccountId) -> None:
        if self._registering_would_orphan(home_fd, account):
            raise AccountRegistryMissingError

    def _registering_would_orphan(self, home_fd: int, account: AccountId) -> bool:
        """A registry created without existing account dirs would sweep them next."""

        state, _ = self._load_registry(home_fd)
        if state is not _Registry.MISSING:
            return False
        return self._has_unregistered_account_directories(home_fd, {str(account)})

    def _has_unregistered_account_directories(
        self, home_fd: int, registered: set[str]
    ) -> bool:
        return any(
            root == "accounts"
            for root, *_ in self._list_orphan_directories(home_fd, registered)
        )

    def _orphan_is_pending_publication(
        self, home_fd: int, root: str, provider: str, name: str
    ) -> bool:
        if root != "accounts":
            return False
        account = AccountId(provider, name)
        try:
            with (
                child_directory(home_fd, root) as root_fd,
                child_directory(root_fd, provider) as provider_fd,
                child_directory(provider_fd, name) as account_fd,
            ):
                pending = _read_publication_pending(account_fd)
                credential = _read_optional_bytes(account_fd, "oauth.json")
        except FileNotFoundError:
            return True
        return (
            pending is not None
            and credential is not None
            and pending == _publication_record(account, credential)
        )

    def _remove_orphan_directory(
        self, home_fd: int, root: str, provider: str, name: str, allowed: set[str]
    ) -> str | None:
        from .logout import remove_account_directory, sweep_warning_path

        try:
            remove_account_directory(home_fd, root, (provider, name), allowed=allowed)
        except (AccountError, OSError, ValueError):
            return sweep_warning_path(self.home / root / provider / name)
        return None

    def _sweep_orphan_directories(
        self, home_fd: int, registered: set[str], warnings: list[str]
    ) -> None:
        refused: list[str] = []
        for root, provider, name, allowed in self._list_orphan_directories(
            home_fd, registered
        ):
            try:
                if self._orphan_is_pending_publication(home_fd, root, provider, name):
                    continue
            except ValueError:
                pass
            path = self._remove_orphan_directory(home_fd, root, provider, name, allowed)
            if path is not None:
                refused.append(path)
        warnings.extend(
            f"could not clean up account directory {path}; check its mode or ownership, then delete it or fix it"
            for path in refused[:_MAX_SWEEP_WARNING_PATHS]
        )
        if len(refused) > _MAX_SWEEP_WARNING_PATHS:
            warnings.append(
                f"and {len(refused) - _MAX_SWEEP_WARNING_PATHS} more account directories could not be cleaned up"
            )

    def _read_registry(self, home_fd: int) -> dict[str, Any]:
        state, value = self._load_registry(home_fd)
        if state is _Registry.BROKEN:
            raise AccountError("accounts.json is invalid; use /accounts")
        return value

    def _load_registry(self, home_fd: int) -> tuple[_Registry, dict[str, Any]]:
        """Read accounts.json; the value is an empty registry unless LOADED."""

        empty: dict[str, Any] = {
            "version": 1,
            "accounts": {},
            "defaults": {},
            "labels": {},
        }
        try:
            for attempt in range(5):
                try:
                    raw = _read_optional(home_fd, "accounts.json")
                    break
                except SessionFileReplaced:
                    if attempt == 4:
                        raise
            if raw is None:
                return _Registry.MISSING, empty
            value = json.loads(raw)
            if (
                not isinstance(value, Mapping)
                or require_closed_variant(
                    "account_registry.version", value.get("version")
                )
                != 1
                or not isinstance(value.get("accounts"), Mapping)
                or not isinstance(value.get("defaults"), Mapping)
                or not isinstance(value.get("labels", {}), Mapping)
            ):
                raise ValueError
            accounts = dict(value["accounts"])
            defaults = dict(value["defaults"])
            labels = dict(value.get("labels", {}))
            normalized = {
                "version": 1,
                "accounts": accounts,
                "defaults": defaults,
                "labels": labels,
            }
            _registry_reference_index(normalized)
            for provider, key in defaults.items():
                if not isinstance(provider, str) or not isinstance(key, str):
                    raise TypeError
                require_closed_variant("account.provider", provider)
                if key not in accounts or not key.startswith(f"{provider}:"):
                    raise ValueError
            return _Registry.LOADED, normalized
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return _Registry.BROKEN, empty
