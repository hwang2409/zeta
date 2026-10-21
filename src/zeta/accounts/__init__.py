"""Named subscription accounts and crash-safe legacy migration."""
from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..oauth import OAuthTokens

_PROVIDERS = {"anthropic", "codex"}
_SLUG = re.compile(r"^[a-z0-9_-]{1,32}$")


class AccountError(RuntimeError):
    """A safe account-store error."""


@dataclass(frozen=True, slots=True)
class AccountId:
    provider: str
    name: str

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS or not _SLUG.fullmatch(self.name.lower()):
            raise ValueError("invalid account id")
        object.__setattr__(self, "name", self.name.lower())

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


class AccountStore:
    """Own the registry, account credentials, and process locks for one home."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.registry = self.home / "accounts.json"
        self.accounts = self.home / "accounts"
        self.locks = self.home / "account-locks"
        self.migration_lock = self.home / ".accounts-migration.lock"
        self.registry_lock = self.home / ".accounts.lock"

    @contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        with path.open("a+") as handle:
            os.chmod(path, 0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def lifecycle_lock(self, account: AccountId) -> Iterator[None]:
        with self._lock(self.locks / account.provider / f"{account.name}.lock"):
            yield

    def _read(self) -> dict:
        if not self.registry.exists():
            return {"version": 1, "accounts": {}, "defaults": {}}
        try:
            value = json.loads(self.registry.read_text())
            if not isinstance(value, Mapping) or value.get("version") != 1:
                raise ValueError
            return {"version": 1, "accounts": dict(value.get("accounts", {})), "defaults": dict(value.get("defaults", {}))}
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AccountError("accounts.json is invalid; use /accounts") from exc

    def _write(self, value: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.home, 0o700)
        temporary = self.registry.with_name(f".accounts.{secrets.token_hex(8)}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.registry)
            os.chmod(self.registry, 0o600)
            dfd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        finally:
            temporary.unlink(missing_ok=True)

    def list(self) -> list[AccountId]:
        with self._lock(self.registry_lock):
            return [AccountId(*key.split(":", 1)) for key in self._read()["accounts"]]

    def default(self, provider: str) -> AccountId | None:
        with self._lock(self.registry_lock):
            value = self._read()["defaults"].get(provider)
            return AccountId(*value.split(":", 1)) if value else None

    def set_default(self, account: AccountId) -> None:
        with self._lock(self.registry_lock):
            value = self._read()
            if str(account) not in value["accounts"]:
                raise AccountError(f"account {account} does not exist; use /accounts")
            value["defaults"][account.provider] = str(account)
            self._write(value)

    def credential_path(self, account: AccountId) -> Path:
        return self.accounts / account.provider / account.name / "oauth.json"

    def ensure_account(self, account: AccountId, tokens: OAuthTokens, *, replace: bool = False) -> None:
        with self._lock(self.registry_lock), self.lifecycle_lock(account):
            value = self._read()
            key = str(account)
            if key in value["accounts"] and not replace:
                raise AccountError(f"account {key} already exists; use --replace")
            destination = self.credential_path(account)
            self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(destination.parent, 0o700)
            _atomic_tokens(destination, tokens)
            value["accounts"][key] = {"provider": account.provider, "name": account.name}
            value["defaults"].setdefault(account.provider, key)
            self._write(value)

    def migrate(self) -> None:
        """Import legacy files without retiring them before the registry commit."""
        with self._lock(self.migration_lock):
            for provider, legacy_name in (("anthropic", "anthropic-oauth.json"), ("codex", "codex-oauth.json")):
                legacy = self.home / legacy_name
                if not legacy.exists():
                    continue
                account = AccountId(provider, "default")
                with self._lock(self.registry_lock), self.lifecycle_lock(account):
                    value = self._read()
                    if str(account) in value["accounts"]:
                        continue
                    try:
                        raw = json.loads(legacy.read_text())
                        tokens = OAuthTokens.from_mapping(raw, error_type=AccountError)
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        raise AccountError(f"{legacy_name} is invalid; login required") from exc
                    destination = self.credential_path(account)
                    _mkdir_durable(destination.parent, self.accounts)
                    os.chmod(destination.parent, 0o700)
                    _atomic_tokens(destination, tokens)
                    if json.loads(destination.read_text()) != tokens.to_dict():
                        raise AccountError("account migration verification failed")
                    value["accounts"][str(account)] = {"provider": provider, "name": "default"}
                    value["defaults"].setdefault(provider, str(account))
                    self._write(value)
                    legacy.unlink()
                    dfd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY)
                    try: os.fsync(dfd)
                    finally: os.close(dfd)


def _mkdir_durable(path: Path, stop: Path) -> None:
    if not stop.exists():
        stop.mkdir(parents=True, mode=0o700)
        dfd = os.open(stop.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    current = stop
    if path == stop:
        return
    for part in path.relative_to(stop).parts:
        current = current / part
        if not current.exists():
            current.mkdir(mode=0o700)
            dfd = os.open(current.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)


def _atomic_tokens(path: Path, tokens: OAuthTokens) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(tokens.to_dict(), handle, separators=(",", ":")); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); os.chmod(path, 0o600)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_account(home: str | Path, account: AccountId | None, provider: str) -> AccountId:
    store = AccountStore(home)
    chosen = account or store.default(provider)
    if chosen is None:
        raise AccountError(f"no {provider} account selected; use /accounts")
    if str(chosen) not in {str(item) for item in store.list()}:
        raise AccountError(f"account {chosen} does not exist; use /accounts")
    return chosen


__all__ = ["AccountError", "AccountId", "AccountStore", "resolve_account"]
