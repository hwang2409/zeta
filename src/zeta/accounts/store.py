"""Private storage implementation for named subscription accounts."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.session_files import child_directory, open_session_file
from ..oauth import OAuthTokens, credential_file_lock

_PROVIDERS = {"anthropic", "codex"}
_SLUG = re.compile(r"^[a-z0-9_-]{1,32}$")


class AccountError(RuntimeError):
    """A credential-safe account-store error."""


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


class AccountStore:
    """Own the durable account catalog, credentials, and process locks."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.registry = self.home / "accounts.json"
        self.accounts = self.home / "accounts"
        self.locks = self.home / "account-locks"
        self.migration_lock = self.home / ".accounts-migration.lock"
        self.registry_lock = self.home / ".accounts.lock"

    @contextmanager
    def _home(self) -> Iterator[int]:
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.home, 0o700)
        fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def _lock_at(
        self, directory_fd: int, name: str, *, wait: bool = True
    ) -> Iterator[None]:
        created = not _exists(directory_fd, name)
        for attempt in range(3):
            try:
                fd = open_session_file(directory_fd, name, os.O_RDWR | os.O_CREAT)
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise
        try:
            os.fchmod(fd, 0o600)
            if created:
                os.fsync(fd)
                _fsync_directory(directory_fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise AccountError(
                    "a login for this provider is already in progress"
                ) from exc
            yield
        finally:
            os.close(fd)

    @contextmanager
    def _registry_guard(self, home_fd: int) -> Iterator[None]:
        with self._lock_at(home_fd, ".accounts.lock"):
            yield

    @contextmanager
    def _lifecycle_guard(self, home_fd: int, account: AccountId) -> Iterator[None]:
        with (
            _account_lock_directory(home_fd, account) as lock_fd,
            self._lock_at(lock_fd, "lifecycle.lock"),
        ):
            yield

    @contextmanager
    def provider_login_lock(self, provider: str) -> Iterator[None]:
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        with self._home() as home_fd:
            locks_fd = _mkdir_child_durable(home_fd, "account-locks")
            try:
                provider_fd = _mkdir_child_durable(locks_fd, provider)
                try:
                    with self._lock_at(provider_fd, "login.lock", wait=False):
                        yield
                finally:
                    os.close(provider_fd)
            finally:
                os.close(locks_fd)

    def list(self) -> list[AccountId]:
        with self._home() as home_fd, self._registry_guard(home_fd):
            return [
                AccountId(*key.split(":", 1))
                for key in self._read_registry(home_fd)["accounts"]
            ]

    def default(self, provider: str) -> AccountId | None:
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        with self._home() as home_fd, self._registry_guard(home_fd):
            value = self._read_registry(home_fd)["defaults"].get(provider)
            return AccountId(*value.split(":", 1)) if value else None

    def contains(self, account: AccountId) -> bool:
        with self._home() as home_fd, self._registry_guard(home_fd):
            return str(account) in self._read_registry(home_fd)["accounts"]

    def set_default(self, account: AccountId) -> None:
        with self._home() as home_fd, self._registry_guard(home_fd):
            value = self._read_registry(home_fd)
            if str(account) not in value["accounts"]:
                raise AccountError(f"account {account} does not exist; use /accounts")
            value["defaults"][account.provider] = str(account)
            self._write_registry(home_fd, value)

    def credential_path(self, account: AccountId) -> Path:
        return self.accounts / account.provider / account.name / "oauth.json"

    def resolve_credential_path(
        self, provider: str, account: AccountId | None = None
    ) -> Path:
        """Migrate legacy auth, then resolve an account or the global default."""

        provider = "anthropic" if provider == "claude" else provider
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        if account is not None and account.provider != provider:
            raise ValueError("account provider does not match")
        self.migrate()
        selected = account or self.default(provider) or AccountId(provider, "default")
        if account is not None and not self.contains(account):
            raise AccountError(f"account {account} does not exist; use /accounts")
        return self.credential_path(selected)

    def ensure_account(
        self, account: AccountId, tokens: OAuthTokens, *, replace: bool = False
    ) -> None:
        with (
            self._home() as home_fd,
            self._registry_guard(home_fd),
            self._lifecycle_guard(home_fd, account),
        ):
            value = self._read_registry(home_fd)
            key = str(account)
            if key in value["accounts"] and not replace:
                raise AccountError(f"account {key} already exists; use --replace")
            with _account_directory(home_fd, account) as account_fd:
                _write_tokens(account_fd, tokens)
                _verify_tokens(account_fd, tokens)
            value["accounts"][key] = {
                "provider": account.provider,
                "name": account.name,
            }
            value["defaults"].setdefault(account.provider, key)
            self._write_registry(home_fd, value)

    def migrate(self) -> None:
        with (
            self._home() as home_fd,
            self._lock_at(home_fd, ".accounts-migration.lock"),
        ):
            for provider in ("anthropic", "codex"):
                self._migrate_provider(home_fd, provider)

    def _migrate_provider(self, home_fd: int, provider: str) -> None:
        account = AccountId(provider, "default")
        legacy_name = f"{provider}-oauth.json"
        legacy_path = self.home / legacy_name
        with (
            self._registry_guard(home_fd),
            self._lifecycle_guard(home_fd, account),
            credential_file_lock(legacy_path),
        ):
            value = self._read_registry(home_fd)
            registered = str(account) in value["accounts"]
            legacy_exists = _exists_regular(home_fd, legacy_name)
            if registered:
                with _open_account_directory(home_fd, account) as account_fd:
                    _read_tokens(account_fd, "oauth.json")
                if legacy_exists:
                    _unlink_legacy(home_fd, legacy_name)
                return
            if not legacy_exists:
                return
            tokens = _read_tokens(home_fd, legacy_name)
            with _account_directory(home_fd, account) as account_fd:
                _write_tokens(account_fd, tokens)
                _verify_tokens(account_fd, tokens)
            value["accounts"][str(account)] = {"provider": provider, "name": "default"}
            value["defaults"].setdefault(provider, str(account))
            self._write_registry(home_fd, value)
            _unlink_legacy(home_fd, legacy_name)

    def _read_registry(self, home_fd: int) -> dict[str, Any]:
        try:
            raw = _read_optional(home_fd, "accounts.json")
            if raw is None:
                return {"version": 1, "accounts": {}, "defaults": {}}
            value = json.loads(raw)
            if (
                not isinstance(value, Mapping)
                or value.get("version") != 1
                or not isinstance(value.get("accounts"), Mapping)
                or not isinstance(value.get("defaults"), Mapping)
            ):
                raise ValueError
            accounts = dict(value["accounts"])
            defaults = dict(value["defaults"])
            for key, metadata in accounts.items():
                if not isinstance(key, str):
                    raise TypeError
                account = AccountId(*key.split(":", 1))
                if key != str(account) or metadata != {
                    "provider": account.provider,
                    "name": account.name,
                }:
                    raise ValueError
            for provider, key in defaults.items():
                if (
                    not isinstance(provider, str)
                    or not isinstance(key, str)
                    or provider not in _PROVIDERS
                    or key not in accounts
                    or not key.startswith(f"{provider}:")
                ):
                    raise ValueError
            return {"version": 1, "accounts": accounts, "defaults": defaults}
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AccountError("accounts.json is invalid; use /accounts") from exc

    def _write_registry(self, home_fd: int, value: dict[str, Any]) -> None:
        data = (
            json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        _atomic_publish(home_fd, "accounts.json", data)
        _fsync_directory(home_fd)


@contextmanager
def _account_lock_directory(home_fd: int, account: AccountId) -> Iterator[int]:
    with ExitStack() as stack:
        locks_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(home_fd, "account-locks"))
        )
        provider_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(locks_fd, account.provider))
        )
        account_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(provider_fd, account.name))
        )
        yield account_fd


@contextmanager
def _account_directory(home_fd: int, account: AccountId) -> Iterator[int]:
    with ExitStack() as stack:
        accounts_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(home_fd, "accounts"))
        )
        provider_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(accounts_fd, account.provider))
        )
        account_fd = stack.enter_context(
            _owned_fd(_mkdir_child_durable(provider_fd, account.name))
        )
        yield account_fd


@contextmanager
def _open_account_directory(home_fd: int, account: AccountId) -> Iterator[int]:
    with (
        child_directory(home_fd, "accounts") as accounts_fd,
        child_directory(accounts_fd, account.provider) as provider_fd,
        child_directory(provider_fd, account.name) as account_fd,
    ):
        yield account_fd


@contextmanager
def _owned_fd(fd: int) -> Iterator[int]:
    try:
        yield fd
    finally:
        os.close(fd)


def _mkdir_child_durable(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    else:
        _fsync_directory(parent_fd)
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    os.fchmod(fd, 0o700)
    return fd


def _atomic_publish(directory_fd: int, name: str, data: bytes) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(
        directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        published = open_session_file(directory_fd, name, os.O_RDONLY)
        try:
            os.fchmod(published, 0o600)
        finally:
            os.close(published)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _write_tokens(directory_fd: int, tokens: OAuthTokens) -> None:
    data = (json.dumps(tokens.to_dict(), separators=(",", ":")) + "\n").encode()
    _atomic_publish(directory_fd, "oauth.json", data)
    _fsync_directory(directory_fd)


def _verify_tokens(directory_fd: int, expected: OAuthTokens) -> None:
    if _read_tokens(directory_fd, "oauth.json") != expected:
        raise AccountError("account credential verification failed")


def _read_tokens(directory_fd: int, name: str) -> OAuthTokens:
    try:
        with os.fdopen(open_session_file(directory_fd, name, os.O_RDONLY)) as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError
        return OAuthTokens.from_mapping(value, error_type=AccountError)
    except AccountError:
        raise AccountError("OAuth credentials are invalid; login required") from None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AccountError("OAuth credentials are invalid; login required") from exc


def _read_optional(directory_fd: int, name: str) -> str | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as handle:
        return handle.read()


def _exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _exists_regular(directory_fd: int, name: str) -> bool:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AccountError("OAuth credentials are invalid; login required")
    return True


def _unlink_legacy(home_fd: int, name: str) -> None:
    _unlink_legacy_file(home_fd, name)
    _fsync_directory(home_fd)


def _unlink_legacy_file(home_fd: int, name: str) -> None:
    os.unlink(name, dir_fd=home_fd)


def _fsync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)
