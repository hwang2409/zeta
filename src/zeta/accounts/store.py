"""Private storage implementation for named subscription accounts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import (
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..core.session_files import child_directory, open_session_file
from ..oauth import OAUTH_REFRESH_TIMEOUT_SECONDS, OAuthTokens, credential_file_lock

_PROVIDERS = {"anthropic", "codex"}
_SLUG = re.compile(r"^[a-z0-9_-]{1,32}$")


class AccountError(RuntimeError):
    """A credential-safe account-store error."""


class AccountCredentialLockTimeoutError(AccountError):
    """Credential publication could not acquire its lifecycle lock in time."""


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
    def _lifecycle_guard(
        self, home_fd: int, account: AccountId, *, wait: bool = True
    ) -> Iterator[None]:
        with (
            _account_lock_directory(home_fd, account) as lock_fd,
            self._lock_at(lock_fd, "lifecycle.lock", wait=wait),
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

    def resolve_account(
        self, provider: str, account: AccountId | None = None
    ) -> AccountId:
        """Migrate legacy auth, then resolve an account or the global default."""

        provider = "anthropic" if provider == "claude" else provider
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        if account is not None and account.provider != provider:
            raise ValueError("account provider does not match")
        self.migrate()
        return account or self.default(provider) or AccountId(provider, "default")

    @contextmanager
    def lifecycle_guard(
        self,
        account: AccountId,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> Iterator[int]:
        """Hold one account's stable lifecycle lock."""

        if timeout is not None:
            with _bounded_guard(
                lambda: self.lifecycle_guard(account, wait=False), timeout
            ) as home_fd:
                yield home_fd
            return
        with (
            self._home() as home_fd,
            self._lifecycle_guard(home_fd, account, wait=wait),
        ):
            yield home_fd

    @contextmanager
    def credential_guard(
        self,
        account: AccountId,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> Iterator[int]:
        """Hold registry then lifecycle locks for account publication."""

        if timeout is not None:
            with _bounded_guard(
                lambda: self.credential_guard(account, wait=False), timeout
            ) as home_fd:
                yield home_fd
            return
        with (
            self._home() as home_fd,
            self._lock_at(home_fd, ".accounts.lock", wait=wait),
            self._lifecycle_guard(home_fd, account, wait=wait),
        ):
            yield home_fd

    def _recover_account(
        self, home_fd: int, account: AccountId
    ) -> OAuthTokens | None:
        value = self._read_registry(home_fd)
        tokens = _read_account_tokens(home_fd, account)
        if str(account) not in value["accounts"] and tokens is not None:
            self._register_account(home_fd, account, value)
        return tokens

    def _register_account(
        self,
        home_fd: int,
        account: AccountId,
        value: dict[str, Any] | None = None,
    ) -> None:
        value = value or self._read_registry(home_fd)
        key = str(account)
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
            staged = _read_account_tokens(home_fd, account)
            if registered:
                if staged is None:
                    raise AccountError("OAuth credentials are invalid; login required")
                if legacy_exists:
                    _unlink_legacy(home_fd, legacy_name)
                return
            if staged is not None:
                self._register_account(home_fd, account, value)
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


class _CredentialDriver(Protocol):
    auth_error_type: type[RuntimeError]
    provider_label: str

    def bootstrap(self) -> OAuthTokens | None: ...

    def login_required_error(self, message: str) -> RuntimeError: ...

    async def refresh(
        self, refresh_token: str, client: httpx.AsyncClient
    ) -> OAuthTokens: ...

    def auth_headers(self, token: str) -> dict[str, str]: ...


class AccountCredentials:
    """Own one account's credential lifecycle behind a provider-facing seam."""

    def __init__(
        self, store: AccountStore, account: AccountId, driver: _CredentialDriver
    ) -> None:
        self.store = store
        self.account = account
        self._driver = driver
        self.path = store.credential_path(account)
        self.auth_error_type = driver.auth_error_type
        self.provider_label = driver.provider_label
        self._async_lock = asyncio.Lock()

    def read(self) -> OAuthTokens | None:
        with self.store.lifecycle_guard(self.account) as home_fd:
            return _read_account_tokens(home_fd, self.account)

    def save(
        self, tokens: OAuthTokens, *, lock_timeout: float | None = None
    ) -> None:
        with self.store.credential_guard(
            self.account, timeout=lock_timeout
        ) as home_fd:
            self._publish(home_fd, tokens)

    def bootstrap(self) -> OAuthTokens | None:
        with self.store.credential_guard(self.account) as home_fd:
            tokens = self.store._recover_account(home_fd, self.account)
            if tokens is not None:
                return tokens
            tokens = self._driver.bootstrap()
            if tokens is not None:
                self._publish(home_fd, tokens)
            return tokens

    async def access_token(self, client: httpx.AsyncClient) -> str:
        async with self._async_lock:
            if not self.store.contains(self.account):
                return await self._bootstrap_access_token(client)
            async with self._async_guard(registry=False) as home_fd:
                tokens = _read_account_tokens(home_fd, self.account)
                if tokens is None:
                    raise self._login_required_error()
                if tokens.is_valid():
                    return tokens.access_token
                return await self._refresh_locked(home_fd, tokens, client)

    async def refresh_token(self, client: httpx.AsyncClient) -> str:
        async with self._async_lock:
            if not self.store.contains(self.account):
                return await self._bootstrap_access_token(client, refresh=True)
            async with self._async_guard(registry=False) as home_fd:
                tokens = _read_account_tokens(home_fd, self.account)
                if tokens is None:
                    raise self._login_required_error()
                return await self._refresh_locked(home_fd, tokens, client)

    async def refresh(
        self, refresh_token: str, client: httpx.AsyncClient
    ) -> OAuthTokens:
        return await self._driver.refresh(refresh_token, client)

    def auth_headers(self, token: str) -> dict[str, str]:
        return self._driver.auth_headers(token)

    def beta_header(self) -> str:
        return self._driver.beta_header()  # type: ignore[attr-defined, no-any-return]

    async def _refresh_locked(
        self, home_fd: int, tokens: OAuthTokens, client: httpx.AsyncClient
    ) -> str:
        try:
            refreshed = await asyncio.wait_for(
                self.refresh(tokens.refresh_token, client),
                timeout=OAUTH_REFRESH_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise self.auth_error_type(
                f"{self.provider_label} OAuth token refresh stalled"
            ) from exc
        self._write(home_fd, refreshed)
        return refreshed.access_token

    async def _bootstrap_access_token(
        self, client: httpx.AsyncClient, *, refresh: bool = False
    ) -> str:
        async with self._async_guard(registry=True) as home_fd:
            tokens = self.store._recover_account(home_fd, self.account)
            if tokens is None:
                tokens = self._driver.bootstrap()
                if tokens is None:
                    raise self._login_required_error()
                self._publish(home_fd, tokens)
            if not refresh and tokens.is_valid():
                return tokens.access_token
        async with self._async_guard(registry=False) as home_fd:
            tokens = _read_account_tokens(home_fd, self.account)
            if tokens is None:
                raise self._login_required_error()
            return await self._refresh_locked(home_fd, tokens, client)

    def _write(self, home_fd: int, tokens: OAuthTokens) -> None:
        with _account_directory(home_fd, self.account) as account_fd:
            _write_tokens(account_fd, tokens)
            _verify_tokens(account_fd, tokens)

    def _publish(self, home_fd: int, tokens: OAuthTokens) -> None:
        self._write(home_fd, tokens)
        self.store._register_account(home_fd, self.account)

    def _login_required_error(self) -> RuntimeError:
        return self._driver.login_required_error(
            f"no {self.provider_label} OAuth login found; log in first"
        )

    @asynccontextmanager
    async def _async_guard(self, *, registry: bool) -> Iterator[int]:
        stack = ExitStack()
        guard = (
            self.store.credential_guard(
                self.account, timeout=OAUTH_REFRESH_TIMEOUT_SECONDS
            )
            if registry
            else self.store.lifecycle_guard(
                self.account, timeout=OAUTH_REFRESH_TIMEOUT_SECONDS
            )
        )
        try:
            home_fd = await asyncio.to_thread(stack.enter_context, guard)
        except AccountCredentialLockTimeoutError as exc:
            raise self.auth_error_type(
                f"{self.provider_label} OAuth credential lock stalled"
            ) from exc
        try:
            yield home_fd
        finally:
            await asyncio.shield(asyncio.to_thread(stack.close))


@contextmanager
def _bounded_guard(
    guard: Callable[[], AbstractContextManager[int]], timeout: float
) -> Iterator[int]:
    deadline = time.monotonic() + timeout
    while True:
        stack = ExitStack()
        try:
            home_fd = stack.enter_context(guard())
        except AccountError as exc:
            stack.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AccountCredentialLockTimeoutError(
                    "OAuth credential lock acquisition timed out"
                ) from exc
            time.sleep(min(0.05, remaining))
        else:
            break
    try:
        yield home_fd
    finally:
        stack.close()


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


def _read_account_tokens(home_fd: int, account: AccountId) -> OAuthTokens | None:
    try:
        with _open_account_directory(home_fd, account) as account_fd:
            if not _exists(account_fd, "oauth.json"):
                return None
            return _read_tokens(account_fd, "oauth.json")
    except FileNotFoundError:
        return None
    except AccountError:
        raise
    except OSError as exc:
        raise AccountError("OAuth credentials are invalid; login required") from exc


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
