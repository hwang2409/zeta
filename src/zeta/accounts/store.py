"""Private storage implementation for named subscription accounts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import (
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
    contextmanager,
)
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..core.session_files import (
    child_directory,
    open_session_file,
    read_session_file,
    require_off_loop_lock,
)
from ..oauth import (
    ACCOUNT_LOCK_TURN_WAIT_SECONDS,
    OAUTH_REFRESH_TIMEOUT_SECONDS,
    OAuthRefreshFailure,
    OAuthTokens,
    classify_oauth_refresh_failure,
    credential_file_lock,
)
from ..storage_format import (
    DeleteValue,
    SetValue,
    open_writer,
)
from . import availability
from .files import (
    exists as _exists,
)
from .files import fsync_directory as _fsync_directory
from .files import mkdir_child_durable as _mkdir_child_durable
from .files import owned_fd as _owned_fd
from .files import (
    read_optional as _read_optional,
)
from .files import (
    read_optional_bytes as _read_optional_bytes,
)
from .publication import atomic_publish, publish_mapping
from .registry import (
    _PROVIDERS,
    _UNSAFE_REGISTRY_WARNINGS,
    AccountError,
    AccountId,
    _account_label,
    _AccountRegistry,
    _publication_record,
    _read_publication_pending,
    _Registry,
    _registry_reference_index,
)

logger = logging.getLogger(__name__)


def credential_wait_timeout_seconds() -> float:
    # Leave one refresh-sized interval for durable publication and lock release.
    return OAUTH_REFRESH_TIMEOUT_SECONDS * 2


class AccountCredentialLockTimeoutError(AccountError):
    """An account operation could not acquire its lifecycle lock in time."""

    code = "account_lock_timeout"
    retryable = True

    def __init__(self, account: AccountId | None = None) -> None:
        self.account = account
        message = "another process holds the account lock"
        if account is not None:
            message = f"cannot access {account}: {message}; retry the turn"
        super().__init__(message)


class AccountLoopThreadLockError(AccountError):
    """A blocking account lock was requested on an asyncio loop thread."""


class AccountStore(_AccountRegistry):
    """Own the durable account catalog, credentials, and process locks."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.registry = self.home / "accounts.json"
        self.accounts = self.home / "accounts"

    @contextmanager
    def _home(self) -> Iterator[int]:
        lease = open_writer(self.home)
        os.chmod(self.home, 0o700)
        fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            yield fd
        finally:
            os.close(fd)
            lease.close()

    @contextmanager
    def _lock_at(
        self, directory_fd: int, name: str, *, wait: bool = True
    ) -> Iterator[None]:
        if wait:
            require_off_loop_lock("blocking account", AccountLoopThreadLockError)
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
    def usage_guard(self, account: AccountId) -> Iterator[None]:
        with (
            self._home() as home_fd,
            _account_lock_directory(home_fd, account) as lock_fd,
            self._lock_at(lock_fd, "usage.lock"),
        ):
            yield

    def _read_usage_bytes(self, account: AccountId) -> bytes | None:
        with self._home() as home_fd:
            try:
                with _open_account_directory(home_fd, account) as account_fd:
                    return _read_optional_bytes(account_fd, "usage.json")
            except FileNotFoundError:
                return None

    def _write_usage_bytes(self, account: AccountId, data: bytes) -> None:
        with (
            self._home() as home_fd,
            _account_directory(home_fd, account) as account_fd,
        ):
            publish_mapping(
                self.home, account_fd, "usage.json", json.loads(data), atomic_publish
            )
            _fsync_directory(account_fd)

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

    def list_catalog(self) -> list[availability.AccountDisplay]:
        with self._home() as home_fd:
            value = self._read_registry(home_fd)
        return availability.catalog_display(value)

    def list_display(self) -> list[availability.AccountDisplay]:
        return availability.list_display(self)

    def require_available(self, account: AccountId) -> None:
        if not any(
            item.account == account and item.available for item in self.list_display()
        ):
            raise AccountError(
                f"account {account} is unavailable: credentials missing; "
                f"run 'zeta accounts login {account.provider} {account.name} --replace'"
            )

    def default(self, provider: str) -> AccountId | None:
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        with self._home() as home_fd:
            value = self._read_registry(home_fd)["defaults"].get(provider)
            return AccountId(*value.split(":", 1)) if value else None

    def contains(self, account: AccountId, *, timeout: float | None = None) -> bool:
        with self.credential_guard(
            account,
            timeout=credential_wait_timeout_seconds() if timeout is None else timeout,
        ) as home_fd:
            return str(account) in self._read_registry(home_fd)["accounts"]

    def set_default(self, account: AccountId) -> None:
        with self._home() as home_fd, self._registry_guard(home_fd):
            value = self._read_registry(home_fd)
            if str(account) not in value["accounts"]:
                raise AccountError(f"account {account} does not exist; use /accounts")
            value["defaults"][account.provider] = str(account)
            self._write_registry(home_fd, value)

    def rename(self, account: AccountId, new_label: str) -> str:
        label = _account_label(new_label)
        with self._home() as home_fd, self._registry_guard(home_fd):
            value = self._read_registry(home_fd)
            key = str(account)
            if key not in value["accounts"]:
                raise AccountError(f"account {account} does not exist")
            if value["labels"].get(key, account.name) == label:
                return "account label unchanged"
            self._check_reference_available(value, account, label)
            if label == account.name:
                value["labels"].pop(key, None)
            else:
                value["labels"][key] = label
            self._write_registry(home_fd, value)
        return f"renamed {account} to {account.provider}:{label}"

    def logout(self, account: AccountId, *, sessions_root: Path) -> str:
        """Remove one local account while no live session pins it."""

        from .logout import (
            remove_account_directory,
            remove_account_lock_directory,
            validate_account_directory,
        )

        with self._home() as home_fd, self._registry_guard(home_fd):
            value = self._read_registry(home_fd)
            key = str(account)
            if key not in value["accounts"]:
                raise AccountError(f"account {account} does not exist")
            with self._lifecycle_guard(home_fd, account):
                if _live_session_pins(sessions_root, account):
                    raise AccountError(
                        f"account {account} is pinned by a live session; close it first"
                    )
                remove_account_lock_directory(home_fd, account, remove=False)
                validate_account_directory(
                    home_fd,
                    "accounts",
                    account,
                    allowed={"oauth.json", "usage.json", "publish-pending.json"},
                )
                cleared_default = value["defaults"].pop(account.provider, None) == key
                value["accounts"].pop(key)
                value["labels"].pop(key, None)
                self._write_registry(home_fd, value)
                remove_account_directory(
                    home_fd,
                    "accounts",
                    account,
                    allowed={"oauth.json", "usage.json", "publish-pending.json"},
                )
                remove_account_lock_directory(home_fd, account, remove=True)
        suffix = f"; cleared {account.provider} default" if cleared_default else ""
        return f"logged out {account}{suffix}"

    def resolve_qualified_reference(self, value: str) -> AccountId:
        try:
            provider, reference = value.split(":", 1)
        except (AttributeError, ValueError) as exc:
            raise AccountError("account reference must be provider:name") from exc
        return self.resolve_reference(provider, reference)

    def resolve_reference(self, provider: str, name_or_label: str) -> AccountId:
        reference = AccountId(provider, name_or_label)
        with self._home() as home_fd, self._registry_guard(home_fd):
            resolved = _registry_reference_index(self._read_registry(home_fd)).get(
                (provider, reference.name)
            )
        if resolved is not None:
            return resolved
        raise AccountError(f"account {reference} does not exist")

    @staticmethod
    def _check_reference_available(
        value: dict[str, Any], account: AccountId, label: str
    ) -> None:
        owner = _registry_reference_index(value).get((account.provider, label))
        if owner is not None and owner != account:
            raise AccountError(f"account {account.provider}:{label} already exists")

    def credential_path(self, account: AccountId) -> Path:
        return self.accounts / account.provider / account.name / "oauth.json"

    def resolve_account(
        self, provider: str, account: AccountId | None = None
    ) -> AccountId:
        """Resolve an account or the global default."""

        provider = "anthropic" if provider == "claude" else provider
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        if account is not None and account.provider != provider:
            raise ValueError("account provider does not match")
        resolved = account or self.default(provider) or AccountId(provider, "default")
        with self.credential_guard(
            resolved, timeout=credential_wait_timeout_seconds()
        ) as home_fd:
            self._recover_account(home_fd, resolved)
        return resolved

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
            try:
                with _bounded_guard(
                    lambda: self.lifecycle_guard(account, wait=False), timeout
                ) as home_fd:
                    yield home_fd
            except AccountCredentialLockTimeoutError as exc:
                raise AccountCredentialLockTimeoutError(account) from exc
            return
        with (
            self._home() as home_fd,
            self._lifecycle_guard(home_fd, account, wait=wait),
        ):
            yield home_fd

    @contextmanager
    def pin_guard(self, accounts: Mapping[str, AccountId]) -> Iterator[None]:
        """Hold registry then account locks while durable pins are published."""

        ordered = sorted(
            set(accounts.values()), key=lambda item: (item.provider, item.name)
        )
        with (
            self._home() as home_fd,
            self._registry_guard(home_fd),
            ExitStack() as stack,
        ):
            for account in ordered:
                stack.enter_context(self._lifecycle_guard(home_fd, account))
            registered = self._read_registry(home_fd)["accounts"]
            if any(str(account) not in registered for account in ordered):
                raise AccountError("account no longer exists; use /accounts")
            yield

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
            try:
                with _bounded_guard(
                    lambda: self.credential_guard(account, wait=False), timeout
                ) as home_fd:
                    yield home_fd
            except AccountCredentialLockTimeoutError as exc:
                raise AccountCredentialLockTimeoutError(account) from exc
            return
        with (
            self._home() as home_fd,
            self._lock_at(home_fd, ".accounts.lock", wait=wait),
            self._lifecycle_guard(home_fd, account, wait=wait),
        ):
            yield home_fd

    def _recover_account(self, home_fd: int, account: AccountId) -> OAuthTokens | None:
        value = self._read_registry(home_fd)
        registered = str(account) in value["accounts"]
        try:
            with _open_account_directory(home_fd, account) as account_fd:
                pending = _read_publication_pending(account_fd)
                if registered:
                    tokens = _read_optional_tokens(account_fd)
                    if pending is not None:
                        _delete_publication_pending(account_fd)
                    return tokens
                if pending is None:
                    return None
                credential = _read_optional_bytes(account_fd, "oauth.json")
                if credential is None or pending != _publication_record(
                    account, credential
                ):
                    _delete_publication_pending(account_fd)
                    return None
                tokens = _tokens_from_bytes(credential)
                self._register_account(home_fd, account, value)
                _delete_publication_pending(account_fd)
                return tokens
        except FileNotFoundError:
            return None

    def _register_account(
        self,
        home_fd: int,
        account: AccountId,
        value: dict[str, Any] | None = None,
    ) -> None:
        value = value or self._read_registry(home_fd)
        self._check_reference_available(value, account, account.name)
        key = str(account)
        self._refuse_orphaning_registration(home_fd, account)
        value["accounts"][key] = {
            "provider": account.provider,
            "name": account.name,
        }
        value["defaults"].setdefault(account.provider, key)
        self._write_registry(home_fd, value)

    def migrate(self) -> tuple[str, ...]:
        warnings: list[str] = []
        with self._home() as home_fd:
            if not any(
                _exists(home_fd, name)
                for name in (
                    "anthropic-oauth.json",
                    "codex-oauth.json",
                    "accounts.json",
                    "accounts",
                    "account-locks",
                )
            ):
                return ()
            with self._lock_at(home_fd, ".accounts-migration.lock"):
                with self._registry_guard(home_fd):
                    state, value = self._load_registry(home_fd)
                    if state is _Registry.LOADED:
                        self._sweep_orphan_directories(
                            home_fd, set(value["accounts"]), warnings
                        )
                    elif (
                        state is _Registry.BROKEN
                        or self._has_unregistered_account_directories(home_fd, set())
                    ):
                        warnings.append(_UNSAFE_REGISTRY_WARNINGS[state])
                    if state is _Registry.BROKEN:
                        # Not logged: every later registry read raises this
                        # same line, so the CLI would print it twice.
                        return tuple(warnings)
                    if state is _Registry.MISSING and warnings:
                        logger.warning("%s", warnings[-1])
                for provider in ("anthropic", "codex"):
                    self._migrate_provider(home_fd, provider, warnings)
        return tuple(warnings)

    def refuse_orphaning_registration(self, account: AccountId) -> None:
        """Fail before a login starts if its registration would be refused."""

        with self._home() as home_fd:
            self._refuse_orphaning_registration(home_fd, account)

    def _migrate_provider(
        self, home_fd: int, provider: str, warnings: list[str]
    ) -> None:
        account = AccountId(provider, "default")
        legacy_name = f"{provider}-oauth.json"
        legacy_path = self.home / legacy_name
        with (
            self._registry_guard(home_fd),
            self._lifecycle_guard(home_fd, account),
            credential_file_lock(legacy_path),
        ):
            if self._registering_would_orphan(home_fd, account):
                return
            if not _exists_regular(home_fd, legacy_name):
                self._recover_account(home_fd, account)
                return
            try:
                self._recover_account(home_fd, account)
            except AccountError:
                pass  # an unreadable account file: _keep_token keeps it
            try:
                with _open_account_directory(home_fd, account) as account_fd:
                    account_tokens = _tokens_or_unreadable(account_fd, "oauth.json")
            except FileNotFoundError:
                account_tokens = None
            keep = _keep_token(
                account_tokens, _tokens_or_unreadable(home_fd, legacy_name)
            )
            if keep is _Keep.BOTH:
                warnings.append(
                    f"kept {self.credential_path(account)} and {legacy_path}: "
                    "could not tell which login is newer; delete the one you "
                    "do not want, then restart zeta"
                )
                logger.warning("%s", warnings[-1])
                return
            with _account_directory(home_fd, account) as account_fd:
                if keep is _Keep.LEGACY:
                    tokens = _read_tokens(home_fd, legacy_name)
                    self._stage_account_tokens(account_fd, account, tokens)
                if str(account) not in self._read_registry(home_fd)["accounts"]:
                    self._register_account(home_fd, account)
                _delete_publication_pending(account_fd)
            _unlink_legacy(home_fd, legacy_name)

    def _stage_account_tokens(
        self, account_fd: int, account: AccountId, tokens: OAuthTokens
    ) -> None:
        tokens = _effective_tokens(account_fd, tokens)
        publish_mapping(
            self.home, account_fd, "oauth.json", tokens.to_dict(), atomic_publish
        )
        credential = _read_optional_bytes(account_fd, "oauth.json")
        if credential is None:
            raise AccountError("OAuth credentials are invalid; login required")
        publish_mapping(
            self.home,
            account_fd,
            "publish-pending.json",
            _publication_record(account, credential),
            atomic_publish,
        )
        _fsync_directory(account_fd)
        _verify_tokens(account_fd, tokens)

    def _write_registry(self, home_fd: int, value: dict[str, Any]) -> None:
        try:
            _registry_reference_index(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise AccountError(
                "account registry change is invalid; accounts.json was not changed"
            ) from exc
        raw = _read_optional(home_fd, "accounts.json")
        current = json.loads(raw) if raw is not None else {}
        mutations = [
            SetValue(("version",), 1),
            SetValue(("accounts",), value["accounts"]),
            SetValue(("defaults",), value["defaults"]),
            SetValue(("labels",), value["labels"]),
        ]
        if isinstance(current, Mapping):
            for field in ("accounts", "defaults", "labels"):
                records = current.get(field)
                if not isinstance(records, Mapping):
                    continue
                for key in records:
                    if key not in value[field] and self._owns_registry_key(field, key):
                        mutations.append(DeleteValue((field, key)))
        with open_writer(self.home) as lease:
            lease.publish_json(
                home_fd,
                "accounts.json",
                mutations,
                atomic_publish,
            )
        _fsync_directory(home_fd)

    @staticmethod
    def _owns_registry_key(field: str, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if field == "defaults":
            return key in _PROVIDERS
        try:
            account = AccountId(*key.split(":", 1))
        except (TypeError, ValueError):
            return False
        return key == str(account)


def rename_account_command(store: AccountStore, args: str) -> str:
    requested = args.split()
    if len(requested) != 3 or requested[0] != "rename":
        return "usage: /accounts [rename provider:old new-label]"
    try:
        account = store.resolve_qualified_reference(requested[1])
        return store.rename(account, requested[2])
    except (AccountError, OSError, ValueError) as exc:
        return f"account unchanged: {exc}"


class _CredentialDriver(Protocol):
    auth_error_type: type[RuntimeError]
    http_error_type: type[RuntimeError]
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
        self.http_error_type = driver.http_error_type
        self.provider_label = driver.provider_label
        self._async_lock = asyncio.Lock()

    def read(self, *, lock_timeout: float | None = None) -> OAuthTokens | None:
        with self.store.lifecycle_guard(
            self.account,
            timeout=credential_wait_timeout_seconds()
            if lock_timeout is None
            else lock_timeout,
        ) as home_fd:
            return _read_account_tokens(home_fd, self.account)

    def save(
        self,
        tokens: OAuthTokens,
        *,
        lock_timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        with self.store.credential_guard(self.account, timeout=lock_timeout) as home_fd:
            if cancel_event is None or not cancel_event.is_set():
                self._publish(home_fd, tokens)

    def bootstrap(self) -> OAuthTokens | None:
        with self.store.credential_guard(
            self.account, timeout=credential_wait_timeout_seconds()
        ) as home_fd:
            tokens = self.store._recover_account(home_fd, self.account)
            if tokens is not None:
                return tokens
            tokens = self._driver.bootstrap()
            if tokens is not None:
                self._publish(home_fd, tokens)
            return tokens

    async def access_token(
        self, client: httpx.AsyncClient, *, lock_timeout: float | None = None
    ) -> str:
        """Return a valid access token, refreshing it under the account lock.

        Each lock wait defaults to the model-turn bound; callers outside a
        turn's retry loop pass ``credential_wait_timeout_seconds()``.
        """
        timeout = ACCOUNT_LOCK_TURN_WAIT_SECONDS if lock_timeout is None else lock_timeout
        async with self._async_lock:
            if not await self._contains(timeout):
                return await self._bootstrap_access_token(client, timeout)
            async with self._async_guard(registry=False, timeout=timeout) as home_fd:
                tokens = _read_account_tokens(home_fd, self.account)
                if tokens is None:
                    raise self.login_required_error()
                if tokens.is_valid():
                    return tokens.access_token
                return await self._refresh_locked(home_fd, tokens, client)

    async def refresh_token(self, client: httpx.AsyncClient) -> str:
        timeout = ACCOUNT_LOCK_TURN_WAIT_SECONDS
        async with self._async_lock:
            if not await self._contains(timeout):
                return await self._bootstrap_access_token(client, timeout, refresh=True)
            return await self._refresh_existing_token(client, timeout)

    async def refresh_existing_token(self, client: httpx.AsyncClient) -> str:
        """Refresh a catalog account without registry lookup or bootstrap."""
        async with self._async_lock:
            return await self._refresh_existing_token(
                client, credential_wait_timeout_seconds()
            )

    async def _contains(self, timeout: float) -> bool:
        return await asyncio.to_thread(self.store.contains, self.account, timeout=timeout)

    async def _refresh_existing_token(
        self, client: httpx.AsyncClient, timeout: float
    ) -> str:
        async with self._async_guard(registry=False, timeout=timeout) as home_fd:
            tokens = _read_account_tokens(home_fd, self.account)
            if tokens is None:
                raise self.login_required_error()
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
            outcome = classify_oauth_refresh_failure(exception=exc)
            raise self.http_error_type(
                f"{self.provider_label} OAuth token refresh stalled",
                retryable=outcome is OAuthRefreshFailure.TRANSIENT,
            ) from exc
        self._write(home_fd, refreshed)
        return refreshed.access_token

    async def _bootstrap_access_token(
        self,
        client: httpx.AsyncClient,
        timeout: float,
        *,
        refresh: bool = False,
    ) -> str:
        async with self._async_guard(registry=True, timeout=timeout) as home_fd:
            tokens = self.store._recover_account(home_fd, self.account)
            if tokens is None:
                tokens = self._driver.bootstrap()
                if tokens is None:
                    raise self.login_required_error()
                self._publish(home_fd, tokens)
            if not refresh and tokens.is_valid():
                return tokens.access_token
        async with self._async_guard(registry=False, timeout=timeout) as home_fd:
            tokens = _read_account_tokens(home_fd, self.account)
            if tokens is None:
                raise self.login_required_error()
            return await self._refresh_locked(home_fd, tokens, client)

    def _write(self, home_fd: int, tokens: OAuthTokens) -> None:
        with _account_directory(home_fd, self.account) as account_fd:
            tokens = _effective_tokens(account_fd, tokens)
            publish_mapping(
                self.store.home,
                account_fd,
                "oauth.json",
                tokens.to_dict(),
                atomic_publish,
            )
            _fsync_directory(account_fd)
            _verify_tokens(account_fd, tokens)

    def _publish(self, home_fd: int, tokens: OAuthTokens) -> None:
        value = self.store._read_registry(home_fd)
        self.store._check_reference_available(value, self.account, self.account.name)
        registered = str(self.account) in value["accounts"]
        if registered:
            self._write(home_fd, tokens)
            return
        self.store._refuse_orphaning_registration(home_fd, self.account)
        with _account_directory(home_fd, self.account) as account_fd:
            self.store._stage_account_tokens(account_fd, self.account, tokens)
        self.store._register_account(home_fd, self.account, value)
        with _open_account_directory(home_fd, self.account) as account_fd:
            _delete_publication_pending(account_fd)

    def login_required_error(self) -> RuntimeError:
        """Return the provider driver's missing-login error."""
        return self._driver.login_required_error(
            f"no {self.provider_label} OAuth login found; log in first"
        )

    @asynccontextmanager
    async def _async_guard(self, *, registry: bool, timeout: float) -> Iterator[int]:
        stack = ExitStack()
        guard = (
            self.store.credential_guard(self.account, timeout=timeout)
            if registry
            else self.store.lifecycle_guard(self.account, timeout=timeout)
        )
        home_fd = await asyncio.to_thread(stack.enter_context, guard)
        try:
            yield home_fd
        finally:
            await asyncio.shield(asyncio.to_thread(stack.close))


@contextmanager
def _bounded_guard(
    guard: Callable[[], AbstractContextManager[int]], timeout: float
) -> Iterator[int]:
    require_off_loop_lock("bounded account", AccountLoopThreadLockError)
    deadline = time.monotonic() + timeout
    while True:
        stack = ExitStack()
        try:
            home_fd = stack.enter_context(guard())
        except AccountError as exc:
            stack.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AccountCredentialLockTimeoutError() from exc
            time.sleep(min(0.05, remaining))
        else:
            break
    try:
        yield home_fd
    finally:
        stack.close()


def _read_child_account_pins(path: Path) -> Mapping[str, Any]:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        lifecycle = json.loads(read_session_file(directory_fd, "agent_lifecycle.json"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AccountError("could not verify live child account pins") from exc
    finally:
        os.close(directory_fd)
    if not isinstance(lifecycle, Mapping) or not isinstance(
        lifecycle.get("selected_accounts"), Mapping
    ):
        raise AccountError("could not verify live child account pins")
    return lifecycle["selected_accounts"]


def _live_child_pins(session_path: Path, account: AccountId) -> bool:
    from ..session_liveness import session_is_live

    agents = session_path / "agents"
    try:
        agents_info = agents.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(agents_info.st_mode):
        return False
    for root, directories, files in os.walk(agents, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(root) / name).is_symlink()
        ]
        if "agent_lifecycle.json" not in files or not session_is_live(Path(root)):
            continue
        pins = _read_child_account_pins(Path(root))
        if pins.get(account.provider) == str(account):
            return True
    return False


def _live_session_pins(sessions_root: Path, account: AccountId) -> bool:
    from ..core.session import SessionError, SessionManager
    from ..session_liveness import session_is_live

    try:
        entries = list(os.scandir(sessions_root))
    except FileNotFoundError:
        return False
    manager = SessionManager(sessions_root.parent)
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        path = sessions_root / entry.name
        if session_is_live(path):
            try:
                metadata = manager.read_metadata(entry.name)
            except (OSError, SessionError, ValueError) as exc:
                raise AccountError(
                    "could not verify live session account pins"
                ) from exc
            if metadata.selected_accounts.get(account.provider) == str(account):
                return True
        if _live_child_pins(path, account):
            return True
    return False


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


def _token_bytes(tokens: OAuthTokens) -> bytes:
    return (json.dumps(tokens.to_dict(), separators=(",", ":")) + "\n").encode()


def _delete_publication_pending(directory_fd: int) -> None:
    try:
        os.unlink("publish-pending.json", dir_fd=directory_fd)
    except FileNotFoundError:
        return
    _fsync_directory(directory_fd)


def _effective_tokens(directory_fd: int, tokens: OAuthTokens) -> OAuthTokens:
    current = _read_optional_bytes(directory_fd, "oauth.json")
    if current is None or (tokens.email is not None and tokens.workspace is not None):
        return tokens
    previous = _tokens_from_bytes(current)
    return OAuthTokens(
        tokens.access_token,
        tokens.refresh_token,
        tokens.expires_at,
        tokens.email or previous.email,
        tokens.workspace or previous.workspace,
    )


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
        credential = _read_optional_bytes(directory_fd, name)
        if credential is None:
            raise FileNotFoundError(name)
        return _tokens_from_bytes(credential)
    except AccountError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AccountError("OAuth credentials are invalid; login required") from exc


class _Unreadable(Enum):
    TOKENS = "unreadable"


class _Keep(Enum):
    """Which login migration keeps when legacy and account credentials meet."""

    LEGACY = "legacy"
    ACCOUNT = "account"
    BOTH = "both"


def _tokens_or_unreadable(
    directory_fd: int, name: str
) -> OAuthTokens | None | _Unreadable:
    try:
        credential = _read_optional_bytes(directory_fd, name)
        return None if credential is None else _tokens_from_bytes(credential)
    except (AccountError, OSError):
        return _Unreadable.TOKENS


def _keep_token(
    account: OAuthTokens | None | _Unreadable, legacy: OAuthTokens | None | _Unreadable
) -> _Keep:
    """Choose between accounts/<p>/default/oauth.json and a legacy <p>-oauth.json.

    The newer login has the later expires_at: every refresh moves it forward.
    mtime is not used because copying or restoring a home resets mtimes, so an
    older token can carry the newer mtime. A missing account file lets the
    legacy login win; anything that cannot prove which is newer (a tie, an
    unreadable file, a missing expires_at, which reads as 0) keeps both files.
    """

    if account is None:
        return _Keep.LEGACY
    # compare parsed tokens only: two unreadable files are equal sentinels
    if not isinstance(account, OAuthTokens) or not isinstance(legacy, OAuthTokens):
        return _Keep.BOTH
    if account == legacy:
        return _Keep.ACCOUNT
    if not min(account.expires_at, legacy.expires_at) > 0:
        return _Keep.BOTH
    if legacy.expires_at > account.expires_at:
        return _Keep.LEGACY
    if account.expires_at > legacy.expires_at:
        return _Keep.ACCOUNT
    return _Keep.BOTH


def _read_optional_tokens(directory_fd: int) -> OAuthTokens | None:
    credential = _read_optional_bytes(directory_fd, "oauth.json")
    return None if credential is None else _tokens_from_bytes(credential)


def _tokens_from_bytes(credential: bytes) -> OAuthTokens:
    try:
        value = json.loads(credential)
        if not isinstance(value, Mapping):
            raise TypeError
        return OAuthTokens.from_mapping(value, error_type=AccountError)
    except AccountError:
        raise AccountError("OAuth credentials are invalid; login required") from None
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AccountError("OAuth credentials are invalid; login required") from exc


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
