"""Private storage implementation for named subscription accounts."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
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


def _account_label(value: str) -> str:
    if type(value) is not str:
        raise ValueError("invalid account label")
    label = value.lower()
    if not _SLUG.fullmatch(label):
        raise ValueError("invalid account label")
    return label


def _registry_reference_index(value: Mapping[str, Any]) -> dict[tuple[str, str], AccountId]:
    accounts = value["accounts"]
    labels = value["labels"]
    references: dict[tuple[str, str], AccountId] = {}
    for key, metadata in accounts.items():
        if not isinstance(key, str) or not isinstance(metadata, Mapping):
            raise TypeError
        account = AccountId(*key.split(":", 1))
        if key != str(account) or metadata != {
            "provider": account.provider,
            "name": account.name,
        }:
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


@dataclass(frozen=True, slots=True)
class AccountDisplay:
    account: AccountId
    label: str
    detail: str | None = None
    usage: Any = None

    def __str__(self) -> str:
        return f"{self.account.provider}:{self.label}"


class AccountStore:
    """Own the durable account catalog, credentials, and process locks."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.registry = self.home / "accounts.json"
        self.accounts = self.home / "accounts"

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
        with self._home() as home_fd, _account_directory(home_fd, account) as account_fd:
            _atomic_publish(account_fd, "usage.json", data)
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

    def list_display(self) -> list[AccountDisplay]:
        result = []
        with self._home() as home_fd:
            value = self._read_registry(home_fd)
            for key in value["accounts"]:
                account = AccountId(*key.split(":", 1))
                tokens = _read_account_tokens(home_fd, account)
                details = (
                    tuple(filter(None, (tokens.email, tokens.workspace)))
                    if tokens
                    else ()
                )
                from .usage import read_usage

                result.append(
                    AccountDisplay(
                        account,
                        value["labels"].get(key, account.name),
                        " · ".join(details) or None,
                        read_usage(self, account),
                    )
                )
        return result

    def default(self, provider: str) -> AccountId | None:
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        with self._home() as home_fd:
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
        """Migrate legacy auth, then resolve an account or the global default."""

        provider = "anthropic" if provider == "claude" else provider
        if provider not in _PROVIDERS:
            raise ValueError("invalid account provider")
        if account is not None and account.provider != provider:
            raise ValueError("account provider does not match")
        self.migrate()
        resolved = account or self.default(provider) or AccountId(provider, "default")
        with self.credential_guard(resolved) as home_fd:
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
                if (
                    credential is None
                    or pending != _publication_record(account, credential)
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
            staged = self._recover_account(home_fd, account)
            if registered or staged is not None:
                if staged is None:
                    raise AccountError("OAuth credentials are invalid; login required")
                if legacy_exists:
                    with _open_account_directory(home_fd, account) as account_fd:
                        if _file_mtime_ns(home_fd, legacy_name) > _file_mtime_ns(
                            account_fd, "oauth.json"
                        ):
                            tokens = _read_tokens(home_fd, legacy_name)
                            _stage_account_tokens(account_fd, account, tokens)
                            _delete_publication_pending(account_fd)
                    _unlink_legacy(home_fd, legacy_name)
                return
            if not legacy_exists:
                return
            tokens = _read_tokens(home_fd, legacy_name)
            with _account_directory(home_fd, account) as account_fd:
                _stage_account_tokens(account_fd, account, tokens)
            self._register_account(home_fd, account, value)
            with _open_account_directory(home_fd, account) as account_fd:
                _delete_publication_pending(account_fd)
            _unlink_legacy(home_fd, legacy_name)

    def _read_registry(self, home_fd: int) -> dict[str, Any]:
        try:
            raw = _read_optional(home_fd, "accounts.json")
            if raw is None:
                return {"version": 1, "accounts": {}, "defaults": {}, "labels": {}}
            value = json.loads(raw)
            if (
                not isinstance(value, Mapping)
                or value.get("version") != 1
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
                if (
                    not isinstance(provider, str)
                    or not isinstance(key, str)
                    or provider not in _PROVIDERS
                    or key not in accounts
                    or not key.startswith(f"{provider}:")
                ):
                    raise ValueError
            return normalized
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AccountError("accounts.json is invalid; use /accounts") from exc

    def _write_registry(self, home_fd: int, value: dict[str, Any]) -> None:
        try:
            _registry_reference_index(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise AccountError("accounts.json is invalid; use /accounts") from exc
        data = (
            json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        _atomic_publish(home_fd, "accounts.json", data)
        _fsync_directory(home_fd)


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
        self,
        tokens: OAuthTokens,
        *,
        lock_timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        with self.store.credential_guard(
            self.account, timeout=lock_timeout
        ) as home_fd:
            if cancel_event is None or not cancel_event.is_set():
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

    async def _access_token_locked(
        self, home_fd: int, client: httpx.AsyncClient
    ) -> str:
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
        if not refreshed.email or not refreshed.workspace:
            refreshed = OAuthTokens(
                refreshed.access_token,
                refreshed.refresh_token,
                refreshed.expires_at,
                refreshed.email or tokens.email,
                refreshed.workspace or tokens.workspace,
            )
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
        value = self.store._read_registry(home_fd)
        self.store._check_reference_available(value, self.account, self.account.name)
        registered = str(self.account) in value["accounts"]
        if registered:
            self._write(home_fd, tokens)
            return
        credential = _token_bytes(tokens)
        with _account_directory(home_fd, self.account) as account_fd:
            _write_publication_pending(account_fd, self.account, credential)
            _write_token_bytes(account_fd, credential)
            _verify_tokens(account_fd, tokens)
        self.store._register_account(home_fd, self.account, value)
        with _open_account_directory(home_fd, self.account) as account_fd:
            _delete_publication_pending(account_fd)

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


def _token_bytes(tokens: OAuthTokens) -> bytes:
    return (json.dumps(tokens.to_dict(), separators=(",", ":")) + "\n").encode()


def _write_tokens(directory_fd: int, tokens: OAuthTokens) -> None:
    _write_token_bytes(directory_fd, _token_bytes(tokens))


def _write_token_bytes(directory_fd: int, credential: bytes) -> None:
    _atomic_publish(directory_fd, "oauth.json", credential)
    _fsync_directory(directory_fd)


def _stage_account_tokens(
    account_fd: int, account: AccountId, tokens: OAuthTokens
) -> None:
    credential = _token_bytes(tokens)
    _write_publication_pending(account_fd, account, credential)
    _write_token_bytes(account_fd, credential)
    _verify_tokens(account_fd, tokens)


def _publication_record(account: AccountId, credential: bytes) -> dict[str, str]:
    return {
        "provider": account.provider,
        "name": account.name,
        "sha256": hashlib.sha256(credential).hexdigest(),
    }


def _write_publication_pending(
    directory_fd: int, account: AccountId, credential: bytes
) -> None:
    data = (
        json.dumps(
            _publication_record(account, credential), separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode()
    _atomic_publish(directory_fd, "publish-pending.json", data)
    _fsync_directory(directory_fd)


def _read_publication_pending(directory_fd: int) -> dict[str, str] | None:
    raw = _read_optional(directory_fd, "publish-pending.json")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"provider", "name", "sha256"}
            or value["provider"] not in _PROVIDERS
            or not isinstance(value["name"], str)
            or not _SLUG.fullmatch(value["name"])
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        ):
            raise ValueError
        return dict(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _delete_publication_pending(directory_fd: int) -> None:
    try:
        os.unlink("publish-pending.json", dir_fd=directory_fd)
    except FileNotFoundError:
        return
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
        credential = _read_optional_bytes(directory_fd, name)
        if credential is None:
            raise FileNotFoundError(name)
        return _tokens_from_bytes(credential)
    except AccountError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AccountError("OAuth credentials are invalid; login required") from exc


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


def _read_optional_bytes(directory_fd: int, name: str) -> bytes | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


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


def _file_mtime_ns(directory_fd: int, name: str) -> int:
    fd = open_session_file(directory_fd, name, os.O_RDONLY)
    try:
        return os.fstat(fd).st_mtime_ns
    finally:
        os.close(fd)


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
