"""Own authentication and account state for one loaded session."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..oauth import OAUTH_REFRESH_TIMEOUT_SECONDS
from .exhaustion import classify_subscription_limit, recovery_alternatives
from .store import AccountDisplay, AccountError, AccountId, AccountStore
from .usage import read_usage, record_subscription_limit

if TYPE_CHECKING:
    from ..core.session import SessionManager, SessionMetadata


ACCOUNT_SWITCH_LOCK_TIMEOUT_SECONDS = OAUTH_REFRESH_TIMEOUT_SECONDS
logger = logging.getLogger(__name__)


async def _shielded(operation: Any) -> Any:
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:  # noqa: BLE001, S110 - caller cancellation wins
            pass
        raise


def account_provider(provider: str) -> str | None:
    if provider == "claude":
        return "anthropic"
    return provider if provider in {"anthropic", "codex"} else None


def api_key_mode(provider: str) -> bool:
    return provider in {"claude", "anthropic"} and os.environ.get(
        "ZETA_ALLOW_API_KEY"
    ) == "1"


@dataclass(frozen=True, slots=True)
class AccountListSnapshot:
    displays: tuple[AccountDisplay, ...]
    selected: dict[str, AccountId]
    defaults: dict[str, AccountId]


@dataclass(frozen=True, slots=True)
class _PreparedAccountState:
    previous_metadata: SessionMetadata
    metadata: SessionMetadata
    pins: dict[str, AccountId]
    switch_sequences: dict[str, int]
    active_provider: str | None
    backend_state: Any


@dataclass(frozen=True, slots=True)
class _UnknownAccountState:
    build_backend: Callable[[SessionMetadata], Any]


@dataclass(frozen=True, slots=True)
class _LockedMetadata:
    lock: AbstractContextManager[int | None]
    metadata: SessionMetadata | None

    def release(self) -> None:
        self.lock.__exit__(None, None, None)


@dataclass(slots=True)
class SessionAccountContext:
    """Resolve credentials and keep provider transitions local to one session."""

    home: Path
    store: AccountStore
    manager: SessionManager | None
    metadata: SessionMetadata
    _pins: dict[str, AccountId] = field(default_factory=dict)
    _frozen: bool = False
    _active_provider: str | None = None
    _switch_sequences: dict[str, int] = field(default_factory=dict)
    _account_state_unknown: bool = False
    _unknown_state: _UnknownAccountState | None = field(default=None, init=False, repr=False)
    _switch_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @classmethod
    def load(
        cls,
        home: str | Path,
        manager: SessionManager,
        metadata: SessionMetadata,
        *,
        validate: bool = True,
    ) -> SessionAccountContext:
        path = Path(home)
        pins = {
            provider: AccountId(*value.split(":", 1))
            for provider, value in metadata.selected_accounts.items()
        }
        context = cls(
            path,
            AccountStore(path),
            manager,
            metadata,
            pins,
            _active_provider=account_provider(metadata.provider),
            _switch_sequences=dict(metadata.account_switch_sequences),
        )
        if validate:
            context._validate_pins()
        if (
            validate
            and context._active_provider is not None
            and not api_key_mode(metadata.provider)
        ):
            context.resolve(metadata.provider)
        return context

    @property
    def auth_mode(self) -> str:
        return (
            "api-key"
            if api_key_mode(self._active_provider or self.metadata.provider)
            else "named-account"
        )

    @property
    def switching_unavailable(self) -> str | None:
        if self.auth_mode == "api-key":
            return "this session uses an API key; account switching is unavailable"
        return None

    @property
    def active_provider(self) -> str | None:
        return self._active_provider

    @property
    def account_state_unknown(self) -> bool:
        return self._account_state_unknown

    @property
    def switch_sequence(self) -> int:
        if self._active_provider is None:
            return 0
        return self._switch_sequences.get(self._active_provider, 0)

    def _validate_pins(self) -> None:
        for provider, account in self._pins.items():
            if api_key_mode(provider):
                continue
            if not self.store.contains(account):
                raise AccountError(
                    f"session account {account} no longer exists; choose another with /accounts"
                )

    def resolve(self, provider: str) -> AccountId:
        canonical = account_provider(provider)
        if canonical is None:
            raise ValueError("provider does not use a named account")
        if api_key_mode(provider):
            raise ValueError("API-key authentication does not use a named account")
        selected = self._pins.get(canonical)
        if selected is not None:
            if not self._frozen and not self.store.contains(selected):
                raise AccountError(
                    f"session account {selected} no longer exists; choose another with /accounts"
                )
            return selected
        if self._frozen:
            raise AccountError(
                f"child account snapshot has no {canonical} account; start a new child"
            )
        selected = self.store.resolve_account(provider)
        if self.store.contains(selected):
            self._pins[canonical] = selected
            assert self.manager is not None
            self.manager.record_account_selection(self.metadata, selected, boundary=0)
        return selected

    def backend_kwargs(self, provider: str) -> dict[str, object]:
        result: dict[str, object] = {"home": self.home}
        canonical = account_provider(provider)
        if canonical is not None and not api_key_mode(provider):
            self.resolve(provider)
            if (account := self._pins.get(canonical)) is not None:
                result["account"] = account
        return result

    def snapshot(self) -> dict[str, AccountId]:
        return dict(self._pins)

    async def list_snapshot(self) -> AccountListSnapshot:
        selected = self.snapshot()

        def load() -> AccountListSnapshot:
            displays = tuple(self.store.list_display())
            providers = {item.account.provider for item in displays}
            defaults = {
                provider: account
                for provider in providers
                if (account := self.store.default(provider)) is not None
            }
            return AccountListSnapshot(displays, selected, defaults)

        return await asyncio.to_thread(load)

    def child_snapshot(self, provider: str | None) -> SessionAccountContext:
        needed = {"codex"}
        canonical = account_provider(provider or "")
        if canonical is not None:
            needed.add(canonical)
        child_pins = self.snapshot()
        for name in needed:
            source = "claude" if name == "anthropic" else name
            if not api_key_mode(source):
                child_pins[name] = self.resolve(source)
        return SessionAccountContext(
            self.home,
            self.store,
            None,
            self.metadata,
            child_pins,
            _frozen=True,
            _active_provider=canonical,
            _switch_sequences=dict(self._switch_sequences),
        )

    async def switch_active(
        self,
        loop: Any,
        account: AccountId,
        *,
        boundary: int,
        provider: str,
        build_backend: Callable[[AccountId], Any],
    ) -> None:
        def prepare(metadata: SessionMetadata) -> SessionMetadata:
            if self.switching_unavailable is not None:
                raise AccountError(self.switching_unavailable)
            if self._frozen:
                raise AccountError("child account snapshots cannot be changed")
            if not self.store.contains(account):
                raise AccountError(f"account {account} does not exist; use /accounts")
            assert self.manager is not None
            return self.manager._apply_account_selection(
                metadata, account, boundary=boundary, switched=True
            )

        active_provider = account_provider(provider)

        def backend_for_metadata(metadata: SessionMetadata) -> Any:
            selected = metadata.selected_accounts.get(account.provider)
            if account_provider(metadata.provider) != account.provider:
                if account_provider(metadata.provider) == self._active_provider:
                    return None
                raise AccountError("disk provider changed during account switch")
            if selected is None:
                raise AccountError("disk account selection is missing")
            return build_backend(AccountId(*selected.split(":", 1)))

        await self._switch(
            loop,
            prepare=prepare,
            build_backend=backend_for_metadata,
            active_provider=active_provider,
        )

    async def switch_model(
        self,
        loop: Any,
        *,
        prepare: Callable[[SessionMetadata], SessionMetadata],
        build_backend: Callable[[SessionMetadata], Any],
    ) -> SessionMetadata:
        return await self._switch(
            loop,
            prepare=prepare,
            build_backend=build_backend,
            active_provider=None,
        )

    async def _switch(
        self,
        loop: Any,
        *,
        prepare: Callable[[SessionMetadata], SessionMetadata],
        build_backend: Callable[[SessionMetadata], Any],
        active_provider: str | None,
    ) -> SessionMetadata:
        async with self._switch_lock:
            if self._account_state_unknown:
                raise AccountError(
                    "session account state is unknown; retry /accounts after the next turn"
                )
            new = await asyncio.to_thread(
                self._prepare,
                loop,
                prepare,
                build_backend,
                active_provider,
            )
            return await _shielded(
                self._commit_and_apply(loop, new, build_backend)
            )

    async def _commit_and_apply(
        self,
        loop: Any,
        new: _PreparedAccountState,
        build_backend: Callable[[SessionMetadata], Any],
    ) -> SessionMetadata:
        locked = await asyncio.to_thread(self._commit_to_disk, new)
        try:
            metadata = locked.metadata
            if metadata is None:
                self._mark_unknown(build_backend)
                return new.previous_metadata
            if self._metadata_equal(metadata, new.metadata):
                self._apply(loop, self._state_from_metadata(new, metadata))
            elif not self._metadata_equal(metadata, new.previous_metadata):
                self._mark_unknown(build_backend)
            return metadata
        finally:
            await asyncio.shield(asyncio.to_thread(locked.release))

    def _mark_unknown(
        self, build_backend: Callable[[SessionMetadata], Any]
    ) -> None:
        self._account_state_unknown = True
        self._unknown_state = _UnknownAccountState(build_backend)

    def _prepare(
        self,
        loop: Any,
        prepare: Callable[[SessionMetadata], SessionMetadata],
        build_backend: Callable[[SessionMetadata], Any],
        active_provider: str | None,
    ) -> _PreparedAccountState:
        previous = copy.deepcopy(self.metadata)
        metadata = prepare(copy.deepcopy(previous))
        backend = build_backend(metadata)
        backend_state = (
            loop.current_backend_state()
            if backend is None
            else loop.prepare_backend_state(backend)
        )
        if hasattr(backend_state, "token_budget"):
            backend_state = replace(
                backend_state, token_budget=metadata.compaction_budget
            )
        return _PreparedAccountState(
            previous,
            metadata,
            {
                provider: AccountId(*value.split(":", 1))
                for provider, value in metadata.selected_accounts.items()
            },
            dict(metadata.account_switch_sequences),
            active_provider
            if active_provider is not None
            else account_provider(metadata.provider),
            backend_state,
        )

    def _commit_to_disk(self, state: _PreparedAccountState) -> _LockedMetadata:
        assert self.manager is not None
        session_id = self.metadata.session_id
        lock = self.manager._metadata_lock(
            session_id, timeout=ACCOUNT_SWITCH_LOCK_TIMEOUT_SECONDS
        )
        directory_fd = lock.__enter__()
        try:
            assert directory_fd is not None
            current = self.manager._read(session_id, directory_fd=directory_fd)
            if not self._metadata_equal(current, state.previous_metadata):
                return _LockedMetadata(lock, current)
            try:
                self.manager._write_unlocked(state.metadata, directory_fd=directory_fd)
            except BaseException as exc:  # noqa: BLE001 - authoritative reread decides
                logger.debug("session transition write reported failure: %s", exc)
            try:
                metadata = self.manager._read(session_id, directory_fd=directory_fd)
            except BaseException:  # noqa: BLE001 - disk outcome cannot be proven
                metadata = None
            return _LockedMetadata(lock, metadata)
        except BaseException:
            lock.__exit__(*sys.exc_info())
            raise

    @staticmethod
    def _metadata_equal(left: SessionMetadata, right: SessionMetadata) -> bool:
        return left.to_storage_dict() == right.to_storage_dict()

    @staticmethod
    def _state_from_metadata(
        state: _PreparedAccountState, metadata: SessionMetadata
    ) -> _PreparedAccountState:
        return _PreparedAccountState(
            state.previous_metadata,
            metadata,
            {
                provider: AccountId(*value.split(":", 1))
                for provider, value in metadata.selected_accounts.items()
            },
            dict(metadata.account_switch_sequences),
            account_provider(metadata.provider),
            state.backend_state,
        )

    def _apply(self, loop: Any, state: _PreparedAccountState) -> None:
        for name in self.metadata.__slots__:
            setattr(self.metadata, name, getattr(state.metadata, name))
        self._pins = state.pins
        self._switch_sequences = state.switch_sequences
        self._active_provider = state.active_provider
        loop.apply_backend_state(state.backend_state)
        self._account_state_unknown = False
        self._unknown_state = None

    async def ensure_account_state(self, loop: Any) -> None:
        if not self._account_state_unknown:
            return
        async with self._switch_lock:
            if not self._account_state_unknown:
                return
            recovery = self._unknown_state
            assert recovery is not None and self.manager is not None
            try:
                await _shielded(self._recover_and_apply(loop, recovery.build_backend))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise AccountError(
                    "session account state is unknown; open /accounts before retrying"
                ) from exc

    async def _recover_and_apply(
        self, loop: Any, build_backend: Callable[[SessionMetadata], Any]
    ) -> None:
        locked, state = await asyncio.to_thread(
            self._recover_locked, loop, build_backend
        )
        try:
            self._apply(loop, state)
        finally:
            await asyncio.shield(asyncio.to_thread(locked.release))

    def _recover_locked(
        self, loop: Any, build_backend: Callable[[SessionMetadata], Any]
    ) -> tuple[_LockedMetadata, _PreparedAccountState]:
        assert self.manager is not None
        session_id = self.metadata.session_id
        lock = self.manager._metadata_lock(
            session_id, timeout=ACCOUNT_SWITCH_LOCK_TIMEOUT_SECONDS
        )
        directory_fd = lock.__enter__()
        try:
            assert directory_fd is not None
            metadata = self.manager._read(session_id, directory_fd=directory_fd)
            state = self._prepare(
                loop,
                lambda _current: copy.deepcopy(metadata),
                build_backend,
                None,
            )
            return _LockedMetadata(lock, metadata), state
        except BaseException:
            lock.__exit__(*sys.exc_info())
            raise

    async def subscription_limit_event(self, failure: object) -> dict[str, object] | None:
        return await asyncio.to_thread(self._subscription_limit_event, failure)

    def _subscription_limit_event(self, failure: object) -> dict[str, object] | None:
        provider = self._active_provider
        if provider is None or api_key_mode(provider):
            return None
        account = self.resolve(provider)
        limit = classify_subscription_limit(provider, account, failure)
        if limit is None:
            return None
        failure.subscription_limit = limit  # type: ignore[attr-defined]
        now = datetime.now(UTC)
        state = limit.usage_state(now)
        assert state.snapshot is not None
        record_subscription_limit(self.store, account, state.snapshot, now)
        states = {
            item: read_usage(self.store, item, now=now)
            for item in self.store.list()
            if item.provider == provider
        }
        alternatives = recovery_alternatives(account, states, now)
        return limit.recovery_event(frozen=self._frozen, alternatives=alternatives)
