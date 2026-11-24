"""Serve account requests and the shared owner for account mutations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from ..accounts import (
    AccountError,
    AccountId,
    AccountStore,
    AccountUsage,
    account_provider,
)
from ..accounts.usage import UsageState, usage_json
from ..core.session import account_home
from .protocol import ErrorDescriptor, ProtocolError
from .resource_protocol import run_owner
from .runtime import ServerRuntime

if TYPE_CHECKING:
    from .control import ServeControl
    from .session_controllers import SessionController, SessionUse


async def migrate(home: Path) -> tuple[str, ...]:
    return await asyncio.to_thread(AccountStore(account_home(home)).migrate)


async def list_accounts(
    runtime: ServerRuntime,
    refreshed: dict[AccountId, UsageState] | None = None,
) -> dict[str, object]:
    context = getattr(runtime.loop, "account_context", None)
    if context is None:
        raise ProtocolError(
            ErrorDescriptor.STATE_UNAVAILABLE, "accounts unavailable for this session"
        )
    snapshot = await context.list_snapshot()
    return {
        "accounts": [
            {
                "account": str(item.account),
                "label": item.label,
                "detail": item.detail,
                "available": item.available,
                "current": snapshot.selected.get(item.account.provider) == item.account,
                "default": snapshot.defaults.get(item.account.provider) == item.account,
                "usage": usage_json(
                    refreshed.get(item.account, item.usage) if refreshed else item.usage
                ),
            }
            for item in snapshot.displays
        ]
    }


async def list_account_catalog(client: Any, runtime: ServerRuntime) -> dict[str, object]:
    context = getattr(runtime.loop, "account_context", None)
    selected = context.snapshot() if context is not None else {}

    def read():
        store = context.store if context is not None else AccountStore(account_home(client.server.home))
        return store.catalog_snapshot()

    def project(snapshot):
        revision, displays, defaults = snapshot
        return {"revision": revision, "data": {"accounts": [
            {
                "account": str(item.account),
                "label": item.label,
                "detail": None,  # Credential metadata is free text and not safe to project.
                "available": item.available,
                "current": selected.get(item.account.provider) == item.account,
                "default": defaults.get(item.account.provider) == str(item.account),
                "usage": usage_json(item.usage),
            }
            for item in displays
        ]}}

    return await run_owner(
        client, read, family="account", project=project,
        owner_error=(AccountError, OSError, RecursionError, ValueError),
        errors={None: ErrorDescriptor.STATE_READ_UNAVAILABLE},
    )


def parse_account(params: dict[str, object]) -> AccountId:
    raw = params.get("account")
    if not isinstance(raw, str) or ":" not in raw:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "account must be provider:name"
        )
    return AccountId(*raw.split(":", 1))


async def logout(
    runtime: ServerRuntime,
    account: AccountId,
) -> dict[str, object]:
    try:
        message = await asyncio.to_thread(
            runtime.loop.account_context.store.logout,
            account,
            sessions_root=runtime.home / "sessions",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, str(exc)) from exc
    return {"account": str(account), "message": message}


async def switch(
    runtime: ServerRuntime,
    account: AccountId,
    authorize_commit: Callable[[], bool] | None = None,
) -> dict[str, object]:
    loop = runtime.loop
    context = loop.account_context
    provider = runtime.metadata.provider
    current_provider = account_provider(provider)

    def prepare_backend(selected: AccountId) -> object | None:
        if selected.provider != current_provider:
            return None
        backend, _ = runtime._build_backend(
            provider,
            loop.backend.model,
            context.home,
            account=selected,
            token_budget=loop.context_assembler.token_budget,
            use_backend_factory=False,
        )
        return backend

    await context.switch_active(
        loop,
        account,
        boundary=loop.store.active_branch_head_sequence(),
        provider=provider,
        build_backend=prepare_backend,
        authorize_commit=authorize_commit,
    )
    return {
        "account": str(account),
        "replay": False,
        "account_state": "unknown" if context.account_state_unknown else "ready",
    }


async def refresh(runtime: ServerRuntime) -> dict[str, object]:
    refreshed = await asyncio.to_thread(
        lambda: asyncio.run(
            AccountUsage(runtime.loop.account_context.store).refresh_stale()
        )
    )
    return await list_accounts(runtime, refreshed)


class AccountOperations:
    def __init__(
        self,
        records: Callable[[], tuple[SessionController, ...]],
        is_multi: Callable[[], bool],
        legacy_controls: Callable[[], Iterator[ServeControl]],
        is_closing: Callable[[ServeControl], bool],
    ) -> None:
        self._records = records
        self._is_multi = is_multi
        self._legacy_controls = legacy_controls
        self._is_closing = is_closing
        self._lock = asyncio.Lock()

    @overload
    async def run(
        self, kind: Literal["login_publication"], use: None,
        operation: Callable[[], Awaitable[object]],
    ) -> object: ...

    @overload
    async def run(
        self, kind: Literal["switch", "logout", "refresh"], use: SessionUse,
        operation: dict[str, object],
    ) -> dict[str, object]: ...

    async def run(
        self, kind: Literal["switch", "logout", "refresh", "login_publication"],
        use: SessionUse | None,
        operation: dict[str, object] | Callable[[], Awaitable[object]],
    ) -> object:
        async with self._lock:
            if kind == "login_publication":
                assert use is None and callable(operation)
                # The arbiter decides on the loop and drains the worker on cancel.
                return await operation()
            assert use is not None and isinstance(operation, dict)
            multi = self._is_multi()
            if not multi:
                for control in self._legacy_controls():
                    await control.require_idle()
            if kind == "switch" and multi:
                self._require_stable(use.record)
            account = parse_account(operation) if kind != "refresh" else None
            if kind == "logout" and multi:
                assert account is not None
                for record in self._records():
                    if self._affected(record, account):
                        self._require_stable(record)

            request = asyncio.current_task()
            assert request is not None

            def authorize_commit() -> bool:
                return not request.cancelling()

            if kind == "switch":
                assert account is not None
                # Legacy cancellation can stop a pending write and mark state unknown.
                if not multi:
                    return await switch(use.runtime, account)
                return await self._finish(switch(use.runtime, account, authorize_commit))

            # Dispatch authorizes the entire worker, including refresh's final display snapshot.
            if not authorize_commit():
                raise asyncio.CancelledError
            if kind == "logout":
                assert account is not None
                return await self._finish(logout(use.runtime, account))
            return await self._finish(refresh(use.runtime))

    async def drain(self) -> None:
        async with self._lock:
            pass

    @staticmethod
    async def _finish[T](work: Coroutine[object, object, T]) -> T:
        completion = asyncio.create_task(work)
        cancelled = None
        while not completion.done():
            try:
                await asyncio.wait((completion,))
            except asyncio.CancelledError as exc:
                cancelled = exc
        if cancelled is not None:
            if not completion.cancelled():
                completion.exception()
            raise cancelled
        return completion.result()

    @staticmethod
    def _affected(record: SessionController, account: AccountId) -> bool:
        runtime = record.runtime
        if record.control.account_snapshot_transition():
            return True
        if runtime.opened is None:
            return False
        context = runtime.loop.account_context
        return (
            account_provider(runtime.metadata.provider) == account.provider
            or (context is not None and context.snapshot().get(account.provider) == account)
        )

    def _require_stable(self, record: SessionController) -> None:
        control = record.control
        if (
            self._is_closing(control) or control.account_snapshot_transition()
            or control.state_change_busy() or control.holds_work()
        ):
            raise ProtocolError(ErrorDescriptor.OPERATION_BUSY, "a turn is already running")
