"""Serialize account mutations from validation through durable and live apply."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import TYPE_CHECKING, Literal, overload

from ..accounts import AccountId, account_provider
from . import accounts
from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
    from .control import ServeControl
    from .session_controllers import SessionController, SessionUse


class AccountOperations:
    def __init__(
        self,
        records: Callable[[], tuple[SessionController, ...]],
        is_multi: Callable[[], bool],
        legacy_controls: Callable[[], Iterator[ServeControl]],
    ) -> None:
        self._records = records
        self._is_multi = is_multi
        self._legacy_controls = legacy_controls
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
            account = accounts.parse_account(operation) if kind != "refresh" else None
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
                    return await accounts.switch(use.runtime, account)
                return await self._finish(accounts.switch(use.runtime, account, authorize_commit))

            # Dispatch authorizes the entire worker, including refresh's final display snapshot.
            if not authorize_commit():
                raise asyncio.CancelledError
            if kind == "logout":
                assert account is not None
                return await self._finish(accounts.logout(use.runtime, account))
            return await self._finish(accounts.refresh(use.runtime))

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

    @staticmethod
    def _require_stable(record: SessionController) -> None:
        control = record.control
        if (
            record.closing or control.account_snapshot_transition()
            or control.state_change_busy() or control.holds_work()
        ):
            raise ProtocolError(ErrorDescriptor.OPERATION_BUSY, "a turn is already running")
