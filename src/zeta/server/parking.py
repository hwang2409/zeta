"""session_park: the busy sessions a client left, and the idle ones being closed."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .connections import last_turn_outcome
from .protocol import ErrorDescriptor, EventDescriptor, ProtocolError

if TYPE_CHECKING:
    from .connections import ConnectionRegistry
    from .control import ServeControl
    from .session_controllers import SessionControllers

logger = logging.getLogger(__name__)


class ParkedPairs:
    """Owns every parked and closing pair: park or close a pair the client left,
    report its row, and close it once it holds no work. Loop thread only."""

    def __init__(
        self,
        connections: Callable[[], ConnectionRegistry | None],
        controllers: SessionControllers,
        limit: int = 3,
    ) -> None:
        self._connections = connections
        self._controllers = controllers
        # parked plus closing pairs; each holds a loop, its store, and its MCP servers
        self.limit = limit
        self._closing: dict[str, asyncio.Task[None]] = {}
        # the last parked_session_changed data sent per parked session
        self._sent: dict[str, dict[str, object]] = {}

    def require_room(self) -> None:
        held = [*(control.runtime.session_id for control in self._controllers.parked_controls()), *self._closing]
        if len(held) >= self.limit:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY,
                "too many parked sessions are running",
                {"reason": "parked_limit", "parked": held},
            )

    def pairs(self) -> tuple[ServeControl, ...]:
        return self._controllers.parked_controls()

    def get(self, session_id: str) -> ServeControl | None:
        return self._controllers.parked_control(session_id)

    def rows(self) -> list[dict[str, object]]:
        """``status.parked_sessions``."""
        return [control.parked_row() for control in self._controllers.parked_controls()]

    def unpark(self, incoming: ServeControl) -> bool:
        """Take ``incoming`` out of the parked set when it is parked."""
        session_id = incoming.runtime.session_id
        if self._controllers.parked_control(session_id) is not incoming:
            return False
        assert self._controllers.unpark(self._controllers.record_for(incoming)) is not None
        self._sent.pop(session_id, None)
        return True

    def leave(self, outgoing: ServeControl) -> None:
        """The client left ``outgoing``: park it while it holds work, else close it."""
        if outgoing.holds_work():
            self._controllers.park(self._controllers.record_for(outgoing))
            self.changed(outgoing)
        else:
            self._close_pair(outgoing)

    def changed(self, control: ServeControl) -> None:
        """Report this parked pair's state and close it once idle; a no-op for a
        pair that is not parked."""
        if control in self._controllers.parked_controls():
            asyncio.get_running_loop().call_soon(self._check, control)

    def changed_from_any_thread(self, control: ServeControl) -> None:
        """``changed`` for a sink that may run off the loop thread."""
        connections = self._connections()
        if connections is None:
            return
        try:
            connections.loop.call_soon_threadsafe(self.changed, control)
        except RuntimeError:
            if not connections.loop.is_closed():
                raise

    def _check(self, control: ServeControl) -> None:
        # a resume may have attached the pair since this check was scheduled
        if control not in self._controllers.parked_controls():
            return
        self._publish(control, control.parked_row())
        if not control.holds_work():
            self.close(control)

    def _publish(self, control: ServeControl, row: dict[str, object]) -> None:
        """Send ``parked_session_changed`` when the row differs from the last one
        sent. The parked control's own frames stay dropped."""
        assert control.runtime.opened is not None
        session_id = control.runtime.session_id
        data = {
            **row,
            "last_turn_outcome": last_turn_outcome(control.runtime.opened.store),
        }
        connections = self._connections()
        if self._sent.get(session_id) == data or connections is None:
            return
        self._sent[session_id] = data
        connections.try_admit_controller(
            EventDescriptor.PARKED_SESSION_CHANGED, session_id, data=data
        )

    def close(
        self, control: ServeControl, settlement_timeout: float | None = None
    ) -> None:
        """Close a parked pair now; its turn ends ``disconnect``."""
        if control in self._controllers.parked_controls():
            session_id = control.runtime.session_id
            self._controllers.remove(self._controllers.record_for(control))
            self._publish(
                control,
                {"session_id": session_id, "state": "closed", "pending_approvals": 0},
            )
            self._sent.pop(session_id, None)
            self._close_pair(control, settlement_timeout)

    def _close_pair(
        self, control: ServeControl, settlement_timeout: float | None = None
    ) -> None:
        session_id = control.runtime.session_id
        control.closing = True  # no wake starts from here on

        async def close() -> None:
            try:
                try:
                    await control.close()
                finally:
                    try:
                        await control.runtime.close()
                    finally:
                        await control.wait_for_approval_settlement(settlement_timeout)
            finally:
                if self._closing.get(session_id) is task:
                    del self._closing[session_id]

        def report(done: asyncio.Task[None]) -> None:
            # a close that no resume awaits would otherwise fail without a trace
            if not done.cancelled() and (exc := done.exception()) is not None:
                logger.error("closing session %s failed", session_id, exc_info=exc)

        task = asyncio.create_task(close())
        task.add_done_callback(report)
        self._closing[session_id] = task

    async def wait_closed(self, session_id: str) -> None:
        """Wait until a closing pair has released ``session_id``."""
        task = self._closing.get(session_id)
        if task is not None:
            # a cancelled request must not cancel the close
            await asyncio.shield(task)

    async def claim(
        self, keep: bool, *, reconnect: bool, require_host_binding: bool
    ) -> None:
        """Reclaim each parked pair for a returning controller; close the ones it
        cannot see (no ``session_park``) or cannot claim."""
        pairs = self._controllers.parked_controls()
        claimed = (
            await asyncio.gather(
                *(
                    control.controller_claimed(
                        reconnect=reconnect, require_host_binding=require_host_binding
                    )
                    for control in pairs
                )
            )
            if keep
            else [False] * len(pairs)
        )
        for control, kept in zip(pairs, claimed, strict=True):
            if not kept:
                self.close(control)

    def close_all(self, settlement_timeout: float | None) -> None:
        for control in self._controllers.parked_controls():
            self.close(control, settlement_timeout)

    def closing_tasks(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._closing.values())
