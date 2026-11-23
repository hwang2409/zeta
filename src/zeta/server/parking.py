"""session_park: the busy sessions a client left, and the idle ones being closed."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from .connections import last_turn_outcome
from .protocol import ErrorDescriptor, EventDescriptor, ProtocolError

if TYPE_CHECKING:
    from .connections import ConnectionRegistry
    from .control import ServeControl
    from .session_controllers import SessionController, SessionControllers

class ParkedPairs:
    """The parking policy: park or close a pair the client left, report its
    row, and close it once it holds no work. ``SessionControllers`` owns the
    records and their close. Loop thread only."""

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
        # the last parked_session_changed data sent per parked session
        self._sent: dict[str, dict[str, object]] = {}
        self._parked_by: dict[ServeControl, object] = {}

    def require_room(self) -> None:
        held = [
            *(control.runtime.session_id for control in self._controllers.parked_controls()),
            *self._controllers.closing_session_ids(),
        ]
        if len(held) >= self.limit:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY,
                "too many parked sessions are running",
                {"reason": "parked_limit", "parked": held},
            )

    def pairs(self) -> tuple[ServeControl, ...]:
        return self._controllers.parked_controls()

    def get(self, session_id: str) -> SessionController | None:
        return self._controllers.parked_record(session_id)

    def rows(self) -> list[dict[str, object]]:
        """``status.parked_sessions``."""
        return [control.parked_row() for control in self._controllers.parked_controls()]

    def forget(self, record: SessionController) -> None:
        """Forget the presentation state after an exact owner claims a record."""
        self._controllers.unpark(record)
        self._parked_by.pop(record.control, None)
        if record.runtime.opened is not None:
            self._sent.pop(record.runtime.session_id, None)

    def unpark(self, incoming: ServeControl) -> bool:
        """Take ``incoming`` out of the parked set when it is parked."""
        session_id = incoming.runtime.session_id
        record = self._controllers.parked_record(session_id)
        if record is None or record.control is not incoming:
            return False
        assert self._controllers.unpark(record) is not None
        self._sent.pop(session_id, None)
        self._parked_by.pop(incoming, None)
        return True

    def leave(self, outgoing: ServeControl, parked_by: object | None) -> None:
        """The client left ``outgoing``: park it while it holds work, else close it."""
        if outgoing.holds_work():
            self._controllers.park(self._controllers.record_for(outgoing))
            if parked_by is not None:
                self._parked_by[outgoing] = parked_by
            self.changed(outgoing)
        else:
            self._controllers.retire(self._controllers.record_for(outgoing))

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
        if (parked_by := self._parked_by.get(control)) is not None:
            connections.publish_connection(
                parked_by, EventDescriptor.PARKED_SESSION_CHANGED, session_id, data=data
            )

    def close(
        self, control: ServeControl, settlement_timeout: float | None = None
    ) -> None:
        """Close a parked pair now; its turn ends ``disconnect``."""
        if control in self._controllers.parked_controls():
            session_id = control.runtime.session_id
            self._controllers.retire(self._controllers.record_for(control), settlement_timeout)
            self._publish(
                control,
                {"session_id": session_id, "state": "closed", "pending_approvals": 0},
            )
            self._parked_by.pop(control, None)
            self._sent.pop(session_id, None)

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
