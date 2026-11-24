"""session_park: the busy sessions a client left, and the idle ones being closed."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from .connections import last_turn_outcome
from .protocol import EventDescriptor

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
    ) -> None:
        self._connections = connections
        self._controllers = controllers

    def pairs(self) -> tuple[ServeControl, ...]:
        return self._controllers.parked_controls()

    def get(self, session_id: str) -> SessionController | None:
        return self._controllers.parked_record(session_id)

    def rows(self, connection: object) -> list[dict[str, object]]:
        """``status.parked_sessions``."""
        return [record.control.parked_row() for record in self._controllers.parked_records(connection)]

    def leave(self, outgoing: SessionController, connection: object) -> None:
        """The client left ``outgoing``: park it while it holds work, else close it."""
        if outgoing.control.holds_work():
            self._controllers.park(outgoing, connection)
            self.changed(outgoing.control)
        else:
            self._controllers.retire(outgoing)

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
        if not control.holds_work() and not self._controllers.parked_in_grace(control):
            self.close(control)

    def _publish(self, control: ServeControl, row: dict[str, object]) -> None:
        """Send ``parked_session_changed`` when the row differs from the last one
        sent. The parked control's own frames stay dropped."""
        record = self._controllers.record_for(control)
        assert control.runtime.opened is not None
        session_id = control.runtime.session_id
        data = {
            **row,
            "last_turn_outcome": last_turn_outcome(control.runtime.opened.store),
        }
        connections = self._connections()
        if record.parked_last_sent == data or connections is None:
            return
        record.parked_last_sent = data
        if (parked_by := record.parked_by) is not None:
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

    async def claim(
        self, keep: bool, *, reconnect: bool, require_host_binding: bool
    ) -> None:
        """Reclaim each parked pair for a returning controller; close the ones it
        cannot see (no ``session_park``) or cannot claim."""
        pairs = tuple(record.control for record in self._controllers.parked_records(None))
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
