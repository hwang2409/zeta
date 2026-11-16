"""Own and resolve the session runtime/control records used by serve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
    from .control import ServeControl
    from .runtime import ServerRuntime


@dataclass(frozen=True, slots=True)
class SessionController:
    """The persistent runtime/control pair for one resident session."""

    control: ServeControl

    @property
    def runtime(self) -> ServerRuntime:
        return self.control.runtime


class SessionControllers:
    """Own the attached record and the records available for selection."""

    def __init__(self, control: ServeControl) -> None:
        self._records: dict[ServeControl, SessionController] = {
            control: SessionController(control)
        }
        self._attached = self._records[control]
        self._parked: dict[str, SessionController] = {}

    def records(self) -> tuple[SessionController, ...]:
        return tuple(self._records.values())

    def record_for(self, control: ServeControl) -> SessionController:
        return self._records[control]

    def register(self, control: ServeControl) -> SessionController:
        record = SessionController(control)
        self._records[control] = record
        return record

    def attached(self) -> SessionController:
        return self._attached

    def owned(self, connection: Any) -> SessionController:
        """Return the persistent record controlled by ``connection``."""
        del connection
        return self._attached

    def selected(
        self,
        connection: Any,
        session_id: str,
        *,
        unattached_error: ProtocolError | None = None,
    ) -> SessionController:
        """Return the persistent resident record selected by ``session_id``.

        ``unattached_error`` replaces the absence error while no session is
        attached, for an operation whose legacy wire text differs."""
        del connection
        attached = self._attached.runtime
        if (
            self._attached in self._records.values()
            and attached.opened is not None
            and attached.session_id == session_id
        ):
            return self._attached
        record = self._parked.get(session_id)
        if record is not None:
            return record
        if attached.opened is None and unattached_error is not None:
            raise unattached_error
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def attach(self, record: SessionController) -> SessionController:
        self._parked.pop(record.runtime.session_id, None)
        self._attached = record
        return record

    def park(self, record: SessionController) -> SessionController:
        self._parked[record.runtime.session_id] = record
        return record

    def unpark(self, record: SessionController) -> SessionController | None:
        session_id = record.runtime.session_id
        if self._parked.get(session_id) is not record:
            return None
        del self._parked[session_id]
        return record

    def parked_controls(self) -> tuple[ServeControl, ...]:
        return tuple(record.control for record in self._parked.values())

    def parked_record(self, session_id: str) -> SessionController | None:
        return self._parked.get(session_id)

    def remove(self, record: SessionController) -> None:
        control = record.control
        self._parked.pop(record.runtime.session_id, None)
        self._records.pop(control, None)

    def require_current(
        self, connection: Any, record: SessionController, session_id: str | None
    ) -> None:
        """Fail unless ``record`` is still resident and still holds ``session_id``."""
        del connection
        runtime = record.runtime
        if record not in self._records.values() or (
            session_id is not None
            and (runtime.opened is None or runtime.session_id != session_id)
        ):
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def is_attached(self, control: ServeControl) -> bool:
        return self._attached.control is control
