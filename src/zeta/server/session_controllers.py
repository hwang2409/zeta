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
        self._attached = SessionController(control)
        self._parked: dict[str, SessionController] = {}

    def attached(self) -> SessionController:
        return self._attached

    def owned(self, connection: Any) -> SessionController:
        """Return the persistent record controlled by ``connection``."""
        del connection
        return self._attached

    def selected(self, connection: Any, session_id: str) -> SessionController:
        """Return the persistent resident record selected by ``session_id``."""
        del connection
        record = self._attached
        try:
            if record.runtime.session_id == session_id:
                return record
        except RuntimeError:
            pass
        try:
            return self._parked[session_id]
        except KeyError as exc:
            raise ProtocolError(
                ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active"
            ) from exc

    def attach(self, control: ServeControl) -> None:
        self._attached = SessionController(control)

    def register_parked(self, control: ServeControl) -> None:
        self._parked[control.runtime.session_id] = SessionController(control)

    def unregister_parked(self, session_id: str) -> None:
        self._parked.pop(session_id, None)

    def is_attached(self, control: ServeControl) -> bool:
        return self._attached.control is control
