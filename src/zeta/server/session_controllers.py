"""Resolve the session runtime and control owned by a serve connection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .control import ServeControl
    from .runtime import ServerRuntime
    from .server import ZetaServer


@dataclass(frozen=True, slots=True)
class SessionController:
    """The runtime/control pair for one resident session."""

    control: ServeControl

    @property
    def runtime(self) -> ServerRuntime:
        return self.control.runtime


class SessionControllers:
    """Own the resolver seam for session-scoped serve work.

    Slice 1 retains the existing single-controller admission and parking model.
    The record is deliberately small; later slices add reservation and use
    settlement without making handlers choose a runtime themselves.
    """

    def __init__(self, server: ZetaServer) -> None:
        self._server = server

    def owned(self, connection: Any) -> SessionController:
        """Return the session controlled by ``connection``."""
        server = connection.server
        control = server.control
        return SessionController(control)

    def selected(self, connection: Any, session_id: str) -> SessionController:
        """Return the resident session selected by ``session_id``."""
        server = connection.server
        control = server.control
        if control.runtime.session_id == session_id:
            return SessionController(control)
        parked = server.parking.get(session_id)
        if parked is not None:
            return SessionController(parked)
        raise KeyError(session_id)

    def record(self, control: ServeControl) -> SessionController:
        """Wrap a newly created or parked pair at its ownership seam."""
        return SessionController(control)
