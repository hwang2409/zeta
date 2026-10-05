"""The desktop runtime seam.

``DesktopBackend`` is the only interface the MCP server uses. A backend owns
one disposable desktop: it starts lazily, executes model-frame actions, and is
destroyed at the end of the session. ``create_backend`` selects the
implementation by name, so a hosted backend (for example Modal or E2B) is a new
entry in ``BACKENDS`` and needs no server change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

DEFAULT_BACKEND = "local"


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One JPEG frame in the model frame size."""

    data: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True, slots=True)
class DesktopOptions:
    """Per-session values every backend receives."""

    session_id: str
    ttl_seconds: int


class DesktopBackend(Protocol):
    """One disposable graphical desktop.

    ``start`` is idempotent and is called before every tool action, so a
    backend starts its desktop on first use and enforces its lifetime there.
    ``destroy`` is idempotent and never raises for an absent desktop; a later
    ``start`` creates a fresh one. ``close`` destroys the desktop and releases
    every backend resource at server exit.
    Coordinates in ``input`` arguments are model-frame values that the caller
    already validated.
    """

    def start(self) -> None: ...

    def destroy(self) -> None: ...

    def close(self) -> None: ...

    def screenshot(self) -> Screenshot: ...

    def input(self, action: str, arguments: Mapping[str, object]) -> None: ...

    def observe(self) -> dict[str, object]: ...

    def settle(self) -> float: ...


@dataclass(frozen=True, slots=True)
class BackendKind:
    """How to create a backend, and how the host removes a session's desktops.

    ``remove_session`` runs in the Zeta process at session end, after the
    server has exited or was killed. It is best effort and must not raise.
    """

    create: Callable[[DesktopOptions], DesktopBackend]
    remove_session: Callable[[str], None]


def _create_local(options: DesktopOptions) -> DesktopBackend:
    from .local import LocalDockerBackend

    return LocalDockerBackend(options)


def _remove_local(session_id: str) -> None:
    from .local import remove_session_desktops

    remove_session_desktops(session_id)


BACKENDS: Mapping[str, BackendKind] = {
    "local": BackendKind(create=_create_local, remove_session=_remove_local),
}


def backend_kind(name: str) -> BackendKind:
    """Return the named backend kind, or raise ``ValueError``."""

    kind = BACKENDS.get(name)
    if kind is None:
        raise ValueError(
            f"unknown computer backend {name!r}; choose one of: {', '.join(sorted(BACKENDS))}"
        )
    return kind


def create_backend(name: str, options: DesktopOptions) -> DesktopBackend:
    return backend_kind(name).create(options)


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "BackendKind",
    "DesktopBackend",
    "DesktopOptions",
    "Screenshot",
    "backend_kind",
    "create_backend",
]
