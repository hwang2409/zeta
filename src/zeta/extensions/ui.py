"""UI methods exposed to extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host import ExtensionOwnerBridge


class ExtensionUI:
    def __init__(self, owner: ExtensionOwnerBridge, extension: str) -> None:
        self._owner = owner
        self._extension = extension

    def notify(self, text: str, *, level: str = "info") -> None:
        """Show a transient notice on the session loop.

        A private-loop hook may call this method; the notice is marshaled to
        the session loop, and the sink runs only there. Other foreign-thread
        calls remain unsupported. Activation notices use the host's bounded
        startup queue until a sink is bound.
        """
        if type(text) is not str or not text:
            raise ValueError("ui.notify text must be a nonempty string")
        if level not in {"info", "warning", "error"}:
            raise ValueError("ui.notify level must be info, warning, or error")
        self._owner.notify(self._extension, text, level)


__all__ = ["ExtensionUI"]
