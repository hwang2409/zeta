"""UI methods exposed to extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host import ExtensionHost


class ExtensionUI:
    def __init__(self, host: ExtensionHost, extension: str) -> None:
        self._host = host
        self._extension = extension

    def notify(self, text: str, *, level: str = "info") -> None:
        """Show a transient notice from the owning event-loop thread.

        Notices emitted during activation queue up to 32 items. Later notices
        are summarized once; after binding, calls from another thread are
        dropped. Use async code or ``zx.run_blocking`` to return to the loop.
        """
        if type(text) is not str or not text:
            raise ValueError("ui.notify text must be a nonempty string")
        if level not in {"info", "warning", "error"}:
            raise ValueError("ui.notify level must be info, warning, or error")
        self._host.notify(self._extension, text, level)


__all__ = ["ExtensionUI"]
