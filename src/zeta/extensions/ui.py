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
        if type(text) is not str or not text:
            raise ValueError("ui.notify text must be a nonempty string")
        if level not in {"info", "warning", "error"}:
            raise ValueError("ui.notify level must be info, warning, or error")
        self._host.notify(self._extension, text, level)


__all__ = ["ExtensionUI"]
