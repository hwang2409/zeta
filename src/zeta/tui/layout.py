"""Shared terminal layout measurements."""

from __future__ import annotations

from ..core.session import _preview_text


CONTENT_MARGIN = 2


def content_width(terminal_width: int) -> int:
    """Return the width between the app's two-column side margins."""

    return max(1, terminal_width - CONTENT_MARGIN * 2)


def resume_picker_line(value: str, width: int) -> str:
    return " " * CONTENT_MARGIN + _preview_text(value, limit=width)
