"""Shared terminal layout measurements."""

from __future__ import annotations


CONTENT_MARGIN = 2


def content_width(terminal_width: int) -> int:
    """Return the width between the app's two-column side margins."""

    return max(1, terminal_width - CONTENT_MARGIN * 2)
