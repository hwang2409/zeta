"""Errors raised while controlling the terminal."""

from __future__ import annotations


class TerminalRenderError(RuntimeError):
    """A redraw failed; unwind the TUI before reporting the failure."""
