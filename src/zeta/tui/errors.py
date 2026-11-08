"""Errors raised while running the terminal interface."""


class TerminalRenderError(RuntimeError):
    """A redraw failed; unwind the TUI before reporting the failure."""
