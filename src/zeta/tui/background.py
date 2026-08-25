"""Background task presentation helpers."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from .theme import DIM


def background_notice(app: Any, message: str) -> None:
    """Print one dim background task notice and refresh the prompt."""

    app._print(Text(message, style=DIM))
    app._invalidate_prompt()
