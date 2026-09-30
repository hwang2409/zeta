"""User message surface shared by live and replayed transcripts."""

from __future__ import annotations

from rich.console import RenderableType
from rich.padding import Padding
from rich.text import Text

from . import theme


def user_message(text: Text) -> RenderableType:
    """Render a sent message across the transcript viewport content width.

    ``expand=True`` uses the width supplied by the transcript's Rich console,
    rather than the terminal's global width. The horizontal padding is part of
    that full-width user surface; transcript separators provide the single
    shared blank row between adjacent units.
    """
    if not theme.USER_BG:
        return text
    return Padding(text, (0, 1), style=theme.USER_BG, expand=True)
