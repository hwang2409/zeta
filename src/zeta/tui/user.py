"""User message surface shared by live and replayed transcripts."""

from __future__ import annotations

from rich.console import RenderableType
from rich.padding import Padding
from rich.text import Text

from ..protocol.types import Message, TextContent
from . import theme

USER_DISPLAY_TEXT_METADATA = "zeta.user_display_text"


def displayed_user_text(message: Message) -> str:
    """Return persisted transcript text, falling back to model-visible text."""

    display_text = message.metadata.get(USER_DISPLAY_TEXT_METADATA)
    if isinstance(display_text, str):
        return display_text
    return next(
        (
            block.text
            for block in message.content
            if isinstance(block, TextContent) and block.path is None
        ),
        "",
    )


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
