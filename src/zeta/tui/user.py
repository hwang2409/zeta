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
    """Render a sent message as a band shaped like the composer.

    One filled row above and below and a one-cell inset mirror the composer's
    padding. ``expand=True`` fills the width of the transcript's Rich console;
    the full-screen layout extends band rows into its side margins so the
    band spans the terminal like the composer does.
    """
    return Padding(text, (1, 1), style=theme.USER_BG, expand=True)
