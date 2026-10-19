"""User message surface shared by live and replayed transcripts."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from rich.console import RenderableType
from rich.padding import Padding
from rich.text import Text

from ..protocol.types import Message, TextContent
from . import theme

USER_DISPLAY_TEXT_METADATA = "zeta.user_display_text"
# Marks full-screen rows of a sent-message band; only these bleed into margins.
USER_BAND_CLASS = "class:user-band"


def mark_band(line: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Tag one rendered row of a user message so the margin painter finds it."""
    return [(f"{USER_BAND_CLASS} {style}", text) for style, text in line]


def mark_bands(
    lines: list[list[tuple[str, str]]],
    locations: Sequence[tuple[object, int]],
    user_units: Collection[object],
) -> list[list[tuple[str, str]]]:
    """Tag rows whose location belongs to a user unit; other rows are unchanged."""
    return [
        mark_band(line)
        if index < len(locations) and locations[index][0] in user_units
        else line
        for index, line in enumerate(lines)
    ]


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

    One filled row above and below mirrors the composer's padding; the text has
    no horizontal inset so it starts in the transcript's text column.
    ``expand=True`` fills the width of the transcript's Rich console;
    the transcript tags band rows with ``USER_BAND_CLASS`` and the full-screen
    layout extends those rows into its side margins.
    """
    return Padding(text, (1, 0), style=theme.USER_BG, expand=True)
