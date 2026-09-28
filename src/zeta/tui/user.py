"""User message surface shared by live and replayed transcripts."""

from __future__ import annotations

from rich.console import RenderableType
from rich.padding import Padding
from rich.text import Text

from . import theme


def user_message(text: Text) -> RenderableType:
    if not theme.USER_BG:
        return text
    return Padding(text, (0, 1), style=theme.USER_BG, expand=True)
