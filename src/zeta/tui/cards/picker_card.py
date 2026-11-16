"""The open picker's card, drawn above the composer.

A picker card is overlay state, not history: it is drawn from the open picker
on every paint, so a history redraw cannot clear it, it never enters the
transcript's pages, and the card and the picker keys cannot disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO

from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.controls import UIContent, UIControl
from rich.console import Console, RenderableType

from ..theme import RICH_THEME

# The highlighted row of every picker starts with this marker.
SELECTED_MARKER = "❯ "
# Kept rows at each end of a cropped card: border and header, hint and border.
_EDGE_ROWS = 2

FragmentLine = list[tuple[str, str]]


def card_height_cap(terminal_rows: int) -> int:
    """Half the terminal: the transcript above keeps the other half."""

    return max(_EDGE_ROWS * 2 + 1, terminal_rows // 2)


def crop_card(lines: list[FragmentLine], cap: int) -> list[FragmentLine]:
    """Keep both card edges and the rows around the highlighted one."""

    if len(lines) <= cap:
        return lines
    body = lines[_EDGE_ROWS:-_EDGE_ROWS]
    room = cap - _EDGE_ROWS * 2
    selected = next(
        (
            index
            for index, line in enumerate(body)
            if SELECTED_MARKER in "".join(text for _style, text in line)
        ),
        0,
    )
    start = max(0, min(selected - room // 2, len(body) - room))
    return [*lines[:_EDGE_ROWS], *body[start : start + room], *lines[-_EDGE_ROWS:]]


class PickerCardControl(UIControl):
    """Draws ``card()`` (None: no picker open) at the window's width."""

    def __init__(self, card: Callable[[], RenderableType | None]) -> None:
        self._card = card

    def lines(self, width: int) -> list[FragmentLine]:
        renderable = self._card()
        if renderable is None:
            return []
        output = StringIO()
        Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            width=max(1, width),
            theme=RICH_THEME,
        ).print(renderable)
        lines = list(split_lines(to_formatted_text(ANSI(output.getvalue().rstrip("\n")))))
        return crop_card(lines, card_height_cap(get_app().output.get_size().rows))

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: object,
    ) -> int:
        return len(self.lines(width))

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self.lines(width)
        return UIContent(get_line=lambda index: lines[index], line_count=len(lines))

    def is_focusable(self) -> bool:
        return False


__all__ = ["SELECTED_MARKER", "PickerCardControl", "card_height_cap", "crop_card"]
