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
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.controls import UIContent, UIControl
from rich.console import Console, RenderableType

from ..theme import RICH_THEME

# The highlighted row of every picker starts with this marker.
SELECTED_MARKER = "❯ "
# Every picker's key hint, its last row, starts with this marker.
HINT_MARKER = "↑/↓"
# Kept rows at the top of a cropped card: border and header.
_EDGE_ROWS = 2

FragmentLine = list[tuple[str, str]]


def card_height_cap(terminal_rows: int) -> int:
    """Half the terminal: the transcript above keeps the other half."""

    # Three body rows at least, so the elision lines never hide the selected one.
    return max(_EDGE_ROWS * 2 + 3, terminal_rows // 2)


def crop_card(lines: list[FragmentLine], cap: int) -> list[FragmentLine]:
    """Keep both card edges and the rows around the highlighted one.

    The bottom edge is the key hint, however many lines it wraps to, and the
    border under it.
    """

    if len(lines) <= cap:
        return lines
    bottom = _bottom_edge_rows(lines)
    body = lines[_EDGE_ROWS:-bottom]
    # A screen too short for the elision lines keeps the selected row instead.
    room = max(1, cap - _EDGE_ROWS - bottom)
    selected = next(
        (index for index, line in enumerate(body) if SELECTED_MARKER in _text(line)),
        0,
    )
    start = max(0, min(selected - room // 2, len(body) - room))
    shown = body[start : start + room]
    if room >= 3:
        # The first and last shown rows say how many rows the crop hides.
        if start:
            shown[0] = _elision(shown[0], f"… {start + 1} more above")
        if start + room < len(body):
            shown[-1] = _elision(shown[-1], f"… {len(body) - start - room + 1} more below")
    return [*lines[:_EDGE_ROWS], *shown, *lines[-bottom:]]


def _bottom_edge_rows(lines: list[FragmentLine]) -> int:
    """Rows from the key hint's first line to the bottom border."""

    for index in range(len(lines) - 1, _EDGE_ROWS - 1, -1):
        if _text(lines[index]).strip("│ ").startswith(HINT_MARKER):
            return len(lines) - index
    return 2


def _text(line: FragmentLine) -> str:
    return "".join(text for _style, text in line)


def _elision(line: FragmentLine, text: str) -> FragmentLine:
    """``line`` with its text between the card borders replaced by ``text``."""

    plain = "".join(part for _style, part in line)
    if len(plain) < 4:
        return line
    border = line[0][0]
    inside = line[1][0] if len(line) > 1 else border
    return [
        (border, plain[0]),
        (inside, f"   {text}".ljust(len(plain) - 2)[: len(plain) - 2]),
        (line[-1][0], plain[-1]),
    ]


class PickerCardControl(UIControl):
    """Draws ``card()`` (None: no picker open) at the window's width."""

    def __init__(self, card: Callable[[], RenderableType | None]) -> None:
        self._card = card
        self._width: int | None = None

    def height(self) -> Dimension:
        """The card's rows, up to half the screen, as the window's minimum.

        A split hands out rows one at a time in turn up to each child's
        preferred height, so with only a preferred height the card would share
        rows with the transcript and lose its selected row and its hint.
        """

        size = get_app().output.get_size()
        lines = self.lines(self._width or size.columns, size.rows)
        return Dimension(min=min(len(lines), size.rows // 2))

    def lines(self, width: int, height: int) -> list[FragmentLine]:
        """The card cropped to half the terminal and to ``height``, the window's room."""

        self._width = width
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
        cap = min(card_height_cap(get_app().output.get_size().rows), height)
        return crop_card(lines, cap)

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: object,
    ) -> int:
        return len(self.lines(width, max_available_height))

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self.lines(width, height)
        return UIContent(get_line=lambda index: lines[index], line_count=len(lines))

    def is_focusable(self) -> bool:
        return False


__all__ = ["HINT_MARKER", "SELECTED_MARKER", "PickerCardControl", "card_height_cap", "crop_card"]
