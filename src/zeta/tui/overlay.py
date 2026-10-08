"""Shared full-screen overlay frame for the TUI's popup panels.

The Ctrl+F finder set the visual language for a Zeta popup: a dim box frame, an
accent title, a thin rule under it, body rows with an accent ``❯`` gutter and a
subtle selection background, and a dim key-hint footer. This module owns that
language once so ``/status``, ``/tasks``, and ``/mcp`` look like the finder
instead of a wireframe, and the finder itself boxes its content through
:func:`frame` so there is a single frame owner.

Every style string is a semantic theme token (PR #385) passed through
:func:`theme.prompt_toolkit_style`, so the panels track the active palette, red
stays reserved for failures, and a background never leaks the Rich ``on
<color>`` spelling into prompt-toolkit (the crash class of PR #409).
"""

from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.utils import get_cwidth

from . import theme

Fragment = tuple[str, str]
FragmentLine = list[Fragment]

OVERLAY_MAX_WIDTH = 96

# A body line equal (by content) to this marker renders as a full-width dim
# rule. Callers build it with :func:`rule`; the width is only known when the
# control paints, so the marker carries no width of its own.
_RULE_MARK = "\x00overlay-rule\x00"


def _s(style: str) -> str:
    return theme.prompt_toolkit_style(style)


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _line_cells(line: FragmentLine) -> int:
    return sum(_cells(text) for _style, text in line)


def _is_rule(line: FragmentLine) -> bool:
    return len(line) == 1 and line[0][1] == _RULE_MARK


# -- line builders ----------------------------------------------------------


def title(text: str) -> FragmentLine:
    """The accent panel title, mirroring the finder's query line weight."""

    return [(_s(f"bold {theme.ACCENT}"), text)]


def rule() -> FragmentLine:
    """A full-width dim separator, expanded to the frame width on paint."""

    return [("", _RULE_MARK)]


def hint(text: str) -> FragmentLine:
    """A dim key-hint footer row."""

    return [(_s(theme.DIM), text)]


def blank() -> FragmentLine:
    """An empty spacer row inside the frame."""

    return []


def label(text: str) -> Fragment:
    return (_s(theme.DIM), text)


def value(text: str, style: str | None = None) -> Fragment:
    return (_s(style or theme.BODY), text)


def field(name: str, text: str, *, width: int, value_style: str | None = None) -> FragmentLine:
    """A dim ``label`` paired with a value, label padded to ``width`` cells."""

    return [label(name.ljust(width)), value(text, value_style)]


def row(
    fragments: Sequence[tuple[str, str]], *, selected: bool
) -> FragmentLine:
    """A selectable list row: accent ``❯`` gutter and a selection background.

    ``fragments`` are ``(style, text)`` pairs expressed in shared theme tokens;
    the selection background is layered on every fragment so the whole row
    reads as one bar, exactly like the finder's current result row.
    """

    suffix = f" on {theme.MENU_BG}" if selected else ""
    gutter_style = f"bold {theme.ACCENT}" if selected else theme.DIM
    out: FragmentLine = [(_s(gutter_style + suffix), "❯ " if selected else "  ")]
    for style, text in fragments:
        out.append((_s(style + suffix), text))
    return out


# -- framing ----------------------------------------------------------------


def _truncate(line: FragmentLine, width: int) -> tuple[FragmentLine, int]:
    """Clip ``line`` to ``width`` cells, returning the kept line and its cells."""

    kept: FragmentLine = []
    cells = 0
    for style, text in line:
        if cells >= width:
            break
        piece: list[str] = []
        for character in text:
            step = max(0, get_cwidth(character))
            if cells + step > width:
                break
            piece.append(character)
            cells += step
        if piece:
            kept.append((style, "".join(piece)))
    return kept, cells


def frame_line(line: FragmentLine, *, content_width: int, border: str) -> FragmentLine:
    """Box one body line between ``│`` sides, padding or clipping to width."""

    if _is_rule(line):
        return [(border, "│"), (_s(theme.DIM), "─" * content_width), (border, "│")]
    clipped, cells = _truncate(line, content_width)
    if cells < content_width:
        clipped = [*clipped, (_s(theme.BODY), " " * (content_width - cells))]
    return [(border, "│"), *clipped, (border, "│")]


def frame(lines: Sequence[FragmentLine], *, width: int) -> list[FragmentLine]:
    """Return ``lines`` wrapped in a dim box frame, sized to ``width`` cells."""

    border = _s(theme.DIM)
    content_width = max(1, width - 2)
    framed: list[FragmentLine] = [[(border, "┌" + "─" * content_width + "┐")]]
    framed.extend(
        frame_line(line, content_width=content_width, border=border) for line in lines
    )
    framed.append([(border, "└" + "─" * content_width + "┘")])
    return framed


# -- control ----------------------------------------------------------------


class OverlayControl(UIControl):
    """A bounded, keyboard-scrollable popup that paints a finder-style frame.

    Callers feed pre-styled :data:`FragmentLine` rows (title, rule, body, hint)
    through :meth:`set_lines`; the control keeps the top and bottom borders
    fixed and scrolls only the rows between them, so the box never scrolls off
    screen on a short terminal.
    """

    def __init__(self) -> None:
        self._lines: tuple[FragmentLine, ...] = ()
        self._offset = 0
        self._capacity = 1

    @property
    def is_focusable(self) -> bool:
        return True

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def set_lines(self, lines: Sequence[FragmentLine], *, keep_offset: bool = False) -> None:
        self._lines = tuple(lines)
        if not keep_offset:
            self._offset = 0
        self._clamp()

    def scroll(self, amount: int) -> None:
        self._offset = min(max(0, self._offset + amount), self._max_offset())

    def page(self, amount: int) -> None:
        self.scroll(amount * max(1, self._capacity - 1))

    def top(self) -> None:
        self._offset = 0

    def bottom(self) -> None:
        self._offset = self._max_offset()

    def _max_offset(self) -> int:
        return max(0, len(self._lines) - self._capacity)

    def _clamp(self) -> None:
        self._offset = min(max(0, self._offset), self._max_offset())

    def preferred_width(self, max_available_width: int) -> int:
        natural = max((_line_cells(line) for line in self._lines), default=0)
        return min(OVERLAY_MAX_WIDTH, max(24, natural + 4))

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: object | None,
    ) -> int:
        del width, max_available_height, wrap_lines, get_line_prefix
        return len(self._lines) + 2

    def create_content(self, width: int, height: int | None) -> UIContent:
        self._capacity = max(1, (height or (len(self._lines) + 2)) - 2)
        self._clamp()
        border = _s(theme.DIM)
        content_width = max(1, width - 2)
        window = self._lines[self._offset : self._offset + self._capacity]
        rows: list[FragmentLine] = [[(border, "┌" + "─" * content_width + "┐")]]
        for line in window:
            rows.append(frame_line(line, content_width=content_width, border=border))
        for _ in range(self._capacity - len(window)):
            rows.append(
                [(border, "│"), (_s(theme.BODY), " " * content_width), (border, "│")]
            )
        rows.append([(border, "└" + "─" * content_width + "┘")])

        return UIContent(
            get_line=rows.__getitem__,
            line_count=len(rows),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    def window(self) -> Window:
        return Window(self, wrap_lines=False, style="class:overlay")


__all__ = [
    "OVERLAY_MAX_WIDTH",
    "Fragment",
    "FragmentLine",
    "OverlayControl",
    "blank",
    "field",
    "frame",
    "frame_line",
    "hint",
    "label",
    "row",
    "rule",
    "title",
    "value",
]
