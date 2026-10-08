"""The full-screen overlay for the transcript message finder.

:class:`FinderControl` is a read-only prompt-toolkit control: it paints the
finder panel (query line, ranked result list, optional preview) from a
:class:`FinderState` snapshot the transcript produces each frame. All input is
routed through key bindings, so the control never owns a focusable buffer; it
draws its own caret and selection bar instead.

Styling pulls exclusively from the semantic theme roles (PR #385) so the panel
tracks the active palette and never hardcodes colour. Tool cards are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.utils import get_cwidth

from .. import overlay, theme
from .message_finder import FinderRow, Role

_MAX_WIDTH = 96
_VISIBLE_ROWS = 12
_PREVIEW_ROWS = 6


@dataclass(frozen=True, slots=True)
class FinderState:
    """An immutable snapshot of the finder, enough to paint one frame."""

    query: str
    rows: tuple[FinderRow, ...]
    selected: int
    preview: tuple[str, ...]
    preview_visible: bool
    total: int
    complete: bool


def _role_style(role: Role) -> str:
    return {
        Role.USER: theme.ACCENT,
        Role.ASSISTANT: theme.AGENT_MAIN,
        Role.TOOL: theme.AGENT_CHILD,
        Role.RESULT: theme.SUCCESS,
        Role.NOTICE: theme.NOTICE,
    }[role]


_ROLE_LABEL = {
    Role.USER: "you",
    Role.ASSISTANT: "zeta",
    Role.TOOL: "tool",
    Role.RESULT: "result",
    Role.NOTICE: "note",
}

_LABEL_WIDTH = max(len(label) for label in _ROLE_LABEL.values())


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    cells = 0
    out: list[str] = []
    for character in text:
        step = max(0, get_cwidth(character))
        if cells + step > width:
            out.append("…")
            break
        out.append(character)
        cells += step
    return "".join(out)


def _list_window(selected: int, count: int, height: int) -> int:
    """Return the first visible row so ``selected`` stays inside the window."""

    if count <= height:
        return 0
    half = height // 2
    offset = selected - half
    return max(0, min(offset, count - height))


class FinderControl(UIControl):
    """Paint the finder overlay from the transcript's live finder state."""

    def __init__(self, get_state: Callable[[], FinderState | None]) -> None:
        self._get_state = get_state
        self._lines: list[list[tuple[str, str]]] = []

    def _result_line(
        self, row: FinderRow, *, selected: bool, width: int
    ) -> list[tuple[str, str]]:
        gutter = "❯ " if selected else "  "
        label = _ROLE_LABEL[row.candidate.role].ljust(_LABEL_WIDTH)
        marker = row.candidate.marker
        prefix_cells = len(gutter) + _LABEL_WIDTH + 1 + _cells(marker) + 1
        excerpt = _truncate(row.excerpt, max(1, width - prefix_cells))
        row_bg = f" on {theme.MENU_BG}" if selected else ""
        base = theme.prompt_toolkit_style(
            (theme.BODY if selected else theme.DIM) + row_bg
        )
        match_style = theme.prompt_toolkit_style(
            (f"bold {theme.ACCENT}" if selected else theme.ACCENT) + row_bg
        )
        highlights = set(row.highlights)
        fragments: list[tuple[str, str]] = [
            (
                theme.prompt_toolkit_style(theme.ACCENT + row_bg)
                if selected
                else base,
                gutter,
            ),
            (theme.prompt_toolkit_style(_role_style(row.candidate.role) + row_bg), label),
            (base, " "),
            (theme.prompt_toolkit_style(theme.DIM + row_bg), marker),
            (base, " "),
        ]
        run: list[str] = []
        run_is_match = False
        for column, character in enumerate(excerpt):
            is_match = column in highlights
            if is_match != run_is_match and run:
                fragments.append((match_style if run_is_match else base, "".join(run)))
                run = []
            run.append(character)
            run_is_match = is_match
        if run:
            fragments.append((match_style if run_is_match else base, "".join(run)))
        return fragments

    def _build(self, state: FinderState, width: int) -> list[list[tuple[str, str]]]:
        content_width = max(1, width - 2)
        lines: list[list[tuple[str, str]]] = []
        # Query line.
        lines.append(
            [
                (theme.ACCENT, "❯ "),
                (theme.BODY, _truncate(state.query, content_width - 3)),
                (f"bold {theme.ACCENT}", "▌"),
            ]
        )
        lines.append([(theme.DIM, "─" * content_width)])
        # Result rows.
        if not state.rows:
            empty = "no messages" if not state.query else "no matches"
            lines.append([(theme.DIM, empty)])
        else:
            window = _list_window(state.selected, len(state.rows), _VISIBLE_ROWS)
            visible = state.rows[window : window + _VISIBLE_ROWS]
            for offset, row in enumerate(visible):
                lines.append(
                    self._result_line(
                        row,
                        selected=(window + offset) == state.selected,
                        width=content_width,
                    )
                )
        # Preview pane.
        if state.preview_visible and state.preview:
            lines.append([(theme.DIM, "─" * content_width)])
            for preview_line in state.preview[:_PREVIEW_ROWS]:
                lines.append([(theme.DIM, _truncate(preview_line, content_width))])
        return lines

    # -- UIControl ---------------------------------------------------------

    def _current_lines(self, width: int) -> list[list[tuple[str, str]]]:
        state = self._get_state()
        if state is None:
            return []
        return self._build(state, width)

    def preferred_width(self, max_available_width: int) -> int:
        lines = self._current_lines(min(max_available_width, _MAX_WIDTH) - 2)
        natural = max(
            (sum(_cells(text) for _style, text in line) for line in lines),
            default=0,
        )
        return min(_MAX_WIDTH, max(24, natural + 4))

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: object | None,
    ) -> int:
        del max_available_height, wrap_lines, get_line_prefix
        return len(self._current_lines(width)) + 2

    def create_content(self, width: int, height: int | None) -> UIContent:
        rows = overlay.frame(self._current_lines(width), width=width)
        return UIContent(
            get_line=rows.__getitem__,
            line_count=len(rows),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    def window(self) -> Window:
        return Window(self, wrap_lines=False, style="class:finder")


__all__ = ["FinderControl", "FinderState"]
