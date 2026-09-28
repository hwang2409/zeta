"""Focused, transient status card for the full-screen terminal UI."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.utils import get_cwidth


def _fit_cells(text: str, width: int) -> str:
    """Truncate and pad text by terminal cells, not Python characters."""

    if width <= 0:
        return ""
    cells = 0
    result: list[str] = []
    for character in text:
        character_width = max(0, get_cwidth(character))
        if cells + character_width > width:
            break
        result.append(character)
        cells += character_width
    return "".join(result) + " " * (width - cells)


class StatusCardControl(UIControl):
    """A bounded, keyboard-scrollable view of one status snapshot."""

    def __init__(self) -> None:
        self._lines: tuple[str, ...] = ()
        self._offset = 0
        self._height = 1
        self._on_change: Callable[[], None] | None = None

    @property
    def is_focusable(self) -> bool:
        return True

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def set_lines(self, lines: Sequence[str]) -> None:
        self._lines = tuple(lines)
        self._offset = 0
        self._notify()

    def scroll(self, amount: int) -> None:
        maximum = max(0, len(self._lines) - self._height)
        old = self._offset
        self._offset = min(max(0, self._offset + amount), maximum)
        if self._offset != old:
            self._notify()

    def page(self, amount: int) -> None:
        self.scroll(amount * max(1, self._height - 2))

    def top(self) -> None:
        self._set_offset(0)

    def bottom(self) -> None:
        self._set_offset(max(0, len(self._lines) - self._height))

    def _set_offset(self, value: int) -> None:
        maximum = max(0, len(self._lines) - self._height)
        value = min(max(0, value), maximum)
        if value != self._offset:
            self._offset = value
            self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def create_content(self, width: int, height: int | None) -> UIContent:
        self._height = max(1, height or 1)
        self._clamp_offset()
        content_width = max(0, width - 4)
        lines = self._lines

        def get_line(index: int) -> list[tuple[str, str]]:
            if width < 4 or index >= len(lines):
                return [("class:status-card", " " * width)]
            text = _fit_cells(lines[index], content_width)
            rendered = f"│ {text} │"
            rendered += " " * max(0, width - get_cwidth(rendered))
            return [("class:status-card.body", rendered)]

        return UIContent(
            get_line=get_line,
            # The window owns scrolling. Returning only the rendered window
            # here makes prompt-toolkit apply the offset a second time.
            line_count=len(lines),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    def _clamp_offset(self) -> None:
        maximum = max(0, len(self._lines) - self._height)
        self._offset = min(max(0, self._offset), maximum)

    def vertical_scroll(self, window: Window) -> int:
        del window
        return self._offset

    def window(self) -> Window:
        return Window(
            self,
            wrap_lines=False,
            get_vertical_scroll=self.vertical_scroll,
            style="class:status-card",
        )


__all__ = ["StatusCardControl"]
