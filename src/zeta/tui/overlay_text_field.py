"""Shared prompt-toolkit text input for full-screen overlays."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.utils import get_cwidth

from . import theme

_MAX_LINES = 4


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _wrap(text: str, width: int) -> list[tuple[str, int, int]]:
    width = max(1, width)
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for source in text.split("\n"):
        if not source:
            lines.append(("", offset, offset))
        else:
            line = ""
            start = offset
            for character in source:
                if line and _cells(line + character) > width:
                    lines.append((line, start, offset))
                    line = ""
                    start = offset
                line += character
                offset += 1
            lines.append((line, start, offset))
        offset += 1
    return lines or [("", 0, 0)]


class OverlayTextField:
    """A bounded Buffer shared by overlay rendering and prompt-toolkit input."""

    def __init__(self, name: str, on_text_changed: Callable[[str], None] | None = None) -> None:
        self.buffer = Buffer(name=name)
        self._text_changed_handler = None
        if on_text_changed is not None:
            self._text_changed_handler = lambda buffer: on_text_changed(buffer.text)
            self.buffer.on_text_changed += self._text_changed_handler
        self.control = BufferControl(buffer=self.buffer)
        self.window = Window(
            self.control,
            height=Dimension(min=0, max=0, preferred=0),
            dont_extend_height=True,
        )

    @property
    def text(self) -> str:
        return self.buffer.text

    def close(self) -> None:
        if self._text_changed_handler is not None:
            self.buffer.on_text_changed -= self._text_changed_handler
            self._text_changed_handler = None

    def reset(self) -> None:
        self.buffer.reset()

    def set_text(self, text: str) -> None:
        self.buffer.set_document(Document(text, len(text)), bypass_readonly=True)

    def lines(self, width: int, *, prefix: str = "❯ ") -> list[list[tuple[str, str]]]:
        available = max(1, width - _cells(prefix))
        wrapped = _wrap(self.text, available)
        cursor = self.buffer.cursor_position
        cursor_line = next(
            (index for index, (_, start, end) in enumerate(wrapped) if start <= cursor <= end),
            len(wrapped) - 1,
        )
        cursor_column = _cells(self.text[wrapped[cursor_line][1] : cursor])
        first_line = max(0, cursor_line - _MAX_LINES + 1)
        visible = wrapped[first_line : first_line + _MAX_LINES]
        vim = None
        try:
            app = get_app()
            if app.editing_mode is EditingMode.VI:
                vim = "normal" if app.vi_state.input_mode is InputMode.NAVIGATION else "insert"
        except RuntimeError:
            vim = None
        result: list[list[tuple[str, str]]] = []
        for index, (line, start, end) in enumerate(visible):
            absolute = first_line + index
            marker = "▌" if absolute == cursor_line else ""
            if marker:
                point = min(cursor_column, _cells(line))
                left = line[:point]
                right = line[point:]
                fragments = [(theme.ACCENT, prefix if index == 0 else " " * _cells(prefix)), (theme.BODY, left), (f"bold {theme.ACCENT}", marker), (theme.BODY, right)]
            else:
                fragments = [(theme.ACCENT, prefix if index == 0 else " " * _cells(prefix)), (theme.BODY, line)]
            result.append(fragments)
        if vim:
            result.append([(theme.DIM, f"{vim} · enter submit · esc cancel")])
        else:
            result.append([(theme.DIM, "enter submit · esc cancel")])
        return result


__all__ = ["OverlayTextField"]
