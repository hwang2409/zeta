"""Shared prompt-toolkit text input for full-screen overlays."""

from __future__ import annotations

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


def _wrap(text: str, width: int) -> list[str]:
    width = max(1, width)
    lines: list[str] = []
    for source in text.split("\n"):
        if not source:
            lines.append("")
            continue
        line = ""
        for character in source:
            if _cells(line + character) >= width:
                lines.append(line)
                line = ""
            line += character
        lines.append(line)
    return lines or [""]


class OverlayTextField:
    """A bounded Buffer shared by overlay rendering and prompt-toolkit input."""

    def __init__(self, name: str) -> None:
        self.buffer = Buffer(name=name)
        self.control = BufferControl(buffer=self.buffer)
        self.window = Window(
            self.control,
            height=Dimension(min=0, max=0, preferred=0),
            dont_extend_height=True,
        )

    @property
    def text(self) -> str:
        return self.buffer.text

    def reset(self) -> None:
        self.buffer.reset()

    def set_text(self, text: str) -> None:
        self.buffer.set_document(Document(text, len(text)), bypass_readonly=True)

    def lines(self, width: int, *, prefix: str = "❯ ") -> list[list[tuple[str, str]]]:
        available = max(1, width - _cells(prefix))
        wrapped = _wrap(self.text, available)
        if len(wrapped) > _MAX_LINES:
            wrapped = wrapped[-_MAX_LINES:]
        cursor = self.buffer.cursor_position
        before = self.text[:cursor]
        cursor_line = len(before.split("\n")) - 1
        cursor_column = _cells(before.rsplit("\n", 1)[-1])
        first_line = max(0, cursor_line - len(wrapped) + 1)
        vim = None
        try:
            app = get_app()
            if app.editing_mode is EditingMode.VI:
                vim = "normal" if app.vi_state.input_mode is InputMode.NAVIGATION else "insert"
        except RuntimeError:
            vim = None
        result: list[list[tuple[str, str]]] = []
        for index, line in enumerate(wrapped):
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
