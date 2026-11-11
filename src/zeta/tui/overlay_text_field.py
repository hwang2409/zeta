"""Shared prompt-toolkit text input for full-screen overlays."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.layout import Dimension, Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.utils import get_cwidth

from . import theme

_MAX_LINES = 4


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _wrap(text: str, width: int) -> list[tuple[str, int, int, int]]:
    width = max(0, width)
    lines: list[tuple[str, int, int, int]] = []
    offset = 0
    for source in text.split("\n"):
        if not source:
            lines.append(("", offset, offset, 0))
        else:
            line = ""
            line_cells = 0
            start = offset
            for character in source:
                character_cells = max(0, get_cwidth(character))
                if line and line_cells + character_cells > width:
                    lines.append((line, start, offset, line_cells))
                    line = ""
                    line_cells = 0
                    start = offset
                if character_cells > width:
                    lines.append(("", offset, offset + 1, 0))
                    offset += 1
                    start = offset
                    continue
                line += character
                line_cells += character_cells
                offset += 1
            if line or start == offset:
                lines.append((line, start, offset, line_cells))
        offset += 1
    return lines or [("", 0, 0, 0)]


class OverlayTextField:
    """A bounded Buffer shared by overlay rendering and prompt-toolkit input."""

    def __init__(self, name: str, on_text_changed: Callable[[str], None] | None = None) -> None:
        self.buffer = Buffer(name=name)
        self._text_changed_handler = None
        self._suppress_text_changed = False
        if on_text_changed is not None:
            self._text_changed_handler = lambda buffer: (
                None if self._suppress_text_changed else on_text_changed(buffer.text)
            )
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

    def focus(self, layout: Layout) -> None:
        """Focus this field's window and start in vi insert mode when editing in VI."""
        try:
            layout.focus(self.window)
        except ValueError:
            return
        app = get_app()
        if app.editing_mode is EditingMode.VI:
            app.vi_state.input_mode = InputMode.INSERT

    def close(self) -> None:
        if self._text_changed_handler is not None:
            self.buffer.on_text_changed -= self._text_changed_handler
            self._text_changed_handler = None

    def reset(self, *, notify: bool = True) -> None:
        self._suppress_text_changed = not notify
        try:
            self.buffer.reset()
        finally:
            self._suppress_text_changed = False

    def set_text(self, text: str, *, notify: bool = True) -> None:
        self.set_document(Document(text, len(text)), notify=notify)

    def set_document(self, document: Document, *, notify: bool = True) -> None:
        self._suppress_text_changed = not notify
        try:
            self.buffer.set_document(document, bypass_readonly=True)
        finally:
            self._suppress_text_changed = False

    def lines(self, width: int, *, prefix: str = "❯ ", enter: str = "submit") -> list[list[tuple[str, str]]]:
        available = max(1, width - _cells(prefix))
        wrapped = _wrap(self.text, available - 1)
        cursor = self.buffer.cursor_position
        cursor_line = next(
            (index for index, (_, start, end, _) in enumerate(wrapped) if start <= cursor <= end),
            len(wrapped) - 1,
        )
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
        for index, (line, start, _end, _line_cells) in enumerate(visible):
            absolute = first_line + index
            marker = "▌" if absolute == cursor_line else ""
            if marker:
                point = min(max(0, cursor - start), len(line))
                left = line[:point]
                right = line[point:]
                fragments = [(theme.ACCENT, prefix if index == 0 else " " * _cells(prefix)), (theme.BODY, left), (f"bold {theme.ACCENT}", marker), (theme.BODY, right)]
            else:
                fragments = [(theme.ACCENT, prefix if index == 0 else " " * _cells(prefix)), (theme.BODY, line)]
            result.append(fragments)
        if vim == "insert":
            hint = f"insert · enter {enter} · esc normal"
        elif vim == "normal":
            hint = f"normal · enter {enter} · esc cancel"
        else:
            hint = f"enter {enter} · esc cancel"
        result.append([(theme.DIM, hint)])
        return result


__all__ = ["OverlayTextField"]
