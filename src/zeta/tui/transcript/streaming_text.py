"""Append-only Rich text optimized for a streaming transcript tail."""

from __future__ import annotations

from typing import Literal

from rich._wrap import divide_line
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text


class StreamingText:
    """Append-only styled text with an incremental Rich wrapping cache."""

    def __init__(
        self, style: str, *, palette_role: Literal["body", "thought"]
    ) -> None:
        self.style = style
        self.palette_role = palette_role
        self.revision = 0
        self._chunks: list[str] = []
        self._plain_chunks: list[str] = []
        self._text = Text("", style=style)
        self._tab_cell_position = 0
        self._wraps: dict[int, tuple[int, list[Text], str]] = {}

    @property
    def plain(self) -> str:
        return "".join(self._plain_chunks)

    def append(self, value: str) -> None:
        raw_value = value
        self._plain_chunks.append(raw_value)
        expanded: list[str] = []
        for character in value:
            if character == "\r":
                continue
            if character == "\n":
                self._tab_cell_position = 0
            elif character == "\t":
                character = " " * (8 - self._tab_cell_position % 8)
            self._tab_cell_position += cell_len(character)
            expanded.append(character)
        value = "".join(expanded)
        self._chunks.append(value)
        self._text.append(raw_value)
        self.revision += 1

    def restyle(self, style: str) -> None:
        """Apply a new palette style and invalidate cached wrapped lines."""

        if style != self.style:
            self.style = style
            self._text.style = style
            self._wraps.clear()
            self.revision += 1

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self._text

    @staticmethod
    def _wrapped(console: Console, value: str, width: int, style: str) -> list[Text]:
        return list(Text(value, style=style).wrap(console, width))

    def tail(self, console: Console, width: int, height: int) -> list[Text]:
        """Return the visible wrapped tail, processing each appended chunk once."""

        consumed, stable, pending = self._wraps.get(width, (0, [], ""))
        pending += "".join(self._chunks[consumed:])
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            stable.extend(self._wrapped(console, line, width, self.style))
        wrapped = self._wrapped(console, pending, width, self.style)
        offsets = divide_line(pending, width)
        retained = height + 2
        if len(wrapped) > retained and len(offsets) >= retained:
            cut = offsets[-retained]
            stable.extend(wrapped[:-retained])
            pending = pending[cut:]
        self._wraps[width] = (len(self._chunks), stable, pending)
        visible = [
            *stable[-height:],
            *self._wrapped(console, pending, width, self.style),
        ]
        return visible[-height:] or [Text("", style=self.style)]
