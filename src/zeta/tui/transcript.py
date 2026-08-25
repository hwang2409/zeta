"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

from io import StringIO

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, Dimension
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.text import Text

from ..types import ToolCall
from .render import render_tool_progress
from .theme import RICH_THEME


class _ToolUnit:
    def __init__(self, call: ToolCall, initial: RenderableType) -> None:
        self.call = call
        self.renderable = initial
        self.output: list[str] = []
        self.finished = False

    def update(self, rendered: RenderableType) -> None:
        text = getattr(rendered, "plain", None)
        if text is not None:
            self.output.append(text)
        if not self.finished:
            self.renderable = render_tool_progress(self.call, "\n".join(self.output))

    def finish(self, rendered: RenderableType) -> None:
        self.finished = True
        self.renderable = rendered


class TranscriptWidget(UIControl):
    """Render logical transcript units at the current width and stay at the bottom."""

    def __init__(self) -> None:
        self._units: list[RenderableType | None | _ToolUnit] = []
        self._tools: dict[str, _ToolUnit] = {}

    @property
    def units(self) -> tuple[RenderableType | None | _ToolUnit, ...]:
        return tuple(self._units)

    def append(self, renderable: RenderableType) -> None:
        self._units.append(renderable)

    def append_blank(self) -> None:
        self._units.append(None)

    def start_tool(
        self, call_id: str, call: ToolCall, renderable: RenderableType
    ) -> None:
        unit = _ToolUnit(call, renderable)
        self._units.append(unit)
        self._tools[call_id] = unit

    def update_tool(self, call_id: str, rendered: RenderableType) -> None:
        unit = self._tools.get(call_id)
        if unit is not None:
            unit.update(rendered)

    def finish_tool(self, call_id: str, rendered: RenderableType) -> None:
        unit = self._tools.pop(call_id, None)
        if unit is not None:
            unit.finish(rendered)
        else:
            self.append(rendered)

    def discard_tools(self) -> None:
        if not self._tools:
            return
        active = set(self._tools.values())
        self._units[:] = [unit for unit in self._units if unit not in active]
        self._tools.clear()

    def render(self, width: int) -> str:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            width=max(1, width),
            theme=RICH_THEME,
        )
        for unit in self._units:
            if unit is None:
                output.write("\n")
                continue
            renderable = unit.renderable if isinstance(unit, _ToolUnit) else unit
            console.print(Padding(renderable, (0, 2, 0, 2)))
        value = output.getvalue().rstrip("\n")
        lines = value.splitlines()
        while lines and not Text.from_ansi(lines[0]).plain.strip():
            lines.pop(0)
        return "\n".join(lines)

    def lines(self, width: int) -> list[str]:
        return self.render(width).splitlines()

    def create_content(self, width: int, height: int | None) -> UIContent:
        del height
        fragments = to_formatted_text(ANSI(self.render(width)))
        lines = list(split_lines(fragments)) or [[]]
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=Point(x=0, y=len(lines) - 1),
            show_cursor=False,
        )

    def window(self, *, height: AnyDimension | None = None) -> Window:
        return Window(
            self,
            height=height or Dimension(weight=1, min=1),
            wrap_lines=False,
        )
