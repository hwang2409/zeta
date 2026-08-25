"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

from io import StringIO

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
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
        self.revision = 0

    def update(self, rendered: RenderableType) -> None:
        text = getattr(rendered, "plain", None)
        if text is not None:
            self.output.append(text)
        if not self.finished:
            self.renderable = render_tool_progress(self.call, "\n".join(self.output))
        self.revision += 1

    def finish(self, rendered: RenderableType) -> None:
        self.finished = True
        self.renderable = rendered
        self.revision += 1


class TranscriptWidget(UIControl):
    """Render logical transcript units at the current width and stay at the bottom."""

    def __init__(self) -> None:
        self._units: list[RenderableType | None | _ToolUnit] = []
        self._tools: dict[str, _ToolUnit] = {}
        self._render_cache: dict[int, tuple[int, int, str]] = {}
        self._follow_tail = True
        self._scroll_offset = 0
        self._viewport_height = 1
        self._content_width = 80

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

    @property
    def follow_tail(self) -> bool:
        return self._follow_tail

    @property
    def scroll_offset(self) -> int:
        return self._scroll_offset

    def _set_scroll_offset(self, value: int) -> None:
        line_count = len(self.lines(self._content_width))
        tail = max(0, line_count - self._viewport_height)
        self._scroll_offset = min(max(0, value), tail)
        self._follow_tail = self._scroll_offset >= tail

    def page_up(self) -> None:
        self._set_scroll_offset(self._scroll_offset - self._viewport_height)
        self._follow_tail = False

    def page_down(self) -> None:
        self._set_scroll_offset(self._scroll_offset + self._viewport_height)

    def scroll_up(self) -> None:
        self._set_scroll_offset(self._scroll_offset - 3)
        self._follow_tail = False

    def scroll_down(self) -> None:
        self._set_scroll_offset(self._scroll_offset + 3)

    def _render_unit(self, unit: RenderableType | _ToolUnit, width: int) -> str:
        revision = unit.revision if isinstance(unit, _ToolUnit) else 0
        key = id(unit)
        cached = self._render_cache.get(key)
        if cached is not None and cached[0] == width and cached[1] == revision:
            return cached[2]
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            width=max(1, width),
            theme=RICH_THEME,
        )
        renderable = unit.renderable if isinstance(unit, _ToolUnit) else unit
        console.print(Padding(renderable, (0, 2, 0, 2)))
        rendered = output.getvalue().rstrip("\n")
        self._render_cache[key] = (width, revision, rendered)
        return rendered

    def render(self, width: int) -> str:
        rendered_units: list[str] = []
        for unit in self._units:
            if unit is None:
                rendered_units.append("")
                continue
            rendered_units.append(self._render_unit(unit, width))
        value = "\n".join(rendered_units).rstrip("\n")
        lines = value.splitlines()
        while lines and not Text.from_ansi(lines[0]).plain.strip():
            lines.pop(0)
        return "\n".join(lines)

    def lines(self, width: int) -> list[str]:
        return self.render(width).splitlines()

    def create_content(self, width: int, height: int | None) -> UIContent:
        self._content_width = max(1, width)
        self._viewport_height = max(1, height or 1)
        fragments = to_formatted_text(ANSI(self.render(width)))
        lines = list(split_lines(fragments)) or [[]]
        if self._follow_tail:
            self._scroll_offset = max(0, len(lines) - self._viewport_height)
        else:
            self._scroll_offset = min(
                self._scroll_offset,
                max(0, len(lines) - self._viewport_height),
            )
        cursor_y = min(self._scroll_offset, len(lines) - 1)
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=Point(x=0, y=cursor_y),
            show_cursor=False,
        )

    def vertical_scroll(self, window: Window) -> int:
        del window
        return self._scroll_offset

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self.scroll_up()
        elif mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self.scroll_down()
        else:
            return NotImplemented
        return None

    def window(self, *, height: AnyDimension | None = None) -> Window:
        return Window(
            self,
            height=height or Dimension(weight=1, min=1),
            wrap_lines=False,
            get_vertical_scroll=self.vertical_scroll,
        )
