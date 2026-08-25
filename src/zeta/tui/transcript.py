"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from ..types import StreamEvent, StreamEventType, ToolCall
from .render import render_event, render_tool_progress
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


class _TranscriptUnit:
    def __init__(
        self, key: int, value: RenderableType | _ToolUnit | None
    ) -> None:
        self.key = key
        self.value = value


class TranscriptWidget(UIControl):
    """Render logical transcript units at the current width and stay at the bottom."""

    def __init__(self) -> None:
        self._units: list[_TranscriptUnit | None] = []
        self._tools: dict[str, _ToolUnit] = {}
        self._render_cache: dict[int, tuple[int, int, str]] = {}
        self._parsed_cache: OrderedDict[int, list[list[tuple[str, str]]]] = (
            OrderedDict()
        )
        self._revision = 0
        self._next_key = 0
        self._follow_tail = True
        self._scroll_offset = 0
        self._viewport_height = 1
        self._content_width = 80
        self._line_locations: list[tuple[_TranscriptUnit | None, int]] = []
        self._anchor: tuple[_TranscriptUnit | None, int] | None = None

    @property
    def units(self) -> tuple[RenderableType | None | _ToolUnit, ...]:
        return tuple(unit.value if unit is not None else None for unit in self._units)

    def _bump_revision(self) -> None:
        self._revision += 1
        self._parsed_cache.clear()

    def _append_unit(self, value: RenderableType | _ToolUnit | None) -> None:
        self._units.append(_TranscriptUnit(self._next_key, value))
        self._next_key += 1
        self._bump_revision()

    def append(self, renderable: RenderableType) -> None:
        self._append_unit(renderable)

    def append_blank(self) -> None:
        self._append_unit(None)

    def start_tool(
        self, call_id: str, call: ToolCall, renderable: RenderableType
    ) -> None:
        unit = _ToolUnit(call, renderable)
        self._append_unit(unit)
        self._tools[call_id] = unit

    def update_tool(self, call_id: str, rendered: RenderableType) -> None:
        unit = self._tools.get(call_id)
        if unit is not None:
            unit.update(rendered)
            self._bump_revision()

    def finish_tool(self, call_id: str, rendered: RenderableType) -> None:
        unit = self._tools.pop(call_id, None)
        if unit is not None:
            unit.finish(rendered)
            self._bump_revision()
        else:
            self.append(rendered)

    def discard_tools(self) -> None:
        if not self._tools:
            return
        active = set(self._tools.values())
        removed_keys = {
            unit.key
            for unit in self._units
            if unit is not None and unit.value in active
        }
        self._units[:] = [
            unit
            for unit in self._units
            if unit is None or unit.value not in active
        ]
        if self._anchor is not None and self._anchor[0] not in self._units:
            self._anchor = None
        for key in removed_keys:
            self._render_cache.pop(key, None)
        self._tools.clear()
        self._bump_revision()

    @property
    def follow_tail(self) -> bool:
        return self._follow_tail

    @property
    def scroll_offset(self) -> int:
        return self._scroll_offset

    def _set_scroll_offset(self, value: int) -> None:
        line_count = len(self._parsed_lines(self._content_width))
        tail = max(0, line_count - self._viewport_height)
        self._scroll_offset = min(max(0, value), tail)
        self._follow_tail = self._scroll_offset >= tail
        locations = self._locations(self._content_width)
        if locations:
            self._anchor = locations[min(self._scroll_offset, len(locations) - 1)]

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

    def _render_unit(self, unit: _TranscriptUnit, width: int) -> str:
        value = unit.value
        if value is None:
            return ""
        revision = value.revision if isinstance(value, _ToolUnit) else 0
        key = unit.key
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
        renderable = value.renderable if isinstance(value, _ToolUnit) else value
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

    def _parsed_lines(self, width: int) -> list[list[tuple[str, str]]]:
        cached = self._parsed_cache.get(width)
        if cached is not None:
            self._parsed_cache.move_to_end(width)
            return cached
        fragments = to_formatted_text(ANSI(self.render(width)))
        lines = list(split_lines(fragments)) or [[]]
        self._parsed_cache[width] = lines
        self._parsed_cache.move_to_end(width)
        while len(self._parsed_cache) > 3:
            self._parsed_cache.popitem(last=False)
        return lines

    def _locations(self, width: int) -> list[tuple[_TranscriptUnit | None, int]]:
        raw_lines: list[tuple[str, _TranscriptUnit | None, int]] = []
        for unit in self._units:
            if unit is None:
                raw_lines.append(("", None, 0))
                continue
            rendered = self._render_unit(unit, width)
            rendered_lines = self._plain_lines(rendered)
            renderable = (
                unit.value.renderable
                if isinstance(unit.value, _ToolUnit)
                else unit.value
            )
            source = getattr(renderable, "plain", None)
            if not isinstance(source, str):
                source = "\n".join(
                    self._strip_padding(line) for line in rendered_lines
                )
            source_offset = 0
            for line in rendered_lines:
                content = self._strip_padding(line)
                offset = source.find(content, source_offset)
                if offset < 0:
                    offset = source_offset
                raw_lines.append((line, unit, offset))
                source_offset = offset + len(content)
        while raw_lines and not raw_lines[0][0].strip():
            raw_lines.pop(0)
        return [(unit, text_offset) for _, unit, text_offset in raw_lines]

    @staticmethod
    def _plain_lines(rendered: str) -> list[str]:
        if "\x1b" in rendered:
            rendered = Text.from_ansi(rendered).plain
        return rendered.splitlines() or [""]

    @staticmethod
    def _strip_padding(line: str) -> str:
        if "\x1b" in line:
            line = Text.from_ansi(line).plain
        plain = line
        if plain.startswith("  "):
            plain = plain[2:]
        return plain.rstrip()

    @staticmethod
    def _anchor_index(
        locations: list[tuple[_TranscriptUnit | None, int]],
        anchor: tuple[_TranscriptUnit | None, int],
    ) -> int | None:
        for index, location in enumerate(locations):
            if location == anchor:
                return index
        unit, text_offset = anchor
        if unit is None:
            return None
        candidates = [
            (index, offset)
            for index, (candidate, offset) in enumerate(locations)
            if candidate is unit
        ]
        if not candidates:
            return None
        preceding = [item for item in candidates if item[1] <= text_offset]
        if preceding:
            return preceding[-1][0]
        return candidates[0][0]

    def create_content(self, width: int, height: int | None) -> UIContent:
        width = max(1, width)
        height = max(1, height or 1)
        self._content_width = max(1, width)
        self._viewport_height = height
        lines = self._parsed_lines(width)
        locations = self._locations(width)
        if self._follow_tail:
            self._scroll_offset = max(0, len(lines) - self._viewport_height)
        elif self._anchor is not None:
            anchor_index = self._anchor_index(locations, self._anchor)
            self._scroll_offset = (
                anchor_index
                if anchor_index is not None
                else min(
                    self._scroll_offset,
                    max(0, len(lines) - self._viewport_height),
                )
            )
        else:
            self._scroll_offset = min(
                self._scroll_offset,
                max(0, len(lines) - self._viewport_height),
            )
        tail = max(0, len(lines) - self._viewport_height)
        if not self._follow_tail and self._scroll_offset >= tail:
            self._follow_tail = True
        if self._follow_tail:
            self._scroll_offset = tail
        self._line_locations = locations
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


@dataclass(frozen=True)
class ToolEventPresentation:
    visible_output: bool = False
    stop_after_tool: bool = False


class TranscriptPresenter:
    """Own transcript output and tool-card presentation for the TUI."""

    def __init__(
        self,
        transcript: TranscriptWidget,
        console: Console,
        full_screen_active: Callable[[], bool],
        print_callback: Callable[[RenderableType | None], None],
    ) -> None:
        self.transcript = transcript
        self.console = console
        self._full_screen_active = full_screen_active
        self._print_callback = print_callback
        self._printed_units = False
        self._assistant_unit_open = False
        self._active_tool_calls: set[str] = set()
        self._pending_tool_renders: list[RenderableType] = []
        self._tool_region: Live | None = None
        self._tool_region_text: Text | None = None
        self._tool_region_call: ToolCall | None = None

    @property
    def tool_region(self) -> Live | None:
        return self._tool_region

    def _append(self, renderable: RenderableType) -> None:
        self.transcript.append(renderable)

    def append_blank(self) -> None:
        self.transcript.append_blank()

    def print(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        self._print_callback(renderable)

    def print_unit(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        if self._printed_units:
            if self._full_screen_active():
                self._append_blank()
            else:
                self.console.print()
        self.print(renderable)
        self._printed_units = True

    def print_assistant(self, renderable: RenderableType | None) -> bool:
        if renderable is None:
            return False
        if not self._assistant_unit_open:
            self.print_unit(renderable)
            self._assistant_unit_open = True
        else:
            self.print(renderable)
        plain = getattr(renderable, "plain", None)
        return plain is None or bool(plain.strip())

    def reset_assistant_unit(self) -> None:
        self._assistant_unit_open = False

    def update_tool_region(self, event: StreamEvent) -> bool:
        rendered = render_event(event)
        if not isinstance(rendered, Text):
            return False
        if self._full_screen_active():
            if event.tool_call is not None:
                self.transcript.update_tool(event.tool_call.id, rendered)
            return bool(event.delta and event.delta.strip())
        if self._tool_region is None:
            self._tool_region_text = Text()
            self._tool_region_call = event.tool_call
            self._tool_region = Live(
                Padding(
                    render_tool_progress(
                        self._tool_region_call,
                        self._tool_region_text.plain,
                    ),
                    (0, 2, 0, 2),
                )
                if self._tool_region_call is not None
                else self._tool_region_text,
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )
        if self._tool_region_text is None:
            return False
        if self._tool_region_text:
            self._tool_region_text.append("\n")
        self._tool_region_text.append(rendered)
        if self._tool_region_call is not None:
            self._tool_region.update(
                Padding(
                    render_tool_progress(
                        self._tool_region_call,
                        self._tool_region_text.plain,
                    ),
                    (0, 2, 0, 2),
                )
            )
        else:
            self._tool_region.update(self._tool_region_text)
        return bool(event.delta and event.delta.strip())

    def handle_tool_event(
        self,
        event: StreamEvent,
        *,
        aborted: bool,
    ) -> ToolEventPresentation | None:
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            self.reset_assistant_unit()
            if event.tool_call is not None:
                self._active_tool_calls.add(event.tool_call.id)
            rendered = render_event(event)
            if rendered is None:
                return ToolEventPresentation()
            if self._full_screen_active() and event.tool_call is not None:
                if self._printed_units:
                    self._append_blank()
                self.transcript.start_tool(
                    event.tool_call.id,
                    event.tool_call,
                    rendered,
                )
                self._printed_units = True
            else:
                self.print_unit(rendered)
            return ToolEventPresentation(visible_output=True)
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            return ToolEventPresentation(
                visible_output=self.update_tool_region(event)
            )
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return None

        if event.tool_call is not None:
            self._active_tool_calls.discard(event.tool_call.id)
        rendered = render_event(event)
        if rendered is not None:
            if self._full_screen_active() and event.tool_call is not None:
                self.transcript.finish_tool(event.tool_call.id, rendered)
            else:
                self._pending_tool_renders.append(rendered)
        if not self._active_tool_calls:
            self.commit_tool_region()
        return ToolEventPresentation(
            visible_output=rendered is not None,
            stop_after_tool=aborted,
        )

    def commit_tool_region(self) -> None:
        final_renders = self._pending_tool_renders
        self._pending_tool_renders = []
        if self._tool_region is not None:
            if final_renders:
                self._tool_region.update(Group(*final_renders))
            self._tool_region.stop()
            self._tool_region = None
            self._tool_region_text = None
            self._tool_region_call = None
        for rendered in final_renders:
            self.print(rendered)
        self.reset_assistant_unit()

    def discard_tool_region(self) -> None:
        self._pending_tool_renders.clear()
        if self._full_screen_active():
            self.transcript.discard_tools()
        if self._tool_region is not None:
            self._tool_region.stop()
            self._tool_region = None
            self._tool_region_text = None
            self._tool_region_call = None

    def clear_active_tool_calls(self) -> None:
        self._active_tool_calls.clear()
