"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

import re
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
from .layout import CONTENT_MARGIN
from .agent_card import AgentCard
from .render import render_event, render_tool_progress
from .theme import RICH_THEME


class _ToolUnit:
    def __init__(
        self,
        call: ToolCall,
        initial: RenderableType,
        start_event: StreamEvent | None = None,
    ) -> None:
        self.call = call
        self.output: list[str] = []
        self.finished = False
        self.revision = 0
        self.card = AgentCard(call)
        if start_event is not None:
            self.card.start(start_event)
        self.renderable = initial

    @property
    def active_card(self) -> bool:
        return self.card.active

    def update(
        self, rendered: RenderableType, event: StreamEvent | None = None
    ) -> None:
        text = getattr(rendered, "plain", None)
        if isinstance(text, str):
            self.output.append(text)
        if not self.finished:
            self.renderable = self.card.update(rendered, event) or render_tool_progress(
                self.call, "\n".join(self.output)
            )
        self.revision += 1

    def refresh(self) -> None:
        rendered = self.card.refresh()
        if rendered is not None:
            self.renderable = rendered
            self.revision += 1

    def finish(self, rendered: RenderableType, event: StreamEvent | None = None) -> None:
        self.finished = True
        self.renderable = self.card.finish(event) or rendered
        self.revision += 1

    def toggle(self) -> bool:
        rendered = self.card.toggle()
        if rendered is None:
            return False
        self.renderable = rendered
        self.revision += 1
        return True


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
        self._background_tools: set[str] = set()
        self._card_units: dict[str, _ToolUnit] = {}
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

    @property
    def has_active_agent(self) -> bool:
        return any(unit.active_card for unit in self._tools.values())

    @property
    def _agent_units(self) -> dict[str, _ToolUnit]:
        """Keep the old diagnostic view while card selection stays generic."""

        return {
            call_id: unit
            for call_id, unit in self._card_units.items()
            if unit.card.supported
        }

    def _bump_revision(self) -> None:
        self._revision += 1
        self._parsed_cache.clear()

    def _append_unit(self, value: RenderableType | _ToolUnit | None) -> _TranscriptUnit:
        unit = _TranscriptUnit(self._next_key, value)
        self._units.append(unit)
        self._next_key += 1
        self._bump_revision()
        return unit

    def append(self, renderable: RenderableType) -> _TranscriptUnit:
        return self._append_unit(renderable)

    def replace(
        self, unit: _TranscriptUnit, renderable: RenderableType
    ) -> _TranscriptUnit:
        if unit not in self._units:
            return self._append_unit(renderable)
        unit.value = renderable
        self._render_cache.pop(unit.key, None)
        self._bump_revision()
        return unit

    def remove(self, unit: _TranscriptUnit) -> None:
        if unit not in self._units:
            return
        self._units.remove(unit)
        if self._anchor is not None and self._anchor[0] is unit:
            self._anchor = None
        self._render_cache.pop(unit.key, None)
        self._bump_revision()

    def append_blank(self) -> None:
        self._append_unit(None)

    def clear(self) -> None:
        """Remove all rendered transcript units."""

        self._units.clear()
        self._tools.clear()
        self._background_tools.clear()
        self._card_units.clear()
        self._render_cache.clear()
        self._parsed_cache.clear()
        self._line_locations.clear()
        self._anchor = None
        self._scroll_offset = 0
        self._follow_tail = True
        self._bump_revision()

    def start_tool(
        self,
        call_id: str,
        call: ToolCall,
        renderable: RenderableType,
        start_event: StreamEvent | None = None,
    ) -> None:
        unit = _ToolUnit(call, renderable, start_event)
        self._append_unit(unit)
        self._tools[call_id] = unit
        self._card_units[call_id] = unit

    def update_tool(
        self,
        call_id: str,
        rendered: RenderableType,
        event: StreamEvent | None = None,
    ) -> None:
        unit = self._tools.get(call_id)
        if unit is not None:
            unit.update(rendered, event)
            self._bump_revision()

    def finish_tool(
        self,
        call_id: str,
        rendered: RenderableType,
        event: StreamEvent | None = None,
    ) -> None:
        unit = self._tools.pop(call_id, None)
        if unit is not None:
            unit.finish(rendered, event)
            self._bump_revision()
        else:
            self.append(rendered)
        self._background_tools.discard(call_id)

    def mark_tool_background(self, call_id: str) -> None:
        """Keep a running child card after its parent tool call returns."""

        if call_id in self._tools:
            self._background_tools.add(call_id)

    def set_tool_child_session_path(self, call_id: str, path: str) -> None:
        unit = self._tools.get(call_id)
        if unit is not None:
            unit.card.set_child_session_path(path)

    def discard_tools(self) -> None:
        if not self._tools:
            return
        active = {
            unit
            for call_id, unit in self._tools.items()
            if call_id not in self._background_tools
        }
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
        self._tools = {
            call_id: unit
            for call_id, unit in self._tools.items()
            if call_id in self._background_tools
        }
        self._bump_revision()

    def toggle_latest_agent(self) -> bool:
        """Toggle the newest child card and keep its tail bounded."""

        for unit in reversed(tuple(self._card_units.values())):
            if unit.toggle():
                self._bump_revision()
                return True
        return False

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
            no_color=False,
            width=max(1, width),
            theme=RICH_THEME,
        )
        renderable = value.renderable if isinstance(value, _ToolUnit) else value
        console.print(renderable)
        rendered = "\n".join(
            line.rstrip(" ") for line in output.getvalue().splitlines()
        )
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
                matched_length = len(content)
                if offset < 0:
                    match = re.search(r"[\w]+(?:[-'][\w]+)*", content)
                    if match is not None:
                        offset = source.find(match.group(), source_offset)
                        matched_length = len(match.group())
                    if offset < 0:
                        offset = source_offset
                raw_lines.append((line, unit, offset))
                source_offset = offset + matched_length
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
        return line.rstrip()

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
        prefix_lines = max(0, self._viewport_height - len(lines))
        visible_lines = ([[] for _ in range(prefix_lines)] + lines)
        cursor_y = min(prefix_lines + self._scroll_offset, len(visible_lines) - 1)
        return UIContent(
            get_line=lambda index: visible_lines[index],
            line_count=len(visible_lines),
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
        self._background_tool_ids: set[str] = set()
        self._pending_tool_renders: list[RenderableType] = []
        self._tool_region_units: dict[str, _ToolUnit] = {}
        self._tool_region: Live | None = None
        self._thinking_live: Live | None = None
        self._thinking_unit: _TranscriptUnit | None = None
        self._assistant_live: Live | None = None
        self._assistant_unit: _TranscriptUnit | None = None
        self._assistant_message_units: list[_TranscriptUnit] = []
        self._assistant_message_region: list[_TranscriptUnit] = []

    @property
    def tool_region(self) -> Live | None:
        return self._tool_region

    @property
    def has_active_agent(self) -> bool:
        return self.transcript.has_active_agent or any(
            unit.active_card for unit in self._tool_region_units.values()
        )

    def _append(self, renderable: RenderableType) -> None:
        self.transcript.append(renderable)

    def append_blank(self) -> None:
        self.transcript.append_blank()

    def print(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        self._print_callback(renderable)

    def print_unit(self, renderable: RenderableType | None) -> _TranscriptUnit | None:
        if renderable is None:
            return None
        if self._printed_units:
            if self._full_screen_active():
                self.append_blank()
            else:
                self.console.print()
        self.print(renderable)
        self._printed_units = True
        if self._full_screen_active() and self.transcript._units:
            return self.transcript._units[-1]
        return None

    def print_assistant(self, renderable: RenderableType | None) -> bool:
        if renderable is None:
            return False
        self.update_assistant(renderable)
        plain = getattr(renderable, "plain", None)
        return plain is None or bool(plain.strip())

    def update_assistant(self, rendered: RenderableType) -> None:
        """Replace the one mutable unit used by an in-flight assistant message."""

        if self._full_screen_active():
            if self._assistant_unit is None:
                unit_start = len(self.transcript._units)
                self._assistant_unit = self.print_unit(rendered)
                self._assistant_message_region.extend(self.transcript._units[unit_start:])
                if self._assistant_unit is not None:
                    self._assistant_message_units.append(self._assistant_unit)
            else:
                self._assistant_unit = self.transcript.replace(
                    self._assistant_unit, rendered
                )
        else:
            if self._assistant_live is None:
                self._assistant_live = Live(
                    Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN)),
                    console=self.console,
                    transient=True,
                    refresh_per_second=20,
                )
                self._assistant_live.start()
            else:
                self._assistant_live.update(
                    Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))
                )
        self._assistant_unit_open = True

    def finish_assistant(
        self,
        rendered: RenderableType,
        *,
        preserve_inline: bool = False,
    ) -> None:
        """Commit the completed assistant message into its existing unit."""

        if self._full_screen_active():
            if self._assistant_unit is None:
                self._assistant_unit = self.print_unit(rendered)
            else:
                self._assistant_unit = self.transcript.replace(
                    self._assistant_unit, rendered
                )
        else:
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            if preserve_inline:
                self.print_unit(rendered)
        self._assistant_unit = None
        self._assistant_unit_open = False

    def finish_assistant_message(self, rendered: RenderableType | None) -> None:
        """Render the complete assistant message in one transcript unit."""

        if self._full_screen_active():
            if rendered is not None:
                unit = next(
                    (
                        candidate
                        for candidate in self._assistant_message_units
                        if candidate in self.transcript._units
                    ),
                    None,
                )
                if unit is None:
                    unit = self.print_unit(rendered)
                else:
                    unit = self.transcript.replace(unit, rendered)
                for candidate in self._assistant_message_units:
                    if candidate is not unit:
                        self.transcript.remove(candidate)
            else:
                for candidate in self._assistant_message_region:
                    self.transcript.remove(candidate)
        else:
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            if rendered is not None:
                self.print_unit(rendered)
        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self._assistant_unit = None
        self._assistant_unit_open = False

    def reset_assistant_unit(self) -> None:
        self._assistant_unit_open = False
        self._assistant_unit = None

    def reset_assistant_message(self) -> None:
        """Forget the units owned by an incomplete assistant message."""

        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self.reset_assistant_unit()

    def start_thinking(self, rendered: Text) -> None:
        self.reset_assistant_unit()
        if self._full_screen_active():
            self._thinking_unit = self.print_unit(rendered)
        else:
            self._thinking_live = Live(
                Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN)),
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )
            self._thinking_live.start()

    def update_thinking(self, rendered: Text) -> None:
        if self._full_screen_active():
            if self._thinking_unit is None:
                self._thinking_unit = self.print_unit(rendered)
            else:
                self._thinking_unit = self.transcript.replace(
                    self._thinking_unit, rendered
                )
        elif self._thinking_live is not None:
            self._thinking_live.update(
                Padding(rendered, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))
            )

    def finish_thinking(self, rendered: Text) -> None:
        if self._full_screen_active():
            if self._thinking_unit is None:
                self._thinking_unit = self.print_unit(rendered)
            else:
                self._thinking_unit = self.transcript.replace(
                    self._thinking_unit, rendered
                )
            self._thinking_unit = None
        elif self._thinking_live is not None:
            self._thinking_live.stop()
            self._thinking_live = None
            self.print_unit(rendered)
        self.reset_assistant_unit()

    def update_tool_region(self, event: StreamEvent) -> bool:
        rendered = render_event(event)
        if not isinstance(rendered, Text):
            return False
        if self._full_screen_active():
            if event.tool_call is not None:
                self.transcript.update_tool(event.tool_call.id, rendered, event)
            return bool(event.delta and event.delta.strip())
        call = event.tool_call
        if call is not None and call.id not in self._tool_region_units:
            self._tool_region_units[call.id] = _ToolUnit(call, rendered, event)
        unit = self._tool_region_units.get(call.id) if call is not None else None
        if unit is None:
            return False
        if self._tool_region is None:
            self._tool_region = Live(
                self._tool_region_renderable(),
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )
        unit.update(rendered, event)
        self._tool_region.update(self._tool_region_renderable())
        return bool(event.delta and event.delta.strip())

    def _tool_region_renderable(self) -> Padding | Group:
        renderables = [unit.renderable for unit in self._tool_region_units.values()]
        content: RenderableType = Group(*renderables) if len(renderables) > 1 else renderables[0]
        return Padding(content, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))

    def refresh_active_agents(self) -> None:
        """Refresh elapsed time without adding child events to the parent store."""

        for unit in self.transcript._tools.values():
            unit.refresh()
        if self._tool_region is not None:
            for unit in self._tool_region_units.values():
                unit.refresh()
            self._tool_region.update(self._tool_region_renderable())

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
                    self.append_blank()
                self.transcript.start_tool(
                    event.tool_call.id,
                    event.tool_call,
                    rendered,
                    event,
                )
                self._printed_units = True
            else:
                self.print_unit(rendered)
            if not self._full_screen_active() and event.tool_call is not None:
                self._tool_region_units[event.tool_call.id] = _ToolUnit(
                    event.tool_call, rendered, event
                )
            return ToolEventPresentation(visible_output=True)
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            return ToolEventPresentation(
                visible_output=self.update_tool_region(event)
            )
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return None

        structured = (
            event.tool_result.structured_content
            if event.tool_result is not None
            else None
        )
        is_background_start = (
            structured is not None and structured.get("status") == "running"
        )
        if event.tool_call is not None:
            self._active_tool_calls.discard(event.tool_call.id)
            if is_background_start:
                self._background_tool_ids.add(event.tool_call.id)
                if self._full_screen_active():
                    self.transcript.mark_tool_background(event.tool_call.id)
                path = structured.get("child_session_path") if structured else None
                if isinstance(path, str) and path:
                    if self._full_screen_active():
                        self.transcript.set_tool_child_session_path(
                            event.tool_call.id, path
                        )
                    else:
                        unit = self._tool_region_units.get(event.tool_call.id)
                        if unit is not None:
                            unit.card.set_child_session_path(path)
                return ToolEventPresentation(visible_output=True)
        rendered = render_event(event)
        if not self._full_screen_active() and event.tool_call is not None:
            unit = self._tool_region_units.get(event.tool_call.id)
            if unit is not None and rendered is not None:
                unit.finish(rendered, event)
                rendered = unit.renderable
        if rendered is not None:
            if self._full_screen_active() and event.tool_call is not None:
                self.transcript.finish_tool(event.tool_call.id, rendered, event)
            else:
                self._pending_tool_renders.append(rendered)
        if event.tool_call is not None:
            self._background_tool_ids.discard(event.tool_call.id)
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
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
            if self._background_tool_ids:
                self._tool_region.update(self._tool_region_renderable())
            else:
                if final_renders:
                    self._tool_region.update(Group(*final_renders))
                self._tool_region.stop()
                self._tool_region = None
        else:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
        for rendered in final_renders:
            self.print(rendered)
        self.reset_assistant_unit()

    def discard_tool_region(self) -> None:
        self._pending_tool_renders.clear()
        if self._full_screen_active():
            self.transcript.discard_tools()
        if self._tool_region is not None:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }
            if not self._background_tool_ids:
                self._tool_region.stop()
                self._tool_region = None
        else:
            self._tool_region_units = {
                call_id: unit
                for call_id, unit in self._tool_region_units.items()
                if call_id in self._background_tool_ids
            }

    def clear_active_tool_calls(self) -> None:
        self._active_tool_calls.clear()

    def clear(self) -> None:
        """Reset presentation state before rebuilding the transcript."""

        self._tool_region = None
        self._tool_region_units.clear()
        self._pending_tool_renders.clear()
        self._active_tool_calls.clear()
        self._thinking_live = None
        self._assistant_live = None
        self._thinking_unit = None
        self._assistant_unit = None
        self._assistant_message_units.clear()
        self._assistant_message_region.clear()
        self._assistant_unit_open = False
        self._printed_units = False
        self.transcript.clear()
