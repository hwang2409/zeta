"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

import re
from collections import OrderedDict
from io import StringIO

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from rich.console import Console, RenderableType
from rich.text import Text

from ..types import (
    RedactedThinkingContent,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from .agent_card import AgentCard
from .render import render_tool_progress
from .theme import RICH_THEME
from .transcript_search import HighlightCache, SearchMatch, find_matches


MAX_TOOL_TAIL_CHARS = 4_096


def stream_key(
    event: StreamEvent,
) -> tuple[str | None, tuple[str, object] | None]:
    content = event.content
    index = event.data.get("index")
    identity = (
        ("index", index)
        if isinstance(index, (int, str, tuple))
        else None
    )
    if isinstance(content, ThinkingContent):
        return "thinking", identity or (
            ("signature", content.signature)
            if content.signature is not None
            else ("kind", "thinking")
        )
    if isinstance(content, RedactedThinkingContent):
        return "redacted-thinking", identity or ("data", content.data)
    if isinstance(content, TextContent) or event.delta is not None:
        return "assistant", ("kind", "assistant")
    return None, None


ToolLifecycleKey = tuple[str | None, str]


def _tool_lifecycle_key(
    call_id: str | ToolLifecycleKey,
    event: StreamEvent | None = None,
) -> ToolLifecycleKey:
    if isinstance(call_id, tuple):
        return call_id
    session_id = event.data.get("agent_instance_id") if event is not None else None
    return (session_id if type(session_id) is str else None, call_id)


def _event_tool_lifecycle_key(event: StreamEvent) -> ToolLifecycleKey | None:
    if event.tool_call is None:
        return None
    return _tool_lifecycle_key(event.tool_call.id, event)


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
            output = "".join(self.output)
            if len(output) > MAX_TOOL_TAIL_CHARS:
                self.output = [output[-MAX_TOOL_TAIL_CHARS:]]
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
        self._tools: dict[ToolLifecycleKey, _ToolUnit] = {}
        self._background_tools: set[ToolLifecycleKey] = set()
        self._card_units: dict[ToolLifecycleKey, _ToolUnit] = {}
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
        self._locations_revision = -1
        self._locations_cache: OrderedDict[
            int, tuple[int, list[tuple[_TranscriptUnit | None, int]]]
        ] = OrderedDict()
        self._anchor: tuple[_TranscriptUnit | None, int] | None = None
        self._user_units: list[_TranscriptUnit] = []
        self._search_active = False
        self._search_query = ""
        self._search_index = 0
        self._search_cache: OrderedDict[
            tuple[int, int, str], list[SearchMatch]
        ] = OrderedDict()
        self._highlight_cache: tuple[tuple[int, int, str], HighlightCache] | None = None

    @property
    def units(self) -> tuple[RenderableType | None | _ToolUnit, ...]:
        return tuple(unit.value if unit is not None else None for unit in self._units)

    @property
    def has_active_agent(self) -> bool:
        return any(unit.active_card for unit in self._tools.values())

    @property
    def _agent_units(self) -> dict[ToolLifecycleKey, _ToolUnit]:
        """Keep the old diagnostic view while card selection stays generic."""

        return {
            call_id: unit
            for call_id, unit in self._card_units.items()
            if unit.card.supported
        }

    def _bump_revision(self) -> None:
        self._revision += 1
        self._parsed_cache.clear()
        self._locations_cache.clear()
        self._search_cache.clear()
        self._highlight_cache = None

    def _append_unit(self, value: RenderableType | _ToolUnit | None) -> _TranscriptUnit:
        unit = _TranscriptUnit(self._next_key, value)
        self._units.append(unit)
        self._next_key += 1
        self._bump_revision()
        return unit

    def mark_user(self, unit: _TranscriptUnit | None) -> None:
        """Mark an existing full-screen unit as a user message."""

        if unit is not None and unit in self._units and unit not in self._user_units:
            self._user_units.append(unit)

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

    def remove(self, unit: _TranscriptUnit, *, leading_blank: bool = False) -> None:
        if unit not in self._units:
            return
        if leading_blank:
            index = self._units.index(unit)
            previous = self._units[index - 1] if index > 0 else None
            if previous is not None and previous.value is None:
                self.remove(previous)
        self._units.remove(unit)
        if self._anchor is not None and self._anchor[0] is unit:
            self._anchor = None
        if unit in self._user_units:
            self._user_units.remove(unit)
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
        self._locations_cache.clear()
        self._anchor = None
        self._user_units.clear()
        self._search_cache.clear()
        self._highlight_cache = None
        self._scroll_offset = 0
        self._follow_tail = True
        self._bump_revision()

    def start_tool(
        self,
        call_id: str | ToolLifecycleKey,
        call: ToolCall,
        renderable: RenderableType,
        start_event: StreamEvent | None = None,
    ) -> None:
        unit = _ToolUnit(call, renderable, start_event)
        lifecycle_key = _tool_lifecycle_key(call_id, start_event)
        self._append_unit(unit)
        self._tools[lifecycle_key] = unit
        self._card_units[lifecycle_key] = unit

    def update_tool(
        self,
        call_id: str | ToolLifecycleKey,
        rendered: RenderableType,
        event: StreamEvent | None = None,
    ) -> None:
        unit = self._tools.get(_tool_lifecycle_key(call_id, event))
        if unit is not None:
            unit.update(rendered, event)
            self._bump_revision()

    def refresh_active_agents(self) -> None:
        """Refresh active cards and invalidate derived transcript caches once."""

        refreshed = False
        for unit in self._tools.values():
            revision = unit.revision
            unit.refresh()
            refreshed = refreshed or unit.revision != revision
        if refreshed:
            self._bump_revision()

    def finish_tool(
        self,
        call_id: str | ToolLifecycleKey,
        rendered: RenderableType,
        event: StreamEvent | None = None,
    ) -> None:
        lifecycle_key = _tool_lifecycle_key(call_id, event)
        unit = self._tools.pop(lifecycle_key, None)
        if unit is not None:
            unit.finish(rendered, event)
            self._bump_revision()
        else:
            self.append(rendered)
        self._background_tools.discard(lifecycle_key)

    def mark_tool_background(self, call_id: str | ToolLifecycleKey) -> None:
        """Keep a running child card after its parent tool call returns."""

        lifecycle_key = _tool_lifecycle_key(call_id)
        if lifecycle_key in self._tools:
            self._background_tools.add(lifecycle_key)

    def set_tool_child_session_path(
        self, call_id: str | ToolLifecycleKey, path: str
    ) -> None:
        unit = self._tools.get(_tool_lifecycle_key(call_id))
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
        self._user_units[:] = [unit for unit in self._user_units if unit in self._units]
        if self._anchor is not None and self._anchor[0] not in self._units:
            self._anchor = None
        for key in removed_keys:
            self._render_cache.pop(key, None)
        self._tools = {
            lifecycle_key: unit
            for lifecycle_key, unit in self._tools.items()
            if lifecycle_key in self._background_tools
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

    def _set_scroll_offset(self, value: int, *, allow_follow_tail: bool = True) -> None:
        line_count = len(self._parsed_lines(self._content_width))
        tail = max(0, line_count - self._viewport_height)
        self._scroll_offset = min(max(0, value), tail)
        self._follow_tail = allow_follow_tail and self._scroll_offset >= tail
        locations = self._locations(self._content_width)
        if locations:
            self._anchor = locations[min(self._scroll_offset, len(locations) - 1)]

    def page_up(self) -> None:
        self._set_scroll_offset(self._scroll_offset - self._viewport_height)

    def page_down(self) -> None:
        self._set_scroll_offset(self._scroll_offset + self._viewport_height)

    def scroll_up(self) -> None:
        self._set_scroll_offset(self._scroll_offset - 3)

    def scroll_down(self) -> None:
        self._set_scroll_offset(self._scroll_offset + 3)

    @property
    def search_active(self) -> bool:
        return self._search_active

    @property
    def search_query(self) -> str:
        return self._search_query

    def begin_search(self) -> None:
        self._search_active = True
        self._search_query = ""
        self._search_index = 0
        self._highlight_cache = None

    def update_search(self, query: str) -> None:
        self._search_active = True
        self._search_query = query
        self._search_index = 0
        self._parsed_cache.clear()
        self._search_cache.clear()
        self._highlight_cache = None
        self._focus_search_match()

    def search_backspace(self) -> None:
        if self._search_query:
            self.update_search(self._search_query[:-1])

    def end_search(self) -> None:
        self._search_active = False
        self._search_query = ""
        self._search_index = 0
        self._parsed_cache.clear()
        self._highlight_cache = None
        line_count = len(self._parsed_lines(self._content_width))
        tail = max(0, line_count - self._viewport_height)
        self._follow_tail = self._scroll_offset >= tail

    def _base_render(self, width: int) -> str:
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

    def _search_matches(self, width: int | None = None) -> list[SearchMatch]:
        actual_width = width or self._content_width
        cache_key = (actual_width, self._revision, self._search_query)
        matches = self._search_cache.get(cache_key)
        if matches is None:
            plain_lines = Text.from_ansi(self._base_render(actual_width)).plain.splitlines()
            matches = find_matches(plain_lines, self._search_query)
            self._search_cache[cache_key] = matches
            self._search_cache.move_to_end(cache_key)
            while len(self._search_cache) > 3:
                self._search_cache.popitem(last=False)
        else:
            self._search_cache.move_to_end(cache_key)
        self._search_index = self._search_index % len(matches) if matches else 0
        return matches

    def _focus_search_match(self) -> None:
        matches = self._search_matches()
        if matches:
            self._set_scroll_offset(
                matches[self._search_index].first_line,
                allow_follow_tail=False,
            )

    def _refresh_search_render_cache(self) -> None:
        cache = self._highlight_cache
        if cache is None or cache[0][0] != self._content_width:
            self._parsed_cache.clear()
            return
        cache[1].render(self._search_index)
        parsed = self._parsed_cache.get(self._content_width)
        if parsed is None:
            return
        for width in tuple(self._parsed_cache):
            if width != self._content_width:
                del self._parsed_cache[width]
        for line in cache[1].changed_lines:
            fragments = to_formatted_text(ANSI(cache[1].fragments[line]))
            updated = list(split_lines(fragments)) or [[]]
            if len(updated) != 1:
                self._parsed_cache.clear()
                return
            parsed[line] = updated[0]

    def next_search_match(self) -> bool:
        matches = self._search_matches()
        if not matches:
            return False
        self._search_index = (self._search_index + 1) % len(matches)
        self._refresh_search_render_cache()
        self._focus_search_match()
        return True

    def previous_search_match(self) -> bool:
        matches = self._search_matches()
        if not matches:
            return False
        self._search_index = (self._search_index - 1) % len(matches)
        self._refresh_search_render_cache()
        self._focus_search_match()
        return True

    def _jump_to_user(self, *, next_message: bool) -> bool:
        if self._locations_revision != self._revision:
            self.create_content(self._content_width, self._viewport_height)
        user_units = set(self._user_units)
        seen: set[_TranscriptUnit] = set()
        targets: list[int] = []
        for index, (unit, _offset) in enumerate(self._line_locations):
            if unit in user_units and unit not in seen:
                seen.add(unit)
                targets.append(index)
        if next_message:
            target = next((index for index in targets if index > self._scroll_offset), None)
        else:
            target = next(
                (index for index in reversed(targets) if index < self._scroll_offset),
                None,
            )
        if target is None:
            return False
        self._set_scroll_offset(target)
        return True

    def next_user_message(self) -> bool:
        return self._jump_to_user(next_message=True)

    def previous_user_message(self) -> bool:
        return self._jump_to_user(next_message=False)

    def search_status(self) -> tuple[int, int] | None:
        if not self._search_active:
            return None
        matches = self._search_matches()
        if not matches:
            return (0, 0)
        return (self._search_index + 1, len(matches))

    def position_indicator(self) -> str | None:
        """Return the top visible line unless the viewport follows the tail."""

        if self._follow_tail:
            return None
        total = len(self._parsed_lines(self._content_width))
        if not total:
            return None
        return f"line {min(self._scroll_offset + 1, total)}/{total}"

    def _highlighted_render(self, width: int, base: str) -> str:
        matches = self._search_matches(width)
        if not matches:
            return base
        cache_key = (width, self._revision, self._search_query)
        if self._highlight_cache is None or self._highlight_cache[0] != cache_key:
            self._highlight_cache = (
                cache_key,
                HighlightCache(base, width, matches),
            )
        return self._highlight_cache[1].render(self._search_index)

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
        return self._highlighted_render(width, self._base_render(width))

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
        cached = self._locations_cache.get(width)
        if cached is not None and cached[0] == self._revision:
            self._locations_cache.move_to_end(width)
            return cached[1]
        locations = self._compute_locations(width)
        self._locations_cache[width] = (self._revision, locations)
        self._locations_cache.move_to_end(width)
        while len(self._locations_cache) > 3:
            self._locations_cache.popitem(last=False)
        return locations

    def _compute_locations(self, width: int) -> list[tuple[_TranscriptUnit | None, int]]:
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
        if not self._search_active and not self._follow_tail and self._scroll_offset >= tail:
            self._follow_tail = True
        if self._follow_tail:
            self._scroll_offset = tail
        self._line_locations = locations
        self._locations_revision = self._revision
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


def __getattr__(name: str):
    if name == "TranscriptPresenter":
        from .transcript_presenter import TranscriptPresenter

        return TranscriptPresenter
    raise AttributeError(name)
