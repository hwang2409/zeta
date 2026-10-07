"""Scrollable transcript control for the full-screen terminal UI."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from typing import TypeVar

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Console, RenderableType
from rich.text import Text

from ...protocol.types import (
    RedactedThinkingContent,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from .. import theme
from ..agent_card import AgentCard
from ..fuzzy import highlight_literal
from ..render import render_tool_progress
from ..theme import RICH_THEME
from .finder_overlay import FinderState
from .message_finder import Candidate, MessageFinder, Role
from .streaming_text import StreamingText
from .transcript_virtual import TranscriptVirtualMixin
from .transcript_search import (
    AnchoredSelection,
    HighlightCache,
    SearchMatch,
    Selection,
    find_matches,
    highlight_fragments,
)


MAX_TOOL_TAIL_CHARS = 4_096
_LAZY_TAIL_MIN_UNITS = 128
_FINDER_TEXT_LIMIT = 2_000
_FINDER_PREVIEW_LINES = 60

_Line = TypeVar("_Line")


def stream_key(
    event: StreamEvent,
) -> tuple[str | None, tuple[str, object] | None]:
    content = event.content
    index = event.data.get("index")
    identity = ("index", index) if isinstance(index, (int, str, tuple)) else None
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
        self.search_renderable = initial

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
            self.search_renderable = rendered
            self.renderable = self.card.update(rendered, event) or render_tool_progress(
                self.call, "\n".join(self.output)
            )
        self.revision += 1

    def refresh(self) -> None:
        rendered = self.card.refresh()
        if rendered is not None:
            self.search_renderable = rendered
            self.renderable = rendered
            self.revision += 1

    def finish(
        self,
        rendered: RenderableType,
        event: StreamEvent | None = None,
        *,
        compact: bool = True,
    ) -> None:
        self.finished = True
        self.search_renderable = rendered
        self.renderable = (
            self.card.finish(event, rendered) if compact else self.card.finish(event)
        ) or rendered
        self.revision += 1

    def toggle(self) -> bool:
        rendered = self.card.toggle()
        if rendered is None:
            return False
        self.renderable = rendered
        self.revision += 1
        return True


class _StreamingText(StreamingText):
    """Internal alias retained for presenter and transcript type checks."""


class _TranscriptUnit:
    def __init__(self, key: int, value: RenderableType | _ToolUnit | None) -> None:
        self.key = key
        self.value = value


@dataclass(frozen=True, slots=True)
class _FinderRestore:
    """The scroll view captured when the finder opens, restored if it cancels."""

    follow_tail: bool
    scroll_offset: int
    anchor: tuple[_TranscriptUnit | None, int] | None
    virtual_start: tuple[int, int] | None


class TranscriptWidget(TranscriptVirtualMixin, UIControl):
    """Render logical transcript units at the current width and stay at the bottom."""

    def __init__(
        self,
        *,
        max_lines: int | None = None,
    ) -> None:
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
        self._search_cache: OrderedDict[tuple[int, int, str], list[SearchMatch]] = (
            OrderedDict()
        )
        self._highlight_cache: tuple[tuple[int, int, str], HighlightCache] | None = None
        # Per-unit paint caches, validated by the identity of the unit's cached
        # render string: a streaming token re-renders one unit, so only that
        # unit is re-parsed and re-mapped instead of the whole transcript.
        self._unit_lines_cache: dict[int, tuple[str, list[list[tuple[str, str]]]]] = {}
        self._unit_locations_cache: dict[int, tuple[str, list[str], list[int]]] = {}
        self._unit_search_cache: dict[int, dict[int, tuple[int, str]]] = {}
        self._unit_search_widths: OrderedDict[int, None] = OrderedDict()
        self._keyed_cache: tuple[int, int, list[tuple[int | None, int]]] | None = None
        self._selection: AnchoredSelection | None = None
        self._prefix_lines = 0
        self._copy_handler: Callable[[str], str | None] | None = None
        self.copy_notice: str | None = None
        self._max_lines = max_lines
        self._line_limit_marker: str | None = None
        self._cache_palette = theme.active_palette()
        self._lazy_viewport = False
        self._mouse_coordinate_base = 0
        self._virtual_start: tuple[int, int] | None = None
        self._virtual_lines: list[list[tuple[str, str]]] = []
        self._virtual_locations: list[tuple[_TranscriptUnit | None, int]] = []
        self._virtual_width = 0
        self._virtual_height = 0
        self._virtual_revision = -1
        self._virtual_unit_count = 0
        self._virtual_start_needs_clamp = False
        self._unit_heights: dict[tuple[int, int, int], int] = {}
        self._pending_virtual_scroll = 0
        self._virtual_search_key: tuple[int, int, str] | None = None
        self._virtual_search_occurrences = []
        self._virtual_search_by_unit = {}
        self._virtual_search_cursor = 0
        self._virtual_search_complete = True
        self._virtual_search_scheduled = False
        # Fuzzy message finder overlay. Built once when opened; the saved view
        # restores the pre-open scroll when the overlay is cancelled.
        self._finder: MessageFinder | None = None
        self._finder_preview = True
        self._finder_restore: _FinderRestore | None = None

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
        self._prime_search_unit(unit)
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
        self._prime_search_unit(unit)
        return unit

    def touch(self, unit: _TranscriptUnit) -> None:
        """Invalidate derived content after an append to a mutable unit."""

        self._render_cache.pop(unit.key, None)
        self._bump_revision()
        self._prime_search_unit(unit)

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
        self._unit_lines_cache.pop(unit.key, None)
        self._unit_locations_cache.pop(unit.key, None)
        self._unit_search_cache.pop(unit.key, None)
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
        self._unit_lines_cache.clear()
        self._unit_locations_cache.clear()
        self._unit_search_cache.clear()
        self._unit_search_widths.clear()
        self._keyed_cache = None
        self._line_locations.clear()
        self._locations_cache.clear()
        self._anchor = None
        self._user_units.clear()
        self._search_cache.clear()
        self._highlight_cache = None
        self._selection = None
        self.copy_notice = None
        self._scroll_offset = 0
        self._follow_tail = True
        self._bump_revision()

    def set_line_limit_marker(self, marker: str | None) -> None:
        self._line_limit_marker = marker
        self._render_cache.clear()
        self._parsed_cache.clear()
        self._locations_cache.clear()

    def _limit_lines(
        self,
        lines: list[_Line],
        *,
        first_line: Callable[[_Line], str],
        marker: Callable[[str], _Line],
    ) -> list[_Line]:
        if self._max_lines is None:
            return lines
        content_limit = max(0, self._max_lines - 1)
        if self._line_limit_marker is None and len(lines) <= self._max_lines:
            return lines
        preserve_header = (
            len(lines) > content_limit
            and self._max_lines > 2
            and first_line(lines[0]).startswith("✱ thought")
        )
        tail_count = content_limit - 1 if preserve_header else content_limit
        tail = lines[-tail_count:] if tail_count else []
        kept = [lines[0], *tail] if preserve_header else tail
        omitted_count = len(lines) - len(kept)
        marker_text = self._line_limit_marker or (
            f"[{omitted_count} older lines omitted]"
        )
        return [marker(marker_text), *kept]

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
            self._prime_search_value(unit)

    def refresh_active_agents(self) -> None:
        """Refresh active cards and invalidate derived transcript caches once."""

        refreshed = False
        for unit in self._tools.values():
            revision = unit.revision
            unit.refresh()
            if unit.revision != revision:
                refreshed = True
                self._prime_search_value(unit)
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
            self._prime_search_value(unit)
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
            unit.refresh()
            self._bump_revision()
            self._prime_search_value(unit)

    def discard_tools(self) -> None:
        if not self._tools:
            return
        active = {
            unit
            for call_id, unit in self._tools.items()
            if call_id not in self._background_tools
        }
        active_ids = {id(unit) for unit in active}
        removed_keys = {
            unit.key
            for unit in self._units
            if (
                unit is not None
                and isinstance(unit.value, _ToolUnit)
                and id(unit.value) in active_ids
            )
        }
        self._units[:] = [
            unit
            for unit in self._units
            if (
                unit is None
                or not isinstance(unit.value, _ToolUnit)
                or id(unit.value) not in active_ids
            )
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
                self._prime_search_value(unit)
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

    def _scroll_by(self, amount: int) -> None:
        if self._follow_tail and amount >= 0:
            return
        if self._uses_virtual_history():
            self._pending_virtual_scroll += amount
            if amount < 0:
                self._follow_tail = False
            return
        self._materialize_for_interaction()
        if self._follow_tail and amount < 0:
            line_count = len(self._parsed_lines(self._content_width))
            tail = max(0, line_count - self._viewport_height)
            self._set_scroll_offset(tail + amount)
            return
        self._set_scroll_offset(self._scroll_offset + amount)

    def page_up(self) -> None:
        self._scroll_by(-self._viewport_height)

    def page_down(self) -> None:
        self._scroll_by(self._viewport_height)

    def scroll_up(self) -> None:
        self._scroll_by(-3)

    def scroll_down(self) -> None:
        self._scroll_by(3)

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
        self._render_cache.clear()
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
        self._render_cache.clear()
        self._parsed_cache.clear()
        self._highlight_cache = None
        if self._uses_virtual_history():
            return
        line_count = len(self._parsed_lines(self._content_width))
        tail = max(0, line_count - self._viewport_height)
        self._follow_tail = self._scroll_offset >= tail

    # -- fuzzy message finder ---------------------------------------------

    @property
    def finder_active(self) -> bool:
        return self._finder is not None

    def open_finder(self) -> None:
        """Open the finder over the current transcript, saving the scroll view."""

        if self._finder is not None:
            return
        self._finder_restore = _FinderRestore(
            self._follow_tail,
            self._scroll_offset,
            self._anchor,
            self._virtual_start,
        )
        self._finder = MessageFinder(self._build_finder_candidates())

    def finder_set_query(self, query: str) -> None:
        if self._finder is not None:
            self._finder.set_query(query)

    def finder_rank_more(self) -> bool:
        """Score another bounded slice; return True once ranking is complete."""

        if self._finder is None:
            return True
        return self._finder.rank_more()

    def finder_move(self, delta: int) -> None:
        if self._finder is not None:
            self._finder.move(delta)

    def finder_toggle_preview(self) -> None:
        self._finder_preview = not self._finder_preview

    def finder_cancel(self) -> None:
        """Close the finder and restore the scroll position from before it opened."""

        restore = self._finder_restore
        self._finder = None
        self._finder_restore = None
        if restore is not None:
            self._follow_tail = restore.follow_tail
            self._scroll_offset = restore.scroll_offset
            self._anchor = restore.anchor
            self._virtual_start = restore.virtual_start

    def finder_accept(self) -> bool:
        """Jump to the selected message and highlight the match for next/prev keys.

        The longest contiguous run of the fuzzy match drives the transcript's
        existing substring highlight, so once the overlay closes the familiar
        next/previous match keys continue to work from the landing position.
        """

        if self._finder is None:
            return False
        row = self._finder.selected
        query = self._finder.query
        self._finder = None
        self._finder_restore = None
        if row is None:
            return False
        self.jump_to_index(row.candidate.index)
        literal = highlight_literal(query, row.candidate.text) if query else None
        if literal:
            self.begin_search()
            self.update_search(literal)
            self._focus_search_on_unit(row.candidate.index)
        return True

    def finder_state(self) -> FinderState | None:
        if self._finder is None:
            return None
        return FinderState(
            query=self._finder.query,
            rows=self._finder.rows,
            selected=self._finder.selected_index,
            preview=self._finder.preview(),
            preview_visible=self._finder_preview,
            total=self._finder.candidate_count,
            complete=self._finder.complete,
        )

    def _finder_role(self, unit: _TranscriptUnit) -> Role:
        if unit in self._user_units:
            return Role.USER
        value = unit.value
        if isinstance(value, _ToolUnit):
            return Role.TOOL
        if isinstance(value, StreamingText):
            return Role.ASSISTANT
        return Role.NOTICE

    def _finder_text(self, unit: _TranscriptUnit) -> tuple[str, tuple[str, ...]]:
        """Return ``(flattened_text, preview_lines)`` cheaply, avoiding renders.

        Plain text comes from the renderable directly where possible so opening
        the finder on a long session does not re-render transcript history. Tool
        cards get a one-line call summary plus the first lines of their output.
        """

        value = unit.value
        if isinstance(value, _ToolUnit):
            arguments = " ".join(
                str(argument)
                for argument in value.call.arguments.values()
                if isinstance(argument, (str, int, float))
            )
            output = "".join(value.output)
            summary = f"{value.call.name} {arguments}".strip()
            lines = [summary, *output.splitlines()] if output else [summary]
        else:
            plain = getattr(value, "plain", None)
            if not isinstance(plain, str):
                plain = Text.from_ansi(
                    self._searchable_text(unit, self._content_width)
                ).plain
            lines = plain.splitlines()
        lines = [line.rstrip() for line in lines if line.strip()]
        if not lines:
            return "", ()
        flat = " ".join(lines)
        if len(flat) > _FINDER_TEXT_LIMIT:
            flat = flat[:_FINDER_TEXT_LIMIT]
        return flat, tuple(lines[:_FINDER_PREVIEW_LINES])

    def _build_finder_candidates(self) -> list[Candidate]:
        candidates: list[Candidate] = []
        turn = 0
        for index, unit in enumerate(self._units):
            if unit is None or unit.value is None:
                continue
            role = self._finder_role(unit)
            if role is Role.USER:
                turn += 1
            text, preview = self._finder_text(unit)
            if not text:
                continue
            candidates.append(
                Candidate(
                    index=index,
                    role=role,
                    marker=f"#{turn}" if turn else "#0",
                    text=text,
                    preview=preview,
                )
            )
        return candidates

    def _focus_search_on_unit(self, unit_index: int) -> None:
        """Make the match inside ``unit_index`` the current one, if any exists."""

        if unit_index >= len(self._units):
            return
        unit = self._units[unit_index]
        matches = self._search_matches()
        if not matches:
            return
        if self._uses_virtual_history():
            for index, occurrence in enumerate(self._virtual_search_occurrences):
                if occurrence.unit is unit:
                    self._search_index = index
                    break
            self._focus_search_match()
            return
        locations = self._locations(self._content_width)
        for index, match in enumerate(matches):
            line = match.first_line
            if 0 <= line < len(locations) and locations[line][0] is unit:
                self._search_index = index
                break
        self._refresh_search_render_cache()
        self._focus_search_match()

    def jump_to_index(self, unit_index: int) -> bool:
        """Scroll so the unit at ``unit_index`` is at the top of the viewport."""

        if unit_index < 0 or unit_index >= len(self._units):
            return False
        unit = self._units[unit_index]
        if unit is None:
            return False
        if self._uses_virtual_history():
            self._virtual_start = (unit_index, 0)
            self._virtual_start_needs_clamp = True
            self._anchor = (unit, 0)
            self._scroll_offset = self._estimated_prefix(
                self._content_width, unit_index, 0
            )
            self._follow_tail = False
            return True
        self._materialize_for_interaction()
        if self._locations_revision != self._revision:
            self.create_content(self._content_width, self._viewport_height)
        locations = self._locations(self._content_width)
        for line, (located, _offset) in enumerate(locations):
            if located is unit:
                self._set_scroll_offset(line, allow_follow_tail=False)
                return True
        return False


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
        lines = self._limit_lines(
            lines,
            first_line=lambda line: Text.from_ansi(line).plain,
            marker=lambda marker_text: marker_text,
        )
        return "\n".join(lines)

    def _search_matches(self, width: int | None = None) -> list[SearchMatch]:
        if not self._search_query:
            self._search_index = 0
            return []
        if self._uses_virtual_history():
            return self._indexed_search_matches()
        actual_width = width or self._content_width
        cache_key = (actual_width, self._revision, self._search_query)
        matches = self._search_cache.get(cache_key)
        if matches is None:
            plain_lines = Text.from_ansi(
                self._base_render(actual_width)
            ).plain.splitlines()
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
        if not matches:
            return
        if self._uses_virtual_history():
            self._focus_virtual_search_match()
            return
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
        if self._uses_virtual_history():
            current = (
                self._virtual_start[0]
                if self._virtual_start is not None
                else len(self._units)
            )
            indexed = [(self._units.index(unit), unit) for unit in self._user_units]
            candidates = (
                (item for item in indexed if item[0] > current)
                if next_message
                else (item for item in reversed(indexed) if item[0] < current)
            )
            target = next(candidates, None)
            if target is None:
                return False
            self._virtual_start = (target[0], 0)
            self._anchor = (target[1], 0)
            self._scroll_offset = self._estimated_prefix(
                self._content_width, target[0], 0
            )
            self._follow_tail = False
            return True
        self._materialize_for_interaction()
        if self._locations_revision != self._revision:
            self.create_content(self._content_width, self._viewport_height)
        if self._locations_revision != self._revision:
            self._line_locations = self._locations(self._content_width)
            self._locations_revision = self._revision
        user_units = set(self._user_units)
        seen: set[_TranscriptUnit] = set()
        targets: list[int] = []
        for index, (unit, _offset) in enumerate(self._line_locations):
            if unit in user_units and unit not in seen:
                seen.add(unit)
                targets.append(index)
        if next_message:
            target = next(
                (index for index in targets if index > self._scroll_offset), None
            )
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
        if self._uses_virtual_history():
            total = self._estimated_total(self._content_width)
            return f"line {min(self._scroll_offset + 1, total)}/~{total}"
        total = len(self._parsed_lines(self._content_width))
        if not total:
            return None
        return f"line {min(self._scroll_offset + 1, total)}/{total}"

    def _highlighted_render(self, width: int, base: str) -> str:
        matches = (
            find_matches(Text.from_ansi(base).plain.splitlines(), self._search_query)
            if self._uses_virtual_history() and self._search_query
            else self._search_matches(width)
        )
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
        revision = (
            value.revision if isinstance(value, (_StreamingText, _ToolUnit)) else 0
        )
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
        if isinstance(value, _ToolUnit):
            renderable = (
                value.search_renderable if self._search_active else value.renderable
            )
        else:
            renderable = value
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

    def _unit_parsed_lines(
        self, unit: _TranscriptUnit, width: int
    ) -> list[list[tuple[str, str]]]:
        """Parse one unit's render into fragment lines, reusing it while unchanged."""

        rendered = self._render_unit(unit, width)
        cached = self._unit_lines_cache.get(unit.key)
        if cached is not None and cached[0] is rendered:
            return cached[1]
        lines = (
            list(split_lines(to_formatted_text(ANSI(rendered)))) if rendered else [[]]
        )
        self._unit_lines_cache[unit.key] = (rendered, lines)
        return lines

    def _finish_assembled_lines(
        self, lines: list[list[tuple[str, str]]]
    ) -> list[list[tuple[str, str]]]:
        while lines and not lines[-1]:
            lines.pop()
        while lines and not "".join(fragment[1] for fragment in lines[0]).strip():
            lines.pop(0)
        return self._limit_lines(
            lines or [[]],
            first_line=lambda line: "".join(fragment[1] for fragment in line),
            marker=lambda marker_text: [(theme.DIM, marker_text)],
        )

    def _assembled_lines(self, width: int) -> list[list[tuple[str, str]]]:
        """Concatenate per-unit fragment lines; mirrors ``_base_render`` trimming."""

        lines: list[list[tuple[str, str]]] = []
        for unit in self._units:
            if unit is None:
                lines.append([])
            else:
                lines.extend(self._unit_parsed_lines(unit, width))
        return self._finish_assembled_lines(lines)

    def _streaming_tail_lines(
        self, value: _StreamingText, width: int, height: int
    ) -> list[list[tuple[str, str]]]:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
            width=width,
            theme=RICH_THEME,
        )
        wrapped = value.tail(console, width, height)
        rendered = Text("", style=value.style)
        for index, line in enumerate(wrapped):
            if index:
                rendered.append("\n")
            rendered.append_text(line)
        console.print(rendered, soft_wrap=True)
        ansi = "\n".join(line.rstrip(" ") for line in output.getvalue().splitlines())
        return list(split_lines(to_formatted_text(ANSI(ansi)))) if ansi else [[]]

    def _tail_lines(self, width: int, height: int) -> list[list[tuple[str, str]]]:
        """Render only enough newest units to fill a follow-tail viewport."""

        lines: list[list[tuple[str, str]]] = []
        trimming_trailing_blanks = True
        reached_start = True
        for unit in reversed(self._units):
            if unit is None:
                unit_lines = [[]]
            elif isinstance(unit.value, _StreamingText):
                unit_lines = self._streaming_tail_lines(unit.value, width, height)
            else:
                unit_lines = self._unit_parsed_lines(unit, width)
            if trimming_trailing_blanks:
                unit_lines = list(unit_lines)
                while unit_lines and not unit_lines[-1]:
                    unit_lines.pop()
                trimming_trailing_blanks = not unit_lines
            if unit_lines:
                lines[:0] = unit_lines
            if len(lines) >= height and not trimming_trailing_blanks:
                reached_start = False
                break
        if reached_start:
            while lines and not "".join(fragment[1] for fragment in lines[0]).strip():
                lines.pop(0)
        return lines[-height:] or [[]]

    def _parsed_lines(self, width: int) -> list[list[tuple[str, str]]]:
        cached = self._parsed_cache.get(width)
        if cached is not None:
            self._parsed_cache.move_to_end(width)
            return cached
        if self._search_active and self._search_query:
            # Search highlights restyle matched lines across the whole render.
            fragments = to_formatted_text(ANSI(self.render(width)))
            lines = list(split_lines(fragments)) or [[]]
        else:
            lines = self._assembled_lines(width)
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

    def _unit_locations(
        self,
        unit: _TranscriptUnit,
        width: int,
        rendered: str | None = None,
    ) -> tuple[list[str], list[int]]:
        """Map one unit's rendered lines to source offsets, reusing while unchanged."""

        if rendered is None:
            rendered = self._render_unit(unit, width)
        cached = self._unit_locations_cache.get(unit.key)
        if cached is not None and cached[0] is rendered:
            return cached[1], cached[2]
        rendered_lines = self._plain_lines(rendered)
        renderable = (
            unit.value.renderable if isinstance(unit.value, _ToolUnit) else unit.value
        )
        source = getattr(renderable, "plain", None)
        if not isinstance(source, str):
            source = "\n".join(self._strip_padding(line) for line in rendered_lines)
        offsets: list[int] = []
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
            offsets.append(offset)
            source_offset = offset + matched_length
        self._unit_locations_cache[unit.key] = (rendered, rendered_lines, offsets)
        return rendered_lines, offsets

    def _compute_locations(
        self, width: int
    ) -> list[tuple[_TranscriptUnit | None, int]]:
        raw_lines: list[tuple[str, _TranscriptUnit | None, int]] = []
        for unit in self._units:
            if unit is None:
                raw_lines.append(("", None, 0))
                continue
            rendered_lines, offsets = self._unit_locations(unit, width)
            raw_lines.extend(
                (line, unit, offset) for line, offset in zip(rendered_lines, offsets)
            )
        while raw_lines and not raw_lines[0][0].strip():
            raw_lines.pop(0)
        raw_lines = self._limit_lines(
            raw_lines,
            first_line=lambda line: line[0],
            marker=lambda marker_text: (marker_text, None, 0),
        )
        return [(unit, text_offset) for _, unit, text_offset in raw_lines]

    def _materialize_for_interaction(self) -> bool:
        was_lazy = self._lazy_viewport
        lines = self._parsed_lines(self._content_width)
        locations = self._locations(self._content_width)
        if self._follow_tail:
            self._scroll_offset = max(0, len(lines) - self._viewport_height)
        self._line_locations = locations
        self._locations_revision = self._revision
        self._lazy_viewport = False
        return was_lazy

    def create_content(self, width: int, height: int | None) -> UIContent:
        palette = theme.active_palette()
        if palette is not self._cache_palette:
            self._cache_palette = palette
            stream_styles = {"body": theme.BODY, "thought": theme.THOUGHT}
            for unit in self._units:
                if unit is not None and isinstance(unit.value, _StreamingText):
                    unit.value.restyle(stream_styles[unit.value.palette_role])
            self._render_cache.clear()
            self._parsed_cache.clear()
            self._unit_lines_cache.clear()
            self._unit_locations_cache.clear()
            self._locations_cache.clear()
            self._locations_revision = -1
        width = max(1, width)
        height = max(1, height or 1)
        self._content_width = max(1, width)
        self._viewport_height = height
        if self._uses_virtual_history():
            return self._virtual_content(width, height)
        # Line-limited transcripts must apply the omission marker on their first
        # frame, so they use the normal eager path instead of the raw lazy tail.
        lazy_tail = (
            self._follow_tail
            and self._max_lines is None
            and not self._search_active
            and self._anchor is None
            and self._selection is None
            and (
                len(self._units) >= _LAZY_TAIL_MIN_UNITS
                or bool(
                    self._units
                    and self._units[-1] is not None
                    and isinstance(self._units[-1].value, _StreamingText)
                )
            )
            and width not in self._parsed_cache
        )
        self._lazy_viewport = lazy_tail
        lines = (
            self._tail_lines(width, self._viewport_height)
            if lazy_tail
            else self._parsed_lines(width)
        )
        # Location data is only needed after an interaction leaves follow-tail.
        # Building it for the first frame would eagerly render all history.
        need_locations = (
            (self._locations_revision != self._revision and not lazy_tail)
            or not self._follow_tail
            or self._anchor is not None
            or self._selection is not None
        )
        locations = self._locations(width) if need_locations else []
        if self._follow_tail:
            self._scroll_offset = (
                0 if lazy_tail else max(0, len(lines) - self._viewport_height)
            )
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
        if (
            not self._search_active
            and not self._follow_tail
            and self._scroll_offset >= tail
        ):
            self._follow_tail = True
        if self._follow_tail:
            self._scroll_offset = tail
        if need_locations:
            self._line_locations = locations
            self._locations_revision = self._revision
        else:
            self._line_locations = []
        prefix_lines = max(0, self._viewport_height - len(lines))
        self._prefix_lines = prefix_lines
        visible_lines = [[] for _ in range(prefix_lines)] + lines
        cursor_y = (
            len(visible_lines) - 1
            if lazy_tail
            else min(prefix_lines + self._scroll_offset, len(visible_lines) - 1)
        )
        selection = self._resolved_selection(locations)
        selection_style = f"bg:{theme.active_palette().search_bg}"

        def get_line(index: int) -> list[tuple[str, str]]:
            line = visible_lines[index]
            if selection is None:
                return line
            length = sum(len(fragment[1]) for fragment in line)
            span = selection.line_span(index - prefix_lines, length)
            if span is None:
                return line
            return highlight_fragments(line, span, selection_style)

        return UIContent(
            get_line=get_line,
            line_count=len(visible_lines),
            cursor_position=Point(x=0, y=cursor_y),
            show_cursor=False,
        )

    def vertical_scroll(self, window: Window) -> int:
        del window
        return 0 if self._uses_virtual_history() else self._scroll_offset

    def set_copy_handler(self, handler: Callable[[str], str | None] | None) -> None:
        """Receive the text of each finished drag; return a footer notice."""

        self._copy_handler = handler

    @property
    def selection(self) -> Selection | None:
        """The current selection resolved to rows, or None when there is none."""

        return self._resolved_selection()

    def clear_selection(self) -> None:
        self._selection = None
        self.copy_notice = None

    def mouse_handler(self, mouse_event: MouseEvent):
        """Scroll on the wheel; turn a left-button drag into a copied selection.

        The terminal reports drags because the session turns on button-event
        tracking; positions arrive in content coordinates, so the blank rows
        padded above a short transcript are subtracted before mapping to lines.
        """

        event_type = mouse_event.event_type
        if event_type is MouseEventType.SCROLL_UP:
            self.scroll_up()
            return None
        if event_type is MouseEventType.SCROLL_DOWN:
            self.scroll_down()
            return None
        if event_type is MouseEventType.MOUSE_DOWN:
            if mouse_event.button is not MouseButton.LEFT:
                return NotImplemented
            if self._uses_virtual_history():
                self._mouse_coordinate_base = 0
            else:
                was_lazy = self._materialize_for_interaction()
                self._mouse_coordinate_base = self._scroll_offset if was_lazy else 0
        row = mouse_event.position.y - self._prefix_lines
        if self._mouse_coordinate_base and row < self._mouse_coordinate_base:
            row += self._mouse_coordinate_base
        cell = (row, mouse_event.position.x)
        if event_type is MouseEventType.MOUSE_DOWN:
            anchor = self._anchor_for(cell)
            self._selection = AnchoredSelection(anchor, anchor)
            self.copy_notice = None
            return None
        selection = self._selection
        if selection is None or not selection.dragging:
            return NotImplemented
        if event_type is MouseEventType.MOUSE_MOVE:
            self._selection = selection.extend(self._anchor_for(cell))
            return None
        if event_type is MouseEventType.MOUSE_UP:
            released = selection.released(self._anchor_for(cell))
            resolved = self._selection_for_copy(released)
            if resolved is None or resolved.is_click:
                self._selection = None
                return None
            self._selection = released
            text = self.selection_text()
            if text and self._copy_handler is not None:
                self.copy_notice = self._copy_handler(text)
            return None
        return NotImplemented

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
