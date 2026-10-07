"""Unit-addressed viewport and source-text search for large transcripts."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any

from prompt_toolkit.application.current import get_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from .. import theme
from ..theme import RICH_THEME
from .streaming_text import StreamingText
from .transcript_search import (
    AnchoredSelection,
    Cell,
    SearchMatch,
    Selection,
    SelectionAnchor,
    find_matches,
    highlight_fragments,
)

_LAZY_TAIL_MIN_UNITS = 128
_VIRTUAL_MARGIN_SCREENS = 1


class _HeightIndex:
    """Fenwick index for O(log n) transcript height updates and prefixes."""

    def __init__(self, size: int) -> None:
        self.values = [1] * size
        self.tree = [0, *(index & -index for index in range(1, size + 1))]

    def ensure(self, size: int) -> None:
        while len(self.values) < size:
            old_size = len(self.values)
            index = old_size + 1
            low = index & -index
            prior = self.prefix(old_size) - self.prefix(index - low)
            self.values.append(1)
            self.tree.append(prior + 1)

    def set(self, index: int, value: int) -> None:
        delta = value - self.values[index]
        if not delta:
            return
        self.values[index] = value
        cursor = index + 1
        while cursor < len(self.tree):
            self.tree[cursor] += delta
            cursor += cursor & -cursor

    def prefix(self, count: int) -> int:
        total = 0
        cursor = min(count, len(self.values))
        while cursor:
            total += self.tree[cursor]
            cursor -= cursor & -cursor
        return total


@dataclass(frozen=True, slots=True)
class _SearchOccurrence:
    unit: Any
    ranges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _SearchRenderSnapshot:
    unit_key: int
    revision: int
    width: int
    renderable: Any
    rich_theme: Theme


class TranscriptVirtualMixin:
    """Bound synchronous transcript work to units in or entering the viewport."""

    @staticmethod
    def _render_search_snapshot(
        renderable: Any, width: int, rich_theme: Theme
    ) -> str:
        """Render copied search input without consulting live widget state."""

        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
            width=max(1, width),
            theme=rich_theme,
        )
        console.print(renderable)
        rendered = "\n".join(
            line.rstrip(" ") for line in output.getvalue().splitlines()
        )
        return Text.from_ansi(rendered).plain

    def _search_rendered(self, unit: Any, width: int) -> str:
        value = unit.value
        if hasattr(value, "search_renderable"):
            value = value.search_renderable
        if value is None:
            return ""
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
            width=max(1, width),
            theme=RICH_THEME,
        )
        console.print(value)
        return "\n".join(line.rstrip(" ") for line in output.getvalue().splitlines())

    def _remember_search_width(self, width: int) -> None:
        self._unit_search_widths[width] = None
        self._unit_search_widths.move_to_end(width)
        while len(self._unit_search_widths) > 2:
            stale_width, _ = self._unit_search_widths.popitem(last=False)
            for entries in self._unit_search_cache.values():
                entries.pop(stale_width, None)

    def _cache_search_text(
        self, unit_key: int, width: int, revision: int, plain: str
    ) -> None:
        self._remember_search_width(width)
        self._unit_search_cache.setdefault(unit_key, {})[width] = (revision, plain)

    def _searchable_text(self, unit: Any, width: int | None = None) -> str:
        """Return exactly the plain text Rich exposes to eager search.

        This has a separate per-unit cache from painting: indexing a card must
        include borders, titles, status text, and nested renderables, without
        causing the viewport render counter to revisit transcript history.
        """

        actual_width = max(1, width or self._content_width)
        self._remember_search_width(actual_width)
        revision = self._unit_revision(unit)
        cached = self._unit_search_cache.get(unit.key, {}).get(actual_width)
        if cached is not None and cached[0] == revision:
            return cached[1]
        if unit.value is None:
            plain = ""
        else:
            rendered = self._render_cache.get(unit.key)
            if rendered is not None and rendered[:2] == (actual_width, revision):
                search_rendered = rendered[2]
            else:
                search_rendered = self._search_rendered(unit, actual_width)
            plain = Text.from_ansi(search_rendered).plain
        self._unit_search_cache.setdefault(unit.key, {})[actual_width] = (
            revision,
            plain,
        )
        return plain

    def _prime_search_unit(self, unit: Any) -> None:
        """Invalidate one unit without rendering transcript history on append."""

        if unit is not None:
            self._unit_search_cache.pop(unit.key, None)

    def _prime_search_value(self, value: Any) -> None:
        unit = next(
            (
                candidate
                for candidate in reversed(self._units)
                if candidate is not None and candidate.value is value
            ),
            None,
        )
        self._prime_search_unit(unit)

    def _search_unit_needs_worker(self, unit: Any) -> bool:
        cached = self._unit_search_cache.get(unit.key, {}).get(self._content_width)
        return unit.value is not None and (
            cached is None or cached[0] != self._unit_revision(unit)
        )

    def _snapshot_search_unit(
        self, unit: Any, width: int
    ) -> _SearchRenderSnapshot | None:
        """Copy mutable Rich input while exclusively on the event-loop thread."""

        value = unit.value
        if hasattr(value, "search_renderable"):
            value = value.search_renderable
        if value is None:
            renderable: Any = Text("")
        elif isinstance(value, StreamingText):
            renderable = Text(value.plain, style=value.style)
        else:
            try:
                renderable = copy.deepcopy(value)
            except Exception:  # noqa: BLE001 - custom Rich renderables may fail freely
                return None
        try:
            rich_theme = copy.deepcopy(RICH_THEME)
        except Exception:  # noqa: BLE001 - preserve loop rendering as the safe fallback
            return None
        return _SearchRenderSnapshot(
            unit.key,
            self._unit_revision(unit),
            width,
            renderable,
            rich_theme,
        )

    async def _render_search_unit_async(
        self, key: tuple[int, int, str], unit: Any
    ) -> None:
        width = key[1]
        snapshot = self._snapshot_search_unit(unit, width)
        if snapshot is None:
            # Uncopyable custom Rich renderables remain on the loop. The indexer
            # schedules at most one such unit per bounded continuation batch.
            plain = Text.from_ansi(self._search_rendered(unit, width)).plain
            revision = self._unit_revision(unit)
            unit_key = unit.key
        else:
            plain = await asyncio.to_thread(
                self._render_search_snapshot,
                snapshot.renderable,
                snapshot.width,
                snapshot.rich_theme,
            )
            revision = snapshot.revision
            unit_key = snapshot.unit_key
        if (
            self._virtual_search_key != key
            or self._revision != key[0]
            or self._content_width != width
            or unit.key != unit_key
            or self._unit_revision(unit) != revision
        ):
            return
        self._cache_search_text(unit_key, width, revision, plain)
        self._virtual_search_scheduled = False
        self._continue_virtual_search_index(key)

    def _continue_virtual_search_index(
        self, key: tuple[int, int, str], *, bounded: bool = True
    ) -> None:
        if self._virtual_search_key != key:
            return
        started = time.perf_counter()
        before = len(self._virtual_search_occurrences)
        while self._virtual_search_cursor < len(self._units):
            unit = self._units[self._virtual_search_cursor]
            if unit is not None and bounded and self._search_unit_needs_worker(unit):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    self._virtual_search_scheduled = True
                    loop.create_task(self._render_search_unit_async(key, unit))
                    return
            self._virtual_search_cursor += 1
            if unit is not None:
                plain_lines = self._searchable_text(
                    unit, self._content_width
                ).splitlines()
                for match in find_matches(plain_lines, self._search_query):
                    occurrence = _SearchOccurrence(unit, match.ranges)
                    index = len(self._virtual_search_occurrences)
                    self._virtual_search_occurrences.append(occurrence)
                    self._virtual_search_by_unit.setdefault(unit, []).append(
                        (index, occurrence)
                    )
            if bounded and time.perf_counter() - started >= 0.02:
                break
        self._virtual_search_complete = self._virtual_search_cursor >= len(self._units)
        if before == 0 and self._virtual_search_occurrences:
            self._focus_virtual_search_match()
        if self._virtual_search_complete:
            self._virtual_search_scheduled = False
        elif not self._virtual_search_scheduled:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._continue_virtual_search_index(key, bounded=False)
            else:
                self._virtual_search_scheduled = True
                loop.call_soon(self._scheduled_virtual_search_batch, key)
        get_app().invalidate()

    def _scheduled_virtual_search_batch(self, key: tuple[int, int, str]) -> None:
        self._virtual_search_scheduled = False
        self._continue_virtual_search_index(key)

    def _indexed_search_matches(self) -> list[SearchMatch]:
        if not self._search_query:
            self._virtual_search_occurrences = []
            return []
        key = (
            self._revision,
            self._content_width,
            self._search_query,
        )
        if self._virtual_search_key != key:
            self._virtual_search_key = key
            self._virtual_search_occurrences = []
            self._virtual_search_by_unit = {}
            self._virtual_search_cursor = 0
            self._virtual_search_complete = False
            self._virtual_search_scheduled = False
            self._continue_virtual_search_index(key)
        matches = [
            SearchMatch(((index, 0, 1),))
            for index in range(len(self._virtual_search_occurrences))
        ]
        self._search_index = self._search_index % len(matches) if matches else 0
        return matches

    def _focus_virtual_search_match(self) -> None:
        occurrence = self._virtual_search_occurrences[self._search_index]
        unit_index = self._units.index(occurrence.unit)
        target_line = occurrence.ranges[0][0]
        self._follow_tail = False
        self._virtual_start = (unit_index, target_line)
        self._virtual_start_needs_clamp = True
        self._anchor = (occurrence.unit, 0)

    def _highlight_virtual_search(
        self,
        lines: list[list[tuple[str, str]]],
        locations: list[tuple[Any | None, int]],
        unit_line_numbers: list[int],
    ) -> None:
        for viewport_line, ((unit, _offset), unit_line) in enumerate(
            zip(locations, unit_line_numbers)
        ):
            if unit is None:
                continue
            for match_index, occurrence in self._virtual_search_by_unit.get(unit, ()):
                for match_line, first, last in occurrence.ranges:
                    if match_line != unit_line:
                        continue
                    style = (
                        theme.SEARCH_CURRENT
                        if match_index == self._search_index
                        else theme.SEARCH_MATCH
                    )
                    lines[viewport_line] = highlight_fragments(
                        lines[viewport_line], (first, last), style
                    )

    def _uses_virtual_history(self) -> bool:
        return self._max_lines is None and len(self._units) >= _LAZY_TAIL_MIN_UNITS

    @staticmethod
    def _unit_revision(unit: Any) -> int:
        value = unit.value
        return int(getattr(value, "revision", 0))

    def _virtual_unit_lines(
        self, index: int, width: int, *, streaming_tail: int | None = None
    ) -> tuple[list[list[tuple[str, str]]], list[tuple[Any | None, int]]]:
        unit = self._units[index]
        if unit is None or unit.value is None:
            return [[]], [(unit, 0)]
        revision = self._unit_revision(unit)
        if isinstance(unit.value, StreamingText) and streaming_tail is not None:
            cached = self._virtual_stream_lines.get(unit.key)
            if cached is not None and cached[:3] == (width, revision, streaming_tail):
                lines = cached[3]
            else:
                lines = self._streaming_tail_lines(
                    unit.value, width, max(1, streaming_tail)
                )
                self._virtual_stream_lines[unit.key] = (
                    width,
                    revision,
                    streaming_tail,
                    lines,
                )
            return lines, [(unit, offset) for offset in range(len(lines))]
        rendered_entry = self._render_cache.get(unit.key)
        line_entry = self._unit_lines_cache.get(unit.key)
        location_entry = self._unit_locations_cache.get(unit.key)
        if (
            rendered_entry is not None
            and rendered_entry[:2] == (width, revision)
            and line_entry is not None
            and line_entry[0] is rendered_entry[2]
            and location_entry is not None
            and location_entry[0] is rendered_entry[2]
        ):
            lines = line_entry[1]
            self._unit_heights[(width, unit.key, revision)] = len(lines)
            self._remember_unit_height(width, index, len(lines))
            return lines, [(unit, offset) for offset in location_entry[2]]
        lines = self._unit_parsed_lines(unit, width)
        self._unit_heights[(width, unit.key, revision)] = len(lines)
        self._remember_unit_height(width, index, len(lines))
        # ``_unit_parsed_lines`` populated the render cache. Avoid even a
        # cached _render_unit call so count-based work remains viewport-bound.
        rendered = self._render_cache[unit.key][2]
        _plain, offsets = self._unit_locations(unit, width, rendered)
        return lines, [(unit, offset) for offset in offsets]

    def _height_index(self, width: int) -> _HeightIndex:
        index = self._height_indexes.get(width)
        if index is None:
            index = _HeightIndex(len(self._units))
            self._height_indexes[width] = index
        else:
            index.ensure(len(self._units))
        return index

    def _remember_unit_height(self, width: int, unit_index: int, height: int) -> None:
        self._height_index(width).set(unit_index, height)

    def _estimated_prefix(self, width: int, unit_index: int, line_offset: int) -> int:
        return self._height_index(width).prefix(unit_index) + line_offset

    def _estimated_total(self, width: int) -> int:
        return self._estimated_prefix(width, len(self._units), 0)

    def _virtual_tail_start(self, width: int, wanted: int) -> tuple[int, int]:
        remaining = wanted
        trimming_blanks = True
        for index in range(len(self._units) - 1, -1, -1):
            unit = self._units[index]
            if trimming_blanks and (unit is None or unit.value is None):
                continue
            trimming_blanks = False
            lines, _ = self._virtual_unit_lines(
                index,
                width,
                streaming_tail=(
                    remaining
                    if isinstance(getattr(unit, "value", None), StreamingText)
                    else None
                ),
            )
            if len(lines) >= remaining:
                return index, max(0, len(lines) - remaining)
            remaining -= len(lines)
        return 0, 0

    def _clamp_virtual_start(
        self, start: tuple[int, int], width: int
    ) -> tuple[int, int]:
        """Keep a virtual viewport start between the first line and the tail."""

        tail = self._virtual_tail_start(width, self._viewport_height)
        if not self._units:
            return tail
        index = min(max(0, start[0]), len(self._units) - 1)
        lines, _ = self._virtual_unit_lines(index, width)
        candidate = (index, min(max(0, start[1]), len(lines) - 1))
        if candidate >= tail:
            if not self._search_active:
                self._follow_tail = True
            return tail
        return candidate

    def _move_virtual_start(
        self, start: tuple[int, int], amount: int, width: int
    ) -> tuple[int, int]:
        index, offset = start
        if amount < 0:
            remaining = -amount
            if offset >= remaining:
                return index, offset - remaining
            remaining -= offset
            while index > 0:
                index -= 1
                lines, _ = self._virtual_unit_lines(index, width)
                if len(lines) >= remaining:
                    return index, len(lines) - remaining
                remaining -= len(lines)
            return 0, 0
        remaining = amount
        while index < len(self._units):
            lines, _ = self._virtual_unit_lines(index, width)
            available = len(lines) - offset
            if remaining < available:
                return self._clamp_virtual_start(
                    (index, offset + remaining), width
                )
            remaining -= available
            index += 1
            offset = 0
        return self._clamp_virtual_start((index, offset), width)

    def _virtual_content(self, width: int, height: int) -> UIContent:
        wanted = max(height, height * _VIRTUAL_MARGIN_SCREENS)
        geometry_changed = self._virtual_width != width or self._virtual_height != height
        content_may_have_shrunk = (
            self._virtual_revision != self._revision
            and 0 < self._virtual_unit_count
            and len(self._units) <= self._virtual_unit_count
        )
        width_changed = self._virtual_width != width
        if self._follow_tail:
            start = self._virtual_tail_start(width, wanted)
        elif self._virtual_start is None:
            anchor = self._anchor
            if anchor is None and self._line_locations:
                anchor = self._line_locations[
                    min(self._scroll_offset, len(self._line_locations) - 1)
                ]
            if anchor is not None and anchor[0] in self._units:
                unit_index = self._units.index(anchor[0])
                _plain, offsets = self._unit_locations(anchor[0], width)
                candidates = [
                    (abs(value - anchor[1]), line) for line, value in enumerate(offsets)
                ]
                start = (
                    unit_index,
                    min(candidates)[1] if candidates else 0,
                )
            else:
                start = self._virtual_tail_start(width, wanted)
        else:
            start = self._virtual_start
            if width_changed and self._anchor is not None:
                anchor_unit = self._units[start[0]]
                if anchor_unit is self._anchor[0] and anchor_unit is not None:
                    _plain, offsets = self._unit_locations(anchor_unit, width)
                    candidates = [
                        (abs(value - self._anchor[1]), line)
                        for line, value in enumerate(offsets)
                    ]
                    start = (start[0], min(candidates)[1] if candidates else 0)
        if self._pending_virtual_scroll:
            start = self._move_virtual_start(start, self._pending_virtual_scroll, width)
            self._pending_virtual_scroll = 0
        elif not self._follow_tail and (
            geometry_changed
            or content_may_have_shrunk
            or self._virtual_start_needs_clamp
        ):
            start = self._clamp_virtual_start(start, width)
        self._virtual_start_needs_clamp = False

        lines: list[list[tuple[str, str]]] = []
        locations: list[tuple[Any | None, int]] = []
        unit_line_numbers: list[int] = []
        index, offset = start
        while index < len(self._units) and len(lines) < wanted:
            unit = self._units[index]
            unit_lines, unit_locations = self._virtual_unit_lines(
                index,
                width,
                streaming_tail=(
                    wanted
                    if self._follow_tail
                    and isinstance(getattr(unit, "value", None), StreamingText)
                    else None
                ),
            )
            lines.extend(unit_lines[offset:])
            locations.extend(unit_locations[offset:])
            unit_line_numbers.extend(range(offset, len(unit_lines)))
            index += 1
            offset = 0
        lines = lines[:wanted] or [[]]
        locations = locations[:wanted] or [(None, 0)]
        unit_line_numbers = unit_line_numbers[:wanted] or [0]
        self._virtual_start = start
        self._virtual_lines = lines
        self._virtual_locations = locations
        self._virtual_width = width
        self._virtual_height = height
        self._virtual_revision = self._revision
        self._virtual_unit_count = len(self._units)
        self._lazy_viewport = True
        self._line_locations = locations
        self._locations_revision = self._revision
        self._anchor = locations[0] if locations else None
        base = self._estimated_prefix(width, *start)
        self._scroll_offset = base

        if self._search_active and self._search_query:
            self._indexed_search_matches()
            self._highlight_virtual_search(lines, locations, unit_line_numbers)

        selection = self._resolved_selection()
        selection_style = f"bg:{theme.active_palette().search_bg}"

        def get_line(line: int) -> list[tuple[str, str]]:
            if not 0 <= line < len(lines):
                return []
            fragments = lines[line]
            if selection is None:
                return fragments
            length = sum(len(fragment[1]) for fragment in fragments)
            span = selection.line_span(line, length)
            return (
                highlight_fragments(fragments, span, selection_style)
                if span is not None
                else fragments
            )

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

    def _keyed_locations(self) -> list[tuple[int | None, int]]:
        """Return paint-local locations, or the cached eager location map."""

        if self._uses_virtual_history():
            return [
                (unit.key if unit is not None else None, offset)
                for unit, offset in self._virtual_locations
            ]
        width = self._content_width
        cached = self._keyed_cache
        if cached is not None and cached[0] == width and cached[1] == self._revision:
            return cached[2]
        keyed = [
            (unit.key if unit is not None else None, offset)
            for unit, offset in self._locations(width)
        ]
        self._keyed_cache = (width, self._revision, keyed)
        return keyed

    def _anchor_for(self, cell: Cell) -> SelectionAnchor:
        """Pin a viewport cell to a stable unit and source offset."""

        line, column = cell
        locations = (
            self._virtual_locations
            if self._uses_virtual_history()
            else self._locations(self._content_width)
        )
        if 0 <= line < len(locations) and locations[line][0] is not None:
            unit, offset = locations[line]
            return SelectionAnchor(unit.key, offset, line, column)
        return SelectionAnchor(None, 0, line, column)

    def _resolved_selection(self, locations: Any = None) -> Selection | None:
        """Resolve a selection for paint without dropping off-screen anchors."""

        del locations
        anchored = self._selection
        if anchored is None:
            return None
        resolved = anchored.resolve(self._keyed_locations())
        if resolved is not None:
            return resolved
        live_keys = {unit.key for unit in self._units if unit is not None}
        anchored_keys = {anchored.anchor.unit_key, anchored.extent.unit_key} - {None}
        if not anchored_keys <= live_keys:
            self._selection = None
        return None

    def _selection_for_copy(
        self, anchored: AnchoredSelection | None = None
    ) -> Selection | None:
        anchored = anchored or self._selection
        if anchored is None:
            return None
        if not self._uses_virtual_history():
            return anchored.resolve(self._keyed_locations())
        keyed = [
            (unit.key if unit is not None else None, offset)
            for unit, offset in self._locations(self._content_width)
        ]
        return anchored.resolve(keyed)

    def selection_text(self) -> str:
        """Materialize virtual history only when the selected text is copied."""

        selection = self._selection_for_copy()
        if selection is None:
            return ""
        lines = self._parsed_lines(self._content_width)

        def line_text(index: int) -> str | None:
            if index < 0 or index >= len(lines):
                return None
            return "".join(fragment[1] for fragment in lines[index])

        return selection.text(line_text)

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
        locations: list[tuple[Any | None, int]],
        anchor: tuple[Any | None, int],
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
