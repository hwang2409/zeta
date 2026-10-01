"""Unit-addressed viewport and source-text search for large transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent
from rich.console import Console
from rich.text import Text

from .. import theme
from ..theme import RICH_THEME
from .transcript_search import (
    AnchoredSelection,
    Cell,
    SearchMatch,
    Selection,
    SelectionAnchor,
    highlight_fragments,
)

_LAZY_TAIL_MIN_UNITS = 128
_VIRTUAL_MARGIN_SCREENS = 1


@dataclass(frozen=True, slots=True)
class _SearchOccurrence:
    unit: Any
    start: int
    end: int


class TranscriptVirtualMixin:
    """Bound synchronous transcript work to units in or entering the viewport."""

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
        return "\n".join(
            line.rstrip(" ") for line in output.getvalue().splitlines()
        )

    def _searchable_text(self, unit: Any, width: int | None = None) -> str:
        """Return exactly the plain text Rich exposes to eager search.

        This has a separate per-unit cache from painting: indexing a card must
        include borders, titles, status text, and nested renderables, without
        causing the viewport render counter to revisit transcript history.
        """

        actual_width = max(1, width or self._content_width)
        revision = self._unit_revision(unit)
        cached = self._unit_search_cache.get(unit.key)
        if cached is not None and cached[:2] == (actual_width, revision):
            return cached[2]
        value = unit.value
        if hasattr(value, "search_renderable"):
            value = value.search_renderable
        if value is None:
            plain = ""
        elif isinstance(value, Text):
            plain = value.plain
        else:
            plain = Text.from_ansi(
                self._search_rendered(unit, actual_width)
            ).plain
        self._unit_search_cache[unit.key] = (actual_width, revision, plain)
        return plain

    def _indexed_search_matches(self) -> list[SearchMatch]:
        if not self._search_query:
            self._virtual_search_occurrences = []
            return []
        key = (
            self._revision,
            self._content_width,
            self._search_query.casefold(),
        )
        if self._virtual_search_key != key:
            pattern = re.compile(re.escape(self._search_query), re.IGNORECASE)
            self._virtual_search_occurrences = [
                _SearchOccurrence(unit, match.start(), match.end())
                for unit in self._units
                if unit is not None
                for match in pattern.finditer(
                    self._searchable_text(unit, self._content_width)
                )
            ]
            self._virtual_search_by_unit = {}
            for index, occurrence in enumerate(self._virtual_search_occurrences):
                self._virtual_search_by_unit.setdefault(occurrence.unit, []).append(
                    (index, occurrence)
                )
            self._virtual_search_key = key
        matches = [
            SearchMatch(((index, 0, 1),))
            for index in range(len(self._virtual_search_occurrences))
        ]
        self._search_index = self._search_index % len(matches) if matches else 0
        return matches

    def _focus_virtual_search_match(self) -> None:
        occurrence = self._virtual_search_occurrences[self._search_index]
        unit_index = self._units.index(occurrence.unit)
        rendered = self._search_rendered(occurrence.unit, self._content_width)
        plain_lines, offsets = self._unit_locations(
            occurrence.unit, self._content_width, rendered
        )
        target_line = 0
        for line_index, (offset, line) in enumerate(zip(offsets, plain_lines)):
            length = len(self._strip_padding(line))
            if offset <= occurrence.start < offset + max(1, length):
                target_line = line_index
                break
            if offset <= occurrence.start:
                target_line = line_index
        self._virtual_start = (unit_index, target_line)
        self._follow_tail = False
        self._anchor = (occurrence.unit, occurrence.start)

    def _highlight_virtual_search(
        self,
        lines: list[list[tuple[str, str]]],
        locations: list[tuple[Any | None, int]],
    ) -> None:
        for line_index, ((unit, offset), fragments) in enumerate(
            zip(locations, lines)
        ):
            if unit is None:
                continue
            length = sum(len(text) for _, text in fragments)
            line_end = offset + length
            for match_index, occurrence in self._virtual_search_by_unit.get(unit, ()):
                first = max(occurrence.start, offset)
                last = min(occurrence.end, line_end)
                if first >= last:
                    continue
                style = (
                    theme.SEARCH_CURRENT
                    if match_index == self._search_index
                    else theme.SEARCH_MATCH
                )
                lines[line_index] = highlight_fragments(
                    lines[line_index], (first - offset, last - offset), style
                )

    def _uses_virtual_history(self) -> bool:
        return self._max_lines is None and len(self._units) >= _LAZY_TAIL_MIN_UNITS

    @staticmethod
    def _unit_revision(unit: Any) -> int:
        value = unit.value
        return int(getattr(value, "revision", 0))

    def _virtual_unit_lines(
        self, index: int, width: int
    ) -> tuple[list[list[tuple[str, str]]], list[tuple[Any | None, int]]]:
        unit = self._units[index]
        if unit is None or unit.value is None:
            return [[]], [(unit, 0)]
        revision = self._unit_revision(unit)
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
            return lines, [
                (unit, offset) for offset in location_entry[2]
            ]
        lines = self._unit_parsed_lines(unit, width)
        self._unit_heights[(width, unit.key, revision)] = len(lines)
        # ``_unit_parsed_lines`` populated the render cache. Avoid even a
        # cached _render_unit call so count-based work remains viewport-bound.
        rendered = self._render_cache[unit.key][2]
        _plain, offsets = self._unit_locations(unit, width, rendered)
        return lines, [(unit, offset) for offset in offsets]

    def _estimated_prefix(self, width: int, unit_index: int, line_offset: int) -> int:
        total = 0
        for unit in self._units[:unit_index]:
            if unit is None:
                total += 1
            else:
                total += self._unit_heights.get(
                    (width, unit.key, self._unit_revision(unit)), 1
                )
        return total + line_offset

    def _virtual_tail_start(self, width: int, wanted: int) -> tuple[int, int]:
        remaining = wanted
        trimming_blanks = True
        for index in range(len(self._units) - 1, -1, -1):
            unit = self._units[index]
            if trimming_blanks and (unit is None or unit.value is None):
                continue
            trimming_blanks = False
            lines, _ = self._virtual_unit_lines(index, width)
            if len(lines) >= remaining:
                return index, max(0, len(lines) - remaining)
            remaining -= len(lines)
        return 0, 0

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
                return index, offset + remaining
            remaining -= available
            index += 1
            offset = 0
        self._follow_tail = True
        return self._virtual_tail_start(width, self._viewport_height)

    def _virtual_content(self, width: int, height: int) -> UIContent:
        wanted = max(height, height * _VIRTUAL_MARGIN_SCREENS)
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
                    (abs(value - anchor[1]), line)
                    for line, value in enumerate(offsets)
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

        lines: list[list[tuple[str, str]]] = []
        locations: list[tuple[Any | None, int]] = []
        index, offset = start
        while index < len(self._units) and len(lines) < wanted:
            unit_lines, unit_locations = self._virtual_unit_lines(index, width)
            lines.extend(unit_lines[offset:])
            locations.extend(unit_locations[offset:])
            index += 1
            offset = 0
        lines = lines[:wanted] or [[]]
        locations = locations[:wanted] or [(None, 0)]
        self._virtual_start = start
        self._virtual_lines = lines
        self._virtual_locations = locations
        self._virtual_width = width
        self._virtual_revision = self._revision
        self._lazy_viewport = True
        self._line_locations = locations
        self._locations_revision = self._revision
        self._anchor = locations[0] if locations else None
        base = self._estimated_prefix(width, *start)
        self._scroll_offset = base

        if self._search_active and self._search_query:
            self._indexed_search_matches()
            self._highlight_virtual_search(lines, locations)

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

