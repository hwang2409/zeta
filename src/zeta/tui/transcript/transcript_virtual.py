"""Unit-addressed viewport and source-text search for large transcripts."""

from __future__ import annotations

import re
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent
from rich.text import Text

from .. import theme
from .transcript_search import SearchMatch, find_matches, highlight_fragments

_LAZY_TAIL_MIN_UNITS = 128
_VIRTUAL_MARGIN_SCREENS = 1


class TranscriptVirtualMixin:
    """Bound synchronous transcript work to units in or entering the viewport."""

    @staticmethod
    def _searchable_text(unit: Any) -> str:
        value = unit.value
        if hasattr(value, "search_renderable"):
            value = value.search_renderable
        plain = getattr(value, "plain", None)
        if isinstance(plain, str):
            return plain
        markup = getattr(value, "markup", None)
        if isinstance(markup, str):
            parsed = getattr(value, "parsed", ())
            blocks: list[str] = []
            for token in parsed:
                children = getattr(token, "children", None)
                if children:
                    blocks.append(
                        "".join(
                            child.content
                            for child in children
                            if getattr(child, "type", "")
                            in {"text", "code_inline"}
                        )
                    )
                elif getattr(token, "type", "") in {"fence", "code_block"}:
                    blocks.append(token.content)
            return "\n".join(blocks) if blocks else markup
        renderables = getattr(value, "renderables", None)
        if renderables is not None:
            return "\n".join(
                text
                for item in renderables
                if isinstance((text := getattr(item, "plain", None)), str)
            )
        return ""

    def _indexed_search_matches(self) -> list[SearchMatch]:
        if not self._search_query:
            self._virtual_search_units = []
            return []
        key = (self._revision, self._search_query.casefold())
        if self._virtual_search_key != key:
            pattern = re.compile(re.escape(self._search_query), re.IGNORECASE)
            self._virtual_search_units = [
                unit
                for unit in self._units
                if unit is not None
                for _match in pattern.finditer(self._searchable_text(unit))
            ]
            self._virtual_search_key = key
        matches = [
            SearchMatch(((index, 0, 1),))
            for index in range(len(self._virtual_search_units))
        ]
        self._search_index = self._search_index % len(matches) if matches else 0
        return matches

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
        if self._follow_tail or self._virtual_start is None:
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
            plain = ["".join(fragment[1] for fragment in line) for line in lines]
            for match_index, match in enumerate(find_matches(plain, self._search_query)):
                style = theme.SEARCH_CURRENT if match_index == 0 else theme.SEARCH_MATCH
                for line_index, first, last in match.ranges:
                    lines[line_index] = highlight_fragments(
                        lines[line_index], (first, last), style
                    )

        def get_line(line: int) -> list[tuple[str, str]]:
            return lines[line] if 0 <= line < len(lines) else []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=0),
            show_cursor=False,
        )

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

