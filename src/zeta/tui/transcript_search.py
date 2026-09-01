"""Search and highlight helpers for the transcript widget."""

from __future__ import annotations

import re
from bisect import bisect_right
from io import StringIO

from rich.console import Console
from rich.text import Text

from .theme import RICH_THEME, SEARCH_CURRENT, SEARCH_MATCH


class SearchMatch:
    """One match and its line-local highlight ranges."""

    def __init__(self, ranges: tuple[tuple[int, int, int], ...]) -> None:
        self.ranges = ranges

    @property
    def first_line(self) -> int:
        return self.ranges[0][0]


class HighlightCache:
    """Cache the rendered transcript and update only changed match lines."""

    def __init__(self, base: str, width: int, matches: list[SearchMatch]) -> None:
        self._width = width
        self._base = Text.from_ansi(base)
        self._lines = list(self._base.split("\n"))
        self._line_matches: dict[int, list[tuple[int, int, int]]] = {}
        for match_index, match in enumerate(matches):
            for line, start, end in match.ranges:
                self._line_matches.setdefault(line, []).append(
                    (match_index, start, end)
                )
        self._fragments: list[str] | None = None
        self._current_index: int | None = None
        self._changed_lines: set[int] = set()
        self._trailing_newline = False
        self._line_fragment_cache: dict[
            int, tuple[list[str], dict[int, int], dict[int, Text]]
        ] = {}

    def _render_text(self, text: Text) -> str:
        output = StringIO()
        Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
            width=max(1, self._width),
            theme=RICH_THEME,
        ).print(text, end="")
        return output.getvalue()

    def _build_line_fragments(
        self, line_index: int, current_index: int
    ) -> tuple[list[str], dict[int, int], dict[int, Text]]:
        line = self._lines[line_index]
        text_fragments: list[Text] = []
        match_fragments: dict[int, int] = {}
        match_texts: dict[int, Text] = {}
        offset = 0
        for match_index, start, end in self._line_matches.get(line_index, ()):
            if offset < start:
                text_fragments.append(line[offset:start])
            match_text = line[start:end]
            match_text.stylize(
                SEARCH_CURRENT if match_index == current_index else SEARCH_MATCH,
                0,
                end - start,
            )
            match_fragments[match_index] = len(text_fragments)
            match_texts[match_index] = line[start:end]
            text_fragments.append(match_text)
            offset = end
        if offset < len(line):
            text_fragments.append(line[offset:])
        combined = Text()
        for index, fragment in enumerate(text_fragments):
            combined.append_text(fragment)
            if index + 1 < len(text_fragments):
                combined.append("\x00")
        return self._render_text(combined).split("\x00"), match_fragments, match_texts

    def _update_line_fragment(
        self,
        line_index: int,
        match_index: int,
        current_index: int,
    ) -> None:
        fragments, match_fragments, match_texts = self._line_fragment_cache[line_index]
        fragment_index = match_fragments[match_index]
        match_text = match_texts[match_index].copy()
        match_text.stylize(
            SEARCH_CURRENT if match_index == current_index else SEARCH_MATCH,
            0,
            len(match_text),
        )
        fragments[fragment_index] = self._render_text(match_text)
        self._fragments[line_index] = "".join(fragments)

    def _render_all(self, current_index: int) -> str:
        text = self._base.copy()
        line_starts: list[int] = []
        offset = 0
        for line in text.plain.splitlines():
            line_starts.append(offset)
            offset += len(line) + 1
        for line_index, ranges in self._line_matches.items():
            for match_index, start, end in ranges:
                line = line_starts[line_index] if line_index < len(line_starts) else 0
                text.stylize(
                    SEARCH_CURRENT if match_index == current_index else SEARCH_MATCH,
                    line + start,
                    line + end,
                )
        return self._render_text(text)

    def render(self, current_index: int) -> str:
        self._changed_lines = set()
        if self._fragments is None:
            rendered = self._render_all(current_index)
            self._trailing_newline = rendered.endswith("\n")
            self._fragments = rendered.splitlines()
        elif self._current_index != current_index:
            previous = self._current_index
            changed_lines: set[int] = set()
            if previous is not None:
                changed_lines.update(
                    line
                    for line, ranges in self._line_matches.items()
                    if any(match_index == previous for match_index, _, _ in ranges)
                )
            changed_lines.update(
                line
                for line, ranges in self._line_matches.items()
                if any(match_index == current_index for match_index, _, _ in ranges)
            )
            for line in changed_lines:
                if line not in self._line_fragment_cache:
                    self._line_fragment_cache[line] = self._build_line_fragments(
                        line, current_index
                    )
                    self._fragments[line] = "".join(
                        self._line_fragment_cache[line][0]
                    )
                else:
                    match_indices = {
                        match_index
                        for match_index, _, _ in self._line_matches[line]
                        if match_index in {previous, current_index}
                    }
                    for match_index in match_indices:
                        self._update_line_fragment(
                            line, match_index, current_index
                        )
            self._changed_lines = changed_lines
        self._current_index = current_index
        rendered = "\n".join(self._fragments)
        return rendered + ("\n" if self._trailing_newline else "")

    @property
    def changed_lines(self) -> frozenset[int]:
        return frozenset(self._changed_lines)

    @property
    def fragments(self) -> tuple[str, ...]:
        return tuple(self._fragments or ())


def find_matches(plain_lines: list[str], query: str) -> list[SearchMatch]:
    """Find case-insensitive query matches in rendered transcript lines."""

    if not query or not plain_lines:
        return []
    plain = "\n".join(plain_lines)
    line_starts: list[int] = []
    offset = 0
    for line in plain_lines:
        line_starts.append(offset)
        offset += len(line) + 1
    matches: list[SearchMatch] = []
    for found in re.finditer(re.escape(query), plain, re.IGNORECASE):
        ranges: list[tuple[int, int, int]] = []
        end = found.end()
        line_index = bisect_right(line_starts, found.start()) - 1
        while line_index < len(plain_lines):
            line_start = line_starts[line_index]
            line_end = line_start + len(plain_lines[line_index])
            start = max(found.start(), line_start) - line_start
            finish = min(end, line_end) - line_start
            if finish > start:
                ranges.append((line_index, start, finish))
            if end <= line_end:
                break
            line_index += 1
        if ranges:
            matches.append(SearchMatch(tuple(ranges)))
    return matches


def highlight(
    base: str,
    width: int,
    matches: list[SearchMatch],
    current_index: int,
) -> str:
    """Apply search styles to rendered text."""

    if not matches:
        return base
    return HighlightCache(base, width, matches).render(current_index)
