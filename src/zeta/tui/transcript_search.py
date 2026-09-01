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
    """Apply search styles to the rendered text with one Rich console pass."""

    if not matches:
        return base
    text = Text.from_ansi(base)
    line_starts: list[int] = []
    offset = 0
    for line in text.plain.splitlines():
        line_starts.append(offset)
        offset += len(line) + 1
    for match_index, match in enumerate(matches):
        style = SEARCH_CURRENT if match_index == current_index else SEARCH_MATCH
        for line, start, end in match.ranges:
            text.stylize(style, line_starts[line] + start, line_starts[line] + end)
    output = StringIO()
    Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=max(1, width),
        theme=RICH_THEME,
    ).print(text, end="")
    return output.getvalue()
