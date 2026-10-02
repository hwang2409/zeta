"""Recognize explicit ``$skill`` references in composer text."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_SKILL_TOKEN_RE = re.compile(r"(?<!\S)\$([A-Za-z0-9][A-Za-z0-9_-]*)")
_COMPLETION_TOKEN_RE = re.compile(r"(?<!\S)\$([A-Za-z0-9_-]*)$")


@dataclass(frozen=True, slots=True)
class DollarSkillMention:
    """One registered skill mention outside Markdown code."""

    name: str
    start: int
    end: int


def dollar_skill_mentions(
    value: str, registered_names: Collection[str]
) -> tuple[DollarSkillMention, ...]:
    """Return distinct registered mentions in source order.

    A leading ``!`` is composer shell mode, where dollar expressions retain
    their shell meaning. Backtick spans, including fenced blocks, are ignored.
    """

    if _is_shell_mode(value):
        return ()
    registered = set(registered_names)
    code_ranges = _backtick_ranges(value)
    seen: set[str] = set()
    mentions: list[DollarSkillMention] = []
    for match in _SKILL_TOKEN_RE.finditer(value):
        name = match.group(1)
        if name not in registered or name in seen or _in_ranges(match.start(), code_ranges):
            continue
        seen.add(name)
        mentions.append(DollarSkillMention(name, match.start(), match.end()))
    return tuple(mentions)


def dollar_completion_prefix(value_before_cursor: str) -> str | None:
    """Return the active dollar-skill prefix, or ``None`` outside that mode."""

    if _is_shell_mode(value_before_cursor):
        return None
    match = _COMPLETION_TOKEN_RE.search(value_before_cursor)
    if match is None or _in_ranges(match.start(), _backtick_ranges(value_before_cursor)):
        return None
    return match.group(1)


def _is_shell_mode(value: str) -> bool:
    return value.lstrip().startswith("!")


def _backtick_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        start = value.find("`", cursor)
        if start < 0:
            break
        run_end = start + 1
        while run_end < len(value) and value[run_end] == "`":
            run_end += 1
        marker = value[start:run_end]
        close = value.find(marker, run_end)
        end = len(value) if close < 0 else close + len(marker)
        ranges.append((start, end))
        cursor = end
    return tuple(ranges)


def _in_ranges(position: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= position < end for start, end in ranges)


__all__ = [
    "DollarSkillMention",
    "dollar_completion_prefix",
    "dollar_skill_mentions",
]
