"""Depth policy for delegated agent trees."""

from __future__ import annotations

DEFAULT_MAX_AGENT_DEPTH = 3
HARD_MAX_AGENT_DEPTH = 5


def child_depth(
    parent_depth: int, max_depth: int, _background: bool
) -> tuple[int, str | None]:
    depth = parent_depth + 1
    if depth > max_depth:
        return depth, f"agent error: maximum agent nesting depth is {max_depth}"
    return depth, None
