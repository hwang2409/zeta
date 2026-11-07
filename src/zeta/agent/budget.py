"""Depth policy for delegated agent trees."""

from __future__ import annotations

DEFAULT_MAX_AGENT_DEPTH = 3
HARD_MAX_AGENT_DEPTH = 5


def agent_tool_available(depth: int, max_depth: int) -> bool:
    return depth < max_depth


def depth_limit(parent_max_depth: int, preset_max_depth: int | None) -> int:
    if preset_max_depth is None:
        return parent_max_depth
    return min(parent_max_depth, preset_max_depth)


def child_depth(
    parent_depth: int, max_depth: int, _background: bool
) -> tuple[int, str | None]:
    depth = parent_depth + 1
    if depth > max_depth:
        return depth, f"agent error: maximum agent nesting depth is {max_depth}"
    return depth, None
