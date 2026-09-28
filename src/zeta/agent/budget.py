"""Depth policy for delegated agent trees."""

from __future__ import annotations

MAX_AGENT_DEPTH = 2


def child_depth(parent_depth: int, _background: bool) -> tuple[int, str | None]:
    depth = parent_depth + 1
    if depth > MAX_AGENT_DEPTH:
        return depth, f"agent error: maximum agent nesting depth is {MAX_AGENT_DEPTH}"
    return depth, None
