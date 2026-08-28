"""Depth and shared-turn policy for sub-agent trees."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ErrorInfo


MAX_AGENT_DEPTH = 2


@dataclass(slots=True)
class SharedTurnBudget:
    """Count turns across all descendants of one parent agent call."""

    limit: int
    remaining: int

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("agent turn budget must be a positive integer")
        self.limit = limit
        self.remaining = limit

    def consume(self) -> bool:
        if self.remaining < 1:
            return False
        self.remaining -= 1
        return True


def configure_budget(
    explicit: int | None, inherited: SharedTurnBudget | None
) -> SharedTurnBudget | None:
    if explicit is not None and inherited is not None:
        raise ValueError("pass only one agent turn budget")
    return inherited or (SharedTurnBudget(explicit) if explicit is not None else None)


def child_depth(parent_depth: int, background: bool) -> tuple[int, str | None]:
    depth = parent_depth + 1
    if depth > MAX_AGENT_DEPTH:
        return depth, f"agent error: maximum agent nesting depth is {MAX_AGENT_DEPTH}"
    if depth == MAX_AGENT_DEPTH and background:
        return depth, "agent error: background grandchildren are not supported"
    return depth, None


def consume_turn(budget: SharedTurnBudget | None) -> ErrorInfo | None:
    if budget is None or budget.consume():
        return None
    return ErrorInfo(
        "agent_turn_budget",
        f"shared agent turn budget exhausted: {budget.limit} turns allocated",
    )
