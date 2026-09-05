"""Depth and shared-turn policy for sub-agent trees."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ErrorInfo

MAX_AGENT_DEPTH = 2
# Hard cap on caller-supplied `max_turns` for one agent tree. Preset defaults
# are 15-25, so 200 leaves ~8-13x headroom for deep multi-child research
# without letting a runaway prompt burn arbitrary turns.
MAX_AGENT_TURN_CAP = 200


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


@dataclass(slots=True)
class AgentTree:
    """Own the shared turn budget for one complete agent tree."""

    budget: SharedTurnBudget | None = None

    def ensure_budget(self, limit: int) -> SharedTurnBudget:
        if self.budget is None:
            self.budget = SharedTurnBudget(limit)
        return self.budget


def child_depth(parent_depth: int, _background: bool) -> tuple[int, str | None]:
    depth = parent_depth + 1
    if depth > MAX_AGENT_DEPTH:
        return depth, f"agent error: maximum agent nesting depth is {MAX_AGENT_DEPTH}"
    return depth, None


def consume_turn(budget: SharedTurnBudget | None) -> ErrorInfo | None:
    if budget is None or budget.consume():
        return None
    return ErrorInfo(
        "agent_turn_budget",
        "shared agent turn budget exhausted for this agent tree: "
        f"{budget.limit} of {budget.limit} turns used",
    )
