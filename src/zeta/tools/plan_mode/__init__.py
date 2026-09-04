"""Shared plan-mode configuration."""

from __future__ import annotations

from ..agent_presets import PLAN_PRESET

# Plan mode and the plan sub-agent mean the same thing by "read-only", so they
# share one definition rather than keeping two that can drift apart.
PLAN_MODE_TOOLS = PLAN_PRESET.tool_names or frozenset()

PLAN_MODE_PREAMBLE = (
    "You are in PLAN MODE. Only read-only tools are available: you cannot edit "
    "files, run shell commands, or change anything on disk.\n"
    "Research the task with the tools you have, then deliver a concise, concrete "
    "plan as your assistant answer. This answer ends the turn. The user decides "
    "when to implement the plan. Do not request approval or implementation with "
    "a tool call."
)


__all__ = [
    "PLAN_MODE_PREAMBLE",
    "PLAN_MODE_TOOLS",
]
