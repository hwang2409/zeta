"""The tool that ends plan mode once the user approves a plan.

Plan mode narrows the assistant to read-only tools until it proposes a plan.
Proposing is an ordinary approval-gated tool call, so the plan reaches the user
through the same approval card, key bindings, and durable pending-request
machinery every other tool uses. ``AgentLoop`` watches for a successful call and
performs the transition; this module only describes the tool.
"""

from __future__ import annotations

from typing import TypedDict

from ...types import StructuredToolResult
from ..agent_presets import PLAN_PRESET
from ..registry import ToolRegistry, _error_result, _success_result, text_block

EXIT_PLAN_MODE = "exit_plan_mode"

# Plan mode and the plan sub-agent mean the same thing by "read-only", so they
# share one definition rather than keeping two that can drift apart.
PLAN_MODE_TOOLS = PLAN_PRESET.tool_names or frozenset()

PLAN_MODE_PREAMBLE = (
    "You are in PLAN MODE. Only read-only tools are available: you cannot edit "
    "files, run shell commands, or change anything on disk.\n"
    "Research the task with the tools you have, then call the "
    f"`{EXIT_PLAN_MODE}` tool with a concise, concrete plan. The user approves "
    "or rejects it. On approval you leave plan mode and carry the plan out; on "
    "rejection you stay in plan mode and revise.\n"
    "Do not ask for permission in prose — proposing the plan is how you ask."
)


class ExitPlanModeArguments(TypedDict):
    plan: str


def _exit_plan_mode(arguments: ExitPlanModeArguments) -> StructuredToolResult:
    plan = arguments.get("plan", "")
    if type(plan) is not str or not plan.strip():
        # An error result keeps plan mode on: the loop only exits on success.
        return _error_result("provide a nonempty plan")
    return _success_result(
        text_block("plan approved — leaving plan mode; carry it out now"),
        structured_content={"approved": True, "plan": plan},
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        EXIT_PLAN_MODE,
        _exit_plan_mode,
        description=(
            "Present a finished plan for the user to approve. Only available in "
            "plan mode. On approval, plan mode ends and you carry out the plan "
            "immediately; on rejection you stay in plan mode and revise."
        ),
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "The plan to carry out, as concise steps.",
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        },
    )


__all__ = [
    "EXIT_PLAN_MODE",
    "PLAN_MODE_PREAMBLE",
    "PLAN_MODE_TOOLS",
    "register",
]
