"""Built-in presets for bounded sub-agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..types import Message, MessageRole, TextContent

AgentType = Literal["general", "explore", "plan", "run"]


@dataclass(frozen=True, slots=True)
class AgentPreset:
    """Define the child prompt, tools, and turn budget for one agent type."""

    name: AgentType
    turn_cap: int
    tool_names: frozenset[str] | None
    preamble: str
    selection_guidance: str


GENERAL_PRESET = AgentPreset(
    name="general",
    turn_cap=25,
    tool_names=None,
    preamble="",
    selection_guidance="full tool set, up to 25 turns",
)
EXPLORE_PRESET = AgentPreset(
    name="explore",
    turn_cap=15,
    tool_names=frozenset(
        {"agent_output", "agent_status", "fetch", "read", "skill", "websearch"}
    ),
    preamble=(
        "You are an explore sub-agent. Use read-only tools to search and inspect. "
        "Summarize the useful findings and return them to the parent."
    ),
    selection_guidance=(
        "read-only agent_output, agent_status, fetch, read, skill, and websearch tools, "
        "up to 15 turns"
    ),
)
PLAN_PRESET = AgentPreset(
    name="plan",
    turn_cap=20,
    tool_names=frozenset(
        {
            "agent_output",
            "agent_status",
            "fetch",
            "read",
            "skill",
            "todo",
            "websearch",
        }
    ),
    preamble=(
        "You are a plan sub-agent. Inspect the task with read-only tools, then "
        "create or update a concise todo plan. Return the plan to the parent."
    ),
    selection_guidance=(
        "read-only agent_output, agent_status, fetch, read, skill, todo, and "
        "websearch tools, "
        "up to 20 turns"
    ),
)

RUN_PRESET = AgentPreset(
    name="run",
    turn_cap=150,
    tool_names=None,
    preamble=(
        "You are a long-horizon agent run. Work the task to completion rather "
        "than returning a summary early. The orchestrator may send follow-up "
        "instructions while you work; you will see them as new user messages "
        "between turns, so re-read the conversation before continuing."
    ),
    selection_guidance=(
        "full tool set, up to 150 turns, runs in the background and accepts "
        "follow-up messages; use for a big task rather than a single lookup"
    ),
)

AGENT_PRESETS: dict[AgentType, AgentPreset] = {
    preset.name: preset
    for preset in (GENERAL_PRESET, EXPLORE_PRESET, PLAN_PRESET, RUN_PRESET)
}


def get_agent_preset(agent_type: object) -> AgentPreset | None:
    """Return a built-in preset, without defaulting unknown values."""

    if type(agent_type) is not str:
        return None
    return next(
        (preset for preset in AGENT_PRESETS.values() if preset.name == agent_type),
        None,
    )


def agent_type_names() -> list[str]:
    """Return the current names from the preset registry."""

    return [preset.name for preset in AGENT_PRESETS.values()]


def agent_type_description() -> str:
    """Describe each registered preset for the agent tool schema."""

    choices = "; ".join(
        f"{preset.name}: {preset.selection_guidance}"
        for preset in AGENT_PRESETS.values()
    )
    return f"Choose one of: {choices}."


def compose_system_prompt(
    system_prompt: str | Message,
    preamble: str,
) -> str | Message:
    """Prepend a typed-agent preamble to the existing child prompt."""

    if not preamble:
        return system_prompt
    if isinstance(system_prompt, Message):
        return Message(
            MessageRole.SYSTEM,
            [TextContent(preamble), *system_prompt.content],
            metadata=dict(system_prompt.metadata),
        )
    return f"{preamble}\n\n{system_prompt}"
