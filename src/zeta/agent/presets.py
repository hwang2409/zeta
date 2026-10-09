"""Built-in presets for delegated sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..protocol.types import Message, MessageRole, TextContent

AgentType = Literal["general", "explore", "plan", "run"]


@dataclass(frozen=True, slots=True)
class AgentPreset:
    """Define the child prompt and tools for one agent type."""

    name: str
    tool_names: frozenset[str] | None
    preamble: str
    selection_guidance: str
    model: str | None = None
    prompt_suffix: str = ""
    source: str = "packaged"
    path: Path | None = field(default=None, compare=False)
    agents_root: Path | None = field(default=None, compare=False)
    allow_delegation: bool = True
    accepts_follow_ups: bool = False

    @property
    def description(self) -> str:
        return self.selection_guidance

    @property
    def tools(self) -> list[str] | None:
        return sorted(self.tool_names) if self.tool_names is not None else None

    @property
    def body(self) -> str:
        return self.prompt_suffix


GENERAL_PRESET = AgentPreset(
    name="general",
    tool_names=None,
    preamble="",
    selection_guidance="full tool set",
    accepts_follow_ups=True,
)
EXPLORE_PRESET = AgentPreset(
    name="explore",
    tool_names=frozenset(
        {"agent_output", "agent_status", "fetch", "read", "skill", "websearch"}
    ),
    preamble=(
        "You are an explore sub-agent. Use read-only tools to search and inspect. "
        "Summarize the useful findings and return them to the parent."
    ),
    selection_guidance=(
        "read-only agent_output, agent_status, fetch, read, skill, and websearch tools"
    ),
)
PLAN_PRESET = AgentPreset(
    name="plan",
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
        "websearch tools"
    ),
)

RUN_PRESET = AgentPreset(
    name="run",
    tool_names=None,
    preamble=(
        "You are a long-horizon agent run. Work the task to completion rather "
        "than returning a summary early. The orchestrator may send follow-up "
        "instructions while you work; you will see them as new user messages "
        "between turns, so re-read the conversation before continuing."
    ),
    selection_guidance=(
        "full tool set, runs in the background and accepts follow-up messages; "
        "use for a big task rather than a single lookup"
    ),
    accepts_follow_ups=True,
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
