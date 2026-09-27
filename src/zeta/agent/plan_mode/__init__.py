"""Shared plan-mode configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ...protocol.types import CompletionBackend, Message, ToolSchema
from ...providers.codex import CodexBackend
from ..presets import PLAN_PRESET, compose_system_prompt

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


def plan_mode_prompt(prompt: Message) -> Message:
    composed = compose_system_prompt(prompt, PLAN_MODE_PREAMBLE)
    assert isinstance(composed, Message)
    return composed


def plan_mode_messages(messages: Sequence[Message]) -> list[Message]:
    if not messages:
        return []
    first, *rest = messages
    return [
        replace(
            first,
            metadata={
                **first.metadata,
                "zeta_allowed_tools": sorted(PLAN_MODE_TOOLS | {"agent"}),
            },
        ),
        *rest,
    ]


def plan_mode_tool_schemas(
    backend: CompletionBackend, schemas: Sequence[ToolSchema]
) -> list[ToolSchema]:
    if isinstance(backend, CodexBackend) and backend.model.startswith("gpt-5.6-"):
        return list(schemas)
    allowed = PLAN_MODE_TOOLS | {"agent"}
    return [schema for schema in schemas if schema.get("name") in allowed]


__all__ = [
    "PLAN_MODE_PREAMBLE",
    "PLAN_MODE_TOOLS",
    "plan_mode_messages",
    "plan_mode_prompt",
    "plan_mode_tool_schemas",
]
