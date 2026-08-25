"""Load one skill prompt from the session's static skill catalog."""

from __future__ import annotations

from ..prompts import load_skill
from .registry import ToolRegistry


def register(registry: ToolRegistry) -> None:
    registry.set_skill_loader(load_skill)
    registry.register(
        "skill",
        lambda arguments, abort_signal: registry.load_skill(arguments["name"]),
        description="Load a skill prompt by name.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )
