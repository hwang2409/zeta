"""Load one skill prompt from the session's static skill catalog."""

from __future__ import annotations

from ...skills import load_skill_prompt
from ..registry import ToolRegistry


async def _load(registry: ToolRegistry, arguments: dict[str, str]) -> str:
    name = arguments["name"]
    try:
        meta = registry.skill_catalog.find(name)
    except ValueError:
        await registry.refresh_skills(force=True)
        meta = registry.skill_catalog.find(name)
    return load_skill_prompt(meta)


def register(registry: ToolRegistry) -> None:
    registry.register(
        "skill",
        lambda arguments, abort_signal: _load(registry, arguments),
        description="Load a skill prompt by name.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )
