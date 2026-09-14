"""Load one skill prompt from the session's static skill catalog."""

from __future__ import annotations

from ..skill_catalog import load_skill
from .registry import ToolRegistry


def _load(registry: ToolRegistry, arguments: dict[str, str]) -> str:
    meta = registry.skill_catalog.find(arguments["name"])
    body = load_skill(meta)
    if meta.path.is_dir():
        body += f"\n\nSkill resources directory: {meta.path}"
    return body


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
