"""Build the tool registry used by an agent loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.store import ConversationStore
from ..protocol.types import ToolSchema
from ..skills import SkillCatalog
from ..skills.agent_catalog import AgentCatalog
from ..tools.registry import ToolHandler, ToolRegistry


def select_tool_registry(
    store: ConversationStore,
    *,
    tools: Mapping[str, ToolHandler] | ToolRegistry | None,
    registry: ToolRegistry | None,
    skill_catalog: SkillCatalog,
    agent_catalog: AgentCatalog | None,
    tool_schemas: Sequence[ToolSchema] | None,
) -> ToolRegistry:
    """Select or construct the loop's registry and validate its catalogs."""

    if registry is not None and tools is not None:
        raise ValueError("pass only one tool registry")
    selected_registry = (
        registry
        if registry is not None
        else tools
        if isinstance(tools, ToolRegistry)
        else None
    )
    if selected_registry is not None:
        if selected_registry.skill_catalog != skill_catalog:
            raise ValueError("loop skill catalog must match the tool registry catalog")
        if agent_catalog is not None and selected_registry.agent_catalog != agent_catalog:
            raise ValueError("loop agent catalog must match the tool registry catalog")
        return selected_registry
    if isinstance(tools, Mapping):
        selected_registry = ToolRegistry(
            store.cwd,
            register_builtin=False,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
        )
        schemas_by_name = {
            schema.get("name"): schema
            for schema in (tool_schemas or [])
            if isinstance(schema.get("name"), str)
        }
        for name, handler in tools.items():
            schema = schemas_by_name.get(name, {})
            parameters = schema.get("parameters", schema.get("input_schema"))
            if parameters is None:
                parameters = {
                    key: value
                    for key, value in schema.items()
                    if key not in {"name", "description", "cache_control"}
                }
            selected_registry.register(
                name,
                handler,
                description=(
                    schema.get("description", "")
                    if isinstance(schema.get("description", ""), str)
                    else ""
                ),
                parameters=parameters,
            )
        return selected_registry
    if tools is None:
        return ToolRegistry(
            store.cwd,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
        )
    raise TypeError("tools must be a mapping or ToolRegistry")


__all__ = ["select_tool_registry"]
