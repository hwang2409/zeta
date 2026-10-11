"""Bounded project memory inspection."""

from __future__ import annotations

from typing import Any

from ...project_registry import ProjectRegistry, ProjectRegistryError
from ...protocol.types import StructuredToolResult
from ..registry import ToolRegistry, _success_result, text_block

_MEMORY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")


def _error(message: str) -> StructuredToolResult:
    return {
        "content": [text_block(message)],
        "isError": True,
        "structuredContent": {"error": {"kind": "project_memory", "message": message}},
    }


def _project(registry: ToolRegistry, store: ProjectRegistry):
    bound_id = registry.project_id
    if bound_id is not None:
        return store.show_project(bound_id)
    project = store.find_for_directory(registry.cwd)
    if project is None:
        raise ProjectRegistryError("current directory is not associated with a project")
    return project


async def _inspect_project(
    registry: ToolRegistry, arguments: dict[str, Any]
) -> StructuredToolResult:
    try:
        projects = registry.project_registry
        if projects is None:
            raise ProjectRegistryError("project registry capability is unavailable")
        project = _project(registry, projects)
        action = arguments.get("action", "inspect")
        if action == "sessions":
            links = projects.list_session_links(project.project_id, limit=100)
            return _success_result(
                text_block(f"{len(links)} recorded sessions for {project.name}"),
                structured_content={
                    "project": project.to_dict(),
                    "sessions": links,
                },
            )
        if projects.memory_format(project.project_id) == 2:
            memory = projects._entry_memory_view(project.project_id, byte_cap=64 * 1024)
        else:
            memory = {
                name: content for name, content in projects.load_memory(project.project_id)
            }
        return _success_result(
            text_block(f"read bounded memory for project {project.name}"),
            structured_content={
                "project": project.to_dict(),
                "memory": memory,
            },
        )
    except (ProjectRegistryError, OSError, TypeError, ValueError) as exc:
        return _error(str(exc))


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "project",
        _inspect_project,
        description="Read the current project's bounded memory (standard filename-to-text files) or recorded session references.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["inspect", "sessions"]}
            },
            "additionalProperties": False,
        },
        requires_approval=False,
    )
