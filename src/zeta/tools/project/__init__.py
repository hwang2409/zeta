"""Bounded project memory inspection and deliberate updates."""

from __future__ import annotations

from typing import Any

from ...core.session import SessionError, SessionManager, env_home
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
    try:
        metadata = SessionManager(env_home()).read_metadata(
            registry.session_store.session_id
        )
    except (SessionError, ValueError):
        metadata = None
    bound_id = registry.project_id
    if bound_id is not None:
        return store.show_project(bound_id)
    if metadata is not None and metadata.project_id:
        return store.show_project(metadata.project_id)
    project = store.find_for_directory(registry.cwd)
    if project is None:
        raise ProjectRegistryError("current directory is not associated with a project")
    return project


async def _inspect_project(
    registry: ToolRegistry, arguments: dict[str, Any]
) -> StructuredToolResult:
    try:
        projects = ProjectRegistry(env_home() / "projects")
        project = _project(registry, projects)
        action = arguments.get("action", "inspect")
        if action == "sessions":
            links = projects.list_session_links(project.project_id, limit=100)
            return _success_result(
                text_block(f"{len(links)} recorded sessions for {project.name}"),
                structured_content={
                    "project": project.to_dict(include_lanes=False),
                    "sessions": links,
                },
            )
        memory = {
            name: content for name, content in projects.load_memory(project.project_id)
        }
        return _success_result(
            text_block(f"read bounded memory for project {project.name}"),
            structured_content={
                "project": project.to_dict(include_lanes=False),
                "memory": memory,
            },
        )
    except (ProjectRegistryError, OSError, TypeError, ValueError) as exc:
        return _error(str(exc))


async def _update_project(
    registry: ToolRegistry, arguments: dict[str, Any]
) -> StructuredToolResult:
    try:
        projects = ProjectRegistry(env_home() / "projects")
        project = _project(registry, projects)
        projects.update_memory(
            project.project_id, {arguments["name"]: arguments["content"]}
        )
        memory = {
            name: content for name, content in projects.load_memory(project.project_id)
        }
        return _success_result(
            text_block(f"updated bounded memory for project {project.name}"),
            structured_content={
                "project": project.to_dict(include_lanes=False),
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
    registry.register_session_tool(
        "project_update",
        _update_project,
        approval_subject="name" if registry.project_id is not None else None,
        description=(
            "With approval, replace exactly one bounded project memory file. "
            "The approval preview includes the exact filename and bounded UTF-8 "
            "content size/preview; no hidden target is used."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(_MEMORY_FILES)},
                "content": {"type": "string", "maxLength": 131072},
            },
            "required": ["name", "content"],
            "additionalProperties": False,
        },
        requires_approval=True,
    )
    # Rules continue to be declared against the public filename subject, but
    # their matching value is immutable-session-project-id/filename rather
    # than a cwd-derived path.  Child registries inherit project_id.
    registry.set_approval_subject_resolver(
        "project_update",
        lambda arguments: (
            f"{registry.project_id}/{arguments.get('name')}"
            if registry.project_id is not None
            and isinstance(arguments.get("name"), str)
            else None
        ),
    )
