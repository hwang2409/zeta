"""Compose isolated project context and guidance for one child runtime."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..core.project_context import (
    ProjectContext,
    discover_project,
    load_project_context,
    resolve_project_context_from_projects,
)
from ..core.session import env_home
from ..path_identity import same_physical_path
from ..project_context.projection import ProjectContextProjection
from ..protocol.types import Message, MessageRole, TextContent
from ..runtime.prompt import (
    PromptFragment,
    RuntimePrompt,
    owned_fragment,
    replace_owned_fragment,
)
from ..skills.agent_catalog import load_agent
from ..tools import ToolRegistry
from .presets import AgentPreset, compose_system_prompt

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop


def _load_child_project_context(
    loop: AgentLoop,
    cwd_override: str | None,
    child_registry: ToolRegistry | None,
    *,
    projection: ProjectContextProjection | None = None,
) -> ProjectContext:
    """Compose one child's project context through the tolerant projection."""

    if child_registry is not None:
        child_registry.verify_cwd_identity()
    child_cwd = Path(
        cwd_override
        or (child_registry.cwd if child_registry is not None else loop.store.cwd)
    )
    home_hint = loop.active_home
    zeta_home = Path(home_hint) if home_hint else env_home()
    discovery = discover_project(child_cwd, user_home=zeta_home.parent)
    registry = loop.project_registry
    if projection is None and registry is not None:
        projection = registry.project_context_projection(loop.root_project_id)
    session_project = projection.session_project if projection is not None else None
    cwd_project = (
        resolve_project_context_from_projects(discovery, projection.projects)
        if projection is not None
        else None
    )
    return load_project_context(
        cwd=child_cwd,
        repo_root=discovery.primary_root or discovery.cwd,
        zeta_home=zeta_home,
        catalog=(
            child_registry.skill_catalog
            if child_registry is not None
            else loop.tool_registry.skill_catalog
        ),
        session_project=session_project,
        cwd_project=cwd_project,
        inbox_enabled=False,
        registry=registry,
        project_projection=projection,
    )


def _child_base_system_prompt(
    loop: AgentLoop,
    cwd_override: str | None,
    child_registry: ToolRegistry | None = None,
) -> ProjectContext:
    """Preserve inherited bytes while giving every child its own health."""

    if cwd_override is not None and not same_physical_path(
        cwd_override, loop.store.cwd
    ):
        return _load_child_project_context(loop, cwd_override, child_registry)
    registry = getattr(loop, "project_registry", None)
    projection = (
        registry.project_context_projection(loop.root_project_id)
        if registry is not None
        else None
    )
    runtime_prompt = (
        loop._skill_prompt_base()
        if hasattr(loop, "_skill_prompt_base")
        else RuntimePrompt(loop.context_assembler.system_prompt, {})
    )
    prompt = runtime_prompt.system_prompt
    if (
        projection is not None
        and isinstance(prompt, str)
        and "project_memory" in runtime_prompt.prompt_components
    ):
        projected = _load_child_project_context(
            loop, cwd_override, child_registry, projection=projection
        )
        projected_prompt = RuntimePrompt(
            projected.system_prompt, projected.prompt_components
        )
        replacement = (
            owned_fragment(projected_prompt, "project_memory")
            if "project_memory" in projected.prompt_components
            else PromptFragment("", {})
        )
        runtime_prompt = replace_owned_fragment(
            runtime_prompt, "project_memory", replacement
        )
        prompt = runtime_prompt.system_prompt
    components = (
        {
            key: dict(component)
            for key, component in runtime_prompt.prompt_components.items()
        }
        if isinstance(prompt, str)
        else {}
    )
    return ProjectContext(
        cast(str, prompt),
        (),
        prompt_recipe="custom",
        prompt_components=components,
        health=projection.health if projection is not None else None,
    )


_EXPLICIT_FINISH_GUIDANCE = (
    "With explicit finish enabled, this run ends only when you call the `finish` "
    "tool with `status` and `summary` (plus `verdict` and `head_sha` when they "
    "apply). Ending a turn without calling `finish` leaves you idle for "
    "follow-ups; call it when the contract's exit criteria are met or when you "
    "are blocked."
)


def _compose_child_project_context(
    context: ProjectContext,
    preset: AgentPreset,
    *,
    explicit_finish: bool = False,
) -> ProjectContext:
    """Apply child guidance while retaining owned-span locations."""

    prompt = _compose_child_system_prompt(
        context.system_prompt,
        preset,
        explicit_finish=explicit_finish,
    )
    if not isinstance(prompt, str):
        return replace(
            context,
            system_prompt=cast(str, prompt),
            prompt_components={},
        )
    shift = len(preset.preamble) + 2 if preset.preamble else 0
    components = {
        key: {**component, "offset": int(component["offset"]) + shift}
        for key, component in context.prompt_components.items()
    }
    return replace(
        context,
        system_prompt=prompt,
        memory_offset=(
            context.memory_offset + shift
            if context.memory_offset is not None
            else None
        ),
        prompt_components=components,
    )


def _compose_child_system_prompt(
    system_prompt: str | Message,
    preset: AgentPreset,
    *,
    explicit_finish: bool = False,
) -> str | Message:
    """Apply the preset preamble and custom body to the child prompt."""

    composed = compose_system_prompt(system_prompt, preset.preamble)
    body = load_agent(preset)
    if explicit_finish:
        body = (
            f"{body}\n\n{_EXPLICIT_FINISH_GUIDANCE}"
            if body
            else _EXPLICIT_FINISH_GUIDANCE
        )
    if not body:
        return composed
    if isinstance(composed, Message):
        return Message(
            MessageRole.SYSTEM,
            [*composed.content, TextContent(body)],
            metadata=dict(composed.metadata),
        )
    return composed + "\n\n" + body
