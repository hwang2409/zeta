"""Safe prompt adoption when a persisted session starts a new run."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..core.project_context import (
    ProjectContext,
    load_project_context,
    refresh_project_memory,
)
from ..core.session import SessionManager, SessionMetadata
from ..skills import SkillCatalog, discover_session_skills
from ..skills.agent_catalog import AgentCatalog, discover_session_agents

ContextLoader = Callable[..., ProjectContext]


@dataclass(frozen=True, slots=True)
class ResumedPrompt:
    """The complete prompt inputs adopted for one resumed runtime."""

    context: ProjectContext
    skill_catalog: SkillCatalog
    agent_catalog: AgentCatalog


def _catalogs(
    metadata: SessionMetadata,
    *,
    home: Path,
    repo_root: Path,
    refresh: bool,
) -> tuple[SkillCatalog, AgentCatalog]:
    if refresh or metadata.skill_catalog is None:
        skills = discover_session_skills(home=home, project_dir=repo_root)
    else:
        skills = SkillCatalog.from_snapshot(metadata.skill_catalog)
    if refresh or metadata.agent_catalog is None:
        agents = discover_session_agents(home=home, project_dir=repo_root)
    else:
        agents = AgentCatalog.from_snapshot(metadata.agent_catalog)
    return skills, agents


def _conservative_context(metadata: SessionMetadata, *, home: Path) -> ProjectContext:
    """Refresh only a digest-owned memory span and preserve all unknown bytes."""

    prompt = metadata.system_prompt
    offset = metadata.project_memory_offset
    length = metadata.project_memory_length
    digest = metadata.project_memory_digest
    owned = False
    if offset is not None and length is not None and digest is not None:
        end = offset + length
        owned = (
            offset >= 0
            and length >= 0
            and end <= len(prompt)
            and hashlib.sha256(prompt[offset:end].encode()).hexdigest() == digest
        )
    refreshed = refresh_project_memory(
        prompt,
        home=home,
        project_id=metadata.project_id,
        memory_offset=offset,
        memory_length=length,
        memory_digest=digest,
    )
    components = {
        key: dict(component) for key, component in metadata.prompt_components.items()
    }
    if owned and offset is not None and length is not None:
        suffix_length = len(prompt) - offset - length
        new_length = len(refreshed) - offset - suffix_length
        new_digest = hashlib.sha256(
            refreshed[offset : offset + new_length].encode()
        ).hexdigest()
        if "project_memory" in components:
            components["project_memory"] = {
                "offset": offset,
                "length": new_length,
                "digest": new_digest,
            }
        length = new_length
        digest = new_digest
    return ProjectContext(
        refreshed,
        tuple(Path(path) for path in metadata.context_files),
        memory_offset=offset,
        memory_length=length,
        memory_project_id=metadata.project_id,
        memory_digest=digest,
        has_override=metadata.prompt_recipe == "custom",
        prompt_recipe=metadata.prompt_recipe,
        prompt_components=components,
    )


def _adopted_context(metadata: SessionMetadata, notices: tuple[str, ...]) -> ProjectContext:
    return ProjectContext(
        metadata.system_prompt,
        tuple(Path(path) for path in metadata.context_files),
        notices,
        memory_offset=metadata.project_memory_offset,
        memory_length=metadata.project_memory_length,
        memory_project_id=metadata.project_id,
        memory_digest=metadata.project_memory_digest,
        has_override=metadata.prompt_recipe == "custom",
        prompt_recipe=metadata.prompt_recipe,
        prompt_components={
            key: dict(component) for key, component in metadata.prompt_components.items()
        },
    )


def resume_prompt(
    metadata: SessionMetadata,
    *,
    manager: SessionManager,
    home: Path,
    repo_root: Path,
    inbox_enabled: bool,
    system_override: str | None = None,
    system_append: str | None = None,
    context_loader: ContextLoader = load_project_context,
) -> ResumedPrompt:
    """Classify, safely recompose, persist, and adopt one resumed prompt.

    A recorded default recipe is fully rebuilt. Every other stored recipe uses
    the conservative flow: unknown prompt bytes remain unchanged and only a
    digest-validated project-memory span can be replaced. Explicit prompt flags
    are a new user-authored custom recipe rather than a default rebuild.
    """

    explicit_custom = system_override is not None or system_append is not None
    rebuild = metadata.prompt_recipe == "default" or explicit_custom
    skills, agents = _catalogs(
        metadata, home=home, repo_root=repo_root, refresh=rebuild
    )
    if rebuild:
        context = context_loader(
            cwd=Path(metadata.cwd),
            repo_root=repo_root,
            zeta_home=home,
            system_override=system_override,
            system_append=system_append,
            catalog=skills,
            project_id=metadata.project_id,
            inbox_enabled=inbox_enabled,
        )
    else:
        context = _conservative_context(metadata, home=home)
    manager.persist_prompt_composition(
        metadata,
        system_prompt=context.system_prompt,
        context_files=[str(path) for path in context.files],
        skill_catalog=skills,
        agent_catalog=agents,
        prompt_recipe=context.prompt_recipe,
        prompt_components=context.prompt_components,
        project_memory_offset=context.memory_offset,
        project_memory_length=context.memory_length,
        project_memory_digest=context.memory_digest,
    )
    adopted = _adopted_context(metadata, context.notices)
    return ResumedPrompt(
        adopted,
        SkillCatalog.from_snapshot(metadata.skill_catalog or []),
        AgentCatalog.from_snapshot(metadata.agent_catalog or []),
    )


__all__ = ["ResumedPrompt", "resume_prompt"]
