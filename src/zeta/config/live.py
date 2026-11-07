"""Prepare immutable live catalogs for one running agent loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..agent.plan_mode import plan_mode_prompt
from ..protocol.types import ToolSchema
from ..runtime.prompt import PromptFragment, RuntimePrompt, replace_owned_fragment
from ..skills.agent_catalog import AgentCatalog, discover_session_agents
from ..skills.catalog import (
    SkillCatalog,
    discover_session_skills,
    skill_roots_signature,
)
from ..tools.agent import agent_tool_schema
from ..tools.registry import ToolDefinition

_FileStat = tuple[int, int, int, int]
type _Manifest = tuple[tuple[str, _FileStat | None], ...]


@dataclass(frozen=True, slots=True)
class LiveConfigScope:
    """Filesystem roots that can change during one loop's lifetime."""

    home: Path
    project_dir: Path


@dataclass(frozen=True, slots=True)
class LiveConfigBase:
    """Loop-owned references captured before worker preparation starts."""

    generation: int
    prompt_generation: int
    runtime_prompt: RuntimePrompt
    prompt_base: RuntimePrompt
    plan_mode: bool
    skill_catalog: SkillCatalog
    agent_catalog: AgentCatalog
    agent_definition: ToolDefinition | None
    tool_schemas: tuple[ToolSchema, ...]
    explicit_tool_schemas: bool


@dataclass(frozen=True, slots=True)
class CatalogDelta:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.updated)


@dataclass(frozen=True, slots=True)
class LiveConfigCandidate:
    """A complete worker-built state transition ready for constant-time commit."""

    base: LiveConfigBase
    manifest: _Manifest
    skill_catalog: SkillCatalog
    agent_catalog: AgentCatalog
    agent_definition: ToolDefinition | None
    tool_schemas: tuple[ToolSchema, ...]
    prompt_base: RuntimePrompt
    runtime_prompt: RuntimePrompt
    skills: CatalogDelta
    agents: CatalogDelta
    diagnostics: tuple[str, ...]
    notice: str | None
    payload: dict[str, object] | None
    changed: bool
    legacy_skill_notice: bool


def _stat(path: Path) -> _FileStat | None:
    try:
        value = path.stat()
    except OSError:
        return None
    return value.st_mtime_ns, value.st_size, value.st_ino, value.st_mode


def _agent_manifest(scope: LiveConfigScope) -> _Manifest:
    manifest: list[tuple[str, _FileStat | None]] = []
    for root in (scope.home / "agents", scope.project_dir / ".zeta" / "agents"):
        try:
            entries = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError:
            manifest.append((str(root), None))
            continue
        manifest.append((str(root), _stat(root)))
        for path in entries:
            if path.suffix.casefold() == ".md":
                manifest.append((str(path), _stat(path)))
    return tuple(manifest)


def _manifest(scope: LiveConfigScope) -> _Manifest:
    return (
        *skill_roots_signature(home=scope.home, project_dir=scope.project_dir),
        *_agent_manifest(scope),
    )


def _retained_skills(current: SkillCatalog, discovered: SkillCatalog) -> SkillCatalog:
    discovered_paths = {skill.path for skill in discovered.skills}
    retained = tuple(
        skill
        for skill in current.skills
        if skill.path not in discovered_paths
        and skill.source != "packaged"
        and (skill.path / "SKILL.md" if skill.path.is_dir() else skill.path).is_file()
    )
    combined = (*discovered.skills, *retained)
    if len({skill.name for skill in combined}) != len(combined):
        return SkillCatalog(current.skills, discovered.notices)
    return SkillCatalog(combined, discovered.notices)


def _retained_agents(current: AgentCatalog, discovered: AgentCatalog) -> AgentCatalog:
    discovered_paths = {agent.path for agent in discovered.agents}
    retained = tuple(
        agent
        for agent in current.agents
        if agent.path is not None
        and agent.path not in discovered_paths
        and agent.source != "packaged"
    )
    combined = (*discovered.agents, *retained)
    missing_notices = tuple(
        f"retained agent profile {agent.name!r}: source is unavailable: {agent.path}"
        for agent in retained
        if agent.path is not None and not agent.path.is_file()
    )
    notices = (*discovered.notices, *missing_notices)
    if len({agent.name for agent in combined}) != len(combined):
        return AgentCatalog(current.agents, notices)
    return AgentCatalog(combined, notices)


def _delta(old: tuple[Any, ...], new: tuple[Any, ...]) -> CatalogDelta:
    old_by_name = {item.name: item for item in old}
    new_by_name = {item.name: item for item in new}
    old_names = set(old_by_name)
    new_names = set(new_by_name)
    return CatalogDelta(
        tuple(sorted(new_names - old_names)),
        tuple(sorted(old_names - new_names)),
        tuple(
            sorted(
                name
                for name in old_names & new_names
                if old_by_name[name] != new_by_name[name]
            )
        ),
    )


def _prepare_prompt(
    base: LiveConfigBase, catalog: SkillCatalog
) -> tuple[RuntimePrompt, RuntimePrompt]:
    prompt = base.prompt_base
    if (
        catalog.skills != base.skill_catalog.skills
        and "skill_index" in prompt.prompt_components
    ):
        prompt = replace_owned_fragment(
            prompt, "skill_index", PromptFragment(catalog.index(), {})
        )
    displayed = (
        RuntimePrompt(plan_mode_prompt(prompt.system_prompt), prompt.prompt_components)
        if base.plan_mode and prompt is not base.prompt_base
        else base.runtime_prompt
        if prompt is base.prompt_base
        else prompt
    )
    return prompt, displayed


def _replace_agent_schema(
    schemas: tuple[ToolSchema, ...], replacement: ToolSchema
) -> tuple[ToolSchema, ...]:
    return tuple(
        replacement if schema.get("name") == "agent" else schema for schema in schemas
    )


def _format_change(kind: str, delta: CatalogDelta) -> str | None:
    changes: list[str] = []
    if delta.added:
        changes.append(f"added {', '.join(delta.added)}")
    if delta.removed:
        changes.append(f"removed {', '.join(delta.removed)}")
    if delta.updated:
        changes.append(f"updated {', '.join(delta.updated)}")
    return f"{kind} changed: {'; '.join(changes)}" if changes else None


def _bounded_diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        value if len(value) <= 400 else f"{value[:397]}..." for value in values
    )


class LiveConfigWatcher:
    """Enumerate and prepare all live catalogs behind one worker interface."""

    def __init__(
        self,
        scope: LiveConfigScope,
        *,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog,
    ) -> None:
        self.scope = scope
        self.manifest = _manifest(scope)
        self.skill_catalog = skill_catalog
        self.agent_catalog = agent_catalog
        self.legacy_skill_notice_emitted = False

    def prepare(
        self, base: LiveConfigBase, *, force: bool = False
    ) -> LiveConfigCandidate | None:
        manifest = _manifest(self.scope)
        if not force and manifest == self.manifest:
            return None
        discovery_diagnostics: list[str] = []
        try:
            discovered_skills = discover_session_skills(
                home=self.scope.home, project_dir=self.scope.project_dir
            )
            prepared_skills = _retained_skills(base.skill_catalog, discovered_skills)
            discovery_diagnostics.extend(
                notice
                for notice in prepared_skills.notices
                if notice not in base.skill_catalog.notices
            )
            skills = (
                base.skill_catalog
                if prepared_skills.skills == base.skill_catalog.skills
                else SkillCatalog(prepared_skills.skills, base.skill_catalog.notices)
            )
        except (OSError, ValueError) as exc:
            skills = base.skill_catalog
            discovery_diagnostics.append(f"skill discovery failed: {exc}")
        try:
            discovered_agents = discover_session_agents(
                home=self.scope.home, project_dir=self.scope.project_dir
            )
            prepared_agents = _retained_agents(base.agent_catalog, discovered_agents)
            discovery_diagnostics.extend(
                notice
                for notice in prepared_agents.notices
                if notice not in base.agent_catalog.notices
            )
            agents = (
                base.agent_catalog
                if prepared_agents.agents == base.agent_catalog.agents
                else AgentCatalog(prepared_agents.agents, base.agent_catalog.notices)
            )
        except (OSError, ValueError) as exc:
            agents = base.agent_catalog
            discovery_diagnostics.append(f"agent discovery failed: {exc}")

        skill_delta = _delta(base.skill_catalog.skills, skills.skills)
        agent_delta = _delta(base.agent_catalog.agents, agents.agents)
        legacy_skill_notice = (
            skill_delta.changed
            and "skill_index" not in base.prompt_base.prompt_components
            and not self.legacy_skill_notice_emitted
        )
        if legacy_skill_notice:
            discovery_diagnostics.append(
                "skills changed, but this session's prompt predates owned skill spans; "
                "restart the session to refresh its skill index"
            )
        diagnostics = _bounded_diagnostics(tuple(discovery_diagnostics))
        semantic_change = (
            skill_delta.changed or agent_delta.changed or bool(diagnostics)
        )
        if not semantic_change:
            return LiveConfigCandidate(
                base,
                manifest,
                base.skill_catalog,
                base.agent_catalog,
                base.agent_definition,
                base.tool_schemas,
                base.prompt_base,
                base.runtime_prompt,
                skill_delta,
                agent_delta,
                (),
                None,
                None,
                False,
                False,
            )

        definition = base.agent_definition
        schemas = base.tool_schemas
        if definition is not None and agent_delta.changed:
            description, parameters = agent_tool_schema(agents)
            definition = replace(
                definition, description=description, parameters=parameters
            )
            if base.explicit_tool_schemas:
                schemas = _replace_agent_schema(schemas, definition.schema())
        prompt_base, runtime_prompt = _prepare_prompt(base, skills)
        notice_parts = tuple(
            value
            for value in (
                _format_change("skills", skill_delta),
                _format_change("agent profiles", agent_delta),
                *diagnostics,
            )
            if value
        )
        notice = "\n".join(notice_parts)
        payload: dict[str, object] = {
            "changes": {
                "skills": {
                    "added": list(skill_delta.added),
                    "removed": list(skill_delta.removed),
                    "updated": list(skill_delta.updated),
                },
                "agents": {
                    "added": list(agent_delta.added),
                    "removed": list(agent_delta.removed),
                    "updated": list(agent_delta.updated),
                },
            },
            "diagnostics": list(diagnostics),
            "message": notice,
        }
        return LiveConfigCandidate(
            base,
            manifest,
            skills,
            agents,
            definition,
            schemas,
            prompt_base,
            runtime_prompt,
            skill_delta,
            agent_delta,
            diagnostics,
            notice,
            payload,
            True,
            legacy_skill_notice,
        )

    def commit(self, candidate: LiveConfigCandidate) -> None:
        self.manifest = candidate.manifest
        self.skill_catalog = candidate.skill_catalog
        self.agent_catalog = candidate.agent_catalog
        if candidate.legacy_skill_notice:
            self.legacy_skill_notice_emitted = True


__all__ = [
    "LiveConfigBase",
    "LiveConfigCandidate",
    "LiveConfigScope",
    "LiveConfigWatcher",
]
