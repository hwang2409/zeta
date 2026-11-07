"""Prepare immutable live catalogs for one running agent loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

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


class LiveConfigListener(Protocol):
    """Prepare a catalog surface off-loop, then commit it by reference."""

    def prepare_live_config(
        self, skill_catalog: SkillCatalog, agent_catalog: AgentCatalog
    ) -> object: ...

    def commit_live_config(self, prepared: object) -> None: ...


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
    listeners: tuple[LiveConfigListener, ...]


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
    listener_updates: tuple[tuple[LiveConfigListener, object], ...]
    skill_snapshot: list[dict[str, object]] | None
    agent_snapshot: list[dict[str, object]] | None
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


_CHANGE_ITEM_LIMIT = 25
_CHANGE_NAME_LIMIT = 200
_DIAGNOSTIC_LIMIT = 10
_DIAGNOSTIC_TEXT_LIMIT = 400


def _bounded_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def _bounded_values(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    selected = tuple(_bounded_text(value, _CHANGE_NAME_LIMIT) for value in values[:limit])
    omitted = len(values) - len(selected)
    return (*selected, f"and {omitted} more") if omitted else selected


def _bounded_delta(delta: CatalogDelta) -> CatalogDelta:
    return CatalogDelta(
        _bounded_values(delta.added, _CHANGE_ITEM_LIMIT),
        _bounded_values(delta.removed, _CHANGE_ITEM_LIMIT),
        _bounded_values(delta.updated, _CHANGE_ITEM_LIMIT),
    )


def _format_change(kind: str, delta: CatalogDelta) -> str | None:
    changes = tuple(
        f"{verb} {', '.join(values)}"
        for verb, values in (
            ("added", delta.added),
            ("removed", delta.removed),
            ("updated", delta.updated),
        )
        if values
    )
    return f"{kind} changed: {'; '.join(changes)}" if changes else None


def _bounded_diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(
        _bounded_text(value, _DIAGNOSTIC_TEXT_LIMIT)
        for value in values[:_DIAGNOSTIC_LIMIT]
    )
    omitted = len(values) - len(selected)
    return (*selected, f"and {omitted} more diagnostics") if omitted else selected


def _discover_skills(
    scope: LiveConfigScope, base: LiveConfigBase
) -> tuple[SkillCatalog, tuple[str, ...]]:
    try:
        discovered = discover_session_skills(
            home=scope.home, project_dir=scope.project_dir
        )
        prepared = _retained_skills(base.skill_catalog, discovered)
    except (OSError, ValueError) as exc:
        return base.skill_catalog, (f"skill discovery failed: {exc}",)
    diagnostics = tuple(
        notice
        for notice in prepared.notices
        if notice not in base.skill_catalog.notices
    )
    catalog = (
        base.skill_catalog
        if prepared.skills == base.skill_catalog.skills
        else SkillCatalog(prepared.skills, base.skill_catalog.notices)
    )
    return catalog, diagnostics


def _discover_agents(
    scope: LiveConfigScope, base: LiveConfigBase
) -> tuple[AgentCatalog, tuple[str, ...]]:
    try:
        discovered = discover_session_agents(
            home=scope.home, project_dir=scope.project_dir
        )
        prepared = _retained_agents(base.agent_catalog, discovered)
    except (OSError, ValueError) as exc:
        return base.agent_catalog, (f"agent discovery failed: {exc}",)
    diagnostics = tuple(
        notice
        for notice in prepared.notices
        if notice not in base.agent_catalog.notices
    )
    catalog = (
        base.agent_catalog
        if prepared.agents == base.agent_catalog.agents
        else AgentCatalog(prepared.agents, base.agent_catalog.notices)
    )
    return catalog, diagnostics


def _delta_payload(
    original: CatalogDelta, bounded: CatalogDelta
) -> dict[str, object]:
    return {
        "added_count": len(original.added),
        "added": list(bounded.added),
        "removed_count": len(original.removed),
        "removed": list(bounded.removed),
        "updated_count": len(original.updated),
        "updated": list(bounded.updated),
    }


def _change_payload(
    skill_delta: CatalogDelta,
    agent_delta: CatalogDelta,
    diagnostics: tuple[str, ...],
) -> tuple[str, dict[str, object]]:
    bounded_skills = _bounded_delta(skill_delta)
    bounded_agents = _bounded_delta(agent_delta)
    notice = "\n".join(
        value
        for value in (
            _format_change("skills", bounded_skills),
            _format_change("agent profiles", bounded_agents),
            *diagnostics,
        )
        if value
    )
    payload: dict[str, object] = {
        "changes": {
            "skills": _delta_payload(skill_delta, bounded_skills),
            "agents": _delta_payload(agent_delta, bounded_agents),
        },
        "diagnostics": list(diagnostics),
        "message": notice.replace("\n", " · "),
    }
    return notice, payload


def _prepared_agent_definition(
    base: LiveConfigBase, agents: AgentCatalog, changed: bool
) -> tuple[ToolDefinition | None, tuple[ToolSchema, ...]]:
    definition = base.agent_definition
    schemas = base.tool_schemas
    if definition is None or not changed:
        return definition, schemas
    description, parameters = agent_tool_schema(agents)
    definition = replace(definition, description=description, parameters=parameters)
    if base.explicit_tool_schemas:
        schemas = _replace_agent_schema(schemas, definition.schema())
    return definition, schemas


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
        self.manifest: _Manifest | None = None
        self.skill_catalog = skill_catalog
        self.agent_catalog = agent_catalog
        self.legacy_skill_notice_emitted = False

    def prepare(
        self, base: LiveConfigBase, *, force: bool = False
    ) -> LiveConfigCandidate | None:
        manifest = _manifest(self.scope)
        if not force and manifest == self.manifest:
            return None
        skills, skill_diagnostics = _discover_skills(self.scope, base)
        agents, agent_diagnostics = _discover_agents(self.scope, base)
        skill_delta = _delta(base.skill_catalog.skills, skills.skills)
        agent_delta = _delta(base.agent_catalog.agents, agents.agents)
        legacy_notice = self._legacy_skill_notice(base, skill_delta)
        diagnostics = _bounded_diagnostics(
            (*skill_diagnostics, *agent_diagnostics, *legacy_notice)
        )
        changed = skill_delta.changed or agent_delta.changed or bool(diagnostics)
        if not changed:
            return LiveConfigCandidate(
                base=base,
                manifest=manifest,
                skill_catalog=base.skill_catalog,
                agent_catalog=base.agent_catalog,
                agent_definition=base.agent_definition,
                tool_schemas=base.tool_schemas,
                prompt_base=base.prompt_base,
                runtime_prompt=base.runtime_prompt,
                skills=skill_delta,
                agents=agent_delta,
                diagnostics=(),
                notice=None,
                payload=None,
                listener_updates=(),
                skill_snapshot=None,
                agent_snapshot=None,
                changed=False,
                legacy_skill_notice=False,
            )
        definition, schemas = _prepared_agent_definition(
            base, agents, agent_delta.changed
        )
        prompt_base, runtime_prompt = _prepare_prompt(base, skills)
        notice, payload = _change_payload(skill_delta, agent_delta, diagnostics)
        listener_updates = tuple(
            (listener, listener.prepare_live_config(skills, agents))
            for listener in base.listeners
        )
        return LiveConfigCandidate(
            base=base,
            manifest=manifest,
            skill_catalog=skills,
            agent_catalog=agents,
            agent_definition=definition,
            tool_schemas=schemas,
            prompt_base=prompt_base,
            runtime_prompt=runtime_prompt,
            skills=skill_delta,
            agents=agent_delta,
            diagnostics=diagnostics,
            notice=notice,
            payload=payload,
            listener_updates=listener_updates,
            skill_snapshot=skills.to_snapshot(),
            agent_snapshot=agents.to_snapshot(),
            changed=True,
            legacy_skill_notice=bool(legacy_notice),
        )

    def _legacy_skill_notice(
        self, base: LiveConfigBase, delta: CatalogDelta
    ) -> tuple[str, ...]:
        if (
            not delta.changed
            or "skill_index" in base.prompt_base.prompt_components
            or self.legacy_skill_notice_emitted
        ):
            return ()
        return (
            (
                "skills changed, but this session's prompt predates owned skill spans; "
                "restart the session to refresh its skill index"
            ),
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
    "LiveConfigListener",
    "LiveConfigScope",
    "LiveConfigWatcher",
]
