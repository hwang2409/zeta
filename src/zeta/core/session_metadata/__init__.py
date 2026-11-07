"""Versioned session metadata parsing and serialization."""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...config.tool_policy import validate_tool_patterns
from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog
from ..session_execution import SessionExecutionLease
from ..session_files import SessionError
from ..session_links import _PROJECT_ROLES
from ..store import ConversationStore
from ..store.prompt_composition import parse_prompt_recipe

logger = logging.getLogger(__name__)
META_VERSION = 1

_MEMORY_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _valid_memory_digest(value: object) -> bool:
    return type(value) is str and _MEMORY_DIGEST.match(value) is not None


def _account_metadata(
    value: Mapping[str, Any], path: Path
) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    selected = value.get("selected_accounts", {})
    sequences = value.get("account_switch_sequences", {})
    boundaries = value.get("native_context_boundaries", {})
    valid_providers = {"anthropic", "codex"}
    if (
        type(selected) is not dict
        or any(
            provider not in valid_providers
            or type(account) is not str
            or not account.startswith(f"{provider}:")
            for provider, account in selected.items()
        )
        or any(
            type(items) is not dict
            or any(
                provider not in valid_providers or type(number) is not int or number < 0
                for provider, number in items.items()
            )
            for items in (sequences, boundaries)
        )
    ):
        raise SessionError(f"session account selection is invalid: {path}")
    return selected, sequences, boundaries


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class SessionMetadata:
    version: int
    session_id: str
    created_at: str
    updated_at: str
    provider: str
    model: str
    cwd: str
    retained_tail: int
    compaction_budget: int
    override_audit: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    skill_catalog: list[dict[str, Any]] | None = None
    agent_catalog: list[dict[str, Any]] | None = None
    vim_mode: bool = True
    budget_pinned: bool = False
    plan_mode: bool = False
    name: str = ""
    approval_mode: str | None = None
    # Previous provider, model, and budget until a frontend selection succeeds.
    model_fallback: tuple[str, str, int] | None = None
    project_id: str | None = None
    project_role: str | None = None
    parent_session_id: str | None = None
    # Structured location of the owned project-memory block within
    # ``system_prompt`` (see project_context.refresh_project_memory).  Absent on
    # legacy sessions, which are refreshed by never touching their prompt.
    project_memory_offset: int | None = None
    project_memory_length: int | None = None
    project_memory_digest: str | None = None
    prompt_recipe: str | None = None
    prompt_components: dict[str, dict[str, int | str]] = field(default_factory=dict)
    tool_allow: tuple[str, ...] | None = None
    tool_deny: tuple[str, ...] = ()
    tool_allow_layers: tuple[tuple[str, ...], ...] = ()
    selected_accounts: dict[str, str] = field(default_factory=dict)
    account_switch_sequences: dict[str, int] = field(default_factory=dict)
    native_context_boundaries: dict[str, int] = field(default_factory=dict)
    _execution_lease: SessionExecutionLease | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        provider: str,
        model: str,
        cwd: str,
        retained_tail: int,
        compaction_budget: int,
        system_prompt: str = "",
        context_files: list[str] | tuple[str, ...] = (),
        skill_catalog: SkillCatalog | None = None,
        agent_catalog: AgentCatalog | None = None,
        vim_mode: bool = True,
        budget_pinned: bool = False,
        plan_mode: bool = False,
        name: str = "",
        project_id: str | None = None,
        project_role: str | None = None,
        parent_session_id: str | None = None,
        project_memory_offset: int | None = None,
        project_memory_length: int | None = None,
        project_memory_digest: str | None = None,
        prompt_recipe: str | None = None,
        prompt_components: dict[str, dict[str, int | str]] | None = None,
        tool_allow: tuple[str, ...] | None = None,
        tool_deny: tuple[str, ...] = (),
        tool_allow_layers: tuple[tuple[str, ...], ...] = (),
    ) -> SessionMetadata:
        timestamp = _now()
        return cls(
            version=META_VERSION,
            session_id=session_id,
            created_at=timestamp,
            updated_at=timestamp,
            provider=provider,
            model=model,
            cwd=cwd,
            retained_tail=retained_tail,
            compaction_budget=compaction_budget,
            system_prompt=system_prompt,
            context_files=list(context_files),
            skill_catalog=skill_catalog.to_snapshot()
            if skill_catalog is not None
            else None,
            agent_catalog=agent_catalog.to_snapshot()
            if agent_catalog is not None
            else None,
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
            project_id=project_id,
            project_role=project_role,
            parent_session_id=parent_session_id,
            project_memory_offset=project_memory_offset,
            project_memory_length=project_memory_length,
            project_memory_digest=project_memory_digest,
            prompt_recipe=prompt_recipe,
            prompt_components=copy.deepcopy(prompt_components or {}),
            tool_allow=tool_allow,
            tool_deny=tool_deny,
            tool_allow_layers=tool_allow_layers,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: Path) -> SessionMetadata:
        if type(value.get("version")) is not int:
            raise SessionError(f"session metadata version is missing: {path}")
        if value["version"] != META_VERSION:
            raise SessionError(
                f"unsupported session metadata version {value['version']} at {path}; "
                f"expected {META_VERSION}"
            )
        required_strings = (
            "session_id",
            "created_at",
            "updated_at",
            "provider",
            "model",
            "cwd",
        )
        if any(
            type(value.get(key)) is not str or not value[key]
            for key in required_strings
        ):
            raise SessionError(f"session metadata is incomplete: {path}")
        retained_tail = value.get("retained_tail")
        compaction_budget = value.get("compaction_budget")
        if value.get("compaction") == "summary":
            logger.warning(
                "session requested removed summary compaction; using eviction"
            )
        if (
            type(retained_tail) is not int
            or retained_tail < 1
            or type(compaction_budget) is not int
            or compaction_budget < 1
        ):
            raise SessionError(f"session metadata budgets are invalid: {path}")
        audit = value.get("override_audit", [])
        if type(audit) is not list or any(type(item) is not dict for item in audit):
            raise SessionError(f"session metadata override audit is invalid: {path}")
        fallback = value.get("model_fallback")
        if fallback is not None and (
            type(fallback) is not list
            or len(fallback) != 3
            or type(fallback[0]) is not str
            or fallback[0] not in {"claude", "codex"}
            or type(fallback[1]) is not str
            or not fallback[1].strip()
            or type(fallback[2]) is not int
            or fallback[2] <= 0
        ):
            raise SessionError(f"session model fallback is invalid: {path}")
        has_system_prompt = "system_prompt" in value
        has_context_files = "context_files" in value
        has_context_snapshot = has_system_prompt and has_context_files
        # Preserve whichever legacy bytes exist even when the old snapshot is
        # incomplete. An incomplete snapshot has no recorded recipe ownership,
        # so resume treats it conservatively instead of rebuilding it.
        system_prompt = value.get("system_prompt", "") if has_system_prompt else ""
        context_files = value.get("context_files", []) if has_context_files else []
        skill_catalog = value.get("skill_catalog")
        agent_catalog = value.get("agent_catalog")
        vim_mode = value.get("vim_mode", True)
        budget_pinned = value.get("budget_pinned", False)
        plan_mode = value.get("plan_mode", False)
        name = value.get("name", "")
        raw_tool_allow = value.get("tool_allow")
        raw_tool_deny = value.get("tool_deny", [])
        raw_tool_allow_layers = value.get("tool_allow_layers")
        (
            selected_accounts,
            account_switch_sequences,
            native_context_boundaries,
        ) = _account_metadata(value, path)
        try:
            tool_allow = validate_tool_patterns(raw_tool_allow, field="tool_allow")
            tool_deny = validate_tool_patterns(raw_tool_deny, field="tool_deny")
            tool_allow_layers = (
                tuple(
                    validate_tool_patterns(layer, field="tool_allow_layers") or ()
                    for layer in raw_tool_allow_layers
                )
                if raw_tool_allow_layers is not None
                else (() if tool_allow is None else (tool_allow,))
            )
        except (TypeError, ValueError) as exc:
            raise SessionError(
                f"session metadata tool policy is invalid: {path}"
            ) from exc
        assert tool_deny is not None
        project_id = value.get("project_id")
        project_role = value.get("project_role")
        parent_session_id = value.get("parent_session_id")
        if any(
            item is not None and (type(item) is not str or not item)
            for item in (project_id, project_role, parent_session_id)
        ) or (project_role is not None and project_role not in _PROJECT_ROLES):
            raise SessionError(f"session project linkage is invalid: {path}")
        raw_offset = (
            value.get("project_memory_offset") if has_context_snapshot else None
        )
        raw_length = (
            value.get("project_memory_length") if has_context_snapshot else None
        )
        raw_digest = (
            value.get("project_memory_digest") if has_context_snapshot else None
        )
        # The owned-memory span is all-or-none: offset, length, and digest must
        # arrive together.  A partial triple (e.g. an offset/length with no
        # digest) is treated as a legacy/absent span rather than authorizing an
        # unverifiable replacement, so all three fields drop to None.
        present = (
            raw_offset is not None,
            raw_length is not None,
            raw_digest is not None,
        )
        if not all(present):
            memory_offset = memory_length = memory_digest = None
        else:
            if (
                type(raw_offset) is not int
                or raw_offset < 0
                or type(raw_length) is not int
                or raw_length < 0
                or not _valid_memory_digest(raw_digest)
            ):
                raise SessionError(f"session project memory span is invalid: {path}")
            memory_offset, memory_length, memory_digest = (
                raw_offset,
                raw_length,
                raw_digest,
            )
        prompt_recipe, prompt_components = parse_prompt_recipe(
            value,
            system_prompt=system_prompt,
            has_context_snapshot=has_context_snapshot,
            path=path,
        )
        if (
            type(system_prompt) is not str
            or type(context_files) is not list
            or any(type(item) is not str for item in context_files)
            or (
                skill_catalog is not None
                and (
                    type(skill_catalog) is not list
                    or any(type(item) is not dict for item in skill_catalog)
                )
            )
            or (
                agent_catalog is not None
                and (
                    type(agent_catalog) is not list
                    or any(type(item) is not dict for item in agent_catalog)
                )
            )
            or type(vim_mode) is not bool
            or type(budget_pinned) is not bool
            or type(plan_mode) is not bool
            or value.get("approval_mode") not in (None, "ask", "allow", "deny")
            or type(name) is not str
        ):
            raise SessionError(f"session metadata context is invalid: {path}")
        if skill_catalog is not None:
            try:
                SkillCatalog.from_snapshot(skill_catalog)
            except ValueError as exc:
                raise SessionError(
                    f"session metadata skill catalog is invalid: {path}"
                ) from exc
        if agent_catalog is not None:
            try:
                AgentCatalog.from_snapshot(agent_catalog)
            except ValueError as exc:
                raise SessionError(
                    f"session metadata agent catalog is invalid: {path}"
                ) from exc
        return cls(
            version=value["version"],
            session_id=value["session_id"],
            created_at=value["created_at"],
            updated_at=value["updated_at"],
            provider=value["provider"],
            model=value["model"],
            cwd=value["cwd"],
            retained_tail=retained_tail,
            compaction_budget=compaction_budget,
            override_audit=[dict(item) for item in audit],
            system_prompt=system_prompt,
            context_files=list(context_files),
            skill_catalog=[dict(item) for item in skill_catalog]
            if skill_catalog is not None
            else None,
            agent_catalog=[dict(item) for item in agent_catalog]
            if agent_catalog is not None
            else None,
            vim_mode=vim_mode,
            budget_pinned=budget_pinned,
            plan_mode=plan_mode,
            name=name,
            approval_mode=value.get("approval_mode"),
            model_fallback=tuple(fallback) if fallback is not None else None,
            project_id=project_id,
            project_role=project_role,
            parent_session_id=parent_session_id,
            project_memory_offset=memory_offset,
            project_memory_length=memory_length,
            project_memory_digest=memory_digest,
            prompt_recipe=prompt_recipe,
            prompt_components=prompt_components,
            tool_allow=tool_allow,
            tool_deny=tool_deny,
            tool_allow_layers=tool_allow_layers,
            selected_accounts=dict(selected_accounts),
            account_switch_sequences=dict(account_switch_sequences),
            native_context_boundaries=dict(native_context_boundaries),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "cwd": self.cwd,
            "retained_tail": self.retained_tail,
            "compaction_budget": self.compaction_budget,
            "override_audit": self.override_audit,
            "system_prompt": self.system_prompt,
            "context_files": self.context_files,
            "skill_catalog": self.skill_catalog,
            "agent_catalog": self.agent_catalog,
            "vim_mode": self.vim_mode,
            "budget_pinned": self.budget_pinned,
            "plan_mode": self.plan_mode,
            "name": self.name,
            "approval_mode": self.approval_mode,
            "project_id": self.project_id,
            "project_role": self.project_role,
            "parent_session_id": self.parent_session_id,
            "project_memory_offset": self.project_memory_offset,
            "project_memory_length": self.project_memory_length,
            "project_memory_digest": self.project_memory_digest,
            "prompt_recipe": self.prompt_recipe,
            "prompt_components": self.prompt_components,
            "tool_allow": list(self.tool_allow)
            if self.tool_allow is not None
            else None,
            "tool_deny": list(self.tool_deny),
            "selected_accounts": self.selected_accounts,
            "account_switch_sequences": self.account_switch_sequences,
            "native_context_boundaries": self.native_context_boundaries,
            **(
                {"tool_allow_layers": [list(layer) for layer in self.tool_allow_layers]}
                if len(self.tool_allow_layers) > 1
                else {}
            ),
        }

    def to_catalog_dict(self) -> dict[str, Any]:
        return {**self.to_dict(), "system_prompt": self.system_prompt[:512]}

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "model_fallback": list(self.model_fallback)
            if self.model_fallback
            else None,
        }


@dataclass(frozen=True, slots=True)
class OpenedSession:
    metadata: SessionMetadata
    store: ConversationStore
