"""Frontend-neutral runtime composition for one zeta session."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.settings import ResolvedConfig
from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.hooks import load_hooks_for_provider
from ..core.project_context import (
    ProjectContext,
    ProjectDiscovery,
    refresh_project_memory,
)
from ..core.session import OpenedSession, SessionManager
from ..core.slash import resolve_session_budget
from ..models.catalog import default_model
from ..protocol.types import CompletionBackend, StreamEvent
from ..skills import SkillCatalog
from ..skills.agent_catalog import AgentCatalog
from ..tools._shared.user_discovery import ExternalToolDiscovery, apply_external_tools
from ..tools.registry import ToolRegistry
from .loop import AgentLoop

BackendBuilder = Callable[..., tuple[CompletionBackend, str]]
BackgroundEventSink = Callable[[StreamEvent], None]


@dataclass(slots=True)
class RuntimeComposition:
    """The shared state that both native frontends need."""

    opened: OpenedSession
    loop: AgentLoop
    policy: ApprovalPolicy
    model: str
    external_tools: ExternalToolDiscovery
    budget_pinned: bool


def compose_runtime(
    *,
    home: Path,
    cwd: Path,
    manager: SessionManager,
    config: ResolvedConfig,
    provider: str,
    model: str | None,
    project_context: ProjectContext,
    backend_builder: BackendBuilder,
    opened: OpenedSession | None = None,
    on_completion_success: Callable[[], None] | None = None,
    on_plan_mode_change: Callable[[bool], None] | None = None,
    max_turns: int | None = None,
    background_event_sink: BackgroundEventSink | None = None,
    skill_catalog: SkillCatalog,
    agent_catalog: AgentCatalog | None = None,
    auto_project: bool = True,
    project_id: str | None = None,
    project_discovery: ProjectDiscovery | None = None,
) -> RuntimeComposition:
    """Build one session, policy, loop, and tool registry for any frontend."""

    with ExitStack() as cleanup:
        session_model = model if opened is None else model or opened.metadata.model
        budget_model = session_model or default_model(provider) or "unknown"
        if opened is None:
            effective_budget, budget_pinned = resolve_session_budget(
                0, False, provider, budget_model, config.token_budget
            )
        else:
            effective_budget, budget_pinned = resolve_session_budget(
                opened.metadata.compaction_budget,
                opened.metadata.budget_pinned,
                provider,
                budget_model,
                config.token_budget,
            )
        if opened is None or config.compaction_pinned:
            compaction = config.compaction
            compaction_pinned = config.compaction_pinned
        else:
            compaction = opened.metadata.compaction
            compaction_pinned = opened.metadata.compaction_pinned
        backend_kwargs: dict[str, object] = {
            "home": home,
            "stall_seconds": config.stream_stall_seconds,
            "stall_retries": config.stream_stall_retries,
            "token_budget": effective_budget,
        }
        backend, selected_model = backend_builder(
            provider, session_model, **backend_kwargs
        )
        if opened is None:
            opened = manager.create(
                provider=provider,
                model=selected_model,
                cwd=cwd,
                compaction_budget=effective_budget,
                compaction=compaction,
                compaction_pinned=compaction_pinned,
                system_prompt=project_context.system_prompt,
                context_files=[str(path) for path in project_context.files],
                skill_catalog=skill_catalog,
                agent_catalog=agent_catalog,
                budget_pinned=budget_pinned,
                project_memory_offset=project_context.memory_offset,
                project_memory_length=project_context.memory_length,
                project_memory_digest=project_context.memory_digest,
                auto_project=(
                    config.auto_project and auto_project and project_discovery is None
                ),
                project_id=(
                    project_discovery.project.project_id
                    if project_discovery is not None
                    and project_discovery.project is not None
                    else project_id
                ),
            )
            cleanup.enter_context(opened.store)
        else:
            if (
                effective_budget != opened.metadata.compaction_budget
                or budget_pinned != opened.metadata.budget_pinned
            ):
                manager.record_budget(
                    opened.metadata,
                    budget=effective_budget,
                    pinned=budget_pinned,
                    touch=False,
                )
            if (
                compaction != opened.metadata.compaction
                or compaction_pinned != opened.metadata.compaction_pinned
            ):
                manager.record_compaction(
                    opened.metadata,
                    compaction=compaction,
                    pinned=compaction_pinned,
                    touch=False,
                )

        metadata = opened.metadata
        if opened is not None:
            project_context = ProjectContext(
                refresh_project_memory(
                    metadata.system_prompt,
                    home=home,
                    cwd=cwd,
                    project_id=metadata.project_id,
                    memory_offset=metadata.project_memory_offset,
                    memory_length=metadata.project_memory_length,
                    memory_digest=metadata.project_memory_digest,
                ),
                project_context.files,
                project_context.notices,
            )
        if metadata.skill_catalog is None:
            raise ValueError("session has no persisted skill catalog")
        skill_catalog = SkillCatalog.from_snapshot(metadata.skill_catalog)
        if metadata.agent_catalog is None:
            raise ValueError("session has no persisted agent catalog")
        agent_catalog = AgentCatalog.from_snapshot(metadata.agent_catalog)
        completion_callback = on_completion_success or (lambda: manager.touch(metadata))
        policy = ApprovalPolicy(
            store=opened.store,
            default=metadata.approval_mode
            or (ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK),
            always_allow=config.approval_allow,
            always_deny=config.approval_deny,
            always_ask=config.approval_ask,
        )
        loop_kwargs: dict[str, Any] = {
            "approval_policy": policy,
            "hooks": load_hooks_for_provider(home, provider),
            "token_budget": effective_budget,
            "retained_tail": metadata.retained_tail,
            "compaction": metadata.compaction,
            "on_completion_success": completion_callback,
            "on_plan_mode_change": on_plan_mode_change,
            "system_prompt": project_context.system_prompt,
        }
        if max_turns is not None and max_turns > 0:
            loop_kwargs["max_turns"] = max_turns
        registry = ToolRegistry(
            opened.store.cwd,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
            project_id=metadata.project_id,
            project_registry=manager.project_registry,
            compaction=metadata.compaction,
        )
        cleanup.callback(registry.background_tasks.release_directory)
        loop = AgentLoop(
            backend,
            opened.store,
            registry=registry,
            skill_catalog=skill_catalog,
            root_project_id=metadata.project_id,
            project_registry=manager.project_registry,
            **loop_kwargs,
        )
        loop.manager = manager
        loop.session_metadata = metadata
        if metadata.plan_mode:
            loop.set_plan_mode(True)
        repo_root = (
            project_discovery.primary_root or project_discovery.cwd
            if project_discovery is not None
            else Path(metadata.cwd).resolve()
        )
        loop.set_mcp_scope(home=home, project_dir=repo_root)
        external_tools = apply_external_tools(
            loop.tool_registry,
            home=home,
            project_dir=repo_root / ".zeta",
        )
        loop.tool_schemas = list(loop.tool_registry.schemas)
        if background_event_sink is not None:
            loop.set_background_event_sink(background_event_sink)
        composition = RuntimeComposition(
            opened=opened,
            loop=loop,
            policy=policy,
            model=selected_model,
            external_tools=external_tools,
            budget_pinned=budget_pinned,
        )
        cleanup.pop_all()
        return composition


__all__ = ["RuntimeComposition", "compose_runtime"]
