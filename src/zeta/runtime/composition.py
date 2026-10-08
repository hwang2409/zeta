"""Frontend-neutral runtime composition for one zeta session."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.settings import ResolvedConfig
from ..config.tool_policy import ToolPolicy
from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.hooks import load_hooks_for_provider
from ..core.project_context import ProjectContext, ProjectDiscovery
from ..core.session import OpenedSession, SessionManager
from ..core.slash import resolve_session_budget
from ..media.image_policy import image_policy_for_provider
from ..memory.auto import AutoMemoryConfig, AutoMemoryReconciler
from ..memory.provider import complete_reconciliation
from ..memory.reconciler import ReconciliationResponse
from ..models.catalog import default_model, provider_for_model
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
    memory_reconciler: AutoMemoryReconciler | None = None


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
    agent_catalog: AgentCatalog,
    auto_project: bool = True,
    project_id: str | None = None,
    project_discovery: ProjectDiscovery | None = None,
    memory_notice: Callable[[str], None] | None = None,
    post_stream_provider_retry: bool = True,
) -> RuntimeComposition:
    """Build one session, policy, loop, and tool registry for any frontend."""

    with ExitStack() as cleanup:
        resuming = opened is not None
        invocation_policy = ToolPolicy.create(
            config.tool_allow,
            config.tool_deny,
            allow_layers=config.tool_allow_layers,
        )
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
        if opened is None:
            compaction = config.compaction
            compaction_pinned = config.compaction_pinned
        else:
            # A session's persisted mode is part of its request shape. Resume
            # must not change it because ambient settings changed later.
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
                prompt_recipe=project_context.prompt_recipe,
                prompt_components=project_context.prompt_components,
                auto_project=(
                    config.auto_project and auto_project and project_discovery is None
                ),
                project_id=(
                    project_discovery.project.project_id
                    if project_discovery is not None
                    and project_discovery.project is not None
                    else project_id
                ),
                tool_allow=config.tool_allow,
                tool_deny=config.tool_deny,
                tool_allow_layers=config.tool_allow_layers,
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
        metadata = opened.metadata
        if resuming:
            persisted_policy = ToolPolicy.create(
                metadata.tool_allow,
                metadata.tool_deny,
                allow_layers=metadata.tool_allow_layers,
            )
            effective_policy = persisted_policy.narrowed_by(invocation_policy)
            manager.persist_tool_policy(
                metadata,
                tool_allow=effective_policy.allow,
                tool_deny=effective_policy.deny,
                tool_allow_layers=effective_policy.allow_layers,
            )
        completion_callback = on_completion_success or (lambda: manager.touch(metadata))
        policy = ApprovalPolicy(
            store=opened.store,
            default=metadata.approval_mode
            or (ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK),
            always_allow=config.approval_allow,
            always_deny=config.approval_deny,
            always_ask=config.approval_ask,
        )
        tool_policy = ToolPolicy.create(
            metadata.tool_allow,
            metadata.tool_deny,
            allow_layers=metadata.tool_allow_layers,
        )
        from ..attention_forks import ATTENTION_FORK_POLICY, read_attention_fork

        attention_fork = read_attention_fork(
            opened.store.session_dir, directory_fd=opened.store.directory_fd
        )
        if attention_fork is not None:
            tool_policy = tool_policy.narrowed_by(ATTENTION_FORK_POLICY)
        hooks_restricted = tool_policy.restricted
        loop_kwargs: dict[str, Any] = {
            "approval_policy": policy,
            "hooks": (
                load_hooks_for_provider(home, provider)
                if config.allow_hooks or not hooks_restricted
                else None
            ),
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
            inbox_enabled=config.inbox_enabled,
            compaction=metadata.compaction,
            tool_allow=tool_policy.allow,
            tool_deny=tool_policy.deny,
            tool_allow_layers=tool_policy.allow_layers,
            image_policy=image_policy_for_provider(provider),
            required_tool_names=tuple(
                dict.fromkeys(
                    (
                        *tool_policy.required_exact_names,
                        *invocation_policy.exact_allow_names,
                    )
                )
            ),
        )
        if attention_fork is not None:
            from ..tools.resolve_attention import register_fork

            registry.unregister("request_attention")
            register_fork(registry)
        cleanup.callback(registry.background_tasks.release_directory)
        loop = AgentLoop(
            backend,
            opened.store,
            registry=registry,
            skill_catalog=skill_catalog,
            root_project_id=metadata.project_id,
            parent_session_id=metadata.parent_session_id,
            project_registry=manager.project_registry,
            **loop_kwargs,
        )
        loop.post_stream_provider_retry = post_stream_provider_retry
        loop.manager = manager
        loop.session_metadata = metadata
        memory_reconciler = None
        if (
            attention_fork is None
            and config.memory_auto
            and metadata.project_id is not None
        ):
            memory_backend: CompletionBackend | None = None

            async def invoke_memory(prompt: str) -> ReconciliationResponse:
                nonlocal memory_backend
                if memory_backend is None:
                    memory_provider = provider_for_model(config.memory_model)
                    memory_backend, _ = backend_builder(
                        memory_provider,
                        config.memory_model,
                        home=home,
                        stall_seconds=config.stream_stall_seconds,
                        stall_retries=config.stream_stall_retries,
                        token_budget=effective_budget,
                    )
                return await complete_reconciliation(memory_backend, prompt)

            memory_reconciler = AutoMemoryReconciler(
                registry=manager.project_registry,
                project_id=metadata.project_id,
                session_id=metadata.session_id,
                session_dir=opened.store.session_dir,
                invoke=invoke_memory,
                config=AutoMemoryConfig(
                    model=config.memory_model,
                    token_threshold=config.memory_token_threshold,
                    idle_seconds=config.memory_idle_minutes * 60,
                ),
                notice=memory_notice,
            )
            loop.memory_reconciler = memory_reconciler
            opened.store.on_persisted_activity = memory_reconciler.activity
            loop.context_assembler.on_before_eviction = (
                memory_reconciler.before_eviction
            )
            if resuming:
                memory_reconciler.catch_up()
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
            allow_external_tools=config.allow_external_tools,
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
            memory_reconciler=memory_reconciler,
        )
        cleanup.pop_all()
        return composition


__all__ = ["RuntimeComposition", "compose_runtime"]
