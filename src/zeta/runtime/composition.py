"""Frontend-neutral runtime composition for one zeta session."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..accounts import SessionAccountContext, account_provider
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
from .loop.skill_refresh import RuntimePrompt

BackendBuilder = Callable[..., tuple[CompletionBackend, str]]
BackgroundEventSink = Callable[[StreamEvent], None]


def _runtime_prompt(context: ProjectContext) -> RuntimePrompt:
    components = {
        key: dict(component) for key, component in context.prompt_components.items()
    }
    if (
        "project_memory" not in components
        and context.memory_offset is not None
        and context.memory_length is not None
        and context.memory_digest is not None
    ):
        components["project_memory"] = {
            "offset": context.memory_offset,
            "length": context.memory_length,
            "digest": context.memory_digest,
        }
    return RuntimePrompt(context.system_prompt, components)


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
    on_completion_success: Callable[[], Awaitable[None] | None] | None = None,
    on_plan_mode_change: Callable[[bool], Awaitable[None] | None] | None = None,
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
        backend_kwargs: dict[str, object] = {
            "home": home,
            "stall_seconds": config.stream_stall_seconds,
            "stall_retries": config.stream_stall_retries,
            "token_budget": effective_budget,
        }
        selected_model = session_model or default_model(provider) or "unknown"
        if opened is None:
            opened = manager.create(
                provider=provider,
                model=selected_model,
                cwd=cwd,
                compaction_budget=effective_budget,
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
        metadata = opened.metadata
        account_context = SessionAccountContext.load(home, manager, metadata)
        backend_kwargs.update(account_context.backend_kwargs(provider))
        backend, selected_model = backend_builder(
            provider, session_model, **backend_kwargs
        )
        if resuming and (
            effective_budget != metadata.compaction_budget
            or budget_pinned != metadata.budget_pinned
        ):
            manager.record_budget(
                metadata,
                budget=effective_budget,
                pinned=budget_pinned,
                touch=False,
            )
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
        completion_callback = on_completion_success or (
            lambda: asyncio.to_thread(manager.refresh_recency, metadata.session_id)
        )
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
        memory_enabled = (
            attention_fork is None
            and config.memory_auto
            and metadata.project_id is not None
        )
        memory_account = None
        if memory_enabled:
            memory_provider = provider_for_model(config.memory_model)
            if account_provider(memory_provider) is not None:
                memory_account = account_context.resolve(memory_provider)
        loop_kwargs: dict[str, Any] = {
            "approval_policy": policy,
            "hooks": (
                load_hooks_for_provider(home, provider)
                if config.allow_hooks or not hooks_restricted
                else None
            ),
            "token_budget": effective_budget,
            "retained_tail": metadata.retained_tail,
            "on_completion_success": completion_callback,
            "on_plan_mode_change": on_plan_mode_change,
            "skill_home": home,
            "skill_project_dir": (
                project_discovery.primary_root
                if project_discovery is not None
                and project_discovery.primary_root is not None
                else cwd
            ),
        }
        if project_context.prompt_recipe is None:
            loop_kwargs["system_prompt"] = project_context.system_prompt
        else:
            loop_kwargs["runtime_prompt"] = _runtime_prompt(project_context)
        if max_turns is not None and max_turns > 0:
            loop_kwargs["max_turns"] = max_turns
        registry = ToolRegistry(
            opened.store.cwd,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
            project_id=metadata.project_id,
            project_registry=manager.project_registry,
            inbox_enabled=config.inbox_enabled,
            command_niceness=config.command_niceness,
            tool_allow=tool_policy.allow,
            tool_deny=tool_policy.deny,
            tool_allow_layers=tool_policy.allow_layers,
            image_policy=image_policy_for_provider(provider),
            account_context=account_context,
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
        loop.explicit_finish = config.agents_explicit_finish
        loop.post_stream_provider_retry = post_stream_provider_retry
        loop.manager = manager
        loop.session_metadata = metadata
        loop.account_context = account_context
        memory_reconciler = None
        if memory_enabled:
            memory_backend: CompletionBackend | None = None
            memory_backend_lock = asyncio.Lock()

            async def invoke_memory(prompt: str) -> ReconciliationResponse:
                nonlocal memory_backend
                if memory_backend is None:
                    async with memory_backend_lock:
                        if memory_backend is None:
                            memory_provider = provider_for_model(config.memory_model)
                            memory_backend, _ = await asyncio.to_thread(
                                backend_builder,
                                memory_provider,
                                config.memory_model,
                                home=home,
                                account=memory_account,
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
            loop.context_assembler.on_before_eviction = (
                memory_reconciler.before_eviction
            )
            if resuming:
                memory_reconciler.catch_up()
        activity_callbacks = [
            callback
            for callback in (
                memory_reconciler.activity if memory_reconciler is not None else None,
            )
            if callback is not None
        ]
        if activity_callbacks:
            def persisted_activity(seq: int) -> None:
                for callback in activity_callbacks:
                    callback(seq)

            opened.store.on_persisted_activity = persisted_activity
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
