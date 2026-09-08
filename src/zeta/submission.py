"""Shared submission state and frontend-neutral runtime composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.approval import ApprovalDecision, ApprovalPolicy
from .core.hooks import load_hooks_for_provider
from .core.project_context import ProjectContext, discover_repo_root
from .core.session import OpenedSession, SessionManager
from .core.slash import resolve_session_budget
from .loop import AgentLoop
from .settings import ResolvedConfig
from .tools._user_discovery import ExternalToolDiscovery, apply_external_tools
from .types import CompletionBackend, StreamEvent


@dataclass(frozen=True, slots=True)
class Submission:
    """One immutable submission identity and its captured composer state."""

    id: int
    text: str
    draft_revision: int = 0
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1
    steer: bool = True


BackendBuilder = Callable[..., tuple[CompletionBackend, str]]
BackgroundEventSink = Callable[[StreamEvent], None]


@dataclass(slots=True)
class RuntimeComposition:
    """The shared state that both native frontends need."""

    opened: OpenedSession
    loop: AgentLoop
    policy: ApprovalPolicy
    provider: str
    model: str
    external_tools: ExternalToolDiscovery
    effective_token_budget: int
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
) -> RuntimeComposition:
    """Build one session, policy, loop, and tool registry for any frontend."""

    session_model = model if opened is None else model or opened.metadata.model
    backend, selected_model = backend_builder(
        provider,
        session_model,
        home=home,
        stall_seconds=config.stream_stall_seconds,
        stall_retries=config.stream_stall_retries,
    )
    if opened is None:
        effective_budget, budget_pinned = resolve_session_budget(
            0, False, provider, selected_model, config.token_budget
        )
        opened = manager.create(
            provider=provider,
            model=selected_model,
            cwd=cwd,
            compaction_budget=effective_budget,
            system_prompt=project_context.system_prompt,
            context_files=[str(path) for path in project_context.files],
            budget_pinned=budget_pinned,
        )
    else:
        effective_budget, budget_pinned = resolve_session_budget(
            opened.metadata.compaction_budget,
            opened.metadata.budget_pinned,
            provider,
            selected_model,
            config.token_budget,
        )
        if (
            effective_budget != opened.metadata.compaction_budget
            or budget_pinned != opened.metadata.budget_pinned
        ):
            manager.record_budget(
                opened.metadata, budget=effective_budget, pinned=budget_pinned
            )

    metadata = opened.metadata
    completion_callback = on_completion_success or (lambda: manager.touch(metadata))
    policy = ApprovalPolicy(
        store=opened.store,
        default=ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK,
        always_allow=config.approval_allow,
        always_deny=config.approval_deny,
        always_ask=config.approval_ask,
    )
    loop_kwargs: dict[str, Any] = {
        "approval_policy": policy,
        "hooks": load_hooks_for_provider(home, provider),
        "token_budget": effective_budget,
        "retained_tail": metadata.retained_tail,
        "on_completion_success": completion_callback,
        "on_plan_mode_change": on_plan_mode_change,
        "system_prompt": project_context.system_prompt,
    }
    if max_turns is not None and max_turns > 0:
        loop_kwargs["max_turns"] = max_turns
    loop = AgentLoop(backend, opened.store, **loop_kwargs)
    if metadata.plan_mode:
        loop.set_plan_mode(True)
    repo_root = discover_repo_root(Path(metadata.cwd))
    loop.set_mcp_scope(home=home, project_dir=repo_root)
    external_tools = apply_external_tools(
        loop.tool_registry,
        home=home,
        project_dir=repo_root / ".zeta",
    )
    loop.tool_schemas = list(loop.tool_registry.schemas)
    if background_event_sink is not None:
        loop.set_background_event_sink(background_event_sink)
    loop.session_start()
    return RuntimeComposition(
        opened=opened,
        loop=loop,
        policy=policy,
        provider=provider,
        model=selected_model,
        external_tools=external_tools,
        effective_token_budget=effective_budget,
        budget_pinned=budget_pinned,
    )


__all__ = ["RuntimeComposition", "Submission", "compose_runtime"]
