"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import os
import shlex
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from ...agent.background import (
    BackgroundAgentOwner,
    recover_agent_children,
)
from ...agent.budget import MAX_AGENT_DEPTH
from ...agent.durable import durable_message
from ...agent.notifications import AgentNotificationMixin
from ...agent.plan_mode import (
    PLAN_MODE_TOOLS,
    plan_mode_messages,
    plan_mode_prompt,
    plan_mode_tool_schemas,
)
from ...agent.receipt import (
    TerminalState,
    finalize_agent_results,
    terminal_state,
)
from ...agent.runner import run_agent_tool
from ...agent.tool_results import validated_tool_result
from ...core.abort import AbortSignal as ToolAbortSignal
from ...core.approval import ApprovalPolicy
from ...core.context import ContextAssembler
from ...core.hooks import HookManager
from ...core.slash import effective_budget_for_model
from ...core.store import ConversationStore
from ...core.tool_dispatch import dispatch_tool_calls
from ...mcp import (
    MCPManagementService,
    MCPMount,
    home_config_path,
    load_mcp_config_overlay,
    project_config_path,
)
from ...mcp.commands import (
    MCP_USAGE,
    MCPCommandError,
    add_and_mount,
    parse_add_command,
    remove_and_unshadow,
    render_mcp_status,
    run_mcp_auth,
    run_mcp_resource_attach,
    run_mcp_resources_list,
)
from ...mcp.prompt_commands import SlashModelInput
from ...prompts import load_identity
from ...protocol.types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    CompletionBackend,
    ContentBlock,
    ContextWindowBackend,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)
from ...providers.retry_policy import ProviderRetryBudget, apply_retry_budget
from ...providers.stream_diagnostics import fd_diagnostics
from ...runtime.tool_setup import select_tool_registry
from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog
from ...tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from ...tools.agent import MAX_AGENT_RESULT_BYTES, agent_result
from ...tools.registry import (
    ToolExecutionContext,
    _validate_unique_tool_call_ids,
)
from ._completion import (
    ProviderAttemptState,
    _error_info,
    assistant_reset_event,
    can_retry_context,
    close_completion,
    provider_retry_notice,
    task_is_cancelling,
    wait_for_provider_retry,
)
from ._store_writes import StoreWriteMixin
from .cache_trace import CacheTrace
from .empty_turn import (
    annotate_turn_metadata,
    build_nudge_message,
    read_turn_metadata,
    should_nudge_empty_turn,
)
from .mcp_session import MCPSession
from .tool_schema import canonical_tool_schemas

TaskResult = TypeVar("TaskResult")
_validated_tool_result = validated_tool_result
class AgentLoop(StoreWriteMixin, AgentNotificationMixin, MCPSession):
    def notify_background_persisted(self) -> None:
        """Wake the root loop after a durable background notification."""

        # Child-owned notifications stay in the child store and must not wake
        # the shared root owner.
        if self.agent_depth > 0:
            return
        self._background_owner.notify_wake()

    def __init__(
        self,
        backend: CompletionBackend,
        store: ConversationStore,
        *,
        tools: Mapping[str, ToolHandler] | ToolRegistry | None = None,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog | None = None,
        registry: ToolRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int | None = 150,
        context_assembler: ContextAssembler | None = None,
        system_prompt: str | Message | None = None,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        compaction: str = "summary",
        on_completion_success: Callable[[], None] | None = None,
        on_plan_mode_change: Callable[[bool], None] | None = None,
        hooks: HookManager | None = None,
        skip_mcp_mount: bool = False,
        agent_depth: int = 0,
        agent_instance_id: str | None = None,
        root_project_id: str | None = None,
        root_session_dir: Any = None,
        project_registry: Any = None,
        background_owner: BackgroundAgentOwner | None = None,
        usage_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if type(agent_depth) is not int or not 0 <= agent_depth <= MAX_AGENT_DEPTH:
            raise ValueError(f"agent depth must be between 0 and {MAX_AGENT_DEPTH}")
        self.backend = backend
        self.store = store
        self.agent_depth = agent_depth
        self.agent_instance_id = agent_instance_id
        self.root_project_id = root_project_id
        # Directory of the ROOT session that owns the durable child-link index;
        # threaded down every loop so nested children publish their lineage
        # intent into a single flat directory the root can reconcile.
        self.root_session_dir = (
            root_session_dir if root_session_dir is not None else store.session_dir
        )
        self.project_registry = project_registry
        self._background_owner = background_owner or BackgroundAgentOwner(store)
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_child_stores: dict[str, ConversationStore] = {}
        self._agent_child_turns: dict[str, int] = {}
        self._agent_child_types: dict[str, str] = {}
        self._background_child_cancellers: dict[str, Callable[[], None]] = {}
        self._background_child_watchers: dict[str, asyncio.Task[Any]] = {}
        self._background_event_sink: Callable[[StreamEvent], None] | None = None
        self._mcp_notice_sink: Callable[[str], None] | None = None
        self._mcp_prompt_refresh: Callable[[MCPMount], None] | None = None
        self._activated = False
        self._closed = False
        self._turn_active = False
        # Partial assistant persistence uses these values so provider metadata is
        # retained even when the stream later fails or is cancelled.
        self._turn_stop_reason: str | None = None
        self._turn_output_tokens: int | None = None
        self._turn_provider_retry_records: list[dict[str, object]] = []
        self._cache_trace = CacheTrace.from_environment(
            agent_instance_id or store.session_id, agent_depth
        )
        recover_agent_children(self)
        self.tool_registry = select_tool_registry(
            store,
            tools=tools,
            registry=registry,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
            tool_schemas=tool_schemas,
            project_id=root_project_id,
            project_registry=project_registry,
            compaction=compaction,
        )
        self._mcp_mount: MCPMount | None = None
        self._mcp_mount_attempted = skip_mcp_mount
        self._mcp_mount_task: asyncio.Task[None] | None = None
        self._mcp_home_hint: str | None = None
        self._mcp_project_dir_value: Path | None = (
            Path(self.store.cwd).expanduser().resolve()
        )
        self._mcp_config_error: str | None = None
        self._mcp_schema_names: set[str] = set()
        self._provided_tool_schemas = tool_schemas is not None
        self.tool_registry._agent_owner = self._background_owner
        self.tool_registry.bind_session_store(store)
        # Route model-owned task exits into this loop's depth-aware wake so a
        # child exit lands in the child store without waking the shared root.
        self.tool_registry.background_tasks.set_notification_sink(
            self.store, self.notify_background_persisted
        )
        self.agent_catalog = self.tool_registry.agent_catalog
        if (
            approval_policy is not None
            and self.tool_registry.approval_policy is not None
            and self.tool_registry.approval_policy is not approval_policy
        ):
            raise ValueError("pass only one approval policy")
        if approval_policy is not None:
            self.tool_registry.set_approval_policy(approval_policy)
        if self.tool_registry.approval_policy is not None:
            self.tool_registry.bind_approval_store(store)
        self.tool_schemas = list(
            tool_schemas if tool_schemas is not None else self.tool_registry.schemas
        )
        self.max_turns = max_turns
        if system_prompt is None:
            system_prompt = load_identity(catalog=self.tool_registry.skill_catalog)
        self.context_assembler = context_assembler or ContextAssembler(
            store,
            token_budget=token_budget,
            retained_tail=retained_tail,
            compaction=compaction,
            system_prompt=system_prompt,
            backend=backend,
            on_completion_success=on_completion_success,
            usage_sink=usage_sink,
        )
        self.on_completion_success = on_completion_success
        self._on_plan_mode_change = on_plan_mode_change
        self.hooks = hooks
        if self.hooks is not None:
            self.hooks.bind_session(store.session_id)
            if self.tool_registry.pre_execute_hook is None:
                self.tool_registry.set_pre_execute_hook(self.hooks.pre_tool)
        self._plan_mode = False
        self._plan_mode_prior_prompt: Message | None = None
        self._steering_queue: deque[Message] = deque()
        if "agent" in self.tool_registry.definitions_by_name:
            self.tool_registry.set_agent_runner(self._run_agent_tool)

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        """Restrict the assistant to read-only tools, or lift the restriction.
        GPT-5.6 keeps tool schemas stable; other models advertise a subset.
        """
        if enabled == self._plan_mode:
            return
        assembler = self.context_assembler
        if enabled:
            self._plan_mode_prior_prompt = assembler.system_prompt
            assembler.system_prompt = plan_mode_prompt(assembler.system_prompt)
        elif self._plan_mode_prior_prompt is not None:
            assembler.system_prompt = self._plan_mode_prior_prompt
            self._plan_mode_prior_prompt = None
        self._plan_mode = enabled
        if self._on_plan_mode_change is not None:
            self._on_plan_mode_change(enabled)

    def plan_mode_allows(self, tool_name: str) -> bool:
        """Check the current plan-mode allowlist at dispatch time."""
        return not self._plan_mode or tool_name in PLAN_MODE_TOOLS | {"agent"}

    @property
    def background_work_descriptions(self) -> tuple[str, ...]:
        process_work = tuple(
            record.command
            for record in self.tool_registry.background_tasks.records
            if record.running
        )
        return self._background_owner.active_descriptions + process_work

    def _active_tool_schemas(self) -> list[ToolSchema]:
        schemas = (
            self.tool_registry.schemas
            if not self._provided_tool_schemas
            else self.tool_registry.allowed_schemas(self.tool_schemas)
        )
        if self._plan_mode:
            schemas = plan_mode_tool_schemas(self.backend, schemas)
        return canonical_tool_schemas(schemas)

    def set_model(self, model: str) -> None:
        """Set the model used by subsequent provider completions."""
        if not model.strip():
            raise ValueError("model must be a nonempty name")
        if hasattr(self.backend, "model"):
            self.backend.model = model
        else:
            self._model = model
        self.set_token_budget(self.context_assembler.token_budget)

    def set_token_budget(self, token_budget: int) -> None:
        """Align compaction and provider-side context budgets."""

        provider = getattr(self.backend, "provider", None)
        model = getattr(self.backend, "model", None)
        if isinstance(provider, str) and isinstance(model, str):
            token_budget = effective_budget_for_model(provider, model, token_budget)
        self.context_assembler.token_budget = token_budget
        if isinstance(self.backend, ContextWindowBackend):
            self.backend.set_token_budget(token_budget)

    def abort(self) -> None:
        """Signal the active tool batch before the caller cancels the turn."""
        self.tool_registry.abort()
        self._background_owner.cancel_all()
        self._steering_queue.clear()

    def steer(self, message: Message) -> None:
        """Queue a user message for injection at the next tool boundary.
        The running ``_run_turn`` drains this queue before the next provider
        call, so the message never lands between a tool_call and its
        tool_result. Callers must pass a durable USER-role message.
        """
        if message.role is not MessageRole.USER:
            raise ValueError("steering message must have the user role")
        self._steering_queue.append(message)

    @property
    def has_pending_steering(self) -> bool:
        return bool(self._steering_queue)

    def clear_pending_steering(self) -> None:
        self._steering_queue.clear()

    def set_background_event_sink(
        self, sink: Callable[[StreamEvent], None] | None
    ) -> None:
        """Set the sink for progress from children that outlive their turn."""
        self._background_event_sink = sink

    def set_mcp_notice_sink(self, sink: Callable[[str], None] | None) -> None:
        """Set the sink for MCP mount notices."""
        self._mcp_notice_sink = sink

    def set_mcp_prompt_refresh(
        self, callback: Callable[[MCPMount], None] | None
    ) -> None:
        """Set the owner callback for live MCP prompt commands."""
        self._mcp_prompt_refresh = callback
        if callback is not None and self._mcp_mount is not None:
            callback(self._mcp_mount)

    @property
    def mcp_summary(self) -> str:
        if self._mcp_mount is None:
            return "mcp: 0 mounted, 0 failed"
        return self._mcp_mount.summary

    async def slash_mcp(self, args: str) -> str | SlashModelInput:
        """Show MCP state, reconnect, add, remove, authorize, or attach."""
        await self._ensure_mcp_servers()
        mount = self._mcp_mount
        try:
            parts = shlex.split(args)
        except ValueError as exc:
            return f"mcp error: {exc}"
        if self._mcp_config_error is not None:
            return f"mcp error: {self._mcp_config_error}"
        if mount is None:
            return "mcp: no configured servers"
        if not parts:
            return render_mcp_status(mount, home=self._mcp_home_hint)
        verb = parts[0]
        try:
            if verb == "status" and len(parts) == 1:
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "reconnect":
                if len(parts) != 2:
                    return MCP_USAGE
                await MCPManagementService(
                    home=self._mcp_home_hint,
                    project_dir=self._mcp_project_dir_value,
                    mount=mount,
                ).sync_runtime()
                await mount.reconnect(parts[1], notice_sink=self._mcp_notice_sink)
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "add":
                await add_and_mount(
                    mount,
                    parse_add_command(parts[1:]),
                    target=self._mcp_add_target(),
                    notice_sink=self._mcp_notice_sink,
                )
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "remove":
                if len(parts) != 2:
                    return MCP_USAGE
                await remove_and_unshadow(
                    mount,
                    parts[1],
                    home_path=self._mcp_home_path(),
                    load_home=lambda: (
                        load_mcp_config_overlay(
                            home=self._mcp_home_hint, project_dir=None
                        ).configured_servers
                    ),
                    notice_sink=self._mcp_notice_sink,
                )
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "auth":
                if len(parts) != 2:
                    return MCP_USAGE
                return await run_mcp_auth(
                    mount,
                    parts[1],
                    home=self._mcp_home_hint,
                    notice_sink=self._mcp_notice_sink,
                )
            if verb == "resources":
                if len(parts) == 2:
                    return await run_mcp_resources_list(mount, parts[1])
                if len(parts) == 3:
                    return await run_mcp_resource_attach(mount, parts[1], parts[2])
                return MCP_USAGE
        except (MCPCommandError, ValueError) as exc:
            return f"mcp error: {exc}"
        return MCP_USAGE

    async def slash_mcp_prompt(self, name: str, arguments: dict[str, str]) -> str:
        """Resolve one mounted MCP prompt for the next model turn."""

        await self._ensure_mcp_servers()
        if self._mcp_mount is None:
            raise RuntimeError("MCP mount is unavailable")
        return await self._mcp_mount.get_prompt(name, arguments)

    def _mcp_add_target(self) -> Path:
        project_dir = self._mcp_project_dir_value
        if project_dir is not None:
            return project_config_path(project_dir)
        return home_config_path(self._mcp_home_hint)

    def _mcp_home_path(self) -> Path:
        override = os.environ.get("ZETA_MCP_CONFIG")
        if override:
            return Path(override).expanduser().resolve()
        return home_config_path(self._mcp_home_hint).resolve()

    @property
    def background_children_running(self) -> bool:
        return self._background_owner.running or bool(self._background_child_cancellers)

    def _publish_background_event(self, event: StreamEvent) -> None:
        if self._background_event_sink is not None:
            self._background_event_sink(event)

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, TaskResult],
    ) -> asyncio.Task[TaskResult]:
        task = asyncio.create_task(coroutine)
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

    def _child_result_payload(
        self,
        tool_call_id: str,
        content: str,
        *,
        state: TerminalState | None = None,
        error: bool | None = None,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
        status: str | None = None,
        description: str | None = None,
        depth: int | None = None,
        stats: dict[str, object] | None = None,
        include_stats: bool = True,
        canceled: bool = False,
        notice: str | None = None,
        notice_items: Sequence[str] | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, object]:
        child_store = self._agent_child_stores.get(tool_call_id)
        path = (
            child_session_path
            if child_session_path is not None
            else str(child_store.session_dir)
            if child_store is not None
            else ""
        )
        turns = (
            turns_used
            if turns_used is not None
            else self._agent_child_turns.get(tool_call_id, 0)
        )
        if child_instance_id is None and child_store is not None:
            child_instance_id = child_store.agent_handle()
        result_status = (
            "running"
            if status == "running"
            else state
            or terminal_state(
                error=error is True,
                canceled=canceled,
                status=status,
            )
        )
        return agent_result(
            content,
            tool_call_id=tool_call_id,
            error=result_status == "failed",
            turns_used=turns,
            child_session_path=path,
            agent_type=agent_type if isinstance(agent_type, str) else None,
            status=status if status == "running" else None,
            child_instance_id=child_instance_id,
            description=description,
            depth=depth,
            stats=stats,
            include_stats=include_stats,
            canceled=result_status == "canceled",
            notice=notice,
            notice_items=notice_items,
            max_bytes=(
                max_bytes
                if max_bytes is not None
                else getattr(
                    getattr(self, "tool_registry", None),
                    "max_output_chars",
                    MAX_AGENT_RESULT_BYTES,
                )
            ),
        )

    def _canceled_agent_result(
        self,
        tool_call_id: str,
        *,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
    ) -> ToolResult:
        return validated_tool_result(
            self._child_result_payload(
                tool_call_id,
                "tool execution canceled",
                state="canceled",
                child_session_path=child_session_path,
                turns_used=turns_used,
                agent_type=agent_type,
                child_instance_id=child_instance_id,
            ),
            tool_call_id,
        )

    async def _run_agent_tool(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal,
        publisher: ToolStreamPublisher | None,
        execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, object]:
        return await run_agent_tool(
            self,
            tool_call,
            arguments,
            abort_signal,
            publisher,
            validate_result=validated_tool_result,
            error_message=lambda exc: _error_info(exc).message,
            execution_context=execution_context,
        )

    def prepare_resume_pending_tool(self, request_id: str) -> bool:
        """Reserve the abort generation before resuming an approved tool."""

        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return False
        if self._existing_tool_result(state[0].id) is not None:
            return False
        self.tool_registry.start_batch()
        return True

    def run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._run_turn(
            user_text,
            user_message=user_message,
            persist_user_message=persist_user_message,
            abort_signal=abort_signal,
        )

    async def close(self, *, cancel_background: bool = True) -> tuple[str, ...]:
        """Close session-owned transports and background processes.

        Returns the ids of background tasks killed by this close so a completing
        child can report them to its parent.
        """

        self._closed = True
        killed: tuple[str, ...] = ()
        if self.agent_depth == 0:
            self._background_owner.set_wake_callback(None)
        try:
            if cancel_background and self.agent_depth == 0:
                self.tool_registry.background_tasks.begin_shutdown()
                self._background_owner.cancel_all()
            elif cancel_background:
                for cancel in tuple(self._background_child_cancellers.values()):
                    cancel()
            watchers = tuple(self._background_child_watchers.values())
            if cancel_background and watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
            if cancel_background and self.agent_depth == 0:
                await self._background_owner.wait()
            tracked_tasks = tuple(
                task
                for task in self._tracked_tasks
                if cancel_background
                or task not in self._background_child_watchers.values()
            )
            for task in tracked_tasks:
                task.cancel()
            await asyncio.gather(*tracked_tasks, return_exceptions=True)
            if self.hooks is not None:
                self.hooks.stop()
                await self.hooks.close()
            if self._mcp_mount is not None:
                await self._mcp_mount.close()
                self._mcp_mount = None
            if self._mcp_mount_task is not None and not self._mcp_mount_task.done():
                self._mcp_mount_task.cancel()
                await asyncio.gather(self._mcp_mount_task, return_exceptions=True)
        finally:
            try:
                killed = await self.tool_registry.close()
            finally:
                if self.agent_depth == 0:
                    self._background_owner.store_leases.close()
        return killed

    def session_start(self) -> None:
        if self.hooks is not None:
            self.hooks.session_start()

    async def activate(self) -> None:
        """Run frontend startup hooks after the frontend installs its sinks."""

        if self._activated:
            return
        self._activated = True
        self.session_start()
        # Frontends can render immediately while trusted, enabled MCP servers
        # connect in the background. Operations that require MCP await this task.
        if not self._mcp_mount_attempted and self._mcp_mount_task is None:
            self._mcp_mount_task = asyncio.create_task(self._mount_mcp_servers())

    async def resume_pending_tool(
        self,
        request_id: str,
        *,
        prepared: bool = False,
        event_sink: Callable[[StreamEvent], None] | None = None,
    ) -> ToolResult | None:
        """Finish a durable approval request before starting another turn."""

        await self._ensure_mcp_servers()
        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        if not self.plan_mode_allows(tool_call.name):
            result = ToolResult(
                tool_call.id,
                f"tool execution denied in plan mode: {tool_call.name} is not allowed",
                is_error=True,
            )
            return self._finalize_tool_results([tool_call], [result])[0]
        if not prepared:
            self.tool_registry.start_batch()
        abort_signal = self.tool_registry.abort_signal

        def lifecycle(kind: str) -> None:
            if event_sink is None:
                return
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is not None:
                event_sink(StreamEvent(event_type, tool_call=tool_call))

        try:
            result = await self.tool_registry.execute(
                tool_call,
                abort_signal=abort_signal,
                _scope_signal=abort_signal,
                _lifecycle_sink=lifecycle,
            )
        except asyncio.CancelledError:
            result = self.finalize_canceled(request_id)
            if event_sink is not None and result is not None:
                event_sink(
                    StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                )
            raise
        except Exception as exc:  # noqa: BLE001 - report execution failures
            result = ToolResult(tool_call.id, str(exc), is_error=True)
        result = validated_tool_result(result, tool_call.id)
        if result.is_canceled:
            result = self.finalize_canceled(request_id)
        else:
            result = self._finalize_tool_results([tool_call], [result])[0]
        if event_sink is not None and result is not None:
            event_sink(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=tool_call,
                    tool_result=result,
                )
            )
        return result

    def finalize_canceled(self, request_id: str) -> ToolResult | None:
        """Persist one canceled result for a durable approval request."""

        state = self.store.approval_states().get(request_id)
        if state is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        return self._finalize_tool_results([tool_call], [None])[0]

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None:
        return self.store.tool_result(tool_call_id)

    async def _run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
        system_message: Message | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._turn_active = True
        stream = self._run_turn_impl(
            user_text,
            user_message=user_message,
            persist_user_message=persist_user_message,
            abort_signal=abort_signal,
            system_message=system_message,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await close_completion(stream)
            self._turn_active = False
            if self.store.agent_notifications():
                self.notify_background_persisted()

    async def _run_turn_impl(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
        system_message: Message | None = None,
    ) -> AsyncIterator[StreamEvent]:
        notification_turn = system_message is not None
        if system_message is not None and system_message.role is not MessageRole.SYSTEM:
            raise ValueError("system_message must have the system role")
        if self.hooks is not None and system_message is None:
            self.hooks.user_prompt_submit(user_text)
        if system_message is not None:
            await self._append_turn_message(system_message)
        elif user_message is None:
            user_message = Message(MessageRole.USER, [TextContent(user_text)])
        elif user_message.role is not MessageRole.USER:
            raise ValueError("user_message must have the user role")
        if system_message is None:
            if persist_user_message:
                await self._append_turn_message(user_message)
            elif user_message not in self.store.messages():
                raise ValueError("cannot reuse a user message that is not persisted")
        setup_error: ErrorInfo | None = None
        try:
            await self._ensure_mcp_servers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report setup failures
            setup_error = _error_info(exc)
        async for event in self.drain_notification_batch(
            message_persisted=system_message is not None, message=system_message
        ):
            yield event
        if setup_error is not None:
            self._persist_partial_with_cancelled_tools([], None, failure=setup_error)
            yield StreamEvent(StreamEventType.AGENT_START)
            yield StreamEvent(StreamEventType.ERROR, error=setup_error)
            yield StreamEvent(StreamEventType.AGENT_END)
            return
        yield StreamEvent(StreamEventType.AGENT_START)
        turn_number = 0
        retrying_context = False
        retrying_provider = False
        provider_retry_budget: ProviderRetryBudget
        self._turn_provider_retry_records = []
        nudged_empty_turn = False
        nudge_turn_pending = False
        consuming_notifications = False
        iteration_consuming_notifications = False
        while (
            self.max_turns is None
            or turn_number < self.max_turns
            or self.has_pending_notification_turn(notification_turn)
            or retrying_context
            or retrying_provider
            # A nudge is persisted as a user message, so it must always get
            # exactly one following model call. This recovery call counts as
            # one additional turn, bounded to at most max_turns + 1 calls.
            or nudge_turn_pending
        ):
            if not retrying_context and not retrying_provider:
                provider_retry_budget = ProviderRetryBudget()
                self._turn_provider_retry_records = provider_retry_budget.records
                iteration_consuming_notifications = consuming_notifications
                consuming_notifications = False
                turn_number += 1
                nudge_turn_pending = False
                self._turn_stop_reason = None
                self._turn_output_tokens = None
                while self._steering_queue:
                    steering = self._steering_queue.popleft()
                    await self._append_turn_message(steering)
                async for event in self.drain_notification_batch():
                    yield event
                self.tool_registry.start_batch()
                yield StreamEvent(
                    StreamEventType.TURN_START,
                    data={"turn": turn_number},
                )
            turn_abort_signal = abort_signal or self.tool_registry.abort_signal
            partial_blocks: list[ContentBlock] = []
            assistant_message: Message | None = None
            completion: AsyncIterator[StreamEvent] | None = None
            completion_succeeded = False
            provider_error: ErrorInfo | None = None
            provider_error_source: BaseException | ErrorInfo | None = None
            provider_error_data: dict[str, Any] = {}
            provider_retry_usage: dict[str, Any] | None = None
            attempt_state = ProviderAttemptState()
            try:
                if retrying_context or self.context_assembler.needs_compaction():
                    yield StreamEvent(
                        StreamEventType.COMPACTION_START,
                        data={"turn": turn_number},
                    )
                context_messages = await self.context_assembler.assemble(
                    backend=self.backend, force=retrying_context
                )
                if self._plan_mode:
                    context_messages = plan_mode_messages(context_messages)
                context = self.context_assembler.last_context
                if context is not None and context.compacted:
                    yield StreamEvent(
                        StreamEventType.COMPACTION_END,
                        data={
                            "turn": turn_number,
                            "token_count": context.token_count,
                        },
                    )
                active_tools = self._active_tool_schemas()
                cache_trace = (
                    self._cache_trace.start(
                        context_messages,
                        active_tools,
                        self.backend,
                        turn_number,
                        self.plan_mode,
                        bool(context and context.compacted),
                    )
                    if self._cache_trace is not None
                    else None
                )
                if not provider_retry_budget.start_attempt("loop"):
                    raise RuntimeError("provider retry budget exhausted")
                completion = apply_retry_budget(
                    self.backend.complete(context_messages, active_tools),
                    provider_retry_budget,
                )
                async for event in completion:
                    attempt_state.observe(event)
                    self.context_assembler.observe_event(event)
                    if cache_trace is not None:
                        assert self._cache_trace is not None
                        self._cache_trace.observe(cache_trace, event)
                    if event.type is StreamEventType.ERROR:
                        provider_error = (
                            replace(event.error, provider_error=True)
                            if isinstance(event.error, ErrorInfo)
                            else ErrorInfo(
                                "backend_error",
                                "provider emitted an invalid error event",
                            )
                        )
                        provider_error_source = event.error or provider_error
                        provider_error_data = dict(event.data)
                        break
                    if (
                        event.type is StreamEventType.ASSISTANT_RESET
                        and not completion_succeeded
                    ):
                        partial_blocks = []
                        assistant_message = None
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial_blocks.append(event.content)
                        if event.delta is not None:
                            partial_blocks.append(TextContent(event.delta))
                    if (
                        event.message is not None
                        and event.type is StreamEventType.MESSAGE_END
                    ):
                        assistant_message = event.message
                    if event.type is StreamEventType.MESSAGE_END:
                        usage = event.data.get("usage")
                        if isinstance(usage, Mapping):
                            provider_retry_usage = dict(usage)
                        reason, tokens = read_turn_metadata(event.data)
                        if reason is not None:
                            self._turn_stop_reason = reason
                        if tokens is not None:
                            self._turn_output_tokens = tokens
                    if event.type is StreamEventType.MESSAGE_END and not event.data.get(
                        "truncated"
                    ):
                        completion_succeeded = True
                    yield event
                if provider_error is None and not completion_succeeded:
                    provider_error = ErrorInfo(
                        "stream_error",
                        "provider stream ended before completion",
                    )
            except asyncio.CancelledError:
                await close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except GeneratorExit:
                await close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except Exception as exc:
                await close_completion(completion)
                if task_is_cancelling():
                    self._persist_partial_for_control(partial_blocks, assistant_message)
                    raise asyncio.CancelledError() from exc
                error = _error_info(exc, provider_error=True)
                provider_error = error
                provider_error_source = exc
                completion = None
            cleanup_error = await close_completion(completion)
            if task_is_cancelling():
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise asyncio.CancelledError()
            if cleanup_error is not None and provider_error is None:
                provider_error = _error_info(cleanup_error, provider_error=True)
                provider_error_source = cleanup_error
            if provider_error is not None and can_retry_context(
                provider_error, retrying_context, partial_blocks, assistant_message
            ):
                retrying_context = True
                yield StreamEvent(
                    StreamEventType.RETRY,
                    data={"text": "context limit reached; compacting and retrying"},
                )
                continue
            if provider_error is not None:
                source = provider_error_source or provider_error
                retry_usage = getattr(source, "retry_usage", None)
                if provider_retry_usage is None and isinstance(retry_usage, Mapping):
                    provider_retry_usage = dict(retry_usage)
                plan = (
                    provider_retry_budget.plan(
                        source,
                        owner="loop",
                        event_data=provider_error_data,
                    )
                    if attempt_state.can_retry and not turn_abort_signal.is_set()
                    else None
                )
                if plan is not None:
                    notice = provider_retry_notice(plan)
                    if provider_retry_usage is not None:
                        notice = replace(
                            notice,
                            data={**notice.data, "usage": provider_retry_usage},
                        )
                    yield notice
                    if await wait_for_provider_retry(plan.delay, turn_abort_signal):
                        provider_retry_budget.record_retry(plan)
                        yield assistant_reset_event()
                        partial_blocks = []
                        assistant_message = None
                        retrying_provider = True
                        continue
                if provider_retry_budget.exhausted:
                    original = provider_retry_budget.original_error
                    if isinstance(original, ErrorInfo):
                        provider_error = original
                    elif isinstance(original, BaseException):
                        provider_error = _error_info(original, provider_error=True)
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message, failure=provider_error
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=provider_error,
                    data=provider_error_data,
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            retrying_context = False
            retrying_provider = False
            if completion_succeeded and self.on_completion_success is not None:
                self.on_completion_success()
            if assistant_message is None and partial_blocks:
                assistant_message = Message(MessageRole.ASSISTANT, partial_blocks)
            if assistant_message is None:
                assistant_message = Message(MessageRole.ASSISTANT)
            assistant_message = self._annotate_current_turn(assistant_message)
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            _validate_unique_tool_call_ids(calls)
            approval_requests: list[
                tuple[str, ToolCall] | tuple[str, ToolCall, dict[str, object]]
            ] = []
            for tool_call in calls:
                if not self.plan_mode_allows(tool_call.name):
                    continue
                request = self.tool_registry.prepare_approval(tool_call)
                if request is not None:
                    display = request.audit_display()
                    approval_requests.append(
                        (request.request_id, request.tool_call, display)
                        if display
                        else (request.request_id, request.tool_call)
                    )
            await self._append_turn_message_with_approvals(
                durable_message(assistant_message),
                approval_requests,
            )
            attempt_state.persisted = True
            self._turn_provider_retry_records = []
            if not calls:
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                should_nudge = (
                    not iteration_consuming_notifications
                    and should_nudge_empty_turn(
                        assistant_message,
                        stop_reason=self._turn_stop_reason,
                        notification_turn=notification_turn,
                        already_nudged=nudged_empty_turn,
                    )
                )
                if should_nudge:
                    nudged_empty_turn = True
                    nudge_turn_pending = True
                    await self._append_turn_message(build_nudge_message())
                if self.has_pending_notification_turn(notification_turn):
                    # This continuation consumes the pending notification. It may
                    # share the one max_turns + 1 recovery call with a nudge, but
                    # notification continuations otherwise retain their existing
                    # phase-1 behavior and are never themselves nudge-eligible.
                    consuming_notifications = True
                    continue
                if should_nudge:
                    continue
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            dispatch = dispatch_tool_calls(
                self,
                calls,
                validated_tool_result,
                abort_signal=turn_abort_signal,
            )
            try:
                async for event in dispatch:
                    yield event
            except (asyncio.CancelledError, GeneratorExit):
                # The assistant message is durable before dispatch starts. If
                # cancellation lands in that small gap, the dispatcher's own
                # cleanup has not run yet, leaving the provider with an
                # unpaired function call on resume. Finalization is idempotent,
                # so this also covers cancellation after dispatch has started.
                for call in calls:
                    self.tool_registry.abort_approval(call)
                self._finalize_tool_results(calls, [None] * len(calls))
                raise
            finally:
                await dispatch.aclose()
            yield StreamEvent(
                StreamEventType.TURN_END,
                message=assistant_message,
                data={"turn": turn_number, "tool_calls": len(calls)},
            )
        yield StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("max_turns", f"maximum turns reached: {self.max_turns}"),
        )
        yield StreamEvent(StreamEventType.AGENT_END)

    def _finalize_tool_results(
        self,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]:
        return finalize_agent_results(self, calls, slots)

    def _annotate_current_turn(self, message: Message) -> Message:
        """Apply provider metadata before any assistant message is persisted."""

        annotated = annotate_turn_metadata(
            message,
            stop_reason=self._turn_stop_reason,
            output_tokens=self._turn_output_tokens,
        )
        if not self._turn_provider_retry_records:
            return annotated
        metadata = dict(annotated.metadata)
        metadata["provider_retries"] = [
            dict(record) for record in self._turn_provider_retry_records
        ]
        return Message(
            annotated.role,
            annotated.content,
            tool_result=annotated.tool_result,
            metadata=metadata,
        )

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        if assistant_message is None:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent)
                or not block.text
                or block.signature
            ]
            if not durable_blocks and failure is None:
                return
            assistant_message = Message(MessageRole.ASSISTANT, durable_blocks)
        assistant_message = self._annotate_current_turn(assistant_message)
        if failure is not None:
            metadata = dict(assistant_message.metadata)
            metadata[FAILED_TURN_MARKER] = True
            metadata[FAILED_TURN_ERROR] = failure.to_dict()
            metadata["fd_diagnostics"] = fd_diagnostics()
            assistant_message = Message(
                assistant_message.role,
                assistant_message.content,
                tool_result=assistant_message.tool_result,
                metadata=metadata,
            )
        self.store.append_message(durable_message(assistant_message))

    def _persist_partial_for_control(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        try:
            self._persist_partial_with_cancelled_tools(
                partial_blocks, assistant_message
            )
        except Exception as exc:  # noqa: BLE001 - warn when persistence fails
            try:
                warnings.warn(
                    f"failed to persist partial state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except BaseException:  # noqa: BLE001, S110 - warning failure is ignored
                pass

    def _persist_partial_with_cancelled_tools(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message, failure=failure)
        blocks = (
            assistant_message.content
            if assistant_message is not None
            else partial_blocks
        )
        calls = [
            block.tool_call for block in blocks if isinstance(block, ToolUseContent)
        ]
        self._finalize_tool_results(calls, [None] * len(calls))
