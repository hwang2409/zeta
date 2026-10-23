"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from ...agent.background import BackgroundAgentOwner
from ...agent.budget import MAX_AGENT_DEPTH
from ...agent.durable import durable_message
from ...agent.notifications import AgentNotificationMixin
from ...agent.pending_delivery import (
    SHUTDOWN_DELIVERY_TIMEOUT_SECONDS,
    NotificationInput,
    ProducerResult,
    StepDeliveryFlow,
    delivery_disposition,
)
from ...agent.plan_mode import (
    PLAN_MODE_POLICY,
    plan_mode_messages,
    plan_mode_tool_schemas,
)
from ...agent.receipt import finalize_agent_results
from ...agent.recovery_lease import recover_agent_children_off_loop
from ...agent.result_payload import ChildResultPayloadMixin
from ...agent.runner import run_agent_tool
from ...agent.tool_results import validated_tool_result as _validated_tool_result
from ...core.abort import AbortSignal as ToolAbortSignal
from ...core.approval import ApprovalPolicy
from ...core.context import BudgetExceeded, ContextAssembler
from ...core.hooks import HookManager
from ...core.slash import effective_budget_for_model
from ...core.store import ConversationStore
from ...core.store._approval_display import ApprovalAuditRequest
from ...core.tool_dispatch import dispatch_tool_calls
from ...mcp import (
    MCPMount,
)
from ...memory.auto import AutoMemoryReconciler
from ...prompts import load_identity
from ...protocol.types import (
    ASSISTANT_RESPONSE_ABORTED,
    ASSISTANT_RESPONSE_COMPLETED,
    ASSISTANT_RESPONSE_FAILED,
    ASSISTANT_RESPONSE_STATE,
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    CompletionBackend,
    ContentBlock,
    ContextWindowBackend,
    ErrorInfo,
    Message,
    MessageOrigin,
    MessageRole,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    user_message_for_turn,
)
from ...providers.retry_policy import ProviderRetryBudget, apply_retry_budget
from ...providers.stream_diagnostics import fd_diagnostics
from ...runtime.tool_setup import select_tool_registry
from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog
from ...tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from ...tools._shared.process import TaskExitInput
from ...tools.registry import (
    ToolExecutionContext,
    _validate_unique_tool_call_ids,
)
from ..backend_state import (
    PreparedBackendState,
    apply_backend_state,
    prepare_backend_state,
)
from ._completion import (
    ProviderAttemptState,
    _error_info,
    assistant_reset_event,
    can_retry_context,
    close_completion,
    ensure_context_reduced,
    observed_completion,
    provider_prompt_tokens,
    provider_retry_notice,
    start_provider_attempt,
    subscription_limit_failure,
    task_is_cancelling,
    wait_for_provider_retry,
)
from ._store_writes import StoreWriteMixin
from .empty_turn import (
    annotate_turn_metadata,
    build_nudge_message,
    read_turn_metadata,
    should_nudge_empty_turn,
)
from .mcp_session import MCPSession
from .plan_mode_runtime import PlanModeRuntimeMixin
from .project_inbox import ProjectInboxNotificationMixin
from .skill_refresh import RuntimePrompt, SkillRefreshMixin
from .tool_schema import canonical_tool_schemas


def _sanitize_account_message(message: Message) -> Message:
    content = [
        ThinkingContent(block.text) if isinstance(block, ThinkingContent) else block
        for block in message.content
        if not isinstance(block, RedactedThinkingContent)
    ]
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=message.metadata,
    )


TaskResult = TypeVar("TaskResult")


@dataclass(frozen=True, slots=True)
class _ToollessTransition:
    continue_turn: bool
    consuming_notifications: bool
    nudged_empty_turn: bool
    nudge_turn_pending: bool


class AgentLoop(
    ChildResultPayloadMixin,
    PlanModeRuntimeMixin,
    SkillRefreshMixin,
    StoreWriteMixin,
    AgentNotificationMixin,
    ProjectInboxNotificationMixin,
    MCPSession,
):
    post_stream_provider_retry = True

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
        runtime_prompt: RuntimePrompt | None = None,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        on_completion_success: Callable[[], Awaitable[None] | None] | None = None,
        on_plan_mode_change: Callable[[bool], Awaitable[None] | None] | None = None,
        hooks: HookManager | None = None,
        skip_mcp_mount: bool = False,
        agent_depth: int = 0,
        agent_instance_id: str | None = None,
        root_project_id: str | None = None,
        parent_session_id: str | None = None,
        root_session_dir: Any = None,
        project_registry: Any = None,
        background_owner: BackgroundAgentOwner | None = None,
        usage_sink: Callable[[Mapping[str, Any]], None] | None = None,
        skill_home: Path | None = None,
        skill_project_dir: Path | None = None,
        freeze_skill_catalog: bool = False,
    ) -> None:
        if type(agent_depth) is not int or not 0 <= agent_depth <= MAX_AGENT_DEPTH:
            raise ValueError(f"agent depth must be between 0 and {MAX_AGENT_DEPTH}")
        self.backend = backend
        self.store = store
        self.store.recover_client_deliveries()
        self.agent_depth = agent_depth
        self.agent_instance_id = agent_instance_id
        self.root_project_id = root_project_id
        self.parent_session_id = parent_session_id
        # Directory of the ROOT session that owns the durable child-link index;
        # threaded down every loop so nested children publish their lineage
        # intent into a single flat directory the root can reconcile.
        self.root_session_dir = (
            root_session_dir if root_session_dir is not None else store.session_dir
        )
        self.project_registry = project_registry
        self._configure_transcript_index()
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
        self._children_recovered = False
        self._closed = False
        self._turn_active = False
        self._delivery_wake_pending = False
        self.explicit_finish = False
        self._explicit_finish_report: dict[str, object] | None = None
        self._delivering_prompt_id: str | None = None
        # Partial assistant persistence uses these values so provider metadata is
        # retained even when the stream later fails or is cancelled.
        self._turn_stop_reason: str | None = None
        self._turn_output_tokens: int | None = None
        self.memory_reconciler: AutoMemoryReconciler | None = None
        self._turn_provider_retry_records: list[dict[str, object]] = []
        self.tool_registry = select_tool_registry(
            store,
            tools=tools,
            registry=registry,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
            tool_schemas=tool_schemas,
            project_id=root_project_id,
            project_registry=project_registry,
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
        self._pending_task_recovery = (
            self.tool_registry.background_tasks.set_notification_sink(
                self._produce_task_exit
            )
        )
        self.agent_catalog = self.tool_registry.agent_catalog
        if system_prompt is not None and runtime_prompt is not None:
            raise ValueError("pass only one system prompt")
        if runtime_prompt is not None:
            system_prompt = runtime_prompt.system_prompt
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
            system_prompt=system_prompt,
            backend=backend,
            on_completion_success=on_completion_success,
            usage_sink=usage_sink,
        )
        self._initialize_step_delivery(store)
        self._configure_skill_refresh(
            catalog=self.tool_registry.skill_catalog,
            home=skill_home,
            project_dir=skill_project_dir,
            frozen=freeze_skill_catalog,
            runtime_prompt=runtime_prompt,
        )
        self.on_completion_success = on_completion_success
        self._on_plan_mode_change = on_plan_mode_change
        self.hooks = hooks
        if self.hooks is not None:
            self.hooks.bind_session(store.session_id)
            if self.tool_registry.pre_execute_hook is None:
                self.tool_registry.set_pre_execute_hook(self.hooks.pre_tool)
        self._plan_mode = False
        self._plan_mode_policy = PLAN_MODE_POLICY
        self._plan_mode_prior_prompt: str | Message | None = None
        if "agent" in self.tool_registry.definitions_by_name:
            self.tool_registry.set_agent_runner(self._run_agent_tool)

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
            schemas = plan_mode_tool_schemas(
                self.backend, schemas, policy=self._plan_mode_policy
            )
        return canonical_tool_schemas(schemas)

    def _account_safe_context(self, messages: list[Message]) -> list[Message]:
        account_context = getattr(self, "account_context", None)
        provider = (
            account_context.active_provider if account_context is not None else None
        )
        sequence = account_context.switch_sequence if account_context is not None else 0
        if provider is None or sequence == 0:
            return messages
        return [
            message
            if (
                message.metadata.get("account_provider") == provider
                and message.metadata.get("account_switch_sequence") == sequence
            )
            else _sanitize_account_message(message)
            for message in messages
        ]

    def prepare_backend_state(self, backend: CompletionBackend) -> PreparedBackendState:
        return prepare_backend_state(self, backend)

    def current_backend_state(self) -> PreparedBackendState:
        return PreparedBackendState(self.backend, self.context_assembler.token_budget)

    def apply_backend_state(self, state: PreparedBackendState) -> None:
        apply_backend_state(self, state)

    def set_token_budget(self, token_budget: int) -> None:
        """Align compaction and provider-side context budgets."""
        provider = getattr(self.backend, "provider", None)
        model = getattr(self.backend, "model", None)
        if isinstance(provider, str) and isinstance(model, str):
            token_budget = effective_budget_for_model(provider, model, token_budget)
        self.context_assembler.update_token_budget(token_budget)
        if self.step_delivery is not None:
            self.step_delivery.capacity_changed()
        if isinstance(self.backend, ContextWindowBackend):
            self.backend.set_token_budget(token_budget)

    def abort(
        self,
        *,
        foreground_only: bool = False,
        steering_drop_reason: Literal["abort", "disconnect"] | None = "abort",
    ) -> None:
        """Signal active tools; optionally preserve background work and steering."""
        self.tool_registry.abort()
        if not foreground_only:
            self._background_owner.cancel_all()
            if steering_drop_reason is not None:
                self.store.drop_client_steering(steering_drop_reason)

    def steer(self, message: Message) -> None:
        """Queue a user message for injection at the next tool boundary.
        The running ``_run_turn`` drains this queue before the next provider
        call, so the message never lands between a tool_call and its
        tool_result. Callers must pass a durable USER-role message.
        """
        self.store.queue_client_steering(message)

    @property
    def has_pending_steering(self) -> bool:
        return self.store.has_pending_client_steering

    def clear_pending_steering(
        self,
        reason: Literal["clear", "turn_end"] = "clear",
    ) -> int:
        return self.store.drop_client_steering(reason)

    def drop_pending_steering(
        self,
        reason: Literal["abort", "disconnect", "turn_end", "failed"],
    ) -> int:
        return self.store.drop_client_steering(reason)

    def set_background_event_sink(
        self, sink: Callable[[StreamEvent], None] | None
    ) -> None:
        """Set the sink for progress from children that outlive their turn."""
        self._background_event_sink = sink

    @property
    def background_children_running(self) -> bool:
        return self._background_owner.running or bool(self._background_child_cancellers)

    def _publish_background_event(self, event: StreamEvent) -> None:
        if self._background_event_sink is not None:
            self._background_event_sink(event)

    def _create_task[TaskResult](
        self,
        coroutine: Coroutine[Any, Any, TaskResult],
    ) -> asyncio.Task[TaskResult]:
        task = asyncio.create_task(coroutine)
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

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
            validate_result=_validated_tool_result,
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

    async def _close_delivery_resources(self) -> tuple[str, ...]:
        killed: tuple[str, ...] = ()
        try:
            killed = await self.tool_registry.close()
            late_tasks = tuple(self._tracked_tasks)
            if late_tasks:
                done, pending = await asyncio.wait(
                    late_tasks, timeout=SHUTDOWN_DELIVERY_TIMEOUT_SECONDS
                )
                if pending:
                    for task in pending:
                        task.cancel()
                    raise RuntimeError("shutdown delivery admission timed out")
                for task in done:
                    task.result()
        finally:
            if self.agent_depth == 0:
                await self._background_owner.wait_for_store_closes()
                if self.step_delivery is not None:
                    await self._background_owner.conversation_channel.unregister_delivery_flow(
                        None
                    )
                    self.step_delivery = None
                await self.store._to_thread_store(
                    self._background_owner.store_leases.close
                )
        return killed

    async def close(self, *, cancel_background: bool = True) -> tuple[str, ...]:
        """Close session-owned transports and background processes.

        Returns the ids of background tasks killed by this close so a completing
        child can report them to its parent.
        """

        self._closed = True
        killed: tuple[str, ...] = ()
        if self.memory_reconciler is not None:
            await self.memory_reconciler.close()
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
            if self._mcp_mount_task is not None and not self._mcp_mount_task.done():
                self._mcp_mount_task.cancel()
                await asyncio.gather(self._mcp_mount_task, return_exceptions=True)
            if self._mcp_mount is not None:
                await self._mcp_mount.close()
                self._mcp_mount = None
        finally:
            killed = await self._close_delivery_resources()
        return killed

    def session_start(self) -> None:
        if self.hooks is not None:
            self.hooks.session_start()

    async def _produce_task_exit(self, item: TaskExitInput) -> ProducerResult:
        data: dict[str, object] = {
            "task_id": item.task_id,
            "headline": item.command,
            "exit_code": item.exit_code,
            "output_tail": item.output_tail,
            "background_owner": item.owner,
            "background_phase": item.phase,
        }
        if item.log_path is not None:
            data["log_path"] = item.log_path
        if item.note is not None:
            data["note"] = item.note
        step_delivery = await self._ensure_step_delivery()
        return await step_delivery.produce(
            NotificationInput("task_exit", data, dedupe_key=item.task_id)
        )

    async def activate(self) -> None:
        """Run frontend startup hooks after the frontend installs its sinks."""

        if self._activated:
            return
        self._activated = True
        await self._ensure_step_delivery()
        for item in self._pending_task_recovery:
            await self._produce_task_exit(item)
        self._pending_task_recovery = ()
        if not self._children_recovered:
            await recover_agent_children_off_loop(self)
            self._children_recovered = True
        self.session_start()
        if self.memory_reconciler is not None:
            self.memory_reconciler.activity()
        await self._activate_project_inbox()
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
        if not self.plan_mode_allows(tool_call):
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
        result = _validated_tool_result(result, tool_call.id)
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

    async def _toolless_transition(
        self,
        step_delivery: StepDeliveryFlow,
        assistant_message: Message,
        *,
        iteration_consuming_notifications: bool,
        notification_turn: bool,
        nudged_empty_turn: bool,
    ) -> _ToollessTransition:
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
            await step_delivery.admit_terminal("nudge")
            await self._append_turn_message(build_nudge_message())
        elif not step_delivery.has_pending_edge and self.agent_instance_id is None:
            await step_delivery.admit_terminal("foreground_end")
        consume_pending = step_delivery.has_pending_edge and not notification_turn
        return _ToollessTransition(
            continue_turn=consume_pending or should_nudge,
            consuming_notifications=consume_pending,
            nudged_empty_turn=nudged_empty_turn or should_nudge,
            nudge_turn_pending=should_nudge,
        )

    async def _run_turn_impl(
        self,
        user_text: str,
        *,
        origin: MessageOrigin,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
        system_message: Message | None = None,
        on_persisted: Callable[[], None] | None = None,
        notification_turn: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        step_delivery = await self._ensure_step_delivery()
        await self.refresh_skills()
        if system_message is not None and system_message.role is not MessageRole.SYSTEM:
            raise ValueError("system_message must have the system role")
        await self._check_project_inbox()
        if self.hooks is not None and system_message is None and not notification_turn:
            self.hooks.user_prompt_submit(user_text)
        if system_message is not None and not notification_turn:
            await self._append_turn_message(system_message, on_persisted=on_persisted)
        elif system_message is None and not notification_turn and persist_user_message:
            user_message = user_message_for_turn(
                user_text,
                origin=origin,
                message=user_message,
                reuse_persisted=False,
            )
        elif user_message is not None and user_message not in self.store.messages():
            raise ValueError("cannot reuse a user message that is not persisted")
        if system_message is None and persist_user_message:
            await self._append_turn_message(user_message)
        await self._flush_skill_refresh_notices()
        setup_error: ErrorInfo | None = None
        try:
            await self._ensure_mcp_servers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report setup failures
            setup_error = _error_info(exc)
        inbox_status_message = self._project_inbox_status_message()
        if inbox_status_message is not None:
            scanner = self._inbox_scanner
            statuses = tuple(
                (item["message_id"], item["status"])
                for item in inbox_status_message.metadata["sent_statuses"]
            )
            await self._append_turn_message(
                inbox_status_message,
                on_persisted=(
                    lambda: (
                        scanner.mark_reported(iter(statuses))
                        if scanner is not None
                        else None
                    )
                ),
            )
        if setup_error is not None:
            self._persist_partial_with_cancelled_tools([], None, failure=setup_error)
            yield StreamEvent(StreamEventType.AGENT_START)
            yield StreamEvent(StreamEventType.ERROR, error=setup_error)
            if self.agent_instance_id is None:
                await step_delivery.admit_terminal("foreground_end")
            yield StreamEvent(StreamEventType.AGENT_END)
            return
        yield StreamEvent(StreamEventType.AGENT_START)
        turn_number = 0
        retrying_context = False
        overflow_context_tokens: int | None = None
        retrying_provider = False
        provider_retry_budget: ProviderRetryBudget | None = None
        self._turn_provider_retry_records = []
        nudged_empty_turn = False
        nudge_turn_pending = False
        consuming_notifications = False
        iteration_consuming_notifications = False
        while (
            self.max_turns is None
            or turn_number < self.max_turns
            or step_delivery.has_pending_edge
            or retrying_context
            or retrying_provider
            # A nudge is persisted as a user message, so it must always get
            # exactly one following model call. This recovery call counts as
            # one additional turn, bounded to at most max_turns + 1 calls.
            or nudge_turn_pending
        ):
            if not retrying_context and not retrying_provider:
                provider_retry_budget = None
                self._turn_provider_retry_records = []
                iteration_consuming_notifications = consuming_notifications
                consuming_notifications = False
                turn_number += 1
                nudge_turn_pending = False
                self._turn_stop_reason = None
                self._turn_output_tokens = None
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
                account_context = getattr(self, "account_context", None)
                if (
                    account_context is not None
                    and account_context.account_state_unknown
                ):
                    yield StreamEvent(StreamEventType.ACCOUNT_STATE_UNKNOWN)
                    await account_context.ensure_account_state(self)
                if retrying_context or self.context_assembler.needs_compaction():
                    yield StreamEvent(
                        StreamEventType.COMPACTION_START,
                        data={"turn": turn_number},
                    )
                try:
                    context_messages = await self.context_assembler.assemble(
                        backend=self.backend,
                        force=retrying_context,
                        emergency=retrying_context,
                    )
                except BudgetExceeded:
                    if not notification_turn:
                        raise
                    context_messages = await self.context_assembler.assemble(
                        backend=self.backend,
                        force=True,
                        emergency=True,
                    )
                if self._plan_mode:
                    context_messages = plan_mode_messages(context_messages)
                context_messages = self._account_safe_context(context_messages)
                context = self.context_assembler.last_context
                if retrying_context:
                    current_tokens = None if context is None else context.token_count
                    ensure_context_reduced(current_tokens, overflow_context_tokens)
                if context is not None and context.compacted:
                    yield StreamEvent(
                        StreamEventType.COMPACTION_END,
                        data={
                            "turn": turn_number,
                            "token_count": context.token_count,
                        },
                    )
                active_tools = self._active_tool_schemas()
                steering = self.store.pending_client_steering()
                notification_context = system_message if notification_turn else None
                prepared_delivery = await step_delivery.prepare_request(
                    "retry" if retrying_provider else "after_tools",
                    context_messages,
                    steering,
                    active_tools,
                    notification_context=notification_context,
                )
                if notification_context is not None and (
                    prepared_delivery.contains_turn_context
                ):
                    if on_persisted is not None:
                        on_persisted()
                    system_message = None
                    on_persisted = None
                self._delivery_wake_pending = False
                context_messages = list(prepared_delivery.messages)
                steering = prepared_delivery.steering
                self._delivering_prompt_id = prepared_delivery.prompt_id
                disposition = delivery_disposition(
                    step_delivery,
                    prepared_delivery,
                    notification_turn=notification_turn,
                    notice_sink=self._mcp_notice_sink,
                )
                if disposition == "deferred":
                    yield StreamEvent(StreamEventType.AGENT_END)
                    return
                if isinstance(disposition, ErrorInfo):
                    self._persist_partial_with_cancelled_tools(
                        [], None, failure=disposition
                    )
                    yield StreamEvent(
                        StreamEventType.ERROR, error=disposition
                    )
                    yield StreamEvent(StreamEventType.AGENT_END)
                    return
                provider_retry_budget = start_provider_attempt(provider_retry_budget)
                self._turn_provider_retry_records = provider_retry_budget.records
                admitted_request = prepared_delivery.admitted_request
                assert admitted_request is not None
                completion = apply_retry_budget(
                    self.context_assembler.dispatch_admitted_request(
                        self.backend, admitted_request
                    ),
                    provider_retry_budget,
                )
                has_active_delivery = prepared_delivery.has_active_delivery
                provider_stream = observed_completion(
                    completion,
                    self.store,
                    steering,
                    self._acknowledge_step_delivery if has_active_delivery else None,
                    self._delivery_publication_events if has_active_delivery else None,
                )
                async for event in provider_stream:
                    attempt_state.observe(event)
                    self.context_assembler.observe_event(event)
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
                context = self.context_assembler.last_context
                overflow_context_tokens = (
                    None if context is None else context.token_count
                )
                self.context_assembler.record_context_overflow(
                    provider_prompt_tokens(provider_error_source)
                )
                retrying_context = True
                yield StreamEvent(
                    StreamEventType.RETRY,
                    data={"text": "context limit reached; compacting and retrying"},
                )
                continue
            if provider_error is not None:
                source = provider_error_source or provider_error
                provider_error, limit_event = await subscription_limit_failure(
                    getattr(self, "account_context", None), provider_error, source
                )
                if limit_event is not None:
                    yield limit_event
                retry_usage = getattr(source, "retry_usage", None)
                if provider_retry_usage is None and isinstance(retry_usage, Mapping):
                    provider_retry_usage = dict(retry_usage)
                plan = attempt_state.retry_plan(
                    provider_retry_budget,
                    source,
                    reset_supported=self.post_stream_provider_retry,
                    turn_aborted=turn_abort_signal.is_set(),
                    event_data=provider_error_data,
                )
                if plan is not None:
                    notice = provider_retry_notice(plan)
                    if provider_retry_usage is not None:
                        notice = replace(
                            notice,
                            data={**notice.data, "usage": provider_retry_usage},
                        )
                    yield notice
                    if await wait_for_provider_retry(
                        plan.delay, turn_abort_signal, source
                    ):
                        provider_retry_budget.record_retry(plan)
                        yield assistant_reset_event()
                        partial_blocks = []
                        assistant_message = None
                        retrying_provider = True
                        continue
                if (
                    provider_retry_budget is not None
                    and provider_retry_budget.exhausted
                ):
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
                if self.agent_instance_id is None:
                    await step_delivery.admit_terminal("foreground_end")
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            retrying_context = False
            retrying_provider = False
            if completion_succeeded and self.on_completion_success is not None:
                completion_result = self.on_completion_success()
                if inspect.isawaitable(completion_result):
                    await completion_result
            if assistant_message is None and partial_blocks:
                assistant_message = Message(MessageRole.ASSISTANT, partial_blocks)
            if assistant_message is None:
                assistant_message = Message(MessageRole.ASSISTANT)
            assistant_message = self._annotate_current_turn(assistant_message)
            metadata = {
                **assistant_message.metadata,
                **self.context_assembler.completion_metadata(),
            }
            metadata[ASSISTANT_RESPONSE_STATE] = ASSISTANT_RESPONSE_COMPLETED
            assistant_message = Message(
                assistant_message.role,
                assistant_message.content,
                tool_result=assistant_message.tool_result,
                metadata=metadata,
            )
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            _validate_unique_tool_call_ids(calls)
            approval_requests: list[ApprovalAuditRequest] = []
            for tool_call in calls:
                if not self.plan_mode_allows(tool_call):
                    continue
                request = self.tool_registry.prepare_approval(tool_call)
                if request is not None:
                    approval_requests.append(request.audit_record())
            await self._append_turn_message_with_approvals(
                durable_message(assistant_message),
                approval_requests,
            )
            attempt_state.persisted = True
            self._turn_provider_retry_records = []
            if not calls:
                transition = await self._toolless_transition(
                    step_delivery,
                    assistant_message,
                    iteration_consuming_notifications=(
                        iteration_consuming_notifications
                    ),
                    notification_turn=notification_turn,
                    nudged_empty_turn=nudged_empty_turn,
                )
                nudged_empty_turn = transition.nudged_empty_turn
                nudge_turn_pending = transition.nudge_turn_pending
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                # Foreground turns consume a newly armed step before ending.
                # One idle epoch makes only its existing provider attempt.
                consuming_notifications = transition.consuming_notifications
                if transition.continue_turn:
                    continue
                self._schedule_transcript_index()
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            dispatch = dispatch_tool_calls(
                self,
                calls,
                _validated_tool_result,
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
                await self._background_owner.wait_for_store_closes()
            await self._check_project_inbox()
            yield StreamEvent(
                StreamEventType.TURN_END,
                message=assistant_message,
                data={"turn": turn_number, "tool_calls": len(calls)},
            )
        yield StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("max_turns", f"maximum turns reached: {self.max_turns}"),
        )
        if self.agent_instance_id is None:
            await step_delivery.admit_terminal("foreground_end")
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
        account_context = getattr(self, "account_context", None)
        if account_context is not None and account_context.active_provider is not None:
            metadata = dict(annotated.metadata)
            metadata["account_provider"] = account_context.active_provider
            metadata["account_switch_sequence"] = account_context.switch_sequence
            annotated = Message(
                annotated.role,
                annotated.content,
                tool_result=annotated.tool_result,
                metadata=metadata,
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
        metadata = dict(assistant_message.metadata)
        metadata[ASSISTANT_RESPONSE_STATE] = (
            ASSISTANT_RESPONSE_FAILED
            if failure is not None
            else ASSISTANT_RESPONSE_ABORTED
        )
        if failure is not None:
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
