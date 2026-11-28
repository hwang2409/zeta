"""Run child-agent turns and preserve their bounded result shape."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.agent_state import _lifecycle_elapsed
from ..core.checkpoints.mutation import AppendConversationRows, ConversationRow
from ..core.store import MAX_AGENT_NOTIFICATION_TEXT, ConversationStore
from ..core.store._async_writes import (
    _RegisterAgentChildCommand,
    _SelectChildAccountsCommand,
)
from ..media.image_policy import image_policy_for_provider
from ..path_identity import same_physical_path
from ..project_registry import ProjectRegistryError
from ..protocol.types import (
    ASSISTANT_RESPONSE_STATE,
    ASSISTANT_RESPONSE_SYNTHETIC,
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    assistant_text,
    harness_notice,
    is_owned_completion_backend,
)
from ..providers.rate_limit_hold import (
    ForegroundChildHeld,
    foreground_hold_abandoned,
    foreground_hold_text,
    rate_limit_hold_text,
)
from ..runtime.offload import await_joined_offload
from ..runtime.prompt import RuntimePrompt
from ..tools import ToolRegistry, ToolStreamPublisher
from ..tools._shared.cwd_state import require_bash_cwd
from ..tools.agent import ChildApprovalPolicy, agent_stats
from ..tools.ask_parent import register_ask_parent
from ..tools.finish import register as register_finish
from ..tools.registry import ToolExecutionContext
from . import budget
from . import child_context as _child_context
from .background_lifecycle import (
    _complete_despite_cancellation,
    finish_background_child,
)
from .child_context import (
    _child_base_system_prompt,
    _child_model_and_budget,
    _compose_child_project_context,
    _prepare_child_project_context,
    consume_run,
    resolve_child_backend,
)
from .child_run_owner import ChildControlError, ChildRunTransitionOwner
from .conversation_channel import NO_HOLD_WAKE, HoldWake, has_follow_up_loop
from .explicit_finish import eligible as explicit_finish_eligible
from .explicit_finish import register_handoff
from .finish_gate import FinishGateControl, FinishGateState, run_finish_gate
from .pending_delivery import delivery_preview
from .presets import (
    GENERAL_PRESET,
    RUN_PRESET,
    AgentPreset,
    resolve_child_profile_tools,
    valid_max_turns,
)
from .receipt import (
    MAX_AGENT_RESULT_BYTES,
    TerminalState,
    child_cleanup_details,
    recoverable_error_result,
)
from .resume_descriptor import preset_snapshot
from .tool_results import validated_tool_result
from .turn_limit import (
    ChildTurnControl,
    append_child_start,
    build_turn_limit_result,
    finalize_exhausted_child,
    forward_nested_agent_start,
    make_child_progress_callbacks,
    make_publish_child_lifecycle,
    turn_control_for,
)

_compose_child_system_prompt = _child_context._compose_child_system_prompt

logger = logging.getLogger(__name__)
_EMPTY_METADATA: Mapping[str, object] = MappingProxyType({})
if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop


async def consume_child(
    child_loop: AgentLoop,
    prompt: str,
    *,
    origin: MessageOrigin,
    child_path: str,
    publish: Callable[[str], None],
    child_turns: Callable[[], int],
    update_turns: Callable[[int], Awaitable[None] | None],
    update_step: Callable[[str], None],
    record_tool_call: Callable[[], None],
    finish_lifecycle: Callable[[str, str], dict[str, object]],
    publish_lifecycle: Callable[..., None],
    child_result: Callable[..., dict[str, object]],
    error_message: Callable[[BaseException], str],
    record_report: Callable[[str], None] | None = None,
    notification_turn: bool = False,
    harness_notice_message: Message | None = None,
    finish_gate_state: FinishGateState | None = None,
    delivering_prompt_id: str | None = None,
    explicit_finish_metadata: Mapping[str, object] = _EMPTY_METADATA,
    child_store: ConversationStore | None = None,
    background: bool = False,
    finish_turn_limit: Callable[[str], dict[str, object]] | None = None,
    hold_wake: HoldWake = NO_HOLD_WAKE,
) -> dict[str, object]:
    """Consume one child loop, including nested lifecycle events."""
    final_message: Message | None = None
    last_message: Message | None = None
    failure_message: str | None = None
    failure_error: tuple[str, str] | None = None
    explicit_report: dict[str, object] | None = None

    def terminal_result(
        *, state: TerminalState, text: str, **result_options: object
    ) -> dict[str, object]:
        stats = finish_lifecycle(state, text)
        return child_result(
            text,
            state=state,
            stats=stats,
            **result_options,
        )

    def clean_no_output(text: str) -> dict[str, object]:
        if child_loop.explicit_finish:
            return terminal_result(state="idle", text="", status="running")
        return terminal_result(state="failed", text=text)

    def lifecycle_depth(call: ToolCall | None) -> int:
        if call is not None and call.name.casefold() == "agent":
            return child_loop.agent_depth + 1
        return child_loop.agent_depth

    def emit_lifecycle(
        kind: str, event: StreamEvent, *, tool_result: ToolResult | None = None
    ) -> None:
        publish_lifecycle(
            kind,
            event.tool_call,
            tool_result=tool_result,
            depth=lifecycle_depth(event.tool_call),
            owner=event.data.get("agent_instance_id"),
            approval_origin=event.data.get("approval_origin"),
        )

    async def consume_events(
        events: AsyncIterator[StreamEvent],
        gate_control: FinishGateControl | None = None,
    ) -> None:
        nonlocal failure_error, failure_message, final_message, last_message, explicit_report
        final_message = None
        async for event in events:
            forward_nested_agent_start(child_loop, event)
            if await hold_wake.observe(event):
                if not background and foreground_hold_abandoned(event.data):
                    raise ForegroundChildHeld(foreground_hold_text(event.data))
                account_text = hold_wake.held_account_text()
                hold_text = rate_limit_hold_text(event.data, pinned=account_text is not None)
                status = f"turn {child_turns() + 1}: {hold_text}"
                if account_text is not None:
                    status += f"; {account_text}"
                update_step(status)
                publish(status)
            elif event.type is StreamEventType.TOOL_APPROVAL_START:
                name = event.tool_call.name if event.tool_call is not None else "tool"
                status = f"turn {child_turns() + 1}: approval pending: {name}"
                update_step(status)
                publish(status)
                emit_lifecycle("approval_start", event)
            elif event.type is StreamEventType.TOOL_APPROVAL_END:
                emit_lifecycle("approval_end", event)
            elif event.type is StreamEventType.TOOL_EXECUTION_START:
                record_tool_call()
                name = event.tool_call.name if event.tool_call is not None else "tool"
                arguments = (
                    event.tool_call.arguments if event.tool_call is not None else {}
                )
                summary = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                status = f"turn {child_turns() + 1}: tool: {name} {summary}"
                update_step(status)
                publish(status)
                emit_lifecycle("execution_start", event)
            elif event.type is StreamEventType.TOOL_EXECUTION_END:
                if (
                    event.tool_call is not None
                    and event.tool_call.name == "finish"
                    and event.tool_result is not None
                    and not event.tool_result.is_error
                    and isinstance(event.tool_result.structured_content, dict)
                ):
                    explicit_report = dict(event.tool_result.structured_content)
                emit_lifecycle("execution_end", event, tool_result=event.tool_result)
            elif event.type is StreamEventType.TURN_END:
                last_message = event.message
                turns = child_turns() + 1
                if (write := update_turns(turns)) is not None:
                    await write
                if event.data.get("tool_calls") == 0 and event.message is not None:
                    final_message = event.message
                if gate_control is not None and gate_control.turn_ended():
                    return
            elif event.type is StreamEventType.ERROR and event.error is not None:
                failure_error = (event.error.code, event.error.message)
                failure_message = event.error.message

    def owned_children() -> tuple[tuple[str, str], ...]:
        instance_id = child_loop.agent_instance_id
        if instance_id is None:
            return ()
        return child_loop._background_owner.owned_running(instance_id)

    def owned_tasks() -> tuple[tuple[str, str], ...]:
        return tuple(
            (task.task_id, task.headline)
            for task in child_loop.tool_registry.background_tasks.snapshot()
            if task.running
        )

    try:
        if notification_turn:
            events = child_loop.run_notification_turn()
        elif harness_notice_message is not None:
            events = child_loop.run_harness_notice_turn(harness_notice_message)
        else:
            events = child_loop.run_turn(
                "" if delivering_prompt_id is not None else prompt,
                origin=origin,
                persist_user_message=delivering_prompt_id is None,
            )
        try:
            await consume_events(events)
        finally:
            await events.aclose()
        control = getattr(child_loop, "child_turn_control", None)
        owner = getattr(child_loop, "_child_run_owner", None)
        if (
            control is not None and control.exhausted and owner is not None and owner.durable_questions
            and (await owner.hold_at_limit() or owner.settlement_due())
        ):
            return terminal_result(state="idle", text="waiting for parent answer", status="running")
        limited_result = await finalize_exhausted_child(
            control=owner.exhausted_turn_limit(control) if owner is not None else control,
            text=assistant_text(last_message) if last_message is not None else "",
            child_session_path=child_path,
            child_instance_id=child_loop.agent_instance_id or "unknown",
            child_store=child_store,
            background=background,
            owned_children=owned_children,
            owned_tasks=owned_tasks,
            terminal_result=terminal_result,
            finish_turn_limit=finish_turn_limit,
            record_report=record_report,
        )
        if limited_result is not None:
            return limited_result
        if owner is not None and owner.finish_waits_for_answer:
            return terminal_result(state="idle", text="waiting for parent answer", status="running")
        if explicit_report is not None:
            report_status = explicit_report["status"]
            summary = explicit_report["summary"]
            state: TerminalState = (
                "failed" if report_status == "failed" else "completed"
            )
            text = f"{report_status}: {summary}"
            finish_lifecycle(state, text)
            structured_content = {
                "turns_used": child_turns(),
                "child_session_path": child_path,
                **explicit_finish_metadata,
            }
            return {
                "content": [{"type": "text", "text": text}],
                "isError": state == "failed",
                "structuredContent": structured_content,
                "explicitFinishReport": explicit_report,
            }
        children, tasks = owned_children(), owned_tasks()
        if (
            not child_loop.explicit_finish
            and final_message is not None
            and (children or tasks)
        ):
            fallback_message = final_message
            gate_result = await run_finish_gate(
                child_loop,
                children,
                tasks,
                consume_events,
                lambda: failure_message is not None,
                finish_gate_state,
                delivering_prompt_id,
            )
            if gate_result.outcome == "task_safety_expired":
                child_loop.tool_registry.background_tasks.begin_shutdown(
                    f"finish-gate task safety bound expired after {child_loop._background_owner.finish_gate_task_timeout:g} seconds",
                    phase="task_kill",
                )
            if finish_gate_state is not None:
                finish_gate_state.terminal_admitted = True
            if gate_result.handoff is not None:
                reason, outputs = gate_result.handoff
                final_message = Message(
                    MessageRole.ASSISTANT,
                    [
                        TextContent(
                            "Handed off running background work.\n"
                            f"Reason: {reason}\nOutputs: {outputs}"
                        )
                    ],
                )
            elif gate_result.reason is not None and (owned_children() or owned_tasks()):
                if owned_tasks():
                    fallback_message = Message(
                        MessageRole.ASSISTANT,
                        [
                            TextContent(
                                assistant_text(fallback_message)
                                + f"\n{gate_result.reason}; "
                                "running background tasks were killed."
                            )
                        ],
                    )
                final_message = fallback_message
            elif final_message is None:
                final_message = fallback_message
    except asyncio.CancelledError:
        raise
    except ForegroundChildHeld as held:
        return terminal_result(state="failed", text=str(held))
    except Exception as exc:  # noqa: BLE001 - child failures become receipts
        failure_message = error_message(exc)
    if failure_message is not None:
        if failure_error is not None and failure_error[0] in {
            "account_lock_timeout",
            "no_progress_loop",
        }:
            return recoverable_error_result(terminal_result, failure_message, *failure_error)
        if (
            failure_error is not None
            and failure_error[0] == "pre_dispatch_timeout"
            and background
            and child_loop.step_delivery is not None
            and child_loop.step_delivery.has_pending_edge
        ):
            return recoverable_error_result(terminal_result, failure_message, *failure_error)
        if failure_error is not None and failure_error[0] == "empty_turn":
            return clean_no_output(f"agent error: {failure_message}")
        return terminal_result(state="failed", text=f"agent error: {failure_message}")
    if final_message is None:
        return clean_no_output("agent error: child ended without a final response")
    from ..runtime.loop.empty_turn import OUTPUT_LIMIT_ERROR_MESSAGE, has_visible_output
    final_text = assistant_text(final_message)
    if not has_visible_output(final_message):  # empty or only eviction receipts
        if final_message.metadata.get("stop_reason") == "max_tokens":
            return terminal_result(state="completed", text=OUTPUT_LIMIT_ERROR_MESSAGE)
        return clean_no_output(
            "agent error: child returned an empty final assistant message"
        )
    if record_report is not None:
        record_report(final_text)
    if child_loop.explicit_finish:
        return terminal_result(state="idle", text=final_text, status="running")
    return terminal_result(state="completed", text=final_text)


class ChildRecoveryPolicyError(ValueError):
    """A policy conflict with a safe, user-facing recovery reason."""


async def _build_child_loop(
    loop: AgentLoop,
    *,
    child_backend: CompletionBackend,
    child_store: ConversationStore,
    child_instance_id: str,
    child_depth: int,
    preset: AgentPreset,
    model: object,
    cwd_override: Path | None,
    follow_up_loop: bool,
    description: str,
    child_account_context: Any,
    run_owner: ChildRunTransitionOwner,
    child_turn_control: ChildTurnControl | None = None,
    recovered_explicit_finish: bool | None = None,
) -> tuple[AgentLoop, ChildApprovalPolicy | None]:
    """Build the child under current policy for both spawn and recovery."""
    child_registry: ToolRegistry | None = None
    child_policy: ChildApprovalPolicy | None = None
    child_loop: AgentLoop | None = None

    async def close_construction(*, background_task_cause: str) -> None:
        if child_loop is not None:
            await child_loop.close(background_task_cause=background_task_cause)
        elif is_owned_completion_backend(child_backend):
            await loop.close_backend(child_backend, stage="child_construction")
        if child_registry is not None:
            await child_registry.close(background_task_cause=background_task_cause)

    try:
        recovering = recovered_explicit_finish is not None
        explicit_finish = (
            explicit_finish_eligible(
                global_enabled=loop.explicit_finish,
                preset=preset,
                background=follow_up_loop,
            )
            if recovered_explicit_finish is None
            else recovered_explicit_finish
        )
        runtime_tool_names = {"ask_parent"} if follow_up_loop else set()
        if explicit_finish:
            runtime_tool_names.update(("finish", "agent_handoff"))
        policy_allowed_runtime_names = frozenset(
            name
            for name in runtime_tool_names
            if loop.tool_registry.tool_is_allowed_by_policy(name)
        )
        try:
            excluded_names, profile_tool_allow = resolve_child_profile_tools(
                preset=preset,
                policy_allowed_names=loop.tool_registry.policy_allowed_registered_names,
                runtime_tool_names=policy_allowed_runtime_names,
            )
        except ValueError as exc:
            if recovering:
                raise ChildRecoveryPolicyError(
                    "current tool policy cannot satisfy the saved profile's required tools"
                ) from exc
            raise
        child_provider = getattr(child_backend, "provider", None)
        child_image_policy = (
            image_policy_for_provider(child_provider)
            if isinstance(child_provider, str)
            else loop.tool_registry.image_policy
        )
        child_registry = loop.tool_registry.clone_for_session(
            child_store,
            exclude_names=excluded_names,
            exclude_mcp_servers=set(loop._mcp_bindings),
            profile_tool_allow=profile_tool_allow,
            cwd=cwd_override,
            image_policy=child_image_policy,
        )
        if child_account_context is not None:
            child_registry.account_context = child_account_context
        loop._background_owner.store_leases.callback(
            child_registry.background_tasks.release_directory
        )
        if follow_up_loop:
            register_ask_parent(
                child_registry,
                channel=loop._background_owner.conversation_channel,
                child_instance_id=child_instance_id,
            )
        parent_policy = loop.tool_registry.approval_policy
        if parent_policy is not None:
            child_policy = ChildApprovalPolicy(
                parent_policy,
                child_store,
                description,
                child_instance_id,
                parent_cwd=loop.tool_registry.cwd,
                child_cwd=child_registry.cwd,
            )
            child_registry.set_approval_policy(child_policy)
        from ..runtime.loop import AgentLoop

        child_model, child_budget = _child_model_and_budget(
            loop, child_backend, child_provider, model
        )

        def record_child_usage(usage: Mapping[str, Any]) -> None:
            loop.context_assembler.record_descendant_usage(
                {**usage, "_zeta_model": usage.get("_zeta_model", child_model)}
            )

        child_context = await await_joined_offload(
            _child_base_system_prompt, loop, cwd_override, child_registry
        )
        child_scope = getattr(loop, "_live_config_scope", None)
        child_context = _compose_child_project_context(
            child_context,
            preset,
            explicit_finish=explicit_finish,
        )
        child_loop = AgentLoop(
            child_backend,
            child_store,
            registry=child_registry,
            skill_catalog=child_registry.skill_catalog,
            max_turns=None,
            token_budget=child_budget,
            retained_tail=loop.context_assembler.retained_tail,
            runtime_prompt=RuntimePrompt(
                child_context.system_prompt, child_context.prompt_components
            ),
            agent_catalog=child_registry.agent_catalog,
            skip_mcp_mount=True,
            agent_depth=child_depth,
            max_agent_depth=budget.depth_limit(
                loop.max_agent_depth, preset.max_child_depth
            ),
            agent_instance_id=child_instance_id,
            root_project_id=loop.root_project_id,
            root_session_dir=loop.root_session_dir,
            project_registry=loop.project_registry,
            child_lineage_publisher=loop._child_lineage_publisher,
            background_owner=loop._background_owner,
            usage_sink=record_child_usage,
            child_turn_control=child_turn_control,
            live_config_home=child_scope.home if child_scope is not None else None,
            live_config_project_dir=(
                child_scope.project_dir if child_scope is not None else None
            ),
            freeze_live_config=child_scope is None,
            configured_model=loop.configured_model,
            telemetry=loop.telemetry,
        )

        if not recovering:
            child_loop.agent_profile = preset.name
            await child_loop.refresh_live_config(publish=False)

        inherited_prompt = cwd_override is None or same_physical_path(
            cwd_override, loop.store.cwd
        )
        child_loop._configure_project_context_recovery(
            registry=loop.project_registry,
            project_id=loop.root_project_id,
            health=child_context.health,
            prepare=partial(
                _prepare_child_project_context,
                loop, cwd_override, child_registry, preset, explicit_finish,
            ),
            replace_component=("project_memory" if inherited_prompt else None),
        )
        child_loop._child_run_owner = run_owner
        child_loop.explicit_finish = explicit_finish
        child_loop.one_shot = getattr(loop, "one_shot", False)
        child_loop.account_context = child_registry.account_context
        if child_loop.explicit_finish:
            register_finish(child_registry, child_loop=child_loop)
            register_handoff(child_loop)
            # Only finish is required to complete the saved contract.
            if recovering and "finish" not in child_registry.registered_names:
                raise ChildRecoveryPolicyError(
                    "current policy blocks required explicit-finish tools: finish"
                )
        loop._background_owner.conversation_channel.register_loop(child_instance_id)
        await child_loop._ensure_step_delivery()
        if loop.plan_mode:
            child_loop.set_plan_mode(True)
        child_loop.set_background_event_sink(loop._publish_background_event)
        return child_loop, child_policy
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(
            close_construction(background_task_cause="child spawn setup canceled")
        )
        await _complete_despite_cancellation(cleanup)
        raise
    except BaseException:
        await close_construction(background_task_cause="child spawn setup failed")
        raise


async def run_agent_tool(
    loop: AgentLoop,
    tool_call: ToolCall,
    arguments: dict[str, Any],
    abort_signal: ToolAbortSignal,
    publisher: ToolStreamPublisher | None,
    validate_result: Callable[[object, str], ToolResult],
    error_message: Callable[[BaseException], str],
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    result = await loop._background_owner.start_child_setup(lambda: _run_child(
        loop, tool_call, arguments, abort_signal, publisher, validate_result,
        error_message, execution_context,
    ))
    assert isinstance(result, dict)
    return result


async def spawn_child(
    loop: AgentLoop, *, agent_type: str, task: str, cwd: str | None,
) -> str:
    arguments = {
        "preset": agent_type,
        "prompt": task,
        "description": delivery_preview(task),
        "background": True,
        "cwd": cwd,
    }
    # A disconnected controller must not interrupt the durable start sequence.
    launch = loop._background_owner.start_child_setup(lambda: _run_child(
        loop, None, arguments, ToolAbortSignal(), None, validated_tool_result, str,
    ))
    if await _complete_despite_cancellation(launch):
        raise asyncio.CancelledError
    result = launch.result()
    assert isinstance(result, str)
    return result


async def _run_child(
    loop: AgentLoop,
    tool_call: ToolCall | None,
    arguments: dict[str, Any],
    abort_signal: ToolAbortSignal,
    publisher: ToolStreamPublisher | None,
    validate_result: Callable[[object, str], ToolResult],
    error_message: Callable[[BaseException], str],
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object] | str:
    external = tool_call is None
    if tool_call is None:
        # Existing markers and cards use a call id; no tool execution is admitted.
        tool_call = ToolCall(f"spawn-{uuid4().hex}", "agent", arguments)

    def failure(
        text: str, *, code: str = "invalid_params", public_detail: bool = False
    ) -> dict[str, object]:
        if external:
            raise ChildControlError(code, detail=text if public_detail else None)
        return loop._child_result_payload(tool_call.id, text, state="failed")

    prompt = arguments.get("prompt")
    description = arguments.get("description")
    requested_preset = arguments.get("preset")
    legacy_agent_type = arguments.get("agent_type")
    if requested_preset is not None and legacy_agent_type is not None:
        return failure(
            "agent error: pass only one of preset and agent_type",
        )
    agent_type = (
        requested_preset
        if requested_preset is not None
        else legacy_agent_type
        if legacy_agent_type is not None
        else GENERAL_PRESET.name
    )
    model = arguments.get("model")
    if type(prompt) is not str or not prompt.strip():
        return failure(
            "agent error: prompt must be a nonempty string",
        )
    if type(description) is not str or not description.strip():
        return failure(
            "agent error: description must be a nonempty string",
        )
    catalog = loop.tool_registry.agent_catalog
    if type(agent_type) is not str:
        return failure(
            f"agent error: preset must be one of: {', '.join(catalog.names())}",
        )
    try:
        preset = catalog.find(agent_type)
    except ValueError:
        return failure(
            f"agent error: unknown agent_type/preset {agent_type!r}; expected one of: "
            f"{', '.join(catalog.names())}",
        )
    if (
        loop.plan_mode
        and preset.source == "packaged"
        and preset.name == GENERAL_PRESET.name
    ):
        return failure(
            "agent error: general agents are unavailable in plan mode; "
            "use agent_type 'explore' or 'plan'",
        )
    requested_max_turns = arguments.get("max_turns", preset.max_turns)
    if requested_max_turns is not None and not valid_max_turns(requested_max_turns):
        return failure(
            "agent error: max_turns must be a positive integer",
        )
    if model is not None:
        allowed = loop.allowed_child_models()
        if allowed is not None and model not in allowed:
            return failure(
                "agent error: model override is not allowed; allowed models: "
                + (", ".join(allowed) or "none"),
            )
    raw_cwd = arguments.get("cwd")
    if raw_cwd is None and preset.requires_cwd:
        return failure(
            f"agent error: preset {preset.name!r} requires cwd; pass the intended worktree path",
            public_detail=True,
        )
    child_cwd = str(loop.store.cwd)
    cwd_override: str | None = None
    if raw_cwd is not None:
        if type(raw_cwd) is not str or not raw_cwd.strip():
            return failure("agent error: cwd must be a nonempty string")

        def resolve_cwd() -> str:
            if external and not Path(raw_cwd).is_absolute():
                raise ValueError("cwd must be absolute")
            candidate = Path(raw_cwd).expanduser()
            if not candidate.is_absolute():
                candidate = require_bash_cwd(loop.tool_registry.bash_cwd) / candidate
            resolved = Path(os.path.abspath(candidate))
            if not resolved.is_dir():
                raise ValueError(f"cwd is not an existing directory: {resolved}")
            return str(resolved)

        try:
            child_cwd = cwd_override = await asyncio.to_thread(resolve_cwd)
        except (OSError, RuntimeError, ValueError) as exc:
            return failure(f"agent error: {exc}")
    effective_model = model if model is not None else preset.model
    if getattr(loop, "account_context", None) is None:
        child_backend, backend_error = resolve_child_backend(loop, effective_model)
    else:
        child_backend, backend_error = await asyncio.to_thread(
            resolve_child_backend, loop, effective_model
        )
    if backend_error is not None:
        return failure(
            backend_error, code="state_unavailable",
        )
    # The long kinds default to background -- a run on someone else's model, or
    # one sized for a big task. Waiting on either blocks the orchestrator for
    # the whole thing. An explicit background argument still wins.
    background = arguments.get(
        "background",
        effective_model is not None
        or (preset.source == "packaged" and preset.name == RUN_PRESET.name),
    )
    if type(background) is not bool:
        return failure(
            "agent error: background must be a boolean",
        )
    if getattr(loop, "one_shot", False):
        if external:
            raise ChildControlError("operation_busy")
        # Headless exits after this turn, so a background child would be canceled.
        background = False
    child_depth, nesting_error = budget.child_depth(
        loop.agent_depth, loop.max_agent_depth, background
    )
    if nesting_error is not None:
        return failure(
            nesting_error, code="operation_busy",
        )
    try:
        await loop._ensure_mcp_servers()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - setup failures become receipts
        return failure(
            f"agent error: {error_message(exc)}", code="state_unavailable",
        )
    follow_up_loop = has_follow_up_loop(
        accepts_follow_ups=preset.accepts_follow_ups,
        background=background,
    )
    stored_agent_type = (
        None
        if preset.source == "packaged" and preset.name == GENERAL_PRESET.name
        else preset.name
    )
    child_number = loop.store.allocate_agent_index()
    parent_identity = loop.agent_instance_id or loop.store.session_id
    child_instance_id = f"{parent_identity}:{child_number}"
    agents_root = loop.store.session_dir / "agents"
    opened: list[ConversationStore] = []

    def open_child_store() -> ConversationStore:
        store = ConversationStore(
            agents_root,
            session_id=str(child_number),
            cwd=child_cwd,
            _storage_lease=loop.store._storage_lease,
            _owner_surface="child",
        )
        opened.append(store)
        return store

    try:
        child_store = await loop.store._to_thread_store(open_child_store)
    except asyncio.CancelledError:
        # _to_thread_store finishes the worker before it raises, so close its store.
        for store in opened:
            await store._to_thread_store(store.close)
        raise
    def prepare_child_store() -> None:
        child_store.activate_runtime_lease()
        child_store.accepts_follow_ups = follow_up_loop
        child_store.mark_agent_parent(tool_call.id, agent_type=stored_agent_type)

    try:
        await child_store._to_thread_store(prepare_child_store)
    except BaseException:
        await child_store._to_thread_store(child_store.close)
        raise
    loop._background_owner.store_leases.enter_context(child_store)
    loop._background_owner.track_store(child_store)
    child_path = str(child_store.session_dir)
    explicit_finish = explicit_finish_eligible(
        global_enabled=loop.explicit_finish, preset=preset, background=background,
    )
    account_context = getattr(loop, "account_context", None)

    def start_lifecycle(selected_accounts: dict[str, object]) -> None:
        make_publish_child_lifecycle(
            account_context=account_context,
            selected_accounts=selected_accounts,
            child_store=child_store,
            child_instance_id=child_instance_id,
            child_depth=child_depth,
            preset=preset,
            description=description,
            child_cwd=child_cwd,
        )()

    async def release_setup_store() -> None:
        loop._agent_child_stores.pop(tool_call.id, None)
        loop._agent_child_turns.pop(tool_call.id, None)
        loop._agent_child_types.pop(tool_call.id, None)
        loop._background_owner.conversation_channel.unregister_loop(child_instance_id)
        loop._background_owner.unregister(child_instance_id)
        loop._background_owner.finish_store(child_instance_id, child_store)
        await loop._background_owner.wait_for_store_closes()

    async def prepare_setup_cancel() -> None:
        if child_store.agent_lifecycle() is None:
            await await_joined_offload(start_lifecycle, {})
        if not store_bound:
            child_store.bind_owner_thread()

    async def cancel_setup() -> None:
        try:
            if run_owner.suspending:
                await run_owner.persist_suspension("normal_exit")
            else:
                await run_owner.cancel()
        finally:
            await release_setup_store()

    setup_task = asyncio.current_task()
    assert setup_task is not None
    store_bound = False

    resume = None
    if background and loop.store.child_recovery_enabled:
        provider = getattr(child_backend, "provider", None)
        resolved_model, _ = _child_model_and_budget(loop, child_backend, provider, model)
        resume = {
            "preset": preset_snapshot(preset), "requested_model": model,
            "model": resolved_model, "provider": provider or "unconfigured",
            "requested_max_turns": arguments.get("max_turns"),
            "effective_max_turns": requested_max_turns, "depth": child_depth,
            "cwd": child_cwd, "description": description, "background": background,
            "follow_up": follow_up_loop, "explicit_finish": explicit_finish,
            "account_pins": "pending", "selected_accounts": {},
        }

    async def register_run() -> ChildRunTransitionOwner:
        await loop.store._commit_storage_command(_RegisterAgentChildCommand(
            tool_call=tool_call, child_session_path=child_path, description=description,
            agent_type=stored_agent_type, background=background,
            child_instance_id=child_instance_id, accepts_follow_ups=follow_up_loop,
            explicit_finish=explicit_finish, turns_used=0, resume=resume,
        ))
        return loop._background_owner.begin_run(
            child_instance_id, child_store, tool_call.id, setup_task, prepare_setup_cancel,
            parent_store=loop.store, parent_instance_id=loop.agent_instance_id,
            description=description if background else None, explicit_finish=explicit_finish,
            background=background,
        )

    # Join a submitted marker commit and register its owner even if interrupted.
    try:
        registration = loop._background_owner.register_child_setup(register_run)
        setup_interrupted = await _complete_despite_cancellation(registration)
    except BaseException:
        await release_setup_store()
        raise
    run_owner = registration.result()
    child_policy: ChildApprovalPolicy | None = None
    try:
        if setup_interrupted or run_owner.quiescing:
            if run_owner.recoverable and run_owner.quiescing:
                await run_owner.suspend()
            raise asyncio.CancelledError
        if loop._child_lineage_publisher is not None:
            link = {
                "project_id": loop.root_project_id,
                "session_id": child_instance_id,
                "role": "worker",
                "parent_session_id": parent_identity,
                "transcript_path": child_path,
            }
            try:
                await await_joined_offload(loop._child_lineage_publisher.publish, link)
            except (ProjectRegistryError, OSError, ValueError) as exc:
                logger.warning("could not record child project lineage: %s", exc)
        await append_child_start(
            loop.store,
            child_instance_id,
            parent_identity if loop.agent_instance_id is not None else None,
            stored_agent_type,
            description,
            accepts_follow_ups=follow_up_loop,
        )
        child_account_pins = (
            await await_joined_offload(
                account_context.child_snapshot, getattr(child_backend, "provider", None)
            )
            if account_context is not None
            else None
        )
        selected_accounts = (
            child_account_pins.snapshot() if child_account_pins is not None else {}
        )
        if resume is not None:
            await loop.store._commit_storage_command(_SelectChildAccountsCommand(
                run_id=child_instance_id, pins={provider: str(account) for provider, account in selected_accounts.items()},
            ))
        child_turn_control = turn_control_for(requested_max_turns, child_path)
        await run_owner.start(lambda: start_lifecycle(selected_accounts))
        child_store.bind_owner_thread()
        store_bound = True
        loop._agent_child_stores[tool_call.id] = child_store
        loop._agent_child_turns[tool_call.id] = 0
        loop._agent_child_types[tool_call.id] = preset.name
        if publisher is not None:
            publisher.set_metadata(
                {"child_session_path": child_path, "depth": child_depth}
            )
        child_loop, child_policy = await _build_child_loop(
            loop,
            child_backend=child_backend,
            child_store=child_store,
            child_instance_id=child_instance_id,
            child_depth=child_depth,
            preset=preset,
            model=model,
            cwd_override=cwd_override,
            follow_up_loop=follow_up_loop,
            description=description,
            child_account_context=child_account_pins,
            child_turn_control=child_turn_control,
            run_owner=run_owner,
        )
        loop._background_owner.project_context_loops[child_store.session_id] = (
            child_loop
        )
        child_loop.agent_background = background
    except asyncio.CancelledError:
        await _complete_despite_cancellation(asyncio.create_task(cancel_setup()))
        raise
    except Exception as exc:
        if not store_bound:
            if await _complete_despite_cancellation(asyncio.create_task(release_setup_store())):
                raise asyncio.CancelledError
            raise
        loop._background_owner.conversation_channel.unregister_loop(child_instance_id)
        failure_text = f"agent error: {error_message(exc)}"
        terminal_stats = _foreground_terminal_stats(child_store, "failed", 0)
        failure_payload = loop._child_result_payload(
            tool_call.id,
            failure_text,
            state="failed",
            child_session_path=child_path,
            agent_type=preset.name,
            child_instance_id=child_instance_id,
            depth=child_depth,
            stats=terminal_stats,
        )
        try:
            await _complete_despite_cancellation(asyncio.create_task(
                run_owner.complete_foreground(
                    "failed", failure_payload["content"][0]["text"], 0, canonical=True,
                )
            ))
        finally:
            loop._background_owner.unregister(child_instance_id)
            loop._background_owner.mark_store_finished(child_store)
        if external:
            if await _complete_despite_cancellation(asyncio.create_task(release_setup_store())):
                raise asyncio.CancelledError
            raise ChildControlError("state_unavailable") from None
        return failure_payload
    return await _run_installed_child(
        loop, child_loop, child_store, run_owner, tool_call,
        prompt=prompt, description=description, preset=preset, child_depth=child_depth,
        child_instance_id=child_instance_id, background=background, follow_up_loop=follow_up_loop,
        child_policy=child_policy, child_account_pins=child_account_pins, external=external,
        abort_signal=abort_signal, publisher=publisher, validate_result=validate_result,
        error_message=error_message, execution_context=execution_context,
    )


def _foreground_terminal_stats(
    child_store: ConversationStore, state: str, turns_used: int,
) -> dict[str, object]:
    lifecycle = child_store.agent_lifecycle() or {}
    stats = agent_stats(lifecycle, status=state, turns_used=turns_used)
    stats["turns_used"] = turns_used
    stats["elapsed"] = _lifecycle_elapsed(lifecycle)
    return stats


async def _run_installed_child(
    loop: AgentLoop,
    child_loop: AgentLoop,
    child_store: ConversationStore,
    run_owner: ChildRunTransitionOwner,
    tool_call: ToolCall,
    *,
    prompt: str,
    description: str,
    preset: AgentPreset,
    child_depth: int,
    child_instance_id: str,
    background: bool,
    follow_up_loop: bool,
    child_policy: ChildApprovalPolicy | None,
    child_account_pins: Any,
    external: bool,
    abort_signal: ToolAbortSignal,
    publisher: ToolStreamPublisher | None,
    validate_result: Callable[[object, str], ToolResult],
    error_message: Callable[[BaseException], str],
    execution_context: ToolExecutionContext | None,
    activation: asyncio.Event | None = None,
    resumed_idle: bool = False,
) -> dict[str, object] | str:
    child_path = str(child_store.session_dir)
    setup_task = asyncio.current_task()
    assert setup_task is not None
    lifecycle_sink = (
        execution_context.lifecycle_sink if execution_context is not None else None
    )

    def publish(status: str, *, step: bool = False) -> None:
        if background:
            event_data: dict[str, object] = {"stream": "stdout"}
            if step:
                event_data["step"] = status
            if loop.agent_instance_id is not None:
                event_data["agent_instance_id"] = loop.agent_instance_id
            loop._publish_background_event(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_UPDATE,
                    tool_call=tool_call,
                    delta=f"{description}: {status}\n",
                    data=event_data,
                )
            )
        elif publisher is not None:
            publisher.publish(f"{description}: {status}\n", "stdout")

    def child_turns() -> int:
        return loop._agent_child_turns.get(tool_call.id, 0)

    def update_step(step: str) -> None:
        child_store.update_agent_lifecycle(current_step=step)

    def set_pre_dispatch_step(step: str) -> None:
        status = f"turn {child_turns() + 1}: {step}"
        update_step(status)
        publish(status, step=True)

    def report_pre_dispatch_stall(step: str, error: str) -> None:
        set_pre_dispatch_step(step)

        async def notify_parent() -> None:
            try:
                await loop._background_owner.conversation_channel.publish_idle(
                    child_instance_id=child_instance_id,
                    child_session_path=child_path,
                    description=description,
                    text=error,
                )
            except LookupError:
                # The progress update is still visible if the parent closed.
                pass
            except Exception:
                logger.exception("could not notify parent of stalled child %s", child_instance_id)

        loop._create_task(notify_parent())

    if run_owner.provider_recovery_enabled:
        run_owner.bind_provider_recovery(loop, child_loop.backend, description)
    child_loop._set_pre_dispatch_step = set_pre_dispatch_step
    child_loop._report_pre_dispatch_stall = report_pre_dispatch_stall

    terminal_state: str | None = None
    terminal_text = ""
    canonical_text: str | None = None

    def finish_lifecycle(state: str, text: str) -> dict[str, object]:
        nonlocal terminal_state, terminal_text
        turns_used = child_turns()
        if background:
            child_store.update_agent_lifecycle(turns_used=turns_used)
            return agent_stats(child_store.agent_lifecycle(), turns_used=turns_used)
        terminal_state, terminal_text = state, text
        return _foreground_terminal_stats(child_store, state, turns_used)

    def child_result(
        text: str,
        *,
        state: TerminalState | None = None,
        error: bool | None = None,
        status: str | None = None,
        stats: dict[str, object] | None = None,
        include_stats: bool = not background,
        notice: str | None = None,
        notice_items: Sequence[str] | None = None,
        max_bytes: int | None = None,
        persist_receipt: bool = True,
    ) -> dict[str, object]:
        nonlocal canonical_text
        payload = loop._child_result_payload(
            tool_call.id,
            text,
            state=state,
            error=error,
            child_session_path=child_path,
            agent_type=preset.name,
            status=status,
            child_instance_id=child_instance_id,
            description=description if background else None,
            depth=child_depth,
            stats=stats,
            include_stats=include_stats,
            notice=notice,
            notice_items=notice_items,
            max_bytes=max_bytes,
        )
        if not background and state is not None and persist_receipt:
            content = payload.get("content")
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and isinstance(content[0].get("text"), str)
            ):
                canonical_text = content[0]["text"]
        return payload

    def finish_turn_limit(text: str) -> dict[str, object]:
        nonlocal canonical_text
        stats = finish_lifecycle("completed", text)
        payload = child_result(text, state="completed", stats=stats, persist_receipt=False)
        canonical_text = build_turn_limit_result(payload["content"][0]["text"], child_path)
        return payload

    def publish_lifecycle(
        kind: str,
        call: ToolCall | None,
        *,
        tool_result: ToolResult | None = None,
        depth: int | None = None,
        owner: object = None,
        approval_origin: object = None,
    ) -> None:
        if call is None:
            return
        event_type = {
            "approval_start": StreamEventType.TOOL_APPROVAL_START,
            "approval_end": StreamEventType.TOOL_APPROVAL_END,
            "execution_start": StreamEventType.TOOL_EXECUTION_START,
            "execution_end": StreamEventType.TOOL_EXECUTION_END,
        }.get(kind)
        if event_type is None:
            return
        data = {"depth": child_depth if depth is None else depth}
        data["agent_instance_id"] = owner if type(owner) is str else child_instance_id
        if approval_origin in {"automatic", "user"}:
            data["approval_origin"] = approval_origin
        if background:
            loop._publish_background_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    tool_result=tool_result,
                    data=data,
                )
            )
        elif lifecycle_sink is not None:
            lifecycle_sink(kind, call, data, tool_result)

    update_turns, record_tool_call, save_progress_while_running = make_child_progress_callbacks(
        loop=loop,
        child_store=child_store,
        child_marker_key=child_instance_id,
        child_tool_call_id=tool_call.id,
        recoverable=run_owner.recoverable,
    )

    consume = consume_run if follow_up_loop else consume_child
    loop._background_owner.causal_episodes.associate_new_member(
        child_instance_id, loop.agent_instance_id
    )
    finish_gate_state = FinishGateState()
    receipt_components: dict[str, str] = {}
    run_kwargs: dict[str, Any] = (
        {
            "child_store": child_store,
            "tool_call_id": tool_call.id,
            "max_receipt_bytes": getattr(
                loop.tool_registry, "max_output_chars", MAX_AGENT_RESULT_BYTES
            ),
            "receipt_components": receipt_components,
            "finish_gate_state": finish_gate_state,
            "finish_turn_limit": finish_turn_limit,
            "explicit_finish_metadata": {
                "child_instance_id": child_instance_id,
                "description": description,
                **({"agent_type": preset.name} if preset.name != "general" else {}),
                **({"depth": child_depth} if child_depth != 1 else {}),
            },
        }
        if follow_up_loop
        else {
            "origin": MessageOrigin.AGENT_PROMPT,
            "record_report": lambda report: receipt_components.setdefault(
                "report", report
            ),
            "finish_gate_state": finish_gate_state,
            "child_store": child_store,
            "finish_turn_limit": finish_turn_limit,
        }
    )
    if resumed_idle:
        run_kwargs["resumed_idle"] = True
        consume = consume_run
        run_kwargs.setdefault("tool_call_id", tool_call.id)
        run_kwargs.pop("origin", None)
    if activation is not None and not resumed_idle:
        run_kwargs["harness_notice_message"] = harness_notice(
            "child recovery", prompt, event="child_recovery", visible=False,
        )
    spawn_ready = activation if activation is not None else asyncio.Event() if external else None

    async def commit_foreground_terminal(result: dict[str, object]) -> None:
        if background or terminal_state is None:
            return
        text = canonical_text
        if text is None and follow_up_loop:
            content = result.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                candidate = content[0].get("text")
                if isinstance(candidate, str):
                    text = candidate
        await run_owner.complete_foreground(
            terminal_state, text or terminal_text, child_turns(), canonical=text is not None,
        )

    async def run_child() -> dict[str, object]:
        if spawn_ready is not None:
            await spawn_ready.wait()
        hold_wake = HoldWake(
            loop._background_owner.conversation_channel if background else None,
            child_instance_id, child_path, description, child_account_pins, child_store,
        )
        await hold_wake.start()
        # Joined before this task returns, so no save lands after the suspend or finish write.
        progress_saver = loop._create_task(save_progress_while_running()) if run_owner.recoverable else None
        try:
            result = await consume(
                child_loop,
                prompt,
                **run_kwargs,
                child_path=child_path,
                publish=publish,
                hold_wake=hold_wake,
                child_turns=child_turns,
                update_turns=update_turns,
                record_tool_call=record_tool_call,
                update_step=update_step,
                finish_lifecycle=finish_lifecycle,
                publish_lifecycle=publish_lifecycle,
                child_result=child_result,
                error_message=error_message,
            )
        finally:
            if progress_saver is not None:
                progress_saver.cancel()
                await asyncio.gather(progress_saver, return_exceptions=True)
        await commit_foreground_terminal(result)
        return result

    child_task = loop._create_task(run_child())

    def abort_runtime() -> None:
        child_loop.admit_abort()
        child_loop.tool_registry.abort()

    async def cleanup_canceled_child() -> None:
        loop._background_owner.cancel_subtree(child_instance_id, include_self=False)
        if not child_task.done():
            child_task.cancel()
        await loop._background_owner.wait_for_descendant_watchers(child_instance_id)
        await asyncio.gather(child_task, return_exceptions=True)
        await child_store.drop_client_steering_async("abort")

    run_owner.attach_runtime(child_task, abort_runtime, cleanup_canceled_child, child_loop.admit_abort)
    if not external:
        loop._background_owner.finish_child_setup(setup_task)

    if background:
        # Lifecycle final_result and the notification reuse the bounded
        # background receipt text, so respect both output limits.
        background_receipt_bytes = min(
            getattr(loop.tool_registry, "max_output_chars", MAX_AGENT_RESULT_BYTES),
            MAX_AGENT_NOTIFICATION_TEXT,
        )

        async def cleanup_background_child() -> None:
            await loop._background_owner.wait_for_descendant_watchers(
                child_instance_id
            )
            await loop._background_owner.conversation_channel.unregister_delivery_flow(
                child_instance_id
            )
            loop._agent_child_stores.pop(tool_call.id, None)
            loop._agent_child_turns.pop(tool_call.id, None)
            loop._agent_child_types.pop(tool_call.id, None)
            loop._background_child_cancellers.pop(tool_call.id, None)
            loop._background_child_watchers.pop(tool_call.id, None)
            loop._background_owner.conversation_channel.unregister_loop(
                child_instance_id
            )
            loop._background_owner.project_context_loops.pop(
                child_store.session_id, None
            )
            loop._background_owner.finish_store(child_instance_id, child_store)
            loop._background_owner.unregister(child_instance_id)
            if child_policy is not None:
                await child_policy.cleanup()

        async def close_background_child(cause: str) -> tuple[str, ...]:
            return tuple(
                await child_loop.close(
                    background_task_cause=cause, cancel_background=False
                )
            )

        async def finish_background() -> None:
            lifecycle = asyncio.create_task(
                finish_background_child(
                    child_task=child_task,
                    child_store=child_store,
                    parent_store=loop.store,
                    tool_call=tool_call,
                    child_instance_id=child_instance_id,
                    child_path=child_path,
                    description=description,
                    child_turns=child_turns,
                    build_result=lambda text, error, status, stats, notice=None, notice_items=None: (
                        child_result(
                            text,
                            error=error,
                            status=status,
                            stats=stats,
                            include_stats=True,
                            notice=notice,
                            notice_items=notice_items,
                            max_bytes=background_receipt_bytes,
                        )
                    ),
                    validate_result=validate_result,
                    publish_event=loop._publish_background_event,
                    cleanup=cleanup_background_child,
                    close_child=close_background_child,
                    error_message=error_message,
                    marker_key=child_instance_id,
                    agent_instance_id=loop.agent_instance_id,
                    background_owner=loop._background_owner,
                    max_receipt_bytes=background_receipt_bytes,
                    run_owner=run_owner,
                    child_loop=child_loop,
                    finish_gate_state=finish_gate_state,
                )
            )
            if await _complete_despite_cancellation(lifecycle):
                raise asyncio.CancelledError

        watcher = loop._create_task(finish_background())
        loop._background_child_watchers[tool_call.id] = watcher
        loop._background_child_cancellers[tool_call.id] = run_owner.request_cancel
        loop._background_owner.attach_run_watcher(child_instance_id, watcher)
        # The tree owner now keeps this task pair alive after this loop closes.
        loop._tracked_tasks.discard(child_task)
        loop._tracked_tasks.discard(watcher)
        if activation is not None:
            return child_instance_id
        running_result = child_result(
            (
                f"background agent started: handle={child_instance_id} "
                f"({description}). Completion is announced automatically: "
                "while you run, after your current tool call and before your "
                "next model call; while you are idle, at your next turn. Use "
                "agent_status or agent_output only for on-demand inspection."
            ),
            error=False,
            status="running",
        )
        running_tool_result = validate_result(running_result, tool_call.id)
        # Keep this publication synchronous: parallel agent calls rely on task
        # dispatch order when exposing their immediate running receipts.
        receipt = Message(
            MessageRole.TOOL_RESULT,
            [TextContent(running_tool_result.content)],
            tool_result=running_tool_result,
        )
        if external:
            # Keep the discoverable receipt paired without leaving a pending tool.
            anchor = Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(tool_call)],
                metadata={ASSISTANT_RESPONSE_STATE: ASSISTANT_RESPONSE_SYNTHETIC},
            )
            try:
                await loop.store._commit_address_async(AppendConversationRows((
                    ConversationRow("message", {"message": anchor.to_dict()}),
                    ConversationRow("message", {"message": receipt.to_dict()}),
                )))
            except (Exception, asyncio.CancelledError):
                loop._background_owner.cancel(child_instance_id)
                raise
            loop._publish_background_event(StreamEvent(
                StreamEventType.TOOL_EXECUTION_START, tool_call=tool_call,
                data={"child_session_path": child_path, "depth": child_depth},
            ))
            assert spawn_ready is not None
            spawn_ready.set()
            return child_instance_id
        loop.store.append_message(receipt)
        return running_result

    loop._background_owner.attach_run_watcher(child_instance_id, child_task)
    abort_task = loop._create_task(abort_signal.wait())
    try:
        done, _ = await asyncio.wait(
            (child_task, abort_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        # A completed child wins an abort race so descendants can be adopted by receipt processing.
        if child_task in done:
            return child_task.result()
        if abort_task in done:
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        await _complete_despite_cancellation(asyncio.create_task(run_owner.cancel_and_join()))
        raise
    finally:
        loop._background_owner.conversation_channel.unregister_loop(child_instance_id)
        loop._background_owner.project_context_loops.pop(child_store.session_id, None)
        loop._background_owner.unregister(child_instance_id)
        try:
            if child_policy is not None:
                await child_policy.cleanup()
        finally:
            if not abort_task.done():
                abort_task.cancel()
            try:
                await asyncio.gather(abort_task, return_exceptions=True)
            finally:
                close_task = asyncio.create_task(
                    child_loop.close(
                        cancel_background=False,
                        background_task_cause=child_cleanup_details(
                            "foreground", child_task.cancelled()
                        )[0],
                    )
                )
                if await _complete_despite_cancellation(close_task):
                    raise asyncio.CancelledError()
