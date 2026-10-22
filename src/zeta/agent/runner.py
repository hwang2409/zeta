"""Run child-agent turns and preserve their bounded result shape."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.checkpoints import _now
from ..core.project_context import (
    discover_project,
    load_project_context,
    resolve_project_context,
)
from ..core.session import env_home
from ..core.slash import effective_budget_for_model
from ..core.store import MAX_AGENT_NOTIFICATION_TEXT, ConversationStore
from ..media.image_policy import image_policy_for_provider
from ..models.catalog import provider_for_model
from ..path_identity import same_physical_path
from ..project_registry import ProjectRegistry, ProjectRegistryError
from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    assistant_text,
)
from ..providers.factory import build_backend, credential_store
from ..skills.agent_catalog import load_agent
from ..tools import ToolRegistry, ToolStreamPublisher
from ..tools.agent import ChildApprovalPolicy, agent_stats
from ..tools.ask_parent import register_ask_parent
from ..tools.finish import register as register_finish
from ..tools.registry import ToolExecutionContext
from .background import finish_background_child
from .budget import MAX_AGENT_DEPTH
from .budget import child_depth as next_agent_depth
from .conversation_channel import has_follow_up_loop
from .explicit_finish import eligible as explicit_finish_eligible
from .explicit_finish import register_handoff
from .finish_gate import FinishGateControl, FinishGateState, run_finish_gate
from .presets import (
    GENERAL_PRESET,
    RUN_PRESET,
    AgentPreset,
    compose_system_prompt,
)
from .receipt import MAX_AGENT_RESULT_BYTES, TerminalState
from .run_consumer import consume_run as consume_follow_up_run
from .terminal_fence import TerminalTransitionFence

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
    update_turns: Callable[[int], None],
    update_step: Callable[[str], None],
    update_tool_calls: Callable[[int], None],
    finish_lifecycle: Callable[[str, str], dict[str, object]],
    publish_lifecycle: Callable[..., None],
    child_result: Callable[..., dict[str, object]],
    error_message: Callable[[BaseException], str],
    record_report: Callable[[str], None] | None = None,
    notification_turn: bool = False,
    finish_gate_state: FinishGateState | None = None,
    delivering_prompt_id: str | None = None,
    explicit_finish_metadata: Mapping[str, object] = _EMPTY_METADATA,
) -> dict[str, object]:
    """Consume one child loop, including nested lifecycle events."""

    final_message: Message | None = None
    failure_message: str | None = None
    tool_calls = 0
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

    def lifecycle_depth(call: ToolCall | None) -> int:
        if call is not None and call.name.casefold() == "agent":
            return child_loop.agent_depth + 1
        return child_loop.agent_depth

    async def consume_events(
        events: AsyncIterator[StreamEvent],
        gate_control: FinishGateControl | None = None,
    ) -> None:
        nonlocal failure_message, final_message, tool_calls, explicit_report
        final_message = None
        async for event in events:
            if event.type is StreamEventType.TURN_START:
                status = f"turn {child_turns() + 1}: thinking"
                update_step(status)
                publish(status)
            elif event.type is StreamEventType.TOOL_APPROVAL_START:
                name = event.tool_call.name if event.tool_call is not None else "tool"
                status = f"turn {child_turns() + 1}: approval pending: {name}"
                update_step(status)
                publish(status)
                publish_lifecycle(
                    "approval_start",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_APPROVAL_END:
                publish_lifecycle(
                    "approval_end",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_EXECUTION_START:
                tool_calls += 1
                update_tool_calls(tool_calls)
                name = event.tool_call.name if event.tool_call is not None else "tool"
                arguments = (
                    event.tool_call.arguments if event.tool_call is not None else {}
                )
                summary = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                status = f"turn {child_turns() + 1}: tool: {name} {summary}"
                update_step(status)
                publish(status)
                publish_lifecycle(
                    "execution_start",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_EXECUTION_END:
                if (
                    event.tool_call is not None
                    and event.tool_call.name == "finish"
                    and event.tool_result is not None
                    and not event.tool_result.is_error
                    and isinstance(event.tool_result.structured_content, dict)
                ):
                    explicit_report = dict(event.tool_result.structured_content)
                publish_lifecycle(
                    "execution_end",
                    event.tool_call,
                    tool_result=event.tool_result,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TURN_END:
                turns = child_turns() + 1
                update_turns(turns)
                if event.data.get("tool_calls") == 0 and event.message is not None:
                    final_message = event.message
                if gate_control is not None and gate_control.turn_ended():
                    return
            elif event.type is StreamEventType.ERROR and event.error is not None:
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
        events = (
            child_loop.run_notification_turn()
            if notification_turn
            else child_loop.run_turn(prompt, origin=origin)
        )
        await consume_events(events)
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
        children = owned_children()
        tasks = owned_tasks()
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
    except Exception as exc:  # noqa: BLE001 - child failures become receipts
        failure_message = error_message(exc)
    if failure_message is not None:
        text = f"agent error: {failure_message}"
        return terminal_result(state="failed", text=text)
    if final_message is None:
        text = "agent error: child ended without a final response"
        return terminal_result(state="failed", text=text)
    final_text = assistant_text(final_message)
    if not final_text.strip():
        if final_message.metadata.get("stop_reason") == "max_tokens":
            from ..runtime.loop.empty_turn import MAX_TOKENS_THINKING_NOTICE

            return terminal_result(state="completed", text=MAX_TOKENS_THINKING_NOTICE)
        text = "agent error: child returned an empty final assistant message"
        return terminal_result(state="failed", text=text)
    if record_report is not None:
        record_report(final_text)
    if child_loop.explicit_finish:
        return terminal_result(state="idle", text=final_text, status="running")
    return terminal_result(state="completed", text=final_text)


async def consume_run(
    child_loop: AgentLoop,
    prompt: str,
    *,
    child_store: ConversationStore,
    tool_call_id: str = "",
    max_receipt_bytes: int = MAX_AGENT_RESULT_BYTES,
    receipt_components: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Consume one child run through its follow-up and idle lifecycle."""

    return await consume_follow_up_run(
        consume_child,
        child_loop,
        prompt,
        child_store=child_store,
        tool_call_id=tool_call_id,
        max_receipt_bytes=max_receipt_bytes,
        receipt_components=receipt_components,
        **kwargs,
    )


def resolve_child_backend(
    loop: AgentLoop,
    model: object,
) -> tuple[CompletionBackend | None, str | None]:
    """Pick the backend a child runs on, returning an error message instead of raising.

    Without a model the child inherits the parent's backend, which is what every
    agent did before cross-provider spawning existed.
    """

    if model is None:
        return loop.backend, None
    if type(model) is not str or not model.strip():
        return None, "agent error: model must be a nonempty string"
    try:
        provider = provider_for_model(model)
    except ValueError as exc:
        return None, f"agent error: {exc}"
    context = getattr(loop, "account_context", None)
    account_kwargs = context.backend_kwargs(provider) if context is not None else {}
    # Check credentials up front: the alternative is a child that spawns, burns a
    # turn, and dies on an auth error the parent cannot act on.
    store = credential_store(provider, **account_kwargs)
    if store is not None:
        tokens = store.read() or store.bootstrap()
        if tokens is None:
            return None, (
                f"agent error: not logged in to {provider}; "
                f"run zeta login --provider {provider}"
            )
    try:
        child_budget = effective_budget_for_model(
            provider, model, loop.context_assembler.token_budget
        )
        backend, _ = build_backend(
            provider, model, token_budget=child_budget, **account_kwargs
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"agent error: could not start {provider} backend: {exc}"
    return backend, None


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
) -> tuple[AgentLoop, ChildApprovalPolicy | None]:
    child_registry: ToolRegistry | None = None
    child_policy: ChildApprovalPolicy | None = None
    try:
        excluded_names = {"inbox", "request_attention", "resolve_attention"}
        if child_depth == MAX_AGENT_DEPTH or not preset.allow_delegation:
            excluded_names.add("agent")
        if preset.tool_names is not None:
            allowed_names = set(preset.tool_names)
            if preset.source == "packaged" and child_depth < MAX_AGENT_DEPTH:
                allowed_names.add("agent")
            excluded_names.update(
                set(loop.tool_registry.definitions_by_name) - allowed_names
            )
        child_provider = getattr(child_backend, "provider", None)
        child_image_policy = (
            image_policy_for_provider(child_provider)
            if isinstance(child_provider, str)
            else loop.tool_registry.image_policy
        )
        child_registry = loop.tool_registry.clone_for_session(
            child_store,
            exclude_names=excluded_names,
            cwd=cwd_override,
            image_policy=child_image_policy,
        )
        account_context = getattr(loop, "account_context", None)
        if account_context is not None:
            child_registry.account_context = account_context.child_snapshot(child_provider)
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

        child_model = getattr(child_backend, "model", None)
        if type(child_model) is not str or not child_model:
            child_model = model if type(model) is str and model else "unknown"
        child_budget = loop.context_assembler.token_budget
        if isinstance(child_provider, str):
            child_budget = effective_budget_for_model(
                child_provider, child_model, child_budget
            )

        def record_child_usage(usage: Mapping[str, Any]) -> None:
            loop.context_assembler.record_descendant_usage(
                {**usage, "_zeta_model": usage.get("_zeta_model", child_model)}
            )

        explicit_finish = explicit_finish_eligible(
            global_enabled=loop.explicit_finish,
            preset=preset,
            background=follow_up_loop,
        )
        child_loop = AgentLoop(
            child_backend,
            child_store,
            registry=child_registry,
            skill_catalog=child_registry.skill_catalog,
            max_turns=None,
            token_budget=child_budget,
            retained_tail=loop.context_assembler.retained_tail,
            system_prompt=_compose_child_system_prompt(
                await asyncio.to_thread(
                    _child_base_system_prompt, loop, cwd_override, child_registry
                ),
                preset,
                explicit_finish=explicit_finish,
            ),
            agent_catalog=child_registry.agent_catalog,
            skip_mcp_mount=True,
            agent_depth=child_depth,
            agent_instance_id=child_instance_id,
            root_project_id=loop.root_project_id,
            root_session_dir=loop.root_session_dir,
            project_registry=loop.project_registry,
            background_owner=loop._background_owner,
            usage_sink=record_child_usage,
        )
        child_loop.explicit_finish = explicit_finish
        child_loop.one_shot = getattr(loop, "one_shot", False)
        child_loop.account_context = child_registry.account_context
        if child_loop.explicit_finish:
            register_finish(child_registry, child_loop=child_loop)
            register_handoff(child_loop)
        loop._background_owner.conversation_channel.register_loop(child_instance_id)
        if loop.plan_mode:
            child_loop.set_plan_mode(True)
        child_loop.set_background_event_sink(loop._publish_background_event)
        return child_loop, child_policy
    except asyncio.CancelledError:
        if child_registry is not None:
            cleanup = asyncio.create_task(child_registry.close())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await asyncio.shield(cleanup)
        raise
    except BaseException:
        if child_registry is not None:
            await child_registry.close()
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
    prompt = arguments.get("prompt")
    description = arguments.get("description")
    requested_preset = arguments.get("preset")
    legacy_agent_type = arguments.get("agent_type")
    if requested_preset is not None and legacy_agent_type is not None:
        return loop._child_result_payload(
            tool_call.id,
            "agent error: pass only one of preset and agent_type",
            state="failed",
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
        return loop._child_result_payload(
            tool_call.id,
            "agent error: prompt must be a nonempty string",
            state="failed",
        )
    if type(description) is not str or not description.strip():
        return loop._child_result_payload(
            tool_call.id,
            "agent error: description must be a nonempty string",
            state="failed",
        )
    catalog = loop.tool_registry.agent_catalog
    if type(agent_type) is not str:
        return loop._child_result_payload(
            tool_call.id,
            f"agent error: preset must be one of: {', '.join(catalog.names())}",
            state="failed",
        )
    try:
        preset = catalog.find(agent_type)
    except ValueError:
        return loop._child_result_payload(
            tool_call.id,
            f"agent error: unknown agent_type/preset {agent_type!r}; expected one of: "
            f"{', '.join(catalog.names())}",
            state="failed",
        )
    if (
        loop.plan_mode
        and preset.source == "packaged"
        and preset.name == GENERAL_PRESET.name
    ):
        return loop._child_result_payload(
            tool_call.id,
            "agent error: general agents are unavailable in plan mode; "
            "use agent_type 'explore' or 'plan'",
            state="failed",
        )
    effective_model = model if model is not None else preset.model
    child_backend, backend_error = resolve_child_backend(loop, effective_model)
    if backend_error is not None:
        return loop._child_result_payload(
            tool_call.id,
            backend_error,
            state="failed",
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
        return loop._child_result_payload(
            tool_call.id,
            "agent error: background must be a boolean",
            state="failed",
        )
    if getattr(loop, "one_shot", False):
        # Headless exits after this turn, so a background child would be canceled.
        background = False
    child_depth, nesting_error = next_agent_depth(loop.agent_depth, background)
    if nesting_error is not None:
        return loop._child_result_payload(
            tool_call.id,
            nesting_error,
            state="failed",
        )
    try:
        await loop._ensure_mcp_servers()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - setup failures become receipts
        return loop._child_result_payload(
            tool_call.id,
            f"agent error: {error_message(exc)}",
            state="failed",
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
    raw_cwd = arguments.get("cwd")
    child_cwd = str(loop.store.cwd)
    cwd_override: str | None = None
    if raw_cwd is not None:
        if type(raw_cwd) is not str or not raw_cwd.strip():
            return loop._child_result_payload(
                tool_call.id,
                "agent error: cwd must be a nonempty string",
                state="failed",
            )
        candidate = Path(raw_cwd).expanduser()
        if not candidate.is_absolute():
            candidate = Path(loop.store.cwd) / candidate
        resolved_cwd = Path(os.path.abspath(candidate))
        if not resolved_cwd.is_dir():
            return loop._child_result_payload(
                tool_call.id,
                f"agent error: cwd is not an existing directory: {resolved_cwd}",
                state="failed",
            )
        child_cwd = str(resolved_cwd)
        cwd_override = child_cwd
    child_number = loop.store.allocate_agent_index()
    parent_identity = loop.agent_instance_id or loop.store.session_id
    child_instance_id = f"{parent_identity}:{child_number}"
    agents_root = loop.store.session_dir / "agents"
    child_store = loop._background_owner.store_leases.enter_context(
        ConversationStore(
            agents_root,
            session_id=str(child_number),
            cwd=child_cwd,
        )
    )
    loop._background_owner.track_store(child_store)
    child_store.mark_agent_parent(tool_call.id, agent_type=stored_agent_type)
    child_path = str(child_store.session_dir)
    # Persist the parent marker first. A crash after this point is
    # recoverable, and the marker itself is the source of truth for the child.
    loop.store.register_agent_child(
        tool_call,
        child_session_path=child_path,
        description=description,
        agent_type=stored_agent_type,
        background=background,
        child_instance_id=child_instance_id,
        accepts_follow_ups=follow_up_loop,
        explicit_finish=explicit_finish_eligible(
            global_enabled=loop.explicit_finish,
            preset=preset,
            background=background,
        ),
    )
    if loop.root_project_id is not None and loop.project_registry is not None:
        link = {
            "project_id": loop.root_project_id,
            "session_id": child_instance_id,
            "role": "worker",
            "parent_session_id": parent_identity,
            "transcript_path": child_path,
        }
        try:
            from ..core.session_links import (
                persist_pending_child_link,
                remove_pending_child_link,
            )

            # Record the durable lineage intent in the ROOT session's pending
            # index (directory-fsynced) BEFORE the registry append, so the root's
            # reconciliation can recover this child link after a crash or a
            # failed append -- without ever walking the agents subtree.
            persist_pending_child_link(loop.root_session_dir, link)
            loop.project_registry.record_session(
                loop.root_project_id,
                session_id=child_instance_id,
                transcript_path=child_path,
                role="worker",
                parent_session_id=parent_identity,
            )
            remove_pending_child_link(loop.root_session_dir, child_instance_id)
        except (ProjectRegistryError, OSError, ValueError) as exc:
            logger.warning("could not record child project lineage: %s", exc)
    child_store.start_agent_lifecycle(
        handle=child_instance_id,
        started_at=_now(),
        depth=child_depth,
        agent_type=preset.name,
        description=description,
        cwd=child_cwd,
    )
    child_policy: ChildApprovalPolicy | None = None
    try:
        loop._agent_child_stores[tool_call.id] = child_store
        loop._agent_child_turns[tool_call.id] = 0
        loop._agent_child_types[tool_call.id] = preset.name
        child_marker_key = child_instance_id
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
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - setup failures become receipts
        loop._background_owner.conversation_channel.unregister_loop(child_instance_id)
        failure_text = f"agent error: {error_message(exc)}"
        child_store.finish_agent_lifecycle(
            "failed",
            final_result=failure_text,
            turns_used=0,
        )
        loop._background_owner.mark_store_finished(child_store)
        failure_payload = loop._child_result_payload(
            tool_call.id,
            failure_text,
            state="failed",
            child_session_path=child_path,
            agent_type=preset.name,
            child_instance_id=child_instance_id,
            depth=child_depth,
        )
        failure_content = failure_payload.get("content")
        if (
            isinstance(failure_content, list)
            and failure_content
            and isinstance(failure_content[0], dict)
            and isinstance(failure_content[0].get("text"), str)
        ):
            child_store.update_agent_lifecycle_result(
                failure_content[0]["text"], canonical_receipt=True
            )
        return failure_payload
    lifecycle_sink = (
        execution_context.lifecycle_sink if execution_context is not None else None
    )

    def publish(status: str) -> None:
        if background:
            event_data: dict[str, object] = {"stream": "stdout"}
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

    def finish_lifecycle(state: str, text: str) -> dict[str, object]:
        child_store.finish_agent_lifecycle(
            state,
            final_result=text,
            turns_used=child_turns(),
        )
        return agent_stats(child_store.agent_lifecycle(), turns_used=child_turns())

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
    ) -> dict[str, object]:
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
        if not background and state is not None:
            content = payload.get("content")
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and isinstance(content[0].get("text"), str)
            ):
                child_store.update_agent_lifecycle_result(
                    content[0]["text"], canonical_receipt=True
                )
        return payload

    def publish_lifecycle(
        kind: str,
        call: ToolCall | None,
        *,
        tool_result: ToolResult | None = None,
        depth: int | None = None,
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
        data["agent_instance_id"] = child_instance_id
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

    def update_turns(turns: int) -> None:
        loop._agent_child_turns[tool_call.id] = turns
        child_store.update_agent_lifecycle(turns_used=turns)
        loop.store.update_agent_child_turns(child_marker_key, turns)

    def update_tool_calls(tool_calls: int) -> None:
        child_store.update_agent_lifecycle(tool_calls=tool_calls)

    consume = consume_run if follow_up_loop else consume_child
    receipt_components: dict[str, str] = {}
    run_kwargs: dict[str, Any] = (
        {
            "child_store": child_store,
            "tool_call_id": tool_call.id,
            "max_receipt_bytes": getattr(
                loop.tool_registry, "max_output_chars", MAX_AGENT_RESULT_BYTES
            ),
            "receipt_components": receipt_components,
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
        }
    )
    child_task = loop._create_task(
        consume(
            child_loop,
            prompt,
            **run_kwargs,
            child_path=child_path,
            publish=publish,
            child_turns=child_turns,
            update_turns=update_turns,
            update_tool_calls=update_tool_calls,
            update_step=update_step,
            finish_lifecycle=finish_lifecycle,
            publish_lifecycle=publish_lifecycle,
            child_result=child_result,
            error_message=error_message,
        )
    )
    child_canceled = False
    terminal_fence = getattr(child_loop, "_terminal_transition_fence", None)
    if getattr(child_loop, "explicit_finish", False) and terminal_fence is None:
        terminal_fence = TerminalTransitionFence(child_store)

    def request_child_cancel() -> None:
        """Cancel this child unless a durable finish already owns termination."""
        if terminal_fence is not None and not terminal_fence.request_cancel():
            return
        child_loop.tool_registry.abort()
        child_loop.store.drop_client_steering("abort")
        if not child_task.done():
            child_task.cancel()

    async def cancel_child() -> None:
        nonlocal child_canceled
        if child_canceled:
            return
        if terminal_fence is not None and terminal_fence.accepted_report() is not None:
            return
        child_canceled = True
        loop._background_owner.cancel_subtree(child_instance_id)
        await asyncio.gather(child_task, return_exceptions=True)
        child_store.mark_agent_canceled(tool_call.id)

    if background:
        # Notification text has a stricter durable contract than the foreground
        # tool result. Size the canonical background receipt, including its
        # persisted tool-result envelope, to the smaller limit; lifecycle
        # final_result reuses this same bounded text.
        background_receipt_bytes = min(
            getattr(loop.tool_registry, "max_output_chars", MAX_AGENT_RESULT_BYTES),
            MAX_AGENT_NOTIFICATION_TEXT,
        )

        def request_background_cancel() -> None:
            request_child_cancel()

        def cleanup_background_child() -> None:
            loop._agent_child_stores.pop(tool_call.id, None)
            loop._agent_child_turns.pop(tool_call.id, None)
            loop._agent_child_types.pop(tool_call.id, None)
            loop._background_child_cancellers.pop(tool_call.id, None)
            loop._background_child_watchers.pop(tool_call.id, None)
            loop._background_owner.conversation_channel.unregister_loop(
                child_instance_id
            )
            loop._background_owner.unregister(child_instance_id)
            loop._background_owner.mark_store_finished(child_store)
            loop._background_owner.release_unused_stores()
            if child_policy is not None:
                child_policy.cleanup()

        async def close_background_child() -> tuple[str, ...]:
            return tuple(await child_loop.close(cancel_background=False))

        async def finish_background() -> None:
            try:
                await finish_background_child(
                    child_task=child_task,
                    child_store=child_store,
                    parent_store=loop.store,
                    notification_store=loop._background_owner.notification_store,
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
                    marker_key=child_marker_key,
                    agent_instance_id=loop.agent_instance_id,
                    background_owner=loop._background_owner,
                    max_receipt_bytes=background_receipt_bytes,
                    explicit_finish=terminal_fence is not None,
                )
            finally:
                loop.notify_background_persisted()

        watcher = loop._create_task(finish_background())
        loop._background_child_watchers[tool_call.id] = watcher
        loop._background_child_cancellers[tool_call.id] = request_background_cancel
        loop._background_owner.register(
            child_instance_id,
            request_background_cancel,
            watcher,
            parent_store=loop.store,
            description=description,
            active_store=child_store,
            parent_instance_id=loop.agent_instance_id,
        )
        # The tree owner now keeps this task pair alive after this loop closes.
        loop._tracked_tasks.discard(child_task)
        loop._tracked_tasks.discard(watcher)
        running_result = child_result(
            (
                f"background agent started: handle={child_instance_id} "
                f"({description}). Completion is announced automatically when "
                "idle or at the next turn boundary; use agent_status or "
                "agent_output only for on-demand inspection."
            ),
            error=False,
            status="running",
        )
        running_tool_result = validate_result(running_result, tool_call.id)
        # Keep this publication synchronous: parallel agent calls rely on task
        # dispatch order when exposing their immediate running receipts.
        loop.store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [TextContent(running_tool_result.content)],
                tool_result=running_tool_result,
            )
        )
        return running_result

    loop._background_owner.register(
        child_instance_id,
        request_child_cancel,
        child_task,
        parent_store=loop.store,
        active_store=child_store,
        parent_instance_id=loop.agent_instance_id,
    )
    abort_task = loop._create_task(abort_signal.wait())
    try:
        done, _ = await asyncio.wait(
            (child_task, abort_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        # A completed child wins a same-turn abort race so its descendants can
        # be adopted by receipt processing rather than canceled as unreachable.
        if child_task in done:
            result = child_task.result()
            if follow_up_loop:
                content = result.get("content")
                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], dict)
                    and isinstance(content[0].get("text"), str)
                ):
                    child_store.update_agent_lifecycle_result(
                        content[0]["text"], canonical_receipt=True
                    )
            return result
        if abort_task in done:
            await cancel_child()
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        await cancel_child()
        raise
    finally:
        loop._background_owner.conversation_channel.unregister_loop(child_instance_id)
        loop._background_owner.unregister(child_instance_id)
        if child_policy is not None:
            child_policy.cleanup()
        if not abort_task.done():
            abort_task.cancel()
        await asyncio.gather(abort_task, return_exceptions=True)
        await child_loop.close(cancel_background=False)


def _child_base_system_prompt(
    loop: AgentLoop,
    cwd_override: str | None,
    child_registry: ToolRegistry | None = None,
) -> str | Message:
    """Compose project context from an explicit cwd that differs from the parent."""

    if cwd_override is None or same_physical_path(cwd_override, loop.store.cwd):
        return loop.context_assembler.system_prompt
    if child_registry is not None:
        child_registry.verify_cwd_identity()
    home_hint = loop.active_home
    zeta_home = Path(home_hint) if home_hint else env_home()
    discovery = discover_project(cwd_override, user_home=zeta_home.parent)
    registry = ProjectRegistry(zeta_home / "projects")
    session_project = None
    if loop.root_project_id is not None:
        try:
            session_project = registry.show_project(loop.root_project_id)
        except ProjectRegistryError:
            pass
    cwd_project = resolve_project_context(discovery, registry)
    context = load_project_context(
        cwd=cwd_override,
        repo_root=discovery.primary_root or discovery.cwd,
        zeta_home=zeta_home,
        catalog=loop.tool_registry.skill_catalog,
        session_project=session_project,
        cwd_project=cwd_project,
        inbox_enabled=False,
    )
    return context.system_prompt


_EXPLICIT_FINISH_GUIDANCE = (
    "With explicit finish enabled, this run ends only when you call the `finish` "
    "tool with `status` and `summary` (plus `verdict` and `head_sha` when they "
    "apply). Ending a turn without calling `finish` leaves you idle for "
    "follow-ups; call it when the contract's exit criteria are met or when you "
    "are blocked."
)


def _compose_child_system_prompt(
    system_prompt: str | Message,
    preset: AgentPreset,
    *,
    explicit_finish: bool = False,
) -> str | Message:
    """Apply the preset preamble and custom body to the child prompt."""

    composed = compose_system_prompt(system_prompt, preset.preamble)
    body = load_agent(preset)
    if explicit_finish:
        body = f"{body}\n\n{_EXPLICIT_FINISH_GUIDANCE}" if body else _EXPLICIT_FINISH_GUIDANCE
    if not body:
        return composed
    if isinstance(composed, Message):
        return Message(
            MessageRole.SYSTEM,
            [*composed.content, TextContent(body)],
            metadata=dict(composed.metadata),
        )
    return composed + "\n\n" + body
