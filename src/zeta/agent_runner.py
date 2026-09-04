"""Run child-agent turns and preserve their bounded result shape."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .agent_background import finish_background_child
from .agent_budget import (
    MAX_AGENT_DEPTH,
    AgentTree,
)
from .agent_budget import child_depth as next_agent_depth
from .core.abort import AbortSignal as ToolAbortSignal
from .core.checkpoints import _now
from .core.store import ConversationStore
from .model_catalog import provider_for_model
from .providers.factory import build_backend, credential_store
from .tools import ToolStreamPublisher
from .tools.agent import ChildApprovalPolicy, agent_stats
from .tools.agent_presets import (
    GENERAL_PRESET,
    agent_type_names,
    compose_system_prompt,
    get_agent_preset,
)
from .tools.registry import ToolExecutionContext
from .types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    assistant_text,
)

if TYPE_CHECKING:
    from .loop import AgentLoop


async def consume_child(
    child_loop: AgentLoop,
    prompt: str,
    *,
    turn_cap: int,
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
) -> dict[str, object]:
    """Consume one child loop, including nested lifecycle events."""

    final_message: Message | None = None
    last_assistant_text = ""
    cap_hit = False
    failure_message: str | None = None
    budget_exhausted = False
    tool_calls = 0

    def terminal_result(
        *, state: str, text: str, error: bool, **result_options: object
    ) -> dict[str, object]:
        stats = finish_lifecycle(state, text)
        return child_result(
            text,
            error=error,
            stats=stats,
            **result_options,
        )

    def lifecycle_depth(call: ToolCall | None) -> int:
        if call is not None and call.name.casefold() == "agent":
            return child_loop.agent_depth + 1
        return child_loop.agent_depth

    try:
        async for event in child_loop.run_turn(prompt):
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
                publish_lifecycle(
                    "execution_end",
                    event.tool_call,
                    tool_result=event.tool_result,
                    depth=lifecycle_depth(event.tool_call),
                )
                if (
                    event.tool_result is not None
                    and event.tool_result.structured_content is not None
                    and event.tool_result.structured_content.get("error_code")
                    == "agent_turn_budget"
                ):
                    budget_exhausted = True
                    failure_message = event.tool_result.content
            elif event.type is StreamEventType.TURN_END:
                turns = child_turns() + 1
                update_turns(turns)
                if event.message is not None:
                    last_assistant_text = _assistant_text_snippet(event.message)
                if event.data.get("tool_calls") == 0 and event.message is not None:
                    final_message = event.message
            elif event.type is StreamEventType.ERROR and event.error is not None:
                if event.error.code == "agent_turn_budget":
                    budget_exhausted = True
                    failure_message = event.error.message
                elif event.error.code == "max_turns":
                    cap_hit = True
                else:
                    failure_message = event.error.message
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure_message = error_message(exc)
    if budget_exhausted:
        text = f"agent error: {failure_message or 'shared agent turn budget exhausted'}"
        return terminal_result(
            state="failed",
            text=text,
            error=True,
            budget_exhausted=True,
        )
    if cap_hit:
        text = (
            f"agent error: child reached the {turn_cap}-turn cap; "
            f"partial state is saved at {child_path}; "
            f"last assistant text: {last_assistant_text or '[none]'}; "
            f"turns used: {child_turns()}"
        )
        return terminal_result(
            state="failed",
            text=text,
            error=True,
        )
    if failure_message is not None:
        text = f"agent error: {failure_message}"
        return terminal_result(
            state="failed", text=text, error=True
        )
    if final_message is None:
        text = "agent error: child ended without a final response"
        return terminal_result(
            state="failed", text=text, error=True
        )
    final_text = assistant_text(final_message)
    if not final_text.strip():
        text = "agent error: child returned an empty final assistant message"
        return terminal_result(
            state="failed", text=text, error=True
        )
    return terminal_result(
        state="completed", text=final_text, error=False
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
    # Check credentials up front: the alternative is a child that spawns, burns a
    # turn, and dies on an auth error the parent cannot act on.
    store = credential_store(provider)
    if store is not None:
        tokens = store.read()
        if tokens is None or not tokens.is_valid():
            return None, (
                f"agent error: not logged in to {provider}; "
                f"run zeta login --provider {provider}"
            )
    try:
        backend, _ = build_backend(provider, model)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"agent error: could not start {provider} backend: {exc}"
    return backend, None


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
    agent_type = arguments.get("agent_type", GENERAL_PRESET.name)
    model = arguments.get("model")
    # A run on someone else's model is the long kind, so it defaults to
    # background; an explicit background argument still wins.
    background = arguments.get("background", model is not None)
    if type(prompt) is not str or not prompt.strip():
        return loop._child_result_payload(
            tool_call.id,
            "agent error: prompt must be a nonempty string",
            error=True,
        )
    if type(description) is not str or not description.strip():
        return loop._child_result_payload(
            tool_call.id,
            "agent error: description must be a nonempty string",
            error=True,
        )
    if type(background) is not bool:
        return loop._child_result_payload(
            tool_call.id,
            "agent error: background must be a boolean",
            error=True,
        )
    preset = get_agent_preset(agent_type)
    if preset is None:
        return loop._child_result_payload(
            tool_call.id,
            "agent error: unknown agent_type "
            f"{agent_type!r}; expected one of: {', '.join(agent_type_names())}",
            error=True,
        )
    if loop.plan_mode and preset.name == GENERAL_PRESET.name:
        return loop._child_result_payload(
            tool_call.id,
            "agent error: general agents are unavailable in plan mode; "
            "use agent_type 'explore' or 'plan'",
            error=True,
        )
    child_backend, backend_error = resolve_child_backend(loop, model)
    if backend_error is not None:
        return loop._child_result_payload(
            tool_call.id,
            backend_error,
            error=True,
        )
    child_depth, nesting_error = next_agent_depth(loop.agent_depth, background)
    if nesting_error is not None:
        return loop._child_result_payload(
            tool_call.id,
            nesting_error,
            error=True,
        )
    agent_tree = loop._agent_tree or AgentTree()
    agent_tree.ensure_budget(
        loop._agent_turn_budget
        if loop._agent_turn_budget is not None
        else preset.turn_cap
    )
    await loop._ensure_mcp_servers()
    stored_agent_type = None if preset.name == GENERAL_PRESET.name else preset.name
    child_number = loop.store.allocate_agent_index()
    agents_root = loop.store.session_dir / "agents"
    child_store = ConversationStore(
        agents_root,
        session_id=str(child_number),
        cwd=loop.store.cwd,
    )
    child_store.mark_agent_parent(tool_call.id, agent_type=stored_agent_type)
    child_path = str(child_store.session_dir)
    child_instance_id = (
        f"{loop.agent_instance_id}:{child_number}"
        if loop.agent_instance_id is not None
        else f"{loop.store.session_id}:{child_number}"
    )
    loop.store.register_agent_child(
        tool_call,
        child_session_path=child_path,
        description=description,
        agent_type=stored_agent_type,
        background=background,
        child_instance_id=(child_instance_id if child_depth > 1 else None),
    )
    child_store.start_agent_lifecycle(
        handle=child_instance_id,
        started_at=_now(),
        tree_budget=agent_tree.budget.limit,
        depth=child_depth,
        agent_type=preset.name,
        description=description,
    )
    loop._agent_child_stores[tool_call.id] = child_store
    loop._agent_child_turns[tool_call.id] = 0
    loop._agent_child_types[tool_call.id] = preset.name
    child_marker_key = child_instance_id if child_depth > 1 else tool_call.id
    if publisher is not None:
        publisher.set_metadata({"child_session_path": child_path, "depth": child_depth})
    excluded_names = {"agent"} if child_depth == MAX_AGENT_DEPTH else set()
    if preset.tool_names is not None:
        allowed_names = set(preset.tool_names)
        if child_depth < MAX_AGENT_DEPTH:
            allowed_names.add("agent")
        excluded_names.update(
            set(loop.tool_registry.definitions_by_name) - allowed_names
        )
    child_registry = loop.tool_registry.clone_for_session(
        child_store,
        exclude_names=excluded_names,
    )
    parent_policy = loop.tool_registry.approval_policy
    child_policy: ChildApprovalPolicy | None = None
    if parent_policy is not None:
        child_policy = ChildApprovalPolicy(
            parent_policy,
            child_store,
            description,
            child_instance_id,
        )
        child_registry.set_approval_policy(child_policy)
    from .loop import AgentLoop

    child_loop = AgentLoop(
        child_backend,
        child_store,
        registry=child_registry,
        max_turns=preset.turn_cap,
        token_budget=loop.context_assembler.token_budget,
        retained_tail=loop.context_assembler.retained_tail,
        system_prompt=compose_system_prompt(
            loop.context_assembler.system_prompt,
            preset.preamble,
        ),
        skip_mcp_mount=True,
        agent_depth=child_depth,
        agent_instance_id=child_instance_id,
        agent_tree=agent_tree,
        background_owner=loop._background_owner,
    )
    if loop.plan_mode:
        child_loop.set_plan_mode(True)
    child_loop.set_background_event_sink(loop._publish_background_event)
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
        return agent_stats(
            child_store.agent_lifecycle(),
            turns_used=child_turns(),
        )

    def child_result(
        text: str,
        *,
        error: bool,
        status: str | None = None,
        budget_exhausted: bool = False,
        stats: dict[str, object] | None = None,
        include_stats: bool = not background,
    ) -> dict[str, object]:
        return loop._child_result_payload(
            tool_call.id,
            text,
            error=error,
            child_session_path=child_path,
            agent_type=preset.name,
            status=status,
            child_instance_id=child_instance_id,
            description=description if background else None,
            depth=child_depth,
            budget_exhausted=budget_exhausted,
            stats=stats,
            include_stats=include_stats,
        )

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

    child_task = loop._create_task(
        consume_child(
            child_loop,
            prompt,
            turn_cap=preset.turn_cap,
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

    async def cancel_child() -> None:
        nonlocal child_canceled
        if child_canceled:
            return
        child_canceled = True
        child_loop.abort()
        if not child_task.done():
            child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)
        child_store.mark_agent_canceled(tool_call.id)

    if background:

        def request_background_cancel() -> None:
            child_loop.abort()
            if not child_task.done():
                child_task.cancel()

        def cleanup_background_child() -> None:
            loop._agent_child_stores.pop(tool_call.id, None)
            loop._agent_child_turns.pop(tool_call.id, None)
            loop._agent_child_types.pop(tool_call.id, None)
            loop._background_child_cancellers.pop(tool_call.id, None)
            loop._background_child_watchers.pop(tool_call.id, None)
            loop._background_owner.unregister(child_instance_id)
            if child_policy is not None:
                child_policy.cleanup()

        async def finish_background() -> None:
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
                build_result=lambda text, error, status, stats: child_result(
                    text, error=error, status=status, stats=stats, include_stats=True
                ),
                validate_result=validate_result,
                publish_event=loop._publish_background_event,
                cleanup=cleanup_background_child,
                close_child=lambda: child_loop.close(cancel_background=False),
                error_message=error_message,
                marker_key=child_marker_key,
                agent_instance_id=loop.agent_instance_id,
                background_owner=loop._background_owner,
            )

        watcher = loop._create_task(finish_background())
        loop._background_child_watchers[tool_call.id] = watcher
        loop._background_child_cancellers[tool_call.id] = request_background_cancel
        loop._background_owner.register(
            child_instance_id,
            request_background_cancel,
            watcher,
            parent_store=loop.store,
            description=description,
        )
        # The tree owner now keeps this task pair alive after this loop closes.
        loop._tracked_tasks.discard(child_task)
        loop._tracked_tasks.discard(watcher)
        running_result = child_result(
            f"background agent started: {description}",
            error=False,
            status="running",
        )
        running_tool_result = validate_result(running_result, tool_call.id)
        loop.store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [TextContent(running_tool_result.content)],
                tool_result=running_tool_result,
            )
        )
        return running_result

    abort_task = loop._create_task(abort_signal.wait())
    try:
        done, _ = await asyncio.wait(
            (child_task, abort_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            await cancel_child()
            raise asyncio.CancelledError()
        result = child_task.result()
        return result
    except asyncio.CancelledError:
        await cancel_child()
        raise
    finally:
        if child_policy is not None:
            child_policy.cleanup()
        if not abort_task.done():
            abort_task.cancel()
        await asyncio.gather(abort_task, return_exceptions=True)
        await child_loop.close(cancel_background=False)


def _assistant_text_snippet(message: Message) -> str:
    text = assistant_text(message).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= 160 else f"{text[:157]}..."
