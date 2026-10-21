"""Follow-up and idle lifecycle for one background child run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..core.checkpoints import _now
from ..core.store import ConversationStore
from ..protocol.types import MessageOrigin
from ..tools.agent import agent_stats
from .explicit_finish import IdleNudgeLifecycle, kill_tasks_after_deadline
from .finish_gate import FinishGateState
from .receipt import (
    MAX_AGENT_RESULT_BYTES,
    TerminalState,
    _without_agent_receipt_suffix,
    build_agent_receipt,
)

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop


_IMMEDIATE_FINISH_REMINDER = (
    "you ended your turn without calling `finish`. call `finish` now with your report, "
    "or continue the work."
)
_IDLE_NUDGE_PROMPT = (
    "you have been idle for 10 minutes. call `finish` if the contract is complete, "
    "call `ask_parent` if you are waiting on a decision, or continue the work."
)


def _transition_running(child_loop: AgentLoop, child_store: ConversationStore) -> None:
    if getattr(child_loop, "explicit_finish", False):
        child_store.update_agent_lifecycle(
            state="running",
            current_step="running follow-up",
        )


def _transition_idle(child_store: ConversationStore) -> None:
    child_store.update_agent_lifecycle(
        state="idle",
        current_step="idle, waiting for follow-up",
        idle_since=_now(),
    )


def _finalize_result(
    result: dict[str, object],
    *,
    segment_reports: list[str],
    receipt_components: dict[str, str] | None,
    delivered_follow_up: bool,
    child_store: ConversationStore,
    child_turns: object,
    tool_call_id: str,
    max_receipt_bytes: int,
) -> dict[str, object]:
    if isinstance(result.get("explicitFinishReport"), dict):
        return result
    if segment_reports and receipt_components is not None:
        receipt_components["report"] = segment_reports[
            -2 if len(segment_reports) > 1 else -1
        ]
        if len(segment_reports) > 1:
            receipt_components["reply"] = segment_reports[-1]
    state: TerminalState = "failed" if result.get("isError") else "completed"
    raw_text = ""
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        raw_text = str(content[0].get("text", ""))
    if not segment_reports and raw_text:
        segment_reports.append(_without_agent_receipt_suffix(raw_text))
    report = (
        segment_reports[-2 if len(segment_reports) > 1 else -1]
        if receipt_components is not None
        and segment_reports
        and (len(segment_reports) > 1 or delivered_follow_up)
        else ""
    )
    reply = (
        segment_reports[-1]
        if segment_reports and len(segment_reports) > 1
        else raw_text
    )
    return build_agent_receipt(
        state,
        reply,
        agent_stats(
            child_store.agent_lifecycle(),
            status=state,
            turns_used=child_turns() if callable(child_turns) else 0,
        ),
        structured_content=(
            dict(result["structuredContent"])
            if isinstance(result.get("structuredContent"), dict)
            else None
        ),
        tool_call_id=tool_call_id,
        max_bytes=max_receipt_bytes,
        report=report,
        reply=reply,
    )


async def consume_run(
    consume_child: Callable[..., Awaitable[dict[str, object]]],
    child_loop: AgentLoop,
    prompt: str,
    *,
    child_store: ConversationStore,
    tool_call_id: str = "",
    max_receipt_bytes: int = MAX_AGENT_RESULT_BYTES,
    receipt_components: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Consume a run, delivering queued follow-ups at each turn boundary.

    run_turn is one-shot, so a follow-up cannot be injected into a turn already
    in flight -- splicing a user message between a tool call and its result
    would not survive the provider's message shape. Instead the run takes
    another turn on the same loop once the current one ends, which is how the
    composer already delivers the follow-ups you type while a turn runs.

    The run finishes when it has nothing left to do and nothing queued.
    """

    finish_lifecycle = kwargs.get("finish_lifecycle")
    child_turns = kwargs.get("child_turns")
    final_finish_lifecycle = finish_lifecycle
    idle_lifecycle = IdleNudgeLifecycle(
        clock=kwargs.pop("idle_clock", None) or asyncio.get_running_loop().time,
        wait=kwargs.pop("idle_wait", None),
        nudge_seconds=kwargs.pop("idle_nudge_seconds", None),
    )

    def keep_lifecycle_running(state: str, text: str) -> dict[str, object]:
        if not callable(child_turns):
            return {}
        idle = state == "idle"
        if idle:
            _transition_idle(child_store)
        return agent_stats(
            child_store.agent_lifecycle(),
            status="idle" if idle else "running",
            turns_used=child_turns(),
        )

    if callable(finish_lifecycle):
        kwargs["finish_lifecycle"] = keep_lifecycle_running
    result: dict[str, object] | None = None
    terminal_result: dict[str, object] | None = None
    current_entry = None
    segment_reports: list[str] = []
    kwargs["record_report"] = segment_reports.append
    kwargs["finish_gate_state"] = FinishGateState()

    safety_started: float | None = None
    idle_published = False
    safety_task: asyncio.Task[tuple[str, ...]] | None = None
    immediate_reminded = False
    try:
        result = await consume_child(
            child_loop, prompt, origin=MessageOrigin.AGENT_PROMPT, **kwargs
        )
        while not result.get("isError"):
            structured = result.get("structuredContent")
            idle = (
                getattr(child_loop, "explicit_finish", False)
                and isinstance(structured, dict)
                and structured.get("status") == "running"
            )
            if idle and safety_started is None:
                safety_started = asyncio.get_running_loop().time()
            # Ack only after a follow-up actually reached the child. A cancel or
            # backend error mid-consume_child leaves the prompt pending, so the
            # next run gets a chance to redeliver it instead of it silently gone.
            if current_entry is not None:
                child_store.acknowledge_pending_prompt(current_entry.id)
                current_entry = None
            pending = (
                child_store.close_pending_queue_if_empty()
                if child_store.pending_prompts()
                else []
            )
            if pending:
                immediate_reminded = False
                idle_lifecycle.reset_idle_period()
                current_entry = pending[0]
                kwargs["delivering_prompt_id"] = current_entry.id
                child_loop._delivering_prompt_id = current_entry.id
                _transition_running(child_loop, child_store)
                result = await consume_child(
                    child_loop,
                    current_entry.data["text"],
                    origin=MessageOrigin.AGENT_SEND,
                    **kwargs,
                )
                idle_published = False
                continue
            kwargs["delivering_prompt_id"] = None
            child_loop._delivering_prompt_id = None
            if child_loop.notification_wake.pending_message() is not None:
                immediate_reminded = False
                idle_lifecycle.reset_idle_period()
                _transition_running(child_loop, child_store)
                result = await consume_child(
                    child_loop,
                    "",
                    origin=MessageOrigin.NOTIFICATION,
                    notification_turn=True,
                    **kwargs,
                )
                idle_published = False
                continue
            if idle:
                content = result.get("content")
                ended_with_text = (
                    isinstance(content, list)
                    and bool(content)
                    and isinstance(content[0], dict)
                    and bool(str(content[0].get("text", "")).strip())
                )
                if (
                    not immediate_reminded
                    and ended_with_text
                    and await idle_lifecycle.reserve_immediate_reminder(
                        child_loop._terminal_transition_fence
                    )
                ):
                    immediate_reminded = True
                    _transition_running(child_loop, child_store)
                    result = await consume_child(
                        child_loop,
                        _IMMEDIATE_FINISH_REMINDER,
                        origin=MessageOrigin.AGENT_PROMPT,
                        **kwargs,
                    )
                    idle_published = False
                    continue
                if not idle_published:
                    content = result.get("content")
                    text = "child is idle"
                    if (
                        isinstance(content, list)
                        and content
                        and isinstance(content[0], dict)
                        and isinstance(content[0].get("text"), str)
                    ):
                        text = content[0]["text"] or text
                    metadata = kwargs.get("explicit_finish_metadata")
                    child_path = kwargs.get("child_path")
                    if isinstance(metadata, dict) and isinstance(child_path, str):
                        child_loop._background_owner.conversation_channel.publish_idle(
                            child_instance_id=str(metadata["child_instance_id"]),
                            child_session_path=child_path,
                            description=str(metadata["description"]),
                            text=text,
                        )
                    idle_published = True
                running_tasks = any(
                    task.running
                    for task in child_loop.tool_registry.background_tasks.snapshot()
                )
                if running_tasks and safety_task is None:
                    elapsed = asyncio.get_running_loop().time() - safety_started
                    timeout = max(
                        0.0,
                        child_loop._background_owner.finish_gate_task_timeout - elapsed,
                    )
                    safety_task = asyncio.create_task(
                        kill_tasks_after_deadline(
                            child_loop.tool_registry.background_tasks, timeout
                        )
                    )
                action = await idle_lifecycle.wait_for_idle_action(
                    lambda: child_loop._background_owner.conversation_channel.wait(
                        child_loop.agent_instance_id
                    ),
                    child_loop._terminal_transition_fence,
                )
                if action == "external_input":
                    continue
                _transition_running(child_loop, child_store)
                result = await consume_child(
                    child_loop,
                    _IDLE_NUDGE_PROMPT,
                    origin=MessageOrigin.AGENT_PROMPT,
                    **kwargs,
                )
                idle_published = False
                continue
            pending = child_store.close_pending_queue_if_empty()
            if pending:
                continue
            terminal_result = _finalize_result(
                result,
                segment_reports=segment_reports,
                receipt_components=receipt_components,
                delivered_follow_up=current_entry is not None,
                child_store=child_store,
                child_turns=child_turns,
                tool_call_id=tool_call_id,
                max_receipt_bytes=max_receipt_bytes,
            )
            return terminal_result
        terminal_result = _finalize_result(
            result,
            segment_reports=segment_reports,
            receipt_components=receipt_components,
            delivered_follow_up=current_entry is not None,
            child_store=child_store,
            child_turns=child_turns,
            tool_call_id=tool_call_id,
            max_receipt_bytes=max_receipt_bytes,
        )
        return terminal_result
    finally:
        if safety_task is not None and not safety_task.done():
            safety_task.cancel()
        if safety_task is not None:
            await asyncio.gather(safety_task, return_exceptions=True)
        try:
            child_store.close_pending_queue()
        finally:
            if terminal_result is not None and callable(final_finish_lifecycle):
                content = terminal_result.get("content")
                text = (
                    content[0].get("text")
                    if (
                        isinstance(content, list)
                        and content
                        and isinstance(content[0], dict)
                        and isinstance(content[0].get("text"), str)
                    )
                    else "agent run ended without a final response"
                )
                final_finish_lifecycle(
                    "failed" if terminal_result.get("isError") else "completed",
                    text
                    if receipt_components is not None
                    else _without_agent_receipt_suffix(text),
                )
                if receipt_components is not None:
                    child_store.update_agent_lifecycle_result(
                        text, canonical_receipt=True
                    )
