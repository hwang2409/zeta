"""Shared lifecycle helpers for background agent children."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.agent_progress import live_agent_lifecycle, publish_child_ended
from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)
from . import conversation_channel
from .background_owner import BackgroundAgentOwner
from .child_run_owner import ChildRunTransitionOwner
from .conversation_channel import recover_orphaned_questions
from .finish_gate import FinishGateState
from .pending_delivery import NotificationInput
from .receipt import (
    MAX_AGENT_RESULT_BYTES,
    TerminalState,
    agent_stats,
    build_agent_receipt,
    build_explicit_finish_receipt,
    child_cleanup_details,
    lifecycle_tool_result,
    receipt_tool_result,
)
from .recovery_lease import AgentLoopForRecovery, acquire_recovery_subtree
from .turn_limit import (
    background_turn_limit_state,
    project_payload,
    write_background_turn_limit,
)


def _nested_canceled_result(marker: dict[str, object]) -> ToolResult:
    structured_content: dict[str, object] = {
        "turns_used": marker.get("turns_used", 0),
        "child_session_path": marker["child_session_path"],
    }
    if type(marker.get("agent_type")) is str:
        structured_content["agent_type"] = marker["agent_type"]
    child_instance_id = marker.get("child_instance_id")
    if type(child_instance_id) is str:
        structured_content["child_instance_id"] = child_instance_id
    result = build_agent_receipt(
        "canceled",
        "tool execution canceled",
        agent_stats(
            None,
            status="canceled",
            turns_used=marker.get("turns_used", 0)
            if type(marker.get("turns_used", 0)) is int
            else 0,
        ),
        structured_content=structured_content,
    )
    return receipt_tool_result(ToolCall.from_dict(marker["tool_call"]).id, result)

async def _complete_despite_cancellation(task: asyncio.Task[Any]) -> bool:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
    task.result()
    return cancelled

@dataclass(frozen=True)
class _BackgroundTerminal:
    status: str
    text: str
    stats: dict[str, object]
    payload: dict[str, object]
    report: dict[str, object] | None
    killed_task_ids: object = None
    killed_task_count: object = None
    killed_task_ids_truncated: object = False

def _background_terminal(
    *,
    lifecycle: dict[str, object] | None,
    report: dict[str, object] | None,
    turns_used: int,
    child_instance_id: str,
    fallback_text: str,
    structured_content: dict[str, object] | None = None,
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
) -> _BackgroundTerminal:
    """Build one canonical terminal result for live and recovered children."""

    lifecycle_state = lifecycle.get("state") if lifecycle else None
    status = (
        "error"
        if report is not None and report["status"] == "failed"
        else "completed"
        if report is not None
        else {
            "completed": "completed",
            "failed": "error",
            "canceled": "canceled",
        }.get(lifecycle_state, "canceled")
    )
    text = (
        lifecycle.get("final_result") if lifecycle is not None else None
    )
    if type(text) is not str or not text:
        text = fallback_text
    stats = agent_stats(lifecycle, status=status, turns_used=turns_used)
    state: TerminalState = (
        "completed"
        if status == "completed"
        else "canceled"
        if status == "canceled"
        else "failed"
    )
    payload = (
        build_explicit_finish_receipt(
            report,
            stats,
            structured_content=structured_content,
            child_instance_id=child_instance_id,
            max_bytes=max_bytes,
        )
        if report is not None
        else build_agent_receipt(state, text, stats)
    )
    if (
        report is None
        and lifecycle is not None
        and lifecycle.get("final_result_is_receipt") is True
    ):
        payload["content"][0]["text"] = text
    bounded_report = None
    if report is not None:
        structured = payload["structuredContent"]
        bounded_report = {"status": structured["report_status"], "summary": structured["summary"]}
        for key in ("head_sha", "verdict"):
            if key in structured:
                bounded_report[key] = structured[key]
    return _BackgroundTerminal(
        status=status,
        text=payload["content"][0]["text"],
        stats=stats,
        payload=payload,
        report=bounded_report,
        killed_task_ids=lifecycle.get("killed_task_ids") if lifecycle else None,
        killed_task_count=lifecycle.get("killed_task_count") if lifecycle else None,
        killed_task_ids_truncated=(
            lifecycle.get("killed_task_ids_truncated", False) if lifecycle else False
        ),
    )

def _recovered_completion_input(
    child_instance_id: str,
    *,
    child_session_path: str,
    description: str,
    status: str,
    text: str,
    stats: dict[str, object] | None = None,
    killed_task_ids: list[str] | None = None,
    killed_task_count: int | None = None,
    killed_task_ids_truncated: bool = False,
    report: dict[str, object] | None = None,
) -> NotificationInput:
    data: dict[str, object] = {
        "child_instance_id": child_instance_id,
        "child_session_path": child_session_path,
        "description": description,
        "status": status,
        "text": text,
    }
    if stats is not None:
        data["stats"] = stats
    if killed_task_ids is not None:
        data["killed_task_ids"] = killed_task_ids
    if killed_task_count is not None:
        data["killed_task_count"] = killed_task_count
    if killed_task_ids_truncated:
        data["killed_task_ids_truncated"] = True
    if report is not None:
        data["report_status"] = report["status"]
        for key in ("summary", "head_sha", "verdict"):
            if key in report:
                data[key] = report[key]
    return NotificationInput(
        "child_completion", data, dedupe_key=child_instance_id
    )

def _settle_recovered_child_from_receipt(
    owner: ChildRunTransitionOwner, result: ToolResult
) -> None:
    status = (
        "canceled" if result.is_canceled else "error" if result.is_error else "completed"
    )
    owner.repair_terminal(status, result.content)


def _recover_nested_children(
    store: ConversationStore,
    history_store: ConversationStore,
    notification_store: ConversationStore,
    recovery_stores: dict[Path, ConversationStore],
    home_epoch: int,
) -> None:
    """Cancel descendants left behind when an ancestor session exits.

    ``history_store`` is the root session store. Serve writes every delegated
    ``approval_history`` row (at any depth) to that one store, so the delegated
    end for a grandchild is repaired there, not against the intermediate parent.
    """

    from ..runtime.approval_recovery import repair_delegated_child

    child_markers = store.agent_children()
    if not child_markers:
        return
    for marker_key, marker in child_markers.items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        child_store = recovery_stores.get(child_path)
        child_instance_id = marker.get(
            "child_instance_id", f"{store.session_id}:{child_path.name}"
        )
        with nullcontext():
            if child_store is not None:
                _recover_nested_children(
                    child_store,
                    history_store,
                    notification_store,
                    recovery_stores,
                    home_epoch,
                )
                # Post-order: the child's own approvals are closed and any
                # missing delegated end is written on the root session history
                # before this marker is cleaned up.
                repair_delegated_child(
                    history_store, child_store, child_instance_id, home_epoch
                )
            existing_result = store.tool_result(tool_call.id)
            run_owner = (
                ChildRunTransitionOwner(child_store, tool_call.id)
                if child_store is not None else None
            )
            if marker.get("background"):
                terminal = child_store.agent_lifecycle() if child_store else None
                accepted_report = (
                    run_owner.recovered_report()
                    if run_owner is not None
                    else None
                )
                reason = (
                    "finished"
                    if accepted_report is not None
                    or terminal
                    and terminal.get("state") == "completed"
                    else "canceled"
                )
                store.close_child_questions(child_instance_id, reason=reason)
                if notification_store is not store:
                    notification_store.close_child_questions(
                        child_instance_id, reason=reason
                    )
                existing = _recover_completion_notification(
                    store,
                    notification_store,
                    child_instance_id,
                    child_lifecycle=terminal,
                )
                if existing is None:
                    lifecycle = (
                        child_store.agent_lifecycle()
                        if child_store is not None
                        else None
                    )
                    terminal = _background_terminal(
                        lifecycle=lifecycle,
                        report=accepted_report,
                        turns_used=marker.get("turns_used", 0),
                        child_instance_id=child_instance_id,
                        fallback_text=(
                            "background child canceled when the parent session exited"
                        ),
                    )
                    if run_owner is not None:
                        run_owner.repair_terminal(terminal.status, terminal.text)
                    _recover_completion_notification(
                        store,
                        notification_store,
                        child_instance_id,
                        _recovered_completion_input(
                            child_instance_id,
                            child_session_path=str(child_path),
                            description=marker["description"],
                            status=terminal.status,
                            text=terminal.text,
                            stats=terminal.stats,
                            report=terminal.report,
                        ),
                        child_lifecycle=(
                            child_store.agent_lifecycle() if child_store is not None else None
                        ),
                    )
                elif run_owner is not None:
                    run_owner.repair_terminal(
                        existing.data["status"],
                        str(existing.data["text"]),
                    )
            elif existing_result is not None:
                if run_owner is not None:
                    _settle_recovered_child_from_receipt(run_owner, existing_result)
            else:
                recovered = (
                    lifecycle_tool_result(tool_call, child_store, child_path)
                    if child_store is not None
                    else None
                )
                result = recovered or _nested_canceled_result(marker)
                if run_owner is not None:
                    _settle_recovered_child_from_receipt(run_owner, result)
                receipt = store.append_message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        [
                            TextContent(
                                result.content if recovered else "tool execution canceled"
                            )
                        ],
                        tool_result=result,
                    )
                )
                _publish_recovered_end(child_store, receipt)
            store.finish_agent_child(marker_key)

def _recover_completion_notification(
    owner_store: ConversationStore,
    notification_store: ConversationStore,
    child_instance_id: str,
    fallback: NotificationInput | None = None,
    *,
    child_lifecycle: object = None,
) -> ConversationEntry | None:
    return conversation_channel.recover_completion_mirrors(
        notification_store,
        owner_store,
        child_instance_id,
        dict(fallback.data) if fallback is not None else None,
        child_lifecycle,
    )

def _publish_recovered_end(
    child_store: ConversationStore | None, receipt: ConversationEntry
) -> None:
    if child_store is not None and (handle := child_store.agent_handle()) is not None:
        publish_child_ended(handle, receipt.seq, child_store.agent_lifecycle())


def recover_agent_children(loop: AgentLoopForRecovery, home_epoch: int) -> None:
    """Resolve child markers left by a process exit before resuming."""

    _recover_agent_children(loop, home_epoch)

def _recover_agent_children(
    loop: AgentLoopForRecovery,
    home_epoch: int,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    child_markers = loop.store.agent_children()
    if not child_markers:
        recover_orphaned_questions(loop.store)
        return
    notification_store = loop._background_owner.notification_store
    busy_child_subtrees: set[str] = set()
    for marker_key, marker in child_markers.items():
        if should_stop is not None and should_stop():
            return
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        child_instance_id = marker.get(
            "child_instance_id", f"{loop.store.session_id}:{child_path.name}"
        )
        recovery_stores: dict[Path, ConversationStore] = {}
        with ExitStack() as recovery_leases:
            if not acquire_recovery_subtree(
                loop.store, child_path, recovery_leases, recovery_stores
            ):
                busy_child_subtrees.add(child_instance_id)
                continue
            child_store = recovery_stores.get(child_path)
            if child_store is not None:
                _recover_nested_children(
                    child_store,
                    loop.store,
                    notification_store,
                    recovery_stores,
                    home_epoch,
                )
                # Close this child's own approvals and write any missing
                # delegated end on the root store before marker cleanup.
                from ..runtime.approval_recovery import repair_delegated_child

                repair_delegated_child(
                    loop.store, child_store, child_instance_id, home_epoch
                )
            run_owner = (
                ChildRunTransitionOwner(child_store, tool_call.id)
                if child_store is not None else None
            )
            if marker.get("background"):
                terminal = child_store.agent_lifecycle() if child_store else None
                accepted_report = (
                    run_owner.recovered_report()
                    if run_owner is not None
                    else None
                )
                reason = (
                    "finished"
                    if accepted_report is not None
                    or terminal
                    and terminal.get("state") == "completed"
                    else "canceled"
                )
                loop.store.close_child_questions(child_instance_id, reason=reason)
                if notification_store is not loop.store:
                    notification_store.close_child_questions(
                        child_instance_id, reason=reason
                    )
                notification = _recover_completion_notification(
                    loop.store,
                    notification_store,
                    child_instance_id,
                    child_lifecycle=terminal,
                )
                if notification is None:
                    lifecycle = (
                        child_store.agent_lifecycle()
                        if child_store is not None
                        else None
                    )
                    terminal = _background_terminal(
                        lifecycle=lifecycle,
                        report=accepted_report,
                        turns_used=marker.get("turns_used", 0),
                        child_instance_id=child_instance_id,
                        fallback_text=(
                            "background child canceled when the parent session exited"
                        ),
                    )
                    if run_owner is not None:
                        run_owner.repair_terminal(terminal.status, terminal.text)
                    _recover_completion_notification(
                        loop.store,
                        notification_store,
                        child_instance_id,
                        _recovered_completion_input(
                            child_instance_id,
                            child_session_path=marker["child_session_path"],
                            description=marker["description"],
                            status=terminal.status,
                            text=terminal.text,
                            stats=terminal.stats,
                            killed_task_ids=terminal.killed_task_ids,
                            killed_task_count=terminal.killed_task_count,
                            killed_task_ids_truncated=terminal.killed_task_ids_truncated,
                            report=terminal.report,
                        ),
                        child_lifecycle=(
                            child_store.agent_lifecycle() if child_store is not None else None
                        ),
                    )
                elif run_owner is not None:
                    run_owner.repair_terminal(
                        notification.data["status"],
                        str(notification.data["text"]),
                    )
                loop.store.finish_agent_child(marker_key)
                continue
            existing_result = loop._existing_tool_result(tool_call.id)
            if existing_result is not None:
                if run_owner is not None:
                    _settle_recovered_child_from_receipt(run_owner, existing_result)
                loop.store.finish_agent_child(marker_key)
                continue
            recovered = (
                lifecycle_tool_result(tool_call, child_store, child_path)
                if child_store is not None
                else None
            )
            result = recovered or loop._canceled_agent_result(
                tool_call.id,
                child_session_path=marker["child_session_path"],
                turns_used=marker.get("turns_used", 0),
                agent_type=marker.get("agent_type"),
                child_instance_id=child_instance_id,
            )
            if run_owner is not None:
                _settle_recovered_child_from_receipt(run_owner, result)
            receipt = loop.store.append_message(
                Message(
                    MessageRole.TOOL_RESULT,
                    [
                        TextContent(
                            result.content if recovered else "tool execution canceled"
                        )
                    ],
                    tool_result=result,
                )
            )
            _publish_recovered_end(child_store, receipt)
            loop.store.finish_agent_child(marker_key)
    recover_orphaned_questions(
        loop.store, excluded_child_subtrees=busy_child_subtrees
    )

BuildResult = Callable[..., dict[str, object]]

async def finish_background_child(
    *,
    child_task: asyncio.Task[dict[str, object]],
    child_store: ConversationStore,
    parent_store: ConversationStore,
    tool_call: ToolCall,
    child_instance_id: str,
    child_path: str,
    description: str,
    child_turns: Callable[[], int],
    build_result: BuildResult,
    validate_result: Callable[[object, str], ToolResult],
    publish_event: Callable[[StreamEvent], None],
    cleanup: Callable[[], Awaitable[None] | None],
    close_child: Callable[[str], Awaitable[tuple[str, ...]]],
    error_message: Callable[[BaseException], str],
    background_owner: BackgroundAgentOwner,
    marker_key: str | None = None,
    agent_instance_id: str | None = None,
    max_receipt_bytes: int | None = None,
    run_owner: ChildRunTransitionOwner,
    child_loop: Any | None = None,
    finish_gate_state: FinishGateState | None = None,
) -> None:
    """Persist a background child result and publish its terminal card event."""
    status = "completed"
    notification_text = ""
    result: object = None
    explicit_finish_content: dict[str, object] | None = None
    close_cause, _ = child_cleanup_details("background", False)
    try:
        execution_error: BaseException | None = None
        try:
            result = await child_task
        except asyncio.CancelledError as exc:
            execution_error = exc
        except Exception as exc:  # noqa: BLE001 - child failures become receipts
            execution_error = exc
        canceled, explicit_finish_report = await run_owner.background_outcome()
        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                explicit_finish_content = structured
            content = result.get("content")
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and isinstance(content[0].get("text"), str)
            ):
                notification_text = content[0]["text"]
            else:
                notification_text = "background child returned no text"
        if explicit_finish_report is not None:
            status = "error" if explicit_finish_report["status"] == "failed" else "completed"
        elif canceled or isinstance(execution_error, asyncio.CancelledError):
            status = "canceled"
            notification_text = "background child canceled"
        elif execution_error is not None:
            status = "error"
            notification_text = f"agent error: {error_message(execution_error)}"
        elif isinstance(result, dict) and result.get("isError") is True:
            status = "error"
        turn_limited, turns_used, turn_limit_handoff = background_turn_limit_state(
            result=result,
            child_turns=child_turns,
            child_loop=child_loop,
            child_instance_id=child_instance_id,
            child_session_path=child_path,
            final_text=notification_text,
            background_owner=background_owner,
        )
        if (
            child_loop is not None
            and status != "canceled"
            and explicit_finish_report is None
            and not (
                finish_gate_state is not None
                and finish_gate_state.terminal_admitted
            )
        ):
            from .finish_gate import await_owned_work
            await await_owned_work(child_loop, finish_gate_state)
        close_cause, close_event = child_cleanup_details(
            "background", status == "canceled"
        )
        killed_tasks = list(await close_child(close_cause) or ())
        # Completion metadata is persisted in more than one store. Keep the
        # legacy field bounded even when a child owned an unbounded task list.
        killed_task_metadata = [task_id[:64] for task_id in killed_tasks[:64]]
        killed_task_count = len(killed_tasks)
        killed_task_ids_truncated = len(killed_tasks) > len(
            killed_task_metadata
        ) or any(len(task_id) > 64 for task_id in killed_tasks)
        killed_task_notice = (
            f"\nbackground tasks killed on {close_event}; cause: {close_cause}: "
            + ", ".join(
                f"{task_id} (background command; archive: {child_path}/background-output.archive)"
                for task_id in killed_tasks
            )
            if killed_tasks
            else None
        )
        lifecycle_state = {
            "completed": "completed",
            "canceled": "canceled",
            "error": "failed",
        }[status]
        effective_parent_store, effective_parent_id = background_owner.parent_edge(child_instance_id, parent_store, agent_instance_id)
        lifecycle = child_store.agent_lifecycle() or {}
        live_lifecycle = live_agent_lifecycle(child_store.agent_lifecycle_path) or {}
        terminal_stats = agent_stats(
            {**lifecycle, **live_lifecycle, "turns_used": turns_used}, status=status
        )
        result_text = notification_text or "background child completed"
        notification_report = None
        if explicit_finish_report is not None:
            explicit_terminal = _background_terminal(
                lifecycle={
                    **(child_store.agent_lifecycle() or {}),
                    **(live_agent_lifecycle(child_store.agent_lifecycle_path) or {}),
                },
                report=explicit_finish_report,
                turns_used=turns_used,
                child_instance_id=child_instance_id,
                fallback_text=result_text,
                structured_content=explicit_finish_content,
                max_bytes=max_receipt_bytes or MAX_AGENT_RESULT_BYTES,
            )
            status = explicit_terminal.status
            terminal_stats = explicit_terminal.stats
            terminal_payload = explicit_terminal.payload
            notification_report = explicit_terminal.report
        else:
            build_parameters = inspect.signature(build_result).parameters
            build_kwargs: dict[str, object] = {}
            if "stats" in build_parameters:
                build_kwargs["stats"] = terminal_stats
            if "notice" in build_parameters:
                build_kwargs["notice"] = killed_task_notice
            if "notice_items" in build_parameters:
                build_kwargs["notice_items"] = killed_tasks or None
            if build_kwargs:
                terminal_payload = build_result(
                    result_text,
                    status != "completed",
                    status,
                    **build_kwargs,
                )
            elif "stats" in build_parameters:
                terminal_payload = build_result(
                    result_text,
                    status != "completed",
                    status,
                    terminal_stats,
                )
            else:
                terminal_payload = build_result(
                    result_text,
                    status != "completed",
                    status,
                )
        payload_content = terminal_payload.get("content")
        if (
            isinstance(payload_content, list)
            and payload_content
            and isinstance(payload_content[0], dict)
            and isinstance(payload_content[0].get("text"), str)
        ):
            notification_text = payload_content[0]["text"]
        else:
            notification_text = result_text
        if turn_limited:
            terminal_payload = project_payload(terminal_payload, child_path)

        async def commit_adopt_publish() -> ConversationEntry:
            persisted_text = notification_text
            if turn_limit_handoff is not None:
                persisted_text = await write_background_turn_limit(
                    child_store,
                    turn_limit_handoff,
                    notification_text,
                    child_path,
                )
            transferred = await child_store.finalize_background_agent(
                parent_store=effective_parent_store,
                terminal_state=lifecycle_state,
                final_result=persisted_text,
                turns_used=turns_used,
                killed_task_ids=killed_task_metadata or None,
                killed_task_count=killed_task_count or None,
                killed_task_ids_truncated=killed_task_ids_truncated,
                canceled_tool_call_id=(
                    tool_call.id if lifecycle_state == "canceled" else None
                ),
            )
            for transferred_id in transferred:
                background_owner.adopt(transferred_id, effective_parent_store, effective_parent_id)
            await background_owner.conversation_channel.close_questions(
                child_instance_id,
                reason="canceled" if status == "canceled" else "finished",
            )
            if status != "canceled":
                await background_owner.conversation_channel.drain_terminal(
                    child_store
                )
            return await background_owner.conversation_channel.publish_completion(
                child_instance_id=child_instance_id,
                marker_key=marker_key or tool_call.id,
                child_session_path=child_path,
                description=description,
                status=status,
                text=persisted_text,
                stats=terminal_stats,
                killed_task_ids=killed_task_metadata or None,
                killed_task_count=killed_task_count or None,
                killed_task_ids_truncated=killed_task_ids_truncated,
                report=notification_report,
                child_lifecycle=child_store.agent_lifecycle(),
            )

        transition = asyncio.create_task(run_owner.publish_background(commit_adopt_publish))
        deferred_cancel = False
        while not transition.done():
            try:
                await asyncio.shield(transition)
            except asyncio.CancelledError:
                deferred_cancel = True
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        notification = transition.result()
        event_data: dict[str, object] = {"notification_id": notification.id}
        if agent_instance_id is not None:
            event_data["agent_instance_id"] = agent_instance_id
        publish_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=tool_call,
                tool_result=validate_result(terminal_payload, tool_call.id),
                data=event_data,
                row_seq=notification.seq,
            )
        )
        if deferred_cancel:
            raise asyncio.CancelledError
    finally:
        try:
            await close_child(close_cause)
            # Let the parent finalize the running/terminal tool result before
            # releasing the child store descriptor.
            await asyncio.sleep(0)
        finally:
            cleanup_result = cleanup()
            if inspect.isawaitable(cleanup_result):
                await cleanup_result
            # Do not let the watcher finish before its child store lease closes.
            await background_owner.wait_for_store_closes()
