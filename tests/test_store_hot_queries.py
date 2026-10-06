from __future__ import annotations

from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def _unexpected_history_scan(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("hot state query scanned conversation history")


def test_approval_queries_use_incremental_state_without_replay(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    pending = ToolCall("pending", "read", {"path": "README.md"})
    resolved = ToolCall("resolved", "bash", {"command": "pwd"})
    store.append_message_with_approval_requests(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(pending), ToolUseContent(resolved)],
        ),
        [(pending.id, pending), (resolved.id, resolved)],
    )
    assert store.resolve_approval(resolved.id, "deny")
    store.replay = _unexpected_history_scan  # type: ignore[method-assign]

    assert store.pending_approvals() == [(pending.id, pending)]
    assert store.approval_states() == {
        pending.id: (pending, None),
        resolved.id: (resolved, "deny"),
    }


def test_pending_approvals_sees_external_append(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    reader = ConversationStore(root, session_id="shared")
    writer = ConversationStore(root, session_id="shared")
    request = ToolCall("external", "read", {"path": "README.md"})

    writer.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(request)]),
        [(request.id, request)],
    )

    assert reader.pending_approvals() == [(request.id, request)]


def test_tool_result_query_uses_incremental_state_without_replay(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    result = ToolResult("call-1", "done")
    store.append_message(Message(MessageRole.USER, [TextContent("history")]))
    store.append_message(Message(MessageRole.USER, [], tool_result=result))
    store.replay = _unexpected_history_scan  # type: ignore[method-assign]

    assert store.tool_result("call-1") == result
    assert store.tool_result("missing") is None


def test_notification_queries_use_incremental_state_without_branch_scan(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("history")]))
    notification = store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    store.mark_agent_notification_presented_to_tui(notification.id)
    store.acknowledge_agent_notification(notification.id)
    store._active_branch = _unexpected_history_scan  # type: ignore[method-assign]

    assert [entry.id for entry in store.agent_notifications(pending_only=False)] == [
        notification.id
    ]
    assert store.agent_notifications() == []
    assert store.is_agent_notification_presented_to_tui(notification.id)


def test_index_parity_with_replay(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="parity")

    def assert_parity(operation: str) -> None:
        indexed_pending = store.pending_approvals()
        branch = store.replay()
        approval_states = store._approval_states_from_branch(branch)
        expected_pending = [
            (request_id, call)
            for request_id, (call, decision) in approval_states.items()
            if decision is None
        ]
        assert indexed_pending == expected_pending, operation
        assert store.approval_states() == approval_states, operation

        notifications = {
            entry.id: entry for entry in branch if entry.type == "notification"
        }
        acknowledgements = {
            entry.data["notification_id"]
            for entry in branch
            if entry.type == "notification_ack"
        }
        presentations = {
            entry.data["notification_id"]
            for entry in branch
            if entry.type == "notification_tui_presented"
        }
        assert list(store._active_notifications) == list(notifications), operation
        assert store._active_notification_acks == acknowledgements, operation
        assert store._active_notification_presentations == presentations, operation
        assert [
            entry.id for entry in store.agent_notifications(pending_only=False)
        ] == list(notifications), operation
        assert [entry.id for entry in store.agent_notifications()] == [
            notification_id
            for notification_id in notifications
            if notification_id not in acknowledgements
        ], operation
        assert all(
            store.is_agent_notification_presented_to_tui(notification_id)
            == (notification_id in presentations)
            for notification_id in notifications
        ), operation

        expected_results: dict[str, ToolResult] = {}
        for entry in reversed(branch):
            if entry.type != "message":
                continue
            result = Message.from_dict(entry.data["message"]).tool_result
            if result is not None:
                expected_results.setdefault(result.tool_call_id, result)
        assert store._active_tool_results == expected_results, operation
        assert all(
            store.tool_result(tool_call_id) == result
            for tool_call_id, result in expected_results.items()
        ), operation
        assert store.tool_result("missing-result") is None, operation

    def apply(operation: str, action: object) -> object:
        result = action()  # type: ignore[operator]
        assert_parity(operation)
        return result

    apply(
        "linear user append",
        lambda: store.append_message(
            Message(MessageRole.USER, [TextContent("start")])
        ),
    )
    apply(
        "linear assistant append",
        lambda: store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("ready")])
        ),
    )
    checkpoint = apply("checkpoint", lambda: store.append_checkpoint("safe"))

    call = ToolCall("branch-call", "read", {"path": "README.md"})
    apply(
        "approval request append",
        lambda: store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
        ),
    )
    apply("approval resolution", lambda: store.resolve_approval(call.id, "allow"))
    apply(
        "first result for duplicate id",
        lambda: store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [],
                tool_result=ToolResult(call.id, "first"),
            )
        ),
    )
    apply(
        "last result for duplicate id",
        lambda: store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [],
                tool_result=ToolResult(call.id, "last"),
            )
        ),
    )
    apply(
        "canceled result",
        lambda: store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [],
                tool_result=ToolResult(
                    "canceled-call",
                    "tool execution canceled",
                    is_error=True,
                    is_canceled=True,
                ),
            )
        ),
    )
    notification = apply(
        "notification append",
        lambda: store.append_agent_notification(
            "child-1",
            child_session_path="/tmp/child-1",
            description="child",
            status="completed",
            text="done",
        ),
    )
    apply(
        "notification presentation",
        lambda: store.mark_agent_notification_presented_to_tui(notification.id),
    )
    apply(
        "notification acknowledgement",
        lambda: store.acknowledge_agent_notification(notification.id),
    )
    apply(
        "compaction marker",
        lambda: store.append_compaction_marker("summary", 1, 2),
    )
    eviction_message = Message(MessageRole.ASSISTANT, [TextContent("retained")])
    apply(
        "eviction marker",
        lambda: store.append_compaction_marker(
            "eviction",
            1,
            2,
            kind="evict",
            view=[{"seq": 2, "message": eviction_message.to_dict()}],
            telemetry={"items_evicted": 1},
        ),
    )
    apply("fork branch replacement", lambda: store.append_fork(str(checkpoint.seq)))

    store.close()
    store = ConversationStore(root, session_id="parity")
    assert_parity("reopen")
    apply("explicit refresh", store.refresh)

    writer = ConversationStore(root, session_id="parity")
    external = ToolCall("external-call", "bash", {"command": "pwd"})
    apply(
        "external writer approval append",
        lambda: writer.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(external)]),
            [(external.id, external)],
        ),
    )
    apply(
        "external writer approval resolution",
        lambda: writer.resolve_approval(external.id, "deny"),
    )
    writer.close()

    with store.path.open("ab") as handle:
        handle.write(b'{"seq":999,"id":"torn"')
    with pytest.warns(RuntimeWarning, match="dropped torn final"):
        apply("torn-tail repair", store.refresh)
