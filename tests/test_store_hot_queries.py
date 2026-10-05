from __future__ import annotations

from pathlib import Path

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
