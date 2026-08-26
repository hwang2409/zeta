from pathlib import Path

import pytest

from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationIntegrityError, ConversationStore
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def message(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


def test_checkpoint_and_fork_switch_the_active_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="fork")
    first = store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.ASSISTANT, "reply"))
    checkpoint = store.append_checkpoint("saved")
    abandoned = store.append_message(message(MessageRole.USER, "abandoned"))
    store.append_message(message(MessageRole.ASSISTANT, "later"))

    listed = store.list_checkpoints()
    assert [(entry.seq, preview) for entry, preview in listed] == [
        (checkpoint.seq, "abandoned")
    ]

    fork = store.append_fork("saved")
    assert fork.parent_id == checkpoint.id
    assert [entry.type for entry in store.replay()] == [
        "message",
        "message",
        "checkpoint",
        "fork",
    ]
    assert store.replay()[0].id == first.id
    assert all(entry.id != abandoned.id for entry in store.replay())
    assert len(store.entries) == 6

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    assert reopened.replay()[-1].id == fork.id


def test_checkpoint_and_fork_require_a_turn_boundary(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "in flight"))

    with pytest.raises(ConversationIntegrityError, match="turn is in flight"):
        store.append_checkpoint("blocked")


@pytest.mark.asyncio
async def test_fork_drops_a_compaction_marker_and_restores_originals(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "original user"))
    store.append_message(message(MessageRole.ASSISTANT, "original answer"))
    checkpoint = store.append_checkpoint("before compaction")
    store.append_message(message(MessageRole.USER, "new user"))
    store.append_message(message(MessageRole.ASSISTANT, "new answer"))
    store.append_compaction_marker("stale summary", 1, 2)
    store.append_fork(str(checkpoint.seq))

    context = ContextAssembler(store)
    messages = await context.assemble()
    text = [
        block.text
        for item in messages
        for block in item.content
        if isinstance(block, TextContent)
    ]
    assert "original user" in text
    assert "original answer" in text
    assert "stale summary" not in text


def test_fork_drops_pending_approval_from_the_abandoned_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "start"))
    store.append_message(message(MessageRole.ASSISTANT, "ready"))
    checkpoint = store.append_checkpoint("safe")
    call = ToolCall("pending", "danger", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    store.append_message(
        Message(MessageRole.TOOL_RESULT, [], tool_result=ToolResult(call.id, "done"))
    )

    assert store.pending_approvals() == [(call.id, call)]
    store.append_fork(str(checkpoint.seq))
    assert store.pending_approvals() == []
