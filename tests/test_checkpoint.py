import json
from io import StringIO
from pathlib import Path
from typing import get_type_hints

import pytest
from rich.console import Console
from rich.text import Text

from zeta.core.checkpoints import CheckpointForkMixin
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationIntegrityError, ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp
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


def test_checkpoint_method_type_hints_resolve_conversation_entry() -> None:
    methods = (
        CheckpointForkMixin._validate_checkpoint_or_fork_payload,
        CheckpointForkMixin._validate_fork_entry,
        CheckpointForkMixin.append_checkpoint,
        CheckpointForkMixin.append_fork,
        CheckpointForkMixin.is_turn_boundary,
        CheckpointForkMixin.list_checkpoints,
        CheckpointForkMixin._default_checkpoint_label,
        CheckpointForkMixin._message_preview,
        CheckpointForkMixin._resolve_checkpoint,
    )

    for method in methods:
        assert get_type_hints(method)


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


def test_numeric_checkpoint_labels_are_reserved_for_sequence_selectors(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.ASSISTANT, "reply"))
    store.append_checkpoint("safe")
    store.append_message(message(MessageRole.USER, "second"))
    store.append_message(message(MessageRole.ASSISTANT, "reply"))
    target = store.append_checkpoint("target")

    with pytest.raises(ValueError, match="numeric.*sequence selectors"):
        store.append_checkpoint("6")

    fork = store.append_fork(str(target.seq))
    assert fork.parent_id == target.id


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


def test_naive_checkpoint_timestamp_is_normalized_at_jsonl_boundary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.ASSISTANT, "reply"))
    store.append_checkpoint("saved")

    rows = [json.loads(row) for row in store.path.read_text().splitlines()]
    checkpoint = next(row for row in rows if row.get("type") == "checkpoint")
    checkpoint["data"]["created_at"] = "2026-08-26T12:00:00"
    store.path.write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) for row in rows
        )
        + "\n"
    )

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    listed = reopened.list_checkpoints()
    assert listed[0][0].data["created_at"] == "2026-08-26T12:00:00+00:00"

    app = TUIApp(
        AgentLoop(FakeBackend([]), reopened),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True, color_system="truecolor"),
    )
    assert "ago" in app.slash_fork("")


def test_fork_rebuild_renders_replayed_tool_call(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall("read-1", "read", {"path": "README.md"})
    store.append_message(message(MessageRole.USER, "inspect the readme"))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(call.id, "file contents"),
        )
    )
    store.append_checkpoint("saved")
    store.append_message(message(MessageRole.USER, "abandoned"))
    store.append_message(message(MessageRole.ASSISTANT, "later"))

    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True, color_system="truecolor"),
    )
    app._active_session = app._make_session()

    app.slash_fork("saved")

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "read" in rendered
    assert "README.md" in rendered
