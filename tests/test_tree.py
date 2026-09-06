"""ZETA-80: fork from arbitrary user messages and the /tree branch browser."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
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


def _make_tui(store: ConversationStore) -> TUIApp:
    return TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(
            file=StringIO(), force_terminal=True, color_system="truecolor"
        ),
    )


def _msg(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


def test_list_user_message_forkpoints_covers_active_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first user"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    second = store.append_message(_msg(MessageRole.USER, "second user"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))

    forkpoints = store.list_user_message_forkpoints()
    assert [(index, entry.id, preview) for index, entry, preview in forkpoints] == [
        (1, first.id, "first user"),
        (2, second.id, "second user"),
    ]


def test_append_message_fork_reanchors_at_target_user_message(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first user"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(_msg(MessageRole.USER, "abandoned"))
    store.append_message(_msg(MessageRole.ASSISTANT, "abandoned reply"))

    fork = store.append_message_fork(first.id)
    assert fork.parent_id == first.id
    assert fork.data["from_entry_id"] == first.id
    assert fork.data["from_seq"] == first.seq

    replay = store.replay()
    assert [entry.type for entry in replay] == ["message", "fork"]
    assert replay[0].id == first.id
    replayed_texts = [
        block.text
        for entry in replay
        if entry.type == "message"
        for block in Message.from_dict(entry.data["message"]).content
        if isinstance(block, TextContent)
    ]
    assert "abandoned" not in replayed_texts
    assert "abandoned reply" not in replayed_texts


def test_append_message_fork_rejects_non_user_targets(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "first"))
    assistant = store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    checkpoint = store.append_checkpoint("saved")

    with pytest.raises(ValueError, match="not a user message"):
        store.append_message_fork(assistant.id)
    with pytest.raises(ValueError, match="not a message"):
        store.append_message_fork(checkpoint.id)
    with pytest.raises(ValueError, match="not on the active branch"):
        store.append_message_fork("does-not-exist")


def test_message_fork_never_splits_tool_call_result_pair(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    user_message = store.append_message(_msg(MessageRole.USER, "read the file"))
    call = ToolCall("call-1", "read", {"path": "README.md"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))

    assert store.has_outstanding_tool_calls(store.replay()) is True

    app = _make_tui(store)
    result = app.slash_fork("1")
    assert result == "fork unavailable while a tool call is pending"
    assert not any(
        entry.type == "fork" for entry in store.replay()
    )

    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(call.id, "contents"),
        )
    )
    assert store.has_outstanding_tool_calls(store.replay()) is False
    result = app.slash_fork("1")
    assert "forked to user message 1" in result
    assert store.replay()[0].id == user_message.id


def test_message_fork_persists_and_reopens(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="s1")
    first = store.append_message(_msg(MessageRole.USER, "first user"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message(_msg(MessageRole.USER, "second"))
    store.append_message(_msg(MessageRole.ASSISTANT, "later"))
    fork = store.append_message_fork(first.id)

    reopened = ConversationStore(tmp_path, session_id="s1")
    assert reopened.replay()[-1].id == fork.id
    assert reopened.replay()[0].id == first.id


def test_slash_fork_picker_lists_prior_user_messages(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "prompt one"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_message(_msg(MessageRole.USER, "prompt two"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply two"))
    app = _make_tui(store)

    result = app.slash_fork("")
    assert "prior user messages on the active branch:" in result
    assert "1 · seq 1" in result
    assert "prompt one" in result
    assert "2 · seq 3" in result
    assert "prompt two" in result


def test_slash_fork_picker_when_no_user_messages(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    app = _make_tui(store)
    assert app.slash_fork("") == "no user messages to fork from; run a turn first"


def test_slash_fork_by_index_returns_dim_notice(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "prompt one"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_message(_msg(MessageRole.USER, "prompt two"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply two"))
    app = _make_tui(store)

    result = app.slash_fork("1")
    assert "forked to user message 1 at seq 1" in result
    assert "no workspace snapshot at this message" in result


def test_slash_fork_by_index_out_of_range(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "only one"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    app = _make_tui(store)

    assert app.slash_fork("5") == (
        "fork failed: user message index out of range (1..1)"
    )
    assert app.slash_fork("0") == (
        "fork failed: user message index out of range (1..1)"
    )


def test_slash_tree_renders_single_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    app = _make_tui(store)

    result = app.slash_tree("")
    assert "branches on this session:" in result
    assert "* 1. head seq 2" in result
    assert "first" in result
    assert "use /tree <n>" in result


def test_slash_tree_shows_branch_points_and_current(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(_msg(MessageRole.USER, "second"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)
    app = _make_tui(store)

    result = app.slash_tree("")
    lines = result.splitlines()
    assert lines[0] == "branches on this session:"
    branch_lines = [line for line in lines if line.startswith(("*", " "))]
    assert len(branch_lines) == 2
    assert any(line.startswith("* ") for line in branch_lines)
    assert any("from seq 1" in line for line in branch_lines)


def test_slash_tree_switches_branches_and_reanchors(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(_msg(MessageRole.USER, "second"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)
    app = _make_tui(store)

    branches = store.list_branches()
    other = next(b for b in branches if not b.is_current)
    result = app.slash_tree(str(next(
        index for index, branch in enumerate(branches, start=1)
        if branch.head.id == other.head.id
    )))
    assert result.startswith("switched to branch ")
    replay = store.replay()
    assert replay[-2].id == other.head.id


def test_slash_tree_usage_and_range_errors(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    app = _make_tui(store)

    assert app.slash_tree("bad arg") == "tree: usage /tree [n]"
    assert app.slash_tree("9") == "tree: branch index out of range (1..1)"
    assert app.slash_tree("1") == "tree: already on branch 1"


def test_slash_tree_refuses_pending_tool_call_only_on_switch(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "read"))
    call = ToolCall("call-1", "read", {"path": "README.md"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    app = _make_tui(store)

    # Listing is always allowed even mid-turn.
    result = app.slash_tree("")
    assert "branches on this session:" in result
    # Switching is refused while a tool call is outstanding.
    assert app.slash_tree("1") == "tree unavailable while a tool call is pending"


def test_slash_tree_refuses_background_children_on_switch(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message(_msg(MessageRole.USER, "second"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply again"))
    store.append_message_fork(first.id)
    app = _make_tui(store)
    app.loop._background_child_cancellers["agent-1"] = lambda: None
    other = next(b for b in store.list_branches() if not b.is_current)
    index = next(
        idx for idx, branch in enumerate(store.list_branches(), start=1)
        if branch.head.id == other.head.id
    )
    assert app.slash_tree(str(index)) == (
        "tree unavailable while background agents are running"
    )


def test_resume_after_message_fork_replays_active_branch_only(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="resume")
    first = store.append_message(_msg(MessageRole.USER, "first user"))
    store.append_message(_msg(MessageRole.ASSISTANT, "assistant reply"))
    store.append_message(_msg(MessageRole.USER, "abandoned prompt"))
    store.append_message(_msg(MessageRole.ASSISTANT, "abandoned reply"))
    store.append_message_fork(first.id)

    reopened = ConversationStore(tmp_path, session_id="resume")
    app = _make_tui(reopened)
    app._active_session = app._make_session()
    app._rebuild_transcript()

    from rich.text import Text

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "first user" in rendered
    assert "abandoned prompt" not in rendered
    assert "abandoned reply" not in rendered


def test_switch_between_branches_round_trips_replay(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(_msg(MessageRole.USER, "second"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)

    original_leaf = next(
        branch for branch in store.list_branches() if not branch.is_current
    )
    store.switch_to_branch(original_leaf.head.id)
    replay = store.replay()
    assert [
        block.text
        for entry in replay
        if entry.type == "message"
        for block in Message.from_dict(entry.data["message"]).content
        if isinstance(block, TextContent)
    ] == ["first", "reply 1", "second", "reply 2"]


def test_switch_to_branch_rejects_non_leaf(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))

    with pytest.raises(ValueError, match="branch head not found"):
        store.switch_to_branch("missing")
    with pytest.raises(ValueError, match="not a branch head"):
        store.switch_to_branch(first.id)


def test_switch_to_branch_rejects_current_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "first"))
    tail = store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    with pytest.raises(ValueError, match="already on that branch"):
        store.switch_to_branch(tail.id)


def test_message_fork_rejects_empty_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "first"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    with pytest.raises(ValueError, match="requires a user message entry id"):
        store.append_message_fork("")


def test_switch_to_branch_rejects_empty_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError, match="requires a head entry id"):
        store.switch_to_branch("")


def test_slash_fork_refuses_pending_tool_call(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_msg(MessageRole.USER, "run tool"))
    call = ToolCall("call-1", "read", {"path": "x"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    app = _make_tui(store)
    assert app.slash_fork("") == "fork unavailable while a tool call is pending"
    assert app.slash_fork("1") == "fork unavailable while a tool call is pending"


def test_fork_entry_reopens_without_checkpoint_source(tmp_path: Path) -> None:
    """The relaxed fork validator accepts non-checkpoint sources on reload."""

    store = ConversationStore(tmp_path, session_id="s1")
    first = store.append_message(_msg(MessageRole.USER, "hi"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message_fork(first.id)

    reopened = ConversationStore(tmp_path, session_id="s1")
    assert reopened.replay()[-1].type == "fork"


def test_list_branches_empty_store(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    assert store.list_branches() == []
    with pytest.raises(ValueError, match="branch head not found"):
        store.switch_to_branch("nope")


# --- ZETA-81 sweep coverage -----------------------------------------------


def test_fork_entries_record_source_type_for_each_flavor(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    store = ConversationStore(checkpoint_dir)
    store.append_message(_msg(MessageRole.USER, "one"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_checkpoint("saved")
    assert store.append_fork("saved").data["source_type"] == "checkpoint"

    message_dir = tmp_path / "message"
    store = ConversationStore(message_dir)
    user = store.append_message(_msg(MessageRole.USER, "hello"))
    store.append_message(_msg(MessageRole.ASSISTANT, "hi"))
    assert store.append_message_fork(user.id).data["source_type"] == "message"

    branch_dir = tmp_path / "branch"
    store = ConversationStore(branch_dir, session_id="branchsession")
    root = store.append_message(_msg(MessageRole.USER, "root"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    # Create a second branch to switch to.
    store.append_message_fork(root.id)
    reopened = ConversationStore(branch_dir, session_id="branchsession")
    leaves = [branch.head for branch in reopened.list_branches()]
    current_head_id = reopened.replay()[-1].id
    target = next(leaf for leaf in leaves if leaf.id != current_head_id)
    assert reopened.switch_to_branch(target.id).data["source_type"] == "branch"


def test_fork_rebuild_banner_names_the_source(tmp_path: Path) -> None:
    from zeta.core.checkpoints import ConversationEntry
    from zeta.tui.checkpoints import _fork_banner

    checkpoint_entry = ConversationEntry(
        id="a",
        seq=3,
        type="fork",
        parent_id=None,
        lane="conversation",
        data={"label": "saved", "from_seq": 2, "source_type": "checkpoint"},
    )
    message_entry = ConversationEntry(
        id="b",
        seq=4,
        type="fork",
        parent_id=None,
        lane="conversation",
        data={"label": "hello", "from_seq": 1, "source_type": "message"},
    )
    branch_entry = ConversationEntry(
        id="c",
        seq=5,
        type="fork",
        parent_id=None,
        lane="conversation",
        data={"label": "hello", "from_seq": 2, "source_type": "branch"},
    )
    legacy_entry = ConversationEntry(
        id="d",
        seq=6,
        type="fork",
        parent_id=None,
        lane="conversation",
        data={"label": "saved", "from_seq": 2},
    )
    assert _fork_banner(checkpoint_entry) == "forked to checkpoint 'saved' at seq 2"
    assert _fork_banner(message_entry) == "forked to user message 'hello' at seq 1"
    assert _fork_banner(branch_entry) == "switched to branch 'hello' at seq 2"
    # Legacy entries (no source_type) fall back to the original label.
    assert _fork_banner(legacy_entry) == "forked to checkpoint 'saved' at seq 2"


def test_slash_fork_picker_falls_through_to_checkpoints_without_user_messages(
    tmp_path: Path,
) -> None:
    """Sweep (b): the picker used to hide checkpoints when no user messages
    existed on the branch. Now it renders both sections independently."""

    store = ConversationStore(tmp_path)
    # Append a checkpoint directly (no user messages yet).
    store.append_checkpoint("bare")
    app = _make_tui(store)
    result = app.slash_fork("")
    assert "explicit checkpoint labels:" in result
    assert "bare" in result


def test_slash_tree_soft_caps_long_branch_lists(tmp_path: Path) -> None:
    """Sweep (d): /tree elides overflow with an ellipsis marker.

    The ellipsis count must match the number of rows hidden exactly — an
    off-by-one under-shows or over-reports how many branches were dropped.
    With N total branches and head_count = TREE_SOFT_CAP // 2 rows kept on
    each side, exactly ``N - 2 * head_count`` rows are hidden.
    """

    from zeta.tui.checkpoints import TREE_SOFT_CAP

    store = ConversationStore(tmp_path)
    root = store.append_message(_msg(MessageRole.USER, "root"))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    extra_forks = 5
    for _ in range(TREE_SOFT_CAP + extra_forks):
        store.append_message_fork(root.id)
    app = _make_tui(store)
    result = app.slash_tree("")
    # 1 root branch + (TREE_SOFT_CAP + extra_forks) forks total.
    total_branches = 1 + TREE_SOFT_CAP + extra_forks
    head_count = TREE_SOFT_CAP // 2
    expected_hidden = total_branches - 2 * head_count
    assert f"... {expected_hidden} more branches ..." in result
    # And there is exactly one ellipsis line, not several.
    assert result.count("more branches") == 1
    # Current branch marker still visible in the output.
    assert "* " in result
    # Exactly 2 * head_count branch rows survive the cap.
    branch_rows = [line for line in result.splitlines() if "head seq" in line]
    assert len(branch_rows) == 2 * head_count
