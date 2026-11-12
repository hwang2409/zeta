"""ZETA-80: fork from arbitrary user messages and the /tree branch browser."""

from __future__ import annotations

from tests.support.branch_reads import branch_entries, branch_messages

import asyncio
import contextlib
import inspect
import threading
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from tests.support.owned import close_owned_app
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp


def _make_tui(store: ConversationStore) -> TUIApp:
    return TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(
            file=StringIO(), force_terminal=True, color_system="truecolor"
        ),
    )


def _msg(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


def test_list_user_message_forkpoints_covers_active_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first user"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    second = store.append_message(with_message_origin(_msg(MessageRole.USER, "second user"), MessageOrigin.USER))
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
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first user"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "abandoned"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "abandoned reply"))

    fork = store.append_message_fork(first.id)
    assert fork.parent_id == first.id
    assert fork.data["from_entry_id"] == first.id
    assert fork.data["from_seq"] == first.seq

    replay = branch_entries(store)
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
    store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    assistant = store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    checkpoint = store.append_checkpoint("saved")

    with pytest.raises(ValueError, match="not a user message"):
        store.append_message_fork(assistant.id)
    with pytest.raises(ValueError, match="not a message"):
        store.append_message_fork(checkpoint.id)
    with pytest.raises(ValueError, match="not on the active branch"):
        store.append_message_fork("does-not-exist")


async def test_message_fork_never_splits_tool_call_result_pair(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    user_message = store.append_message(with_message_origin(_msg(MessageRole.USER, "read the file"), MessageOrigin.USER))
    call = ToolCall("call-1", "read", {"path": "README.md"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))

    assert store.has_outstanding_tool_calls() is True

    app = _make_tui(store)
    result = app.slash_fork("1")
    assert result == "fork unavailable while a tool call is pending"
    assert not any(
        entry.type == "fork" for entry in branch_entries(store)
    )

    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(call.id, "contents"),
        )
    )
    assert store.has_outstanding_tool_calls() is False
    result = await app.slash_fork("1")
    assert "forked to user message 1" in result
    assert branch_entries(store)[0].id == user_message.id
    await close_owned_app(app)


async def test_message_fork_cancellation_restores_workspace_and_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(
        with_message_origin(_msg(MessageRole.USER, "first user"), MessageOrigin.USER)
    )
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message(
        with_message_origin(_msg(MessageRole.USER, "later"), MessageOrigin.USER)
    )
    store.append_message(_msg(MessageRole.ASSISTANT, "later reply"))
    app = _make_tui(store)
    started = threading.Event()
    release = threading.Event()
    workspace_file = tmp_path / "workspace.txt"
    workspace_file.write_text("later")
    transcript_rebuilt = False
    original_append = ConversationStore.append_message_fork_async

    async def held_append(self: ConversationStore, entry_id: str):
        started.set()
        await asyncio.to_thread(release.wait, 10)
        return await original_append(self, entry_id)

    def restore_workspace(*args, **kwargs):
        workspace_file.write_text("first")
        return True, None

    def rebuild_transcript() -> None:
        nonlocal transcript_rebuilt
        transcript_rebuilt = True

    # test-contract: seam hold the submitted store command.
    monkeypatch.setattr(ConversationStore, "append_message_fork_async", held_append)
    # test-contract: seam patch the composite operation's workspace restore.
    monkeypatch.setattr("zeta.tui.checkpoints._restore_workspace", restore_workspace)
    # test-contract: seam patch the composite operation's post-commit rebuild.
    monkeypatch.setattr(app, "_rebuild_transcript", rebuild_transcript)
    task = asyncio.ensure_future(app.slash_fork("1"))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert workspace_file.read_text() == "first"
        assert transcript_rebuilt
        assert branch_entries(store)[-1].type == "fork"
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
        await close_owned_app(app)


def test_message_fork_persists_and_reopens(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="s1")
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first user"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "later"))
    fork = store.append_message_fork(first.id)

    reopened = ConversationStore(tmp_path, session_id="s1")
    assert branch_entries(reopened)[-1].id == fork.id
    assert branch_entries(reopened)[0].id == first.id


def test_slash_fork_picker_lists_prior_user_messages(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "prompt one"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "prompt two"), MessageOrigin.USER))
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


async def test_slash_fork_by_index_returns_dim_notice(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "prompt one"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "prompt two"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply two"))
    app = _make_tui(store)

    result = await app.slash_fork("1")
    assert "forked to user message 1 at seq 1" in result
    assert "no workspace snapshot at this message" in result
    await close_owned_app(app)


def test_slash_fork_by_index_out_of_range(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "only one"), MessageOrigin.USER))
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
    store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    app = _make_tui(store)

    result = app.slash_tree("")
    assert "branches on this session:" in result
    assert "* 1. head seq 2" in result
    assert "first" in result
    assert "use /tree <n>" in result


def test_slash_tree_shows_branch_points_and_current(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER))
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


async def test_slash_tree_switches_branches_and_reanchors(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)
    app = _make_tui(store)

    branches = store.list_branches()
    other = next(b for b in branches if not b.is_current)
    result = await app.slash_tree(str(next(
        index for index, branch in enumerate(branches, start=1)
        if branch.head.id == other.head.id
    )))
    assert result.startswith("switched to branch ")
    replay = branch_entries(store)
    assert replay[-2].id == other.head.id
    await close_owned_app(app)


def test_slash_tree_usage_and_range_errors(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    app = _make_tui(store)

    assert app.slash_tree("bad arg") == "tree: usage /tree [n]"
    assert app.slash_tree("9") == "tree: branch index out of range (1..1)"
    assert app.slash_tree("1") == "tree: already on branch 1"


def test_slash_tree_refuses_pending_tool_call_only_on_switch(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "read"), MessageOrigin.USER))
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
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER))
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
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first user"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "assistant reply"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "abandoned prompt"), MessageOrigin.USER))
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
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)

    original_leaf = next(
        branch for branch in store.list_branches() if not branch.is_current
    )
    store.switch_to_branch(original_leaf.head.id)
    replay = branch_entries(store)
    assert [
        block.text
        for entry in replay
        if entry.type == "message"
        for block in Message.from_dict(entry.data["message"]).content
        if isinstance(block, TextContent)
    ] == ["first", "reply 1", "second", "reply 2"]


def test_switch_to_branch_rejects_non_leaf(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))

    with pytest.raises(ValueError, match="branch head not found"):
        store.switch_to_branch("missing")
    with pytest.raises(ValueError, match="not a branch head"):
        store.switch_to_branch(first.id)


def test_switch_to_branch_rejects_current_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    tail = store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    with pytest.raises(ValueError, match="already on that branch"):
        store.switch_to_branch(tail.id)


def test_message_fork_rejects_empty_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    with pytest.raises(ValueError, match="requires a user message entry id"):
        store.append_message_fork("")


def test_switch_to_branch_rejects_empty_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError, match="requires a head entry id"):
        store.switch_to_branch("")


def test_slash_fork_refuses_pending_tool_call(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "run tool"), MessageOrigin.USER))
    call = ToolCall("call-1", "read", {"path": "x"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    app = _make_tui(store)
    assert app.slash_fork("") == "fork unavailable while a tool call is pending"
    assert app.slash_fork("1") == "fork unavailable while a tool call is pending"


def test_fork_entry_reopens_without_checkpoint_source(tmp_path: Path) -> None:
    """The relaxed fork validator accepts non-checkpoint sources on reload."""

    store = ConversationStore(tmp_path, session_id="s1")
    first = store.append_message(with_message_origin(_msg(MessageRole.USER, "hi"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    store.append_message_fork(first.id)

    reopened = ConversationStore(tmp_path, session_id="s1")
    assert branch_entries(reopened)[-1].type == "fork"


def test_list_branches_empty_store(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    assert store.list_branches() == []
    with pytest.raises(ValueError, match="branch head not found"):
        store.switch_to_branch("nope")


# --- ZETA-81 sweep coverage -----------------------------------------------


def test_fork_entries_record_source_type_for_each_flavor(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    store = ConversationStore(checkpoint_dir)
    store.append_message(with_message_origin(_msg(MessageRole.USER, "one"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply one"))
    store.append_checkpoint("saved")
    assert store.append_fork("saved").data["source_type"] == "checkpoint"

    message_dir = tmp_path / "message"
    store = ConversationStore(message_dir)
    user = store.append_message(with_message_origin(_msg(MessageRole.USER, "hello"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "hi"))
    assert store.append_message_fork(user.id).data["source_type"] == "message"

    branch_dir = tmp_path / "branch"
    store = ConversationStore(branch_dir, session_id="branchsession")
    root = store.append_message(with_message_origin(_msg(MessageRole.USER, "root"), MessageOrigin.USER))
    store.append_message(_msg(MessageRole.ASSISTANT, "reply"))
    # Create a second branch to switch to.
    store.append_message_fork(root.id)
    reopened = ConversationStore(branch_dir, session_id="branchsession")
    leaves = [branch.head for branch in reopened.list_branches()]
    current_head_id = branch_entries(reopened)[-1].id
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
    root = store.append_message(with_message_origin(_msg(MessageRole.USER, "root"), MessageOrigin.USER))
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


async def test_tree_switch_runs_off_loop_and_cancel_still_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # /tree <n> switches the active branch. The switch re-anchors the branch,
    # which rebuilds the active-branch sidecar and fold (seconds of work on a
    # long session). This test proves two things that fail on 19eced31, where
    # the switch ran synchronously on the event loop:
    #   1. the switch runs off the loop, so a callback scheduled on the loop
    #      runs while the branch rebuild is held (the fail-first assertion);
    #   2. cancelling the awaiting task before the result installs still
    #      finalizes the durable switch and leaves the resident store consistent
    #      with a cold reopen (FINALIZE_ONCE_SUBMITTED).
    store = ConversationStore(tmp_path, session_id="offloop")
    first = store.append_message(
        with_message_origin(_msg(MessageRole.USER, "first"), MessageOrigin.USER)
    )
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 1"))
    store.append_message(
        with_message_origin(_msg(MessageRole.USER, "second"), MessageOrigin.USER)
    )
    store.append_message(_msg(MessageRole.ASSISTANT, "reply 2"))
    store.append_message_fork(first.id)
    app = _make_tui(store)

    # Gate _sync_history_index, which both the synchronous (main) and the
    # off-loop (this branch) branch-switch paths call during the re-anchor.
    gate: dict[str, threading.Event | None] = {"reached": None, "release": None}
    original_sync = ConversationStore._sync_history_index

    def gated_sync(self: ConversationStore, entries: object) -> object:
        reached = gate["reached"]
        if reached is not None:
            gate["reached"] = None
            release = gate["release"]
            reached.set()
            assert release is not None and release.wait(10)
        return original_sync(self, entries)

    # test-contract: seam hold the real branch rebuild off the event loop.
    monkeypatch.setattr(
        ConversationStore, "_sync_history_index", gated_sync, raising=True
    )

    other = next(b for b in store.list_branches() if not b.is_current)
    index = next(
        position
        for position, branch in enumerate(store.list_branches(), start=1)
        if branch.head.id == other.head.id
    )

    reached1 = threading.Event()
    release1 = threading.Event()
    callback_ran = threading.Event()
    observed: dict[str, bool] = {}

    def watcher() -> None:
        if not reached1.wait(10):
            observed["during_hold"] = False
            release1.set()
            return
        # The branch rebuild is held. On this branch it is held off the loop, so
        # the loop is free to run the scheduled callback; on main it is held on
        # the loop thread, so the callback cannot run until the gate releases.
        observed["during_hold"] = callback_ran.wait(3)
        release1.set()

    watcher_thread = threading.Thread(target=watcher)
    try:
        watcher_thread.start()
        gate["reached"], gate["release"] = reached1, release1
        asyncio.get_running_loop().call_soon(callback_ran.set)
        result = app.slash_tree(str(index))
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, 10)
        watcher_thread.join(15)

        assert observed.get("during_hold"), (
            "the event loop must run a scheduled callback while the branch "
            "switch is held: the switch has to run off the loop"
        )
        assert isinstance(result, str) and result.startswith("switched to branch ")
        # The switch re-anchors by appending a fork row parented at the target
        # leaf, so the target leaf is now second from the tail.
        assert branch_entries(store)[-2].id == other.head.id

        # Phase 2: cancel-before-install still finalizes the durable switch.
        back = next(b for b in store.list_branches() if not b.is_current)
        reached2 = threading.Event()
        release2 = threading.Event()
        gate["reached"], gate["release"] = reached2, release2
        task = asyncio.ensure_future(store.switch_to_branch_async(back.head.id))
        assert await asyncio.to_thread(reached2.wait, 10)
        task.cancel()
        release2.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)

        cold = ConversationStore(tmp_path, session_id="offloop")
        try:
            assert store.active_branch_head_id() == cold.active_branch_head_id()
            assert branch_entries(store)[-2].id == back.head.id
        finally:
            cold.close()
    finally:
        if gate["release"] is not None:
            gate["release"].set()
        if watcher_thread.is_alive():
            watcher_thread.join(5)
        await close_owned_app(app)
