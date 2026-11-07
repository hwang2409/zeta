from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import shutil
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from tests.support.fake_backend import FakeBackend
from zeta.context_eviction import recall_history
from zeta.core.context import ContextAssembler
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.core.store._child_runs import (
    parse_child_run_id,
    validate_child_run_started_data,
)
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)
from zeta.server.ergonomics import history
from zeta.skills import SkillCatalog
from zeta.transcript_search.index import TranscriptIndex, TranscriptSource
from zeta.tui.app import TUIApp
from zeta.tui.cards.agent import AgentCard


@pytest.mark.parametrize("run_id", ["session:4", "session:4:2", "session:4:2:9"])
def test_nested_run_ids_are_parsed(run_id: str) -> None:
    number, parent = parse_child_run_id("session", run_id)
    assert number == int(run_id.rsplit(":", 1)[1])
    assert parent == (run_id.rpartition(":")[0] if run_id.count(":") > 1 else None)


@pytest.mark.parametrize(
    "run_id",
    ["../1", "session/1", "session\\1", "session:%2f1", "session:\x001", "session:+1", "session:0", "session:01", "other:1", "session:"],
)
def test_malformed_run_ids_are_rejected(run_id: str) -> None:
    with pytest.raises(ValueError):
        parse_child_run_id("session", run_id)
    with pytest.raises(ValueError):
        parse_child_run_id("session", "session:" + ":".join("1" for _ in range(65)))
    with pytest.raises(ValueError):
        parse_child_run_id("session", "session:" + "1" * 250)


def _append_in_spawn(path: str) -> None:
    store = ConversationStore(Path(path), session_id="session")
    store.append_child_run_started("session:1", None, "worker", "task", "now")
    os.kill(os.getpid(), 9)


def test_child_run_started_truncates_long_task_and_name_to_utf8_bytes(tmp_path: Path) -> None:
    name = "名" * 400
    task = "作業" * 10_000
    store = ConversationStore(tmp_path, session_id="session")
    try:
        store.append_child_run_started("session:1", None, name, task, "now")
        row = store.entries[-1].data
        assert len(row["name"].encode()) <= 1024
        assert len(row["task"].encode()) <= 1024
        row["name"].encode().decode()
        row["task"].encode().decode()
    finally:
        store.close()


@pytest.mark.parametrize(
    "change",
    [
        {"task": "x" * 1025},
        {"name": "x" * 1025},
        {"started_at": "x" * 65},
        {"run_id": "session:" + ":".join("1" for _ in range(65))},
    ],
    ids=["task", "name", "timestamp", "depth"],
)
def test_child_run_started_reader_rejects_out_of_bound_rows(change: dict[str, str]) -> None:
    data = {"run_id": "session:1", "parent_run_id": None, "task": "task", "started_at": "now"}
    data.update(change)
    if change.keys() == {"run_id"}:
        data["parent_run_id"] = data["run_id"].rpartition(":")[0]
    with pytest.raises(ValueError):
        validate_child_run_started_data(data)


def test_start_row_is_durable_before_child_execution_and_published(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    milestones: list[str] = []
    callback_thread: list[int] = []
    fsync_thread: list[int] = []
    loop_thread = threading.get_ident()

    def installed(rows: object) -> None:
        milestones.append("installed")
        callback_thread.append(threading.get_ident())
        assert rows != "reset"
        assert store.entries[-1].type == "child_run_started"

    store.on_entries_installed = installed

    async def run() -> None:
        original_fsync = os.fsync
        entered = asyncio.Event()
        release = asyncio.Event()

        def held_fsync(fd: int) -> None:
            fsync_thread.append(threading.get_ident())
            entered_loop.call_soon_threadsafe(entered.set)
            while not release.is_set():
                # the wait is deliberately in the worker, not the event loop
                threading.Event().wait(0.001)
            original_fsync(fd)

        entered_loop = asyncio.get_running_loop()
        os.fsync = held_fsync
        try:
            task = asyncio.create_task(store.append_child_run_started_async("session:1", None, "worker", "task", "now"))
            await asyncio.wait_for(entered.wait(), 2)
            ticks = 0
            deadline = entered_loop.time() + .05
            while entered_loop.time() < deadline:
                ticks += 1
                await asyncio.sleep(0)
            assert ticks > 1
            milestones.append("fsync-held")
            release.set()
            await asyncio.wait_for(task, 2)
        finally:
            os.fsync = original_fsync

    try:
        asyncio.run(run())
        assert fsync_thread and fsync_thread[0] != loop_thread
        assert callback_thread
        assert milestones == ["fsync-held", "installed"]
        assert store.entries[-1].type == "child_run_started"
    finally:
        store.close()


def test_crash_after_start_row_recovers(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_append_in_spawn, args=(str(tmp_path),))
    child.start()
    child.join(5)
    assert child.exitcode == -9
    store = ConversationStore(tmp_path, session_id="session", _read_only=True)
    try:
        assert len(store.entries) == 1
        assert store.entries[0].type == "child_run_started"
        assert store.entries[0].data["run_id"] == "session:1"
    finally:
        store.close()


def test_discussion_fork_copy_loads_start_row(tmp_path: Path) -> None:
    source = ConversationStore(tmp_path, session_id="source")
    source.append_child_run_started("source:4", None, "worker", "task", "now")
    source.close()
    fork_dir = tmp_path / "fork"
    shutil.copytree(tmp_path / "source", fork_dir)
    # A discussion fork has a new directory identity, but preserves transcript ids.
    log = fork_dir / "conversation.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[0]["data"]["session_id"] = "fork"
    log.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    fork = ConversationStore(tmp_path, session_id="fork", _read_only=True)
    try:
        assert any(e.type == "child_run_started" for e in fork.entries)
        row = next(e for e in fork.entries if e.type == "child_run_started")
        assert row.data["run_id"] == "source:4"
        fork.replay()
    finally:
        fork.close()


def test_pre_feature_session_loads_unchanged(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="old")
    store.close()
    reopened = ConversationStore(tmp_path, session_id="old", _read_only=True)
    try:
        assert list(reopened.replay()) == []
        assert reopened.entries == []
    finally:
        reopened.close()


def _message(role: MessageRole, text: str) -> Message:
    metadata = (
        {"zeta.origin": MessageOrigin.USER.value}
        if role is MessageRole.USER
        else {"response_state": "completed"}
    )
    return Message(role, [TextContent(text)], metadata=metadata)


def test_store_full_and_incremental_validation_accept_start_row(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    store.append_message(_message(MessageRole.USER, "before"))
    store.append_child_run_started("session:1", None, "worker", "task", "now")
    store.append_message(_message(MessageRole.ASSISTANT, "after"))
    store.close()

    reopened = ConversationStore(tmp_path, session_id="session", _read_only=True)
    try:
        assert [entry.type for entry in reopened.replay()] == [
            "message",
            "child_run_started",
            "message",
        ]
    finally:
        reopened.close()


def test_context_and_eviction_ignore_start_row(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    first = store.append_message(_message(MessageRole.USER, "before"))
    store.append_child_run_started("session:1", None, "worker", "hidden task", "now")
    last = store.append_message(_message(MessageRole.ASSISTANT, "after"))
    store.append_compaction_marker("summary", first.seq, last.seq)
    try:
        assembled = asyncio.run(
            ContextAssembler(
                store,
                token_budget=10_000,
                retained_tail=8,
                system_prompt="system",
            ).assemble_context()
        )
        assert all(
            "hidden task" not in str(message.to_dict())
            for message in assembled.messages
        )
        recalled = recall_history(store, seq_start=first.seq, seq_end=last.seq)
        assert "before" in recalled
        assert "after" in recalled
        assert "hidden task" not in recalled
    finally:
        store.close()


def test_transcript_search_ignores_start_row_payload(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="session")
    store.append_message(_message(MessageRole.USER, "searchable before"))
    store.append_child_run_started(
        "session:1", None, "worker", "private child task", "now"
    )
    store.append_message(_message(MessageRole.ASSISTANT, "searchable after"))
    index = TranscriptIndex(tmp_path / "project", "p_" + "a" * 32)
    try:
        index.append(TranscriptSource("session", store.session_dir))
        assert index.search("searchable before")
        assert index.search("searchable after")
        assert index.search("private child task") == ()
    finally:
        store.close()


def test_tui_replays_start_row_through_real_renderer(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    store.append_message(_message(MessageRole.USER, "before"))
    store.append_child_run_started(
        "session:1", None, "worker", "must stay hidden", "now"
    )
    store.append_message(_message(MessageRole.ASSISTANT, "after"))
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    try:
        app._rebuild_transcript()
        rendered = Text.from_ansi(output.getvalue()).plain
        assert "before" in rendered
        assert "after" in rendered
        assert "must stay hidden" not in rendered
        assert "unknown" not in rendered.casefold()
    finally:
        asyncio.run(app.loop.close())
        store.close()


def test_agent_card_receipt_shows_identity_and_clean_state(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path, session_id="child")
    lifecycle = {
        "handle": "parent:1",
        "state": "completed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "elapsed": 1.5,
        "turns_used": 2,
        "tool_calls": 3,
        "current_step": "done",
        "depth": 1,
        "agent_type": "general",
        "description": "child",
        "cwd": "/repo",
        "final_result": "done",
    }
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")
    call = ToolCall(
        "agent-1",
        "agent",
        {
            "prompt": "inspect",
            "description": "child",
            "agent_type": "general",
            "model": "gpt-5.6-luna",
        },
    )
    rendered = AgentCard.render_receipt(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "done",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(child.session_dir),
                    "status": "completed",
                    "turns_used": 2,
                },
                lifecycle_projection=lifecycle,
            ),
        )
    )
    assert rendered is not None
    assert rendered.plain == (
        "general · child :1 · 2 turns · 1.5s · completed · depth 1 · 3 tool calls · gpt-5.6-luna · done"
    )
    child.close()


def test_serve_session_history_ignores_start_row(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    store.append_message(_message(MessageRole.USER, "before"))
    store.append_child_run_started(
        "session:1", None, "worker", "must stay hidden", "now"
    )
    store.append_message(_message(MessageRole.ASSISTANT, "after"))
    try:
        result = history(SimpleNamespace(opened=SimpleNamespace(store=store)), {})
        rendered = json.dumps(result)
        empty = ConversationStore(tmp_path, session_id="empty")
        try:
            empty_frame = history(
                SimpleNamespace(opened=SimpleNamespace(store=empty)), {}
            )
            assert json.dumps(
                empty_frame, sort_keys=True, separators=(",", ":")
            ) == '{"messages":[],"next_offset":null}'
        finally:
            empty.close()
        assert "before" in rendered
        assert "after" in rendered
        assert "must stay hidden" not in rendered
    finally:
        store.close()
