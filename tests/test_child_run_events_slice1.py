from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import shutil
import threading
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.store import ConversationStore
from zeta.core.store._child_runs import parse_child_run_id
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.tui.render import render_event


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


def test_tui_replays_start_row_through_real_renderer(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session")
    store.append_child_run_started("session:1", None, "worker", "task", "now")
    try:
        # Replay is the production transcript path; the renderer must still accept
        # the surrounding lifecycle event without producing a broken renderable.
        event = StreamEvent(
            StreamEventType.CHILD_MESSAGE_DELIVERED,
            data={"child_label": "session:1", "preview": "replayed"},
        )
        rendered = render_event(event)
        assert rendered is not None
        output = Console(record=True, width=100)
        output.print(rendered)
        assert "replayed" in output.export_text()
        assert [entry.type for entry in store.replay()] == ["child_run_started"]
    finally:
        store.close()
