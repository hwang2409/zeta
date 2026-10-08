"""Tests for session-scoped background process tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    TextContent,
    ToolCall,
    ToolSchema,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools._shared import process as process_module
from zeta.tools._shared.background_output_archive import _BackgroundOutputArchive
from zeta.tools._shared.process import (
    BackgroundTaskRegistry,
    _BackgroundRecord,
    _group_exists,
    _OrderedLogWriter,
)
from zeta.tui.render import format_status


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


async def _wait_for_exit(registry: BackgroundTaskRegistry, task_id: str) -> None:
    await asyncio.wait_for(registry.wait(task_id), timeout=30)


@pytest.mark.asyncio
async def test_shutdown_notice_is_structured_for_field_accessing_sinks(tmp_path: Path) -> None:
    notices = []
    registry = BackgroundTaskRegistry(notice_sink=lambda notice: notices.append(notice.message))
    task_id, _ = await registry.start("sleep 30", tmp_path)
    await registry.close()
    assert any("killed on session exit" in message for message in notices)
    assert task_id in notices[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "background"),
    [
        ("run_background", {"command": "true"}, False),
        ("bash", {"command": "true"}, True),
    ],
)
async def test_real_background_producers_preserve_owner_on_start_and_exit(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, str],
    background: bool,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    notices = []
    registry.background_tasks.set_notice_sink(notices.append)
    result = await registry.execute(
        ToolCall("start", tool_name, arguments),
        _background=background,
    )
    task_id = result["structuredContent"]["task_id"]
    await _wait_for_exit(registry.background_tasks, task_id)
    lifecycle = [notice for notice in notices if notice.task_id == task_id]
    assert [notice.phase for notice in lifecycle] == ["started", "natural_exit"]
    expected_owner = "background_macro" if background else "run_background"
    assert [notice.owner for notice in lifecycle] == [expected_owner, expected_owner]
    await registry.close()


@pytest.mark.asyncio
async def test_canceled_monitor_propagates_when_record_already_stopped() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
    )
    record = _BackgroundRecord(
        task_id="task-cancel",
        command="sleep",
        pid=process.pid,
        process=process,
    )
    registry = BackgroundTaskRegistry()
    monitor = asyncio.create_task(registry._monitor(record))
    await asyncio.sleep(0)
    record.running = False
    monitor.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await monitor
    finally:
        process.kill()
        await process.wait()


async def _collect(events):
    return [event async for event in events]


class _GatedFakeBackend(FakeBackend):
    def __init__(self, turns: Sequence[ScriptedTurn], gated_prompt: str) -> None:
        super().__init__(turns)
        self.gated_prompt = gated_prompt
        self.completion_gate = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        gated = any(
            block.text == self.gated_prompt
            for message in messages
            for block in message.content
            if isinstance(block, TextContent)
        )
        async for event in super().complete(messages, tool_schemas):
            if gated:
                await self.completion_gate.wait()
                gated = False
            yield event


@pytest.mark.asyncio
async def test_background_exit_persists_task_notification(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session")
    wakes = 0

    def wake() -> None:
        nonlocal wakes
        wakes += 1

    registry = BackgroundTaskRegistry(notification_store=store, notification_callback=wake)
    task_id, _ = await registry.start(_python("print('task output')"), Path.cwd())
    await _wait_for_exit(registry, task_id)
    notifications = store.agent_notifications()
    assert len(notifications) == 1
    assert notifications[0].data["kind"] == "task_exited"
    assert notifications[0].data["task_id"] == task_id
    assert notifications[0].data["output_tail"] == "task output"
    assert wakes == 1
    await registry.close()


@pytest.mark.asyncio
async def test_session_close_kills_without_notification(tmp_path: Path) -> None:
    # S1: whole-session shutdown kills running tasks and returns their ids, but
    # must NOT create task_exited notifications.
    store = ConversationStore(tmp_path / "session")
    wakes = 0

    def wake() -> None:
        nonlocal wakes
        wakes += 1

    registry = BackgroundTaskRegistry(notification_store=store, notification_callback=wake)
    task_id, _ = await registry.start(_python("import time; time.sleep(30)"), Path.cwd())
    killed = await registry.close()
    assert task_id in killed
    assert store.agent_notifications() == []
    assert wakes == 0


@pytest.mark.asyncio
async def test_task_exit_recovered_after_restart_exactly_once(tmp_path: Path) -> None:
    # S5: a previously-running row with no existing task_exited notification
    # yields exactly one recovery notification on the normal production path,
    # and repeated resumes never duplicate it.
    from zeta.core.fake import FakeBackend
    from zeta.runtime.loop import AgentLoop
    from zeta.tools._shared.process import _BackgroundRecord

    def seed_running_row(session_store: ConversationStore) -> None:
        seed = BackgroundTaskRegistry(
            session_dir=session_store.session_dir,
            directory_fd=session_store.directory_fd,
        )
        seed._records["task-seed"] = _BackgroundRecord(
            task_id="task-seed", command="sleep 99", pid=4242, process=None, running=True
        )
        seed._persist()
        seed.release_directory()

    def task_exits(session_store: ConversationStore) -> list:
        return [
            entry
            for entry in session_store.agent_notifications(pending_only=False)
            if entry.data.get("kind") == "task_exited"
        ]

    # First resume: recovery must fire through the normal AgentLoop construction.
    store1 = ConversationStore(tmp_path, session_id="restart")
    seed_running_row(store1)
    loop1 = AgentLoop(FakeBackend([]), store1, max_turns=1, skill_catalog=SkillCatalog.empty())
    recovered = task_exits(store1)
    assert len(recovered) == 1
    assert recovered[0].data["task_id"] == "task-seed"
    assert recovered[0].data["exit_code"] is None
    assert recovered[0].data["note"] == "exit not observed (zeta restarted)"
    await loop1.close()
    store1.close()

    # Second resume of the same session (crash left the row running again):
    # the observed-notification check must keep it at exactly one.
    store2 = ConversationStore(tmp_path, session_id="restart")
    seed_running_row(store2)
    loop2 = AgentLoop(FakeBackend([]), store2, max_turns=1, skill_catalog=SkillCatalog.empty())
    assert len(task_exits(store2)) == 1
    await loop2.close()
    store2.close()


@pytest.mark.asyncio
async def test_failed_run_background_start_uses_tool_error_as_canonical_result(tmp_path: Path) -> None:
    # A failed run_background start is rendered by its error tool card; do not
    # persist a second task_exited receipt.
    store = ConversationStore(tmp_path / "session")
    registry = BackgroundTaskRegistry(notification_store=store)
    with pytest.raises(ValueError):
        await registry.start("echo hi", tmp_path / "does-not-exist")
    assert store.agent_notifications() == []
    await registry.close()


@pytest.mark.asyncio
async def test_failed_macro_start_is_visible_once_as_tool_error(tmp_path: Path) -> None:
    # Background macros render their failed start as the macro tool error card;
    # their lifecycle notification is disabled, so it must not add a receipt.
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    notices = []
    registry.background_tasks.set_notice_sink(notices.append)
    result = await registry.execute(
        ToolCall("macro-fail", "bash", {"command": "echo hi", "cwd": str(tmp_path / "does-not-exist")}),
        _background=True,
    )
    assert result["isError"] is True
    assert "could not execute command" in result["content"][0]["text"]
    assert not [
        notice
        for notice in notices
        if getattr(notice, "phase", None)
        in {"natural_exit", "task_kill", "session_shutdown"}
    ]
    await registry.close()


@pytest.mark.asyncio
async def test_task_kill_produces_exactly_one_notification(tmp_path: Path) -> None:
    # S6: killing a modeled task notifies exactly once.
    store = ConversationStore(tmp_path / "session")
    registry = BackgroundTaskRegistry(notification_store=store)
    task_id, _ = await registry.start(_python("import time; time.sleep(30)"), Path.cwd())
    await registry.kill(task_id)
    task_exits = [
        entry
        for entry in store.agent_notifications(pending_only=False)
        if entry.data.get("kind") == "task_exited"
    ]
    assert len(task_exits) == 1
    assert task_exits[0].data["task_id"] == task_id
    await registry.close()
    # Closing after the kill must not append a second notification.
    task_exits = [
        entry
        for entry in store.agent_notifications(pending_only=False)
        if entry.data.get("kind") == "task_exited"
    ]
    assert len(task_exits) == 1


def test_task_notification_dedupe_after_rewind_or_fork(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="task-fork")
    store.append_message(
        with_message_origin(Message(MessageRole.USER, [TextContent("before notification")]), MessageOrigin.USER)
    )
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")]))
    store.append_checkpoint("before-task")
    first = store.append_task_notification(
        task_id="task-reused", command="printf hi", exit_code=0
    )

    store.append_fork("before-task")
    second = store.append_task_notification(
        task_id="task-reused", command="printf again", exit_code=1
    )

    assert second.id != first.id
    active = [
        entry
        for entry in store.agent_notifications(pending_only=False)
        if entry.data.get("kind") == "task_exited"
    ]
    assert len(active) == 1
    assert active[0].id == second.id
    assert active[0].data["exit_code"] == 1






def test_store_replay_accepts_task_and_unknown_notification_kinds(
    tmp_path: Path,
) -> None:
    # S7: replay validation accepts task_exited rows and legacy rows without a
    # kind (agent_completion), and tolerates unknown kinds for forward compat.
    store = ConversationStore(tmp_path, session_id="kinds")
    store.append_task_notification(
        task_id="task-1", command="printf hi", exit_code=0, output_tail="hi"
    )
    store._append_row(
        "notification",
        {
            "child_instance_id": "child-legacy",
            "child_session_path": "agents/1",
            "description": "background child",
            "status": "completed",
            "text": "done",
        },
    )
    store._append_row("notification", {"kind": "monitor_alert", "note": "heads up"})

    # A fresh store on the same session forces load + validation of every row.
    reloaded = ConversationStore(tmp_path, session_id="kinds")
    kinds = [
        entry.data.get("kind", "agent_completion")
        for entry in reloaded.replay()
        if entry.type == "notification"
    ]
    assert kinds == ["task_exited", "agent_completion", "monitor_alert"]
    assert len(reloaded.agent_notifications(pending_only=False)) == 3


def test_tui_renders_task_exit_and_ignores_unknown_kind() -> None:
    # S7: the TUI renders task_exited receipts and tolerates unknown kinds
    # without ever rendering them as agent completions.
    from zeta.protocol.types import StreamEvent, StreamEventType
    from zeta.tui.render import render_agent_notification

    def render(data: dict) -> str:
        return render_agent_notification(
            StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=data)
        ).plain

    task_line = render(
        {
            "kind": "task_exited",
            "task_id": "task-1",
            "exit_code": 0,
            "headline": "printf hi",
        }
    )
    assert "task task-1 exited (0)" in task_line
    assert "printf hi" in task_line

    legacy_line = render(
        {
            "child_instance_id": "child-1",
            "child_session_path": "agents/1",
            "description": "background child",
            "status": "completed",
            "text": "done",
        }
    )
    assert "background child" in legacy_line
    assert "completed" in legacy_line

    unknown_line = render({"kind": "monitor_alert", "text": "heads up"})
    assert unknown_line == "background agent notification unavailable"


@pytest.mark.asyncio
async def test_agent_store_closes_after_each_foreground_completion(
    tmp_path: Path,
) -> None:
    calls = [
        ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "child", "description": "child"},
        )
        for index in (1, 2)
    ]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[calls[0]]),
            ScriptedTurn([TextContent("first done")]),
            ScriptedTurn(tool_calls=[calls[1]]),
            ScriptedTurn([TextContent("second done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    child_stores = []
    owner = loop._background_owner
    track_store = owner.track_store
    owner.track_store = lambda child_store: (
        child_stores.append(child_store), track_store(child_store)
    )

    try:
        await _collect(loop.run_turn("first", origin=MessageOrigin.USER))
        await _collect(loop.run_turn("second", origin=MessageOrigin.USER))

        for child_store in child_stores:
            with pytest.raises(OSError):
                os.fstat(child_store.directory_fd)
        assert store.directory_fd >= 0
    finally:
        try:
            await loop.close()
        finally:
            store.close()


@pytest.mark.asyncio
async def test_background_agent_store_closes_before_root_shutdown(tmp_path: Path) -> None:
    call = ToolCall(
        "background",
        "agent",
        {"prompt": "child", "description": "background", "background": True},
    )
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    child_stores = []
    owner = loop._background_owner
    track_store = owner.track_store
    owner.track_store = lambda child_store: (
        child_stores.append(child_store), track_store(child_store)
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    for _ in range(100):
        if not owner._stores:
            break
        await asyncio.sleep(0.01)
    child_store = child_stores[0]
    with pytest.raises(OSError):
        os.fstat(child_store.directory_fd)
    assert store.directory_fd >= 0
    await loop.close()


@pytest.mark.asyncio
async def test_adopted_background_grandchild_keeps_foreground_ancestor_open(
    tmp_path: Path,
) -> None:
    foreground = ToolCall(
        "foreground",
        "agent",
        {"prompt": "child", "description": "child"},
    )
    grandchild = ToolCall(
        "grandchild",
        "agent",
        {
            "prompt": "grandchild",
            "description": "grandchild",
            "background": True,
        },
    )
    backend = _GatedFakeBackend(
        [
            ScriptedTurn(tool_calls=[foreground]),
            ScriptedTurn(tool_calls=[grandchild]),
            ScriptedTurn([TextContent("done")]),
        ],
        gated_prompt="grandchild",
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    tracked = []
    owner = loop._background_owner
    track_store = owner.track_store
    owner.track_store = lambda child_store: (
        tracked.append(child_store),
        track_store(child_store),
    )

    try:
        await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        for _ in range(100):
            if len(tracked) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(tracked) == 2
        os.fstat(tracked[0].directory_fd)

        backend.completion_gate.set()
        await asyncio.wait_for(owner.wait(), timeout=5)
        for child_store in tracked:
            with pytest.raises(OSError):
                os.fstat(child_store.directory_fd)
        assert store.directory_fd >= 0
    finally:
        backend.completion_gate.set()
        try:
            await loop.close()
        finally:
            store.close()


@pytest.mark.asyncio
async def test_background_start_poll_and_kill_round_trip(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    started = await registry.execute(
        ToolCall("start", "run_background", {"command": _python("print('hello')")})
    )
    task_id = started["structuredContent"]["task_id"]
    await _wait_for_exit(registry.background_tasks, task_id)
    output = await registry.execute(
        ToolCall("output", "task_output", {"task_id": task_id})
    )
    assert output["structuredContent"]["output"] == "hello\n"
    assert output["structuredContent"]["running"] is False
    visible = output["content"][0]["text"]
    assert "cursor:" in visible
    assert "running: False" in visible
    assert "exit_code: 0" in visible

    long_running = await registry.execute(
        ToolCall("start-2", "run_background", {"command": "sleep 30"})
    )
    killed = await registry.execute(
        ToolCall(
            "kill",
            "task_kill",
            {"task_id": long_running["structuredContent"]["task_id"]},
        )
    )
    assert killed["structuredContent"]["running"] is False
    await registry.close()


@pytest.mark.asyncio
async def test_background_stdin_round_trip_and_eof(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry()
    task_id, _ = await tasks.start(
        _python("import sys; data=sys.stdin.buffer.read(); print(data.decode(), end='')"),
        tmp_path,
    )
    first = await tasks.input(task_id, "hello ")
    stdin_writer = tasks.records[0].stdin
    assert stdin_writer is not None
    second = await tasks.input(task_id, "world", eof=True)
    assert tasks.records[0].stdin is None
    assert stdin_writer.is_closing()
    assert first == {
        "task_id": task_id, "bytes_written": 6, "eof": False,
        "status": "committed", "retry": False,
    }
    assert second == {
        "task_id": task_id, "bytes_written": 5, "eof": True,
        "status": "committed", "retry": False,
    }
    assert await tasks.input(task_id, eof=True) == {
        "task_id": task_id, "bytes_written": 0, "eof": True,
        "status": "committed", "retry": False,
    }
    await _wait_for_exit(tasks, task_id)
    assert (await tasks.output(task_id))["output"] == "hello world"
    await tasks.close()


@pytest.mark.asyncio
async def test_background_stdin_cancellation_returns_committed_bytes(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(stdin_drain_timeout=1)
    task_id, _ = await tasks.start(
        _python(
            "import sys, time; time.sleep(.05); data=sys.stdin.buffer.read(); "
            "print(len(data), end='')"
        ),
        tmp_path,
    )
    writer = tasks.records[0].stdin
    assert writer is not None
    original_drain = writer.drain
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()

    async def delayed_drain():
        drain_started.set()
        await release_drain.wait()
        await original_drain()

    writer.drain = delayed_drain  # type: ignore[method-assign]
    pending = asyncio.create_task(tasks.input(task_id, "cancel-me"))
    await drain_started.wait()
    pending.cancel()
    release_drain.set()
    result = await pending
    assert result["status"] == "committed"
    assert result["retry"] is False
    await tasks.input(task_id, eof=True)
    await _wait_for_exit(tasks, task_id)
    assert (await tasks.output(task_id))["output"] == "9"
    await tasks.close()


@pytest.mark.asyncio
async def test_background_stdin_timeout_aborts_fd_and_forbids_retry(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(stdin_drain_timeout=0.01, term_grace=0.1)
    task_id, _ = await tasks.start(
        _python(
            "import sys; data=sys.stdin.buffer.read(); print(len(data), end='')"
        ),
        tmp_path,
    )
    writer = tasks.records[0].stdin
    assert writer is not None
    drain_started = asyncio.Event()

    async def blocked_drain():
        drain_started.set()
        await asyncio.Future()

    writer.drain = blocked_drain  # type: ignore[method-assign]
    result = await tasks.input(task_id, "indeterminate")
    assert result["status"] == "indeterminate"
    assert result["retry"] is False
    assert writer.is_closing()
    await _wait_for_exit(tasks, task_id)
    assert (await tasks.output(task_id))["output"] == str(len("indeterminate"))
    await tasks.close()


@pytest.mark.asyncio
async def test_background_stdin_serializes_writes_and_bounds_input(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry()
    task_id, _ = await tasks.start(
        _python("import sys; sys.stdout.write(sys.stdin.read()); sys.stdout.flush()"),
        tmp_path,
    )
    parts = [f"{index:03d}" for index in range(20)]
    await asyncio.gather(*(tasks.input(task_id, part) for part in parts))
    await tasks.input(task_id, eof=True)
    await _wait_for_exit(tasks, task_id)
    assert (await tasks.output(task_id))["output"] == "".join(parts)
    with pytest.raises(ValueError, match="65536-byte limit"):
        await tasks.input(task_id, "x" * (64 * 1024 + 1))
    await tasks.close()


@pytest.mark.asyncio
async def test_background_stdin_ownership_is_session_local(tmp_path: Path) -> None:
    owner = BackgroundTaskRegistry()
    other = BackgroundTaskRegistry()
    task_id, _ = await owner.start("sleep 30", tmp_path)
    with pytest.raises(ValueError, match="unknown background task"):
        await other.input(task_id, "x")
    await owner.close()
    await other.close()


@pytest.mark.asyncio
async def test_background_stdin_unknown_finished_and_kill_races(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry()
    with pytest.raises(ValueError, match="unknown background task"):
        await tasks.input("not-owned", "x")
    task_id, _ = await tasks.start("true", tmp_path)
    await _wait_for_exit(tasks, task_id)
    with pytest.raises(ValueError, match="not running"):
        await tasks.input(task_id, "x")
    task_id, _ = await tasks.start("sleep 30", tmp_path)
    writes = [tasks.input(task_id, "x" * 100) for _ in range(10)]
    await asyncio.gather(tasks.kill(task_id), *writes, return_exceptions=True)
    assert tasks.records[-1].stdin is None
    await tasks.close()

    shutting_down = BackgroundTaskRegistry()
    shutdown_id, _ = await shutting_down.start("sleep 30", tmp_path)
    shutdown_writes = [shutting_down.input(shutdown_id, "x" * 100) for _ in range(10)]
    await asyncio.gather(shutting_down.close(), *shutdown_writes, return_exceptions=True)
    assert shutting_down.records[-1].stdin is None


@pytest.mark.asyncio
async def test_task_output_waits_without_canceling_background_monitor(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    started = await registry.execute(
        ToolCall("start", "run_background", {"command": "sleep 2; printf done"})
    )
    task_id = started["structuredContent"]["task_id"]
    first = await registry.execute(
        ToolCall("first", "task_output", {"task_id": task_id, "wait_seconds": 1})
    )
    assert first["structuredContent"]["running"] is True
    second = await registry.execute(
        ToolCall("second", "task_output", {"task_id": task_id, "wait_seconds": 3})
    )
    assert second["structuredContent"]["running"] is False
    assert second["structuredContent"]["exit_code"] == 0
    assert second["structuredContent"]["output"] == "done"
    invalid = await registry.execute(
        ToolCall("invalid", "task_output", {"task_id": task_id, "wait_seconds": 301})
    )
    assert invalid["isError"] is True
    await registry.close()


@pytest.mark.asyncio
async def test_background_output_cursor_and_ring_overflow(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(
        output_limit=8,
        call_limit=4,
    )
    task_id, _ = await tasks.start("printf 0123456789abcdef", tmp_path)
    await _wait_for_exit(tasks, task_id)

    first = await tasks.output(task_id)
    assert first["output"] == "0123"
    assert first["cursor"] == 4
    second = await tasks.output(task_id, since=first["cursor"])
    assert second["output"] == "4567"
    assert second["cursor"] == 8
    await tasks.close()


@pytest.mark.asyncio
async def test_archived_output_survives_resume(tmp_path: Path) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="archive-resume") as store:
        first = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        task_id, _ = await first.start("printf durable-output", tmp_path)
        await _wait_for_exit(first, task_id)
        await first.close()

        resumed = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        result = await resumed.output(task_id, since=0)

        assert result["output"] == "durable-output"
        assert result["cursor"] == len(b"durable-output")
        await resumed.close()


def test_two_archive_instances_append_concurrently_keep_both(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    first = _BackgroundOutputArchive(directory_fd)
    second = _BackgroundOutputArchive(directory_fd)
    payloads = {
        "task-concurrent-a": b"a" * (2 * 1024 * 1024),
        "task-concurrent-b": b"b" * (2 * 1024 * 1024),
    }
    source_fds = []
    try:
        assert first.recover() == {}
        assert second.recover() == {}
        for task_id, payload in payloads.items():
            path = tmp_path / f"source-{task_id}"
            path.write_bytes(payload)
            source_fds.append(os.open(path, os.O_RDONLY))
        barrier = threading.Barrier(2)

        def commit(archive, task_id: str, source_fd: int) -> None:
            barrier.wait()
            archive.append(task_id, source_fd, len(payloads[task_id]))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(commit, first, "task-concurrent-a", source_fds[0]),
                pool.submit(commit, second, "task-concurrent-b", source_fds[1]),
            ]
            for future in futures:
                future.result(timeout=10)

        recovered = _BackgroundOutputArchive(directory_fd)
        try:
            assert recovered.recover() == {
                task_id: len(payload) for task_id, payload in payloads.items()
            }
            for task_id, payload in payloads.items():
                assert recovered.pread(task_id, 0, len(payload)) == payload
        finally:
            recovered.close()
    finally:
        for source_fd in source_fds:
            os.close(source_fd)
        first.close()
        second.close()
        os.close(directory_fd)


def test_recover_skips_log_held_by_live_writer(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    writer_archive = _BackgroundOutputArchive(directory_fd)
    recovering_archive = _BackgroundOutputArchive(directory_fd)
    task_id = "task-live-writer"
    payload = b"still being written"
    writer_fd = -1
    try:
        writer_archive.recover()
        writer_fd = writer_archive.open_task_log(task_id)
        os.write(writer_fd, payload)
        os.fsync(writer_fd)

        assert recovering_archive.recover() == {}
        assert (tmp_path / f"background-{task_id}.log").exists()

        writer_archive.append(task_id, writer_fd, len(payload))
        assert recovering_archive.recover() == {task_id: len(payload)}
        assert recovering_archive.pread(task_id, 0, len(payload)) == payload
    finally:
        if writer_fd >= 0:
            os.close(writer_fd)
        writer_archive.close()
        recovering_archive.close()
        os.close(directory_fd)


def test_append_reads_only_new_journal_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    archive = _BackgroundOutputArchive(directory_fd)
    source_path = tmp_path / "source"
    source_path.write_bytes(b"x")
    source_fd = os.open(source_path, os.O_RDONLY)
    real_pread = os.pread
    journal_reads: list[int] = []

    def track_read(fd: int, size: int, offset: int) -> bytes:
        if fd == archive._journal_fd:
            journal_reads.append(size)
        return real_pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", track_read)
    try:
        archive.recover()
        for index in range(400):
            archive.append(f"task-{index:04d}", source_fd, 1)

        # Each append replays only the newly appended frame, not the complete
        # journal. The first append reads zero bytes because the journal is empty.
        assert sum(journal_reads[-50:]) <= sum(journal_reads[:50]) * 2
    finally:
        os.close(source_fd)
        archive.close()
        os.close(directory_fd)

    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    instances_fd = os.open(instances_dir, os.O_RDONLY | os.O_DIRECTORY)
    first = _BackgroundOutputArchive(instances_fd)
    second = _BackgroundOutputArchive(instances_fd)
    first_source = instances_dir / "first-source"
    second_source = instances_dir / "second-source"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    first_source_fd = os.open(first_source, os.O_RDONLY)
    second_source_fd = os.open(second_source, os.O_RDONLY)
    try:
        assert first.recover() == {}
        assert second.recover() == {}
        first.append("task-first", first_source_fd, 5)
        second.append("task-second", second_source_fd, 6)
        assert first.pread("task-second", 0, 6) == b"second"
        assert second.pread("task-first", 0, 5) == b"first"
    finally:
        os.close(first_source_fd)
        os.close(second_source_fd)
        first.close()
        second.close()
        os.close(instances_fd)


def test_journal_commit_cost_is_bounded_and_torn_tail_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    archive = _BackgroundOutputArchive(directory_fd)
    source_path = tmp_path / "source"
    source_path.write_bytes(b"x")
    source_fd = os.open(source_path, os.O_RDONLY)
    journal_writes: list[int] = []
    try:
        archive.recover()
        real_pwrite_all = archive._pwrite_all

        def track_write(fd: int, data: bytes, offset: int) -> None:
            if fd == archive._journal_fd:
                journal_writes.append(len(data))
            real_pwrite_all(fd, data, offset)

        monkeypatch.setattr(archive, "_pwrite_all", track_write)
        for index in range(100):
            archive.append(f"task-{index:04d}", source_fd, 1)

        assert len(journal_writes) == 100
        assert max(journal_writes) < 2 * min(journal_writes)
        committed_journal_size = (tmp_path / "background-output.journal").stat().st_size
        archive.close()

        with (tmp_path / "background-output.journal").open("ab") as journal:
            journal.write(b"ZBO1\x00")
        with (tmp_path / "background-output.archive").open("ab") as output:
            output.write(b"uncommitted")

        recovered = _BackgroundOutputArchive(directory_fd)
        try:
            lengths = recovered.recover()
            assert len(lengths) == 100
            assert set(lengths.values()) == {1}
            assert recovered.pread("task-0099", 0, 1) == b"x"
            assert (
                tmp_path / "background-output.journal"
            ).stat().st_size == committed_journal_size
            assert (tmp_path / "background-output.archive").stat().st_size == 100
        finally:
            recovered.close()
    finally:
        os.close(source_fd)
        archive.close()
        os.close(directory_fd)


@pytest.mark.asyncio
async def test_cancel_first_close_caller_during_archive_worker_wait_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="archive-cancel") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        archive = tasks._archive
        assert archive is not None
        real_append = archive.append
        entered = threading.Event()
        release = threading.Event()
        append_errors: list[BaseException] = []

        def blocked_append(task_id: str, source_fd: int, length: int) -> None:
            entered.set()
            assert release.wait(timeout=10)
            try:
                real_append(task_id, source_fd, length)
            except BaseException as exc:
                append_errors.append(exc)
                raise

        monkeypatch.setattr(archive, "append", blocked_append)
        task_id, _ = await tasks.start("printf cancellation-output", tmp_path)
        assert await asyncio.to_thread(entered.wait, 10)

        real_wait = tasks._wait_for_archive_workers
        wait_entered = threading.Event()

        async def tracked_wait() -> None:
            wait_entered.set()
            await real_wait()

        monkeypatch.setattr(tasks, "_wait_for_archive_workers", tracked_wait)
        close_task = asyncio.create_task(tasks.close())
        assert await asyncio.to_thread(wait_entered.wait, 10)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()
        assert not archive._closed

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert append_errors == []
        assert archive._closed

        resumed = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        assert (await resumed.output(task_id, since=0))["output"] == "cancellation-output"
        await resumed.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_archive_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="archive-shutdown") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        archive = tasks._archive
        assert archive is not None
        real_append = archive.append
        entered = threading.Event()
        release = threading.Event()

        def slow_append(task_id: str, source_fd: int, length: int) -> None:
            entered.set()
            assert release.wait(timeout=10)
            real_append(task_id, source_fd, length)

        monkeypatch.setattr(archive, "append", slow_append)
        task_id, _ = await tasks.start("printf shutdown-output", tmp_path)
        assert await asyncio.to_thread(entered.wait, 10)
        close_task = asyncio.create_task(tasks.close())
        await asyncio.sleep(0.05)
        assert not close_task.done()
        release.set()
        await close_task

        resumed = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        result = await resumed.output(task_id, since=0)
        assert result["output"] == "shutdown-output"
        await resumed.close()


@pytest.mark.asyncio
async def test_crash_between_data_and_manifest_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="archive-crash") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        committed_id, _ = await tasks.start("printf committed", tmp_path)
        await _wait_for_exit(tasks, committed_id)

        archive = tasks._archive
        assert archive is not None
        real_append_commit = archive._append_commit
        failed = False

        def fail_commit_once(task_id: str, offset: int, length: int) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("simulated crash before journal commit")
            real_append_commit(task_id, offset, length)

        monkeypatch.setattr(archive, "_append_commit", fail_commit_once)
        interrupted_id, _ = await tasks.start("printf interrupted", tmp_path)
        with pytest.raises(OSError, match="simulated crash"):
            await _wait_for_exit(tasks, interrupted_id)
        tasks._close_storage()

        resumed = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        committed = await resumed.output(committed_id, since=0)
        interrupted = await resumed.output(interrupted_id, since=0)
        archive = (store.session_dir / "background-output.archive").read_bytes()

        assert committed["output"] == "committed"
        assert interrupted["output"] == "interrupted"
        assert archive == b"committedinterrupted"

        real_unlink = os.unlink
        unlink_failed = False

        def fail_log_unlink_once(path, *args, **kwargs):
            nonlocal unlink_failed
            if (
                str(path).startswith("background-task-")
                and str(path).endswith(".log")
                and not unlink_failed
            ):
                unlink_failed = True
                raise OSError("simulated crash after manifest commit")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", fail_log_unlink_once)
        after_manifest_id, _ = await resumed.start("printf after-manifest", tmp_path)
        with pytest.raises(OSError, match="simulated crash"):
            await _wait_for_exit(resumed, after_manifest_id)
        resumed._close_storage()
        monkeypatch.setattr(os, "unlink", real_unlink)

        recovered_again = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        after_manifest = await recovered_again.output(after_manifest_id, since=0)
        archive = (store.session_dir / "background-output.archive").read_bytes()

        assert after_manifest["output"] == "after-manifest"
        assert archive == b"committedinterruptedafter-manifest"
        assert not (
            store.session_dir / f"background-{after_manifest_id}.log"
        ).exists()
        await recovered_again.close()


@pytest.mark.asyncio
async def test_active_task_log_recovered_after_crash(tmp_path: Path) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="active-log-crash") as store:
        task_id = "task-crashed-active"
        payload = b"output-written-before-crash"
        (store.session_dir / f"background-{task_id}.log").write_bytes(payload)
        (store.session_dir / "background_tasks.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": task_id,
                        "command": "crashed command",
                        "pid": 12345,
                        "running": True,
                        "exit_code": None,
                    }
                ]
            ),
            encoding="utf-8",
        )

        resumed = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        result = await resumed.output(task_id, since=0)

        assert result["output"] == payload.decode()
        assert result["cursor"] == len(payload)
        assert not (store.session_dir / f"background-{task_id}.log").exists()
        await resumed.close()


def test_writer_handles_short_writes() -> None:
    class ShortWriter:
        def __init__(self) -> None:
            self.data = bytearray()
            self.flushes = 0

        def write(self, data: bytes | memoryview) -> int:
            count = min(3, len(data))
            self.data.extend(data[:count])
            return count

        def flush(self) -> None:
            self.flushes += 1

    async def write() -> tuple[bytes, int]:
        handle = ShortWriter()
        writer = _OrderedLogWriter(handle)  # type: ignore[arg-type]
        await writer.write(b"short writes must preserve every byte")
        await writer.close()
        return bytes(handle.data), handle.flushes

    data, flushes = asyncio.run(write())
    assert data == b"short writes must preserve every byte"
    assert flushes == 2


@pytest.mark.asyncio
async def test_task_output_reads_archive_not_replaceable_path(tmp_path: Path) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="archive-read") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        task_id, _ = await tasks.start("printf original-output", tmp_path)
        await _wait_for_exit(tasks, task_id)
        old_log = Path(tasks.records[-1].log_path or "")
        assert not old_log.exists()
        old_log.write_text("replacement-secret", encoding="utf-8")
        archive_path = store.session_dir / "background-output.archive"
        original_archive = archive_path.with_suffix(".original")
        archive_path.rename(original_archive)
        archive_path.write_text("replacement-archive", encoding="utf-8")

        result = await tasks.output(task_id, since=0)

        assert "original-output" in result["output"]
        assert "replacement-secret" not in result["output"]
        await tasks.close()


@pytest.mark.asyncio
async def test_output_beyond_ring_recoverable_active_and_finished(tmp_path: Path) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="concurrent-output") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=7,
            call_limit=5,
        )
        payloads = (b"alpha-0123456789", b"beta-ABCDEFGHIJ", b"gamma-uvwxyz")
        task_ids = []
        for payload in payloads:
            code = (
                "import sys,time; data="
                + repr(payload)
                + "; sys.stdout.buffer.write(data[:8]); sys.stdout.flush(); "
                + "time.sleep(0.4); sys.stdout.buffer.write(data[8:]); sys.stdout.flush()"
            )
            task_id, _ = await tasks.start(_python(code), tmp_path)
            task_ids.append(task_id)

        async def wait_for_prefix(task_id: str) -> None:
            for _ in range(100):
                record = next(item for item in tasks.records if item.task_id == task_id)
                if record.total_bytes >= 8:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("task did not produce its active prefix")

        async def read_exact(task_id: str, length: int) -> bytes:
            cursor = 0
            output = bytearray()
            while cursor < length:
                result = await tasks.output(task_id, since=cursor)
                output.extend(result["output"].encode())
                assert result["cursor"] > cursor
                cursor = result["cursor"]
            return bytes(output)

        await asyncio.gather(*(wait_for_prefix(task_id) for task_id in task_ids))
        active = await asyncio.gather(
            *(read_exact(task_id, 8) for task_id in task_ids)
        )
        assert active == [payload[:8] for payload in payloads]

        await asyncio.gather(*(_wait_for_exit(tasks, task_id) for task_id in task_ids))
        finished = await asyncio.gather(
            *(read_exact(task_id, len(payload)) for task_id, payload in zip(task_ids, payloads, strict=True))
        )
        assert finished == list(payloads)
        await tasks.close()


def test_many_finished_tasks_hold_bounded_descriptors(tmp_path: Path) -> None:
    script = r'''
import asyncio, json, os, resource, sys
from pathlib import Path
from zeta.core.store import ConversationStore
from zeta.tools._shared.process import BackgroundTaskRegistry

async def main(root: Path) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, hard), hard))
    with ConversationStore(root / "sessions", session_id="fd-bound") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
        )
        baseline = len(os.listdir("/dev/fd"))
        task_ids = []
        for index in range(200):
            task_id, _ = await tasks.start(
                f"printf task-{index:03d}-output", root
            )
            await tasks.wait(task_id)
            task_ids.append(task_id)
        for index, task_id in enumerate(task_ids):
            result = await tasks.output(task_id, since=0)
            assert f"task-{index:03d}-output" in result["output"]
        final = len(os.listdir("/dev/fd"))
        print(json.dumps({"baseline": baseline, "final": final}))
        assert final <= baseline + 2
        await tasks.close()

asyncio.run(main(Path(sys.argv[1])))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    counts = json.loads(completed.stdout.strip().splitlines()[-1])
    assert counts["final"] <= counts["baseline"] + 2


@pytest.mark.asyncio
async def test_archive_private_and_session_scoped(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="fake", model="fake", cwd=tmp_path)
    store = opened.store
    tasks = BackgroundTaskRegistry(
        session_dir=store.session_dir,
        directory_fd=store.directory_fd,
        output_limit=4,
    )
    task_id, _ = await tasks.start("printf private-output", tmp_path)
    await _wait_for_exit(tasks, task_id)
    result = await tasks.output(task_id, since=0)
    archive_path = store.session_dir / "background-output.archive"

    assert result["output_location"] == f"task-output://{task_id}"
    assert archive_path.parent == store.session_dir
    assert stat.S_IMODE(store.session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
    journal_path = store.session_dir / "background-output.journal"
    assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
    assert journal_path.stat().st_size > 0

    await tasks.close()
    store.close()
    manager.delete(opened.metadata.session_id)
    assert not archive_path.exists()


@pytest.mark.asyncio
async def test_background_slow_log_writer_does_not_block_loop_and_cursors_stay_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = BackgroundTaskRegistry(output_limit=4, call_limit=4)
    original_open_log = tasks._open_task_log

    class SlowLog:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return self._handle.fileno()

        def write(self, data: bytes) -> int:
            time.sleep(0.25)
            return self._handle.write(data)

        def flush(self) -> None:
            self._handle.flush()

        def close(self) -> None:
            self._handle.close()

    monkeypatch.setattr(
        tasks, "_open_task_log", lambda task_id: SlowLog(original_open_log(task_id))
    )
    ticks: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    task_id, _ = await tasks.start("printf 0123456789abcdef", tmp_path)
    await _wait_for_exit(tasks, task_id)
    stop.set()
    await ticker_task

    gaps = [later - earlier for earlier, later in pairwise(ticks)]
    assert gaps and max(gaps) < 0.1
    cursor = 0
    for expected in ("0123", "4567", "89ab", "cdef"):
        result = await tasks.output(task_id, since=cursor)
        assert result["output"] == expected
        cursor = result["cursor"]
    assert cursor == 16
    await tasks.close()


@pytest.mark.asyncio
async def test_background_output_ring_trims_at_utf8_boundary(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(output_limit=4)
    task_id, _ = await tasks.start(
        _python("import sys; sys.stdout.buffer.write('a€bc'.encode())"),
        tmp_path,
    )
    await _wait_for_exit(tasks, task_id)

    result = await tasks.output(task_id)
    assert result["output"] == "a€bc"
    assert "�" not in result["output"]
    assert result["output"].encode().decode() == result["output"]
    await tasks.close()


@pytest.mark.asyncio
async def test_background_output_cursor_tracks_outer_tool_cap(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, max_output_chars=10_000, skill_catalog=SkillCatalog.empty())
    started = await registry.execute(
        ToolCall(
            "start",
            "run_background",
            {"command": "printf " + "x" * 20_000},
        )
    )
    task_id = started["structuredContent"]["task_id"]
    await _wait_for_exit(registry.background_tasks, task_id)

    first = await registry.execute(
        ToolCall("output", "task_output", {"task_id": task_id})
    )
    first_cursor = first["structuredContent"]["cursor"]
    assert first_cursor == len(first["structuredContent"]["output"])
    assert 0 < first_cursor < 10_000
    assert len(first["content"][0]["text"]) <= 10_000
    assert first["structuredContent"]["has_more"] is True

    second = await registry.execute(
        ToolCall(
            "output-2",
            "task_output",
            {"task_id": task_id, "since": first["structuredContent"]["cursor"]},
        )
    )
    second_cursor = second["structuredContent"]["cursor"]
    assert second_cursor == first_cursor + len(second["structuredContent"]["output"])
    assert first_cursor < second_cursor <= 20_000
    assert len(second["content"][0]["text"]) <= 10_000
    await registry.close()


@pytest.mark.asyncio
async def test_task_output_text_announces_more_and_location(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path, max_output_chars=10_000, skill_catalog=SkillCatalog.empty()
    )
    started = await registry.execute(
        ToolCall(
            "start-notice",
            "run_background",
            {"command": "printf " + "x" * 40_000},
        )
    )
    task_id = started["structuredContent"]["task_id"]
    await _wait_for_exit(registry.background_tasks, task_id)

    result = await registry.execute(
        ToolCall("output-notice", "task_output", {"task_id": task_id})
    )
    text = result["content"][0]["text"]
    cursor = result["structuredContent"]["cursor"]

    assert result["structuredContent"]["has_more"] is True
    assert "has_more: true" in text
    assert f"next_cursor: {cursor}" in text
    assert "total_bytes: 40000" in text
    assert f"remaining_bytes: {40_000 - cursor}" in text
    assert f"task-output://{task_id}" in text
    assert f"retrieve with task_output (since={cursor}); do not use read" in text
    assert len(text) <= registry.max_output_chars
    await registry.close()


@pytest.mark.asyncio
async def test_background_output_caps_at_utf8_boundary(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(call_limit=4)
    task_id, _ = await tasks.start(
        _python("import sys; sys.stdout.buffer.write('ab€x'.encode())"),
        tmp_path,
    )
    await _wait_for_exit(tasks, task_id)

    first = await tasks.output(task_id)
    assert first["output"].startswith("ab")
    assert "�" not in first["output"]
    assert first["cursor"] == 2
    second = await tasks.output(task_id, since=first["cursor"])
    assert second["output"] == "€x"
    await tasks.close()


@pytest.mark.asyncio
async def test_background_task_stays_running_for_detached_group_member(
    tmp_path: Path,
) -> None:
    tasks = BackgroundTaskRegistry()
    task_id, pid = await tasks.start(
        _python(
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        ),
        tmp_path,
    )
    await asyncio.sleep(0.1)
    assert (await tasks.output(task_id))["running"] is True
    assert _group_exists(pid)

    result = await tasks.kill(task_id)
    assert result["running"] is False
    assert not _group_exists(pid)
    await tasks.close()


@pytest.mark.asyncio
async def test_background_kill_escalates_for_term_ignoring_process(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.03)
    task_id, _ = await tasks.start(
        _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
        tmp_path,
    )
    result = await tasks.kill(task_id)
    assert result["running"] is False
    assert result["exit_code"] in {-signal.SIGTERM, -signal.SIGKILL}
    assert not _group_exists(tasks.records[0].pid)
    await tasks.close()


@pytest.mark.asyncio
async def test_start_during_shutdown_is_rejected_and_nothing_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.03)
    await tasks.start("sleep 30", tmp_path)
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    original_close_stdin = tasks._close_stdin

    async def pause_close_stdin(record: _BackgroundRecord) -> None:
        shutdown_started.set()
        await release_shutdown.wait()
        await original_close_stdin(record)

    monkeypatch.setattr(tasks, "_close_stdin", pause_close_stdin)
    closing = asyncio.create_task(tasks.close())
    try:
        await shutdown_started.wait()
        with pytest.raises(RuntimeError, match="registry is closing"):
            await tasks.start("sleep 30", tmp_path)
        with pytest.raises(RuntimeError, match="registry is closing"):
            tasks.open_log(tmp_path / "late.log")
        directory_fd = os.open(tmp_path, os.O_RDONLY)
        try:
            with pytest.raises(RuntimeError, match="registry is closing"):
                tasks.bind_session_dir(tmp_path, directory_fd)
        finally:
            os.close(directory_fd)

        release_shutdown.set()
        await closing
        assert all(not _group_exists(record.pid) for record in tasks.records)
    finally:
        release_shutdown.set()
        await asyncio.gather(closing, return_exceptions=True)
        for record in tasks.records:
            if _group_exists(record.pid):
                os.killpg(record.pid, signal.SIGKILL)


def test_asyncio_run_teardown_during_close_terminates_all_groups(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import os
import signal
import sys

from zeta.tools._shared.process import BackgroundTaskRegistry, _group_exists


async def main():
    tasks = BackgroundTaskRegistry(term_grace=0.25)
    _, pid = await tasks.start(
        f"{sys.executable} -c \\\"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)\\\"",
        sys.argv[1],
    )
    await asyncio.sleep(0.1)
    asyncio.create_task(tasks.close())
    await asyncio.sleep(0.05)
    return tasks, pid


tasks, pid = asyncio.run(main())
alive = _group_exists(pid)
closed = tasks._closed
if alive:
    os.killpg(pid, signal.SIGKILL)
print({"group_alive_after_asyncio_run_teardown": alive, "closed": closed})
raise SystemExit(1 if alive or not closed else 0)
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_asyncio_run_teardown_right_after_close_starts_terminates_all_groups(
    tmp_path: Path,
) -> None:
    # Teardown lands one loop step after close() begins: a separately
    # scheduled shutdown task would be cancelled before its first instruction.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import os
import signal
import sys

from zeta.tools._shared.process import BackgroundTaskRegistry, _group_exists


async def main():
    tasks = BackgroundTaskRegistry(term_grace=0.25)
    _, pid = await tasks.start(
        f"{sys.executable} -c \\\"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)\\\"",
        sys.argv[1],
    )
    await asyncio.sleep(0.1)
    asyncio.create_task(tasks.close())
    await asyncio.sleep(0)
    return tasks, pid


tasks, pid = asyncio.run(main())
alive = _group_exists(pid)
closed = tasks._closed
if alive:
    os.killpg(pid, signal.SIGKILL)
print({"group_alive_after_asyncio_run_teardown": alive, "closed": closed})
raise SystemExit(1 if alive or not closed else 0)
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_close_cancelled_midway_still_terminates_all_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.25)
    task_ids = [
        await tasks.start(
            _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
            tmp_path,
        )
        for _ in range(3)
    ]
    term_sent = asyncio.Event()
    original_signal_group = process_module._signal_group

    def signal_group(process: object, signum: signal.Signals) -> None:
        if signum == signal.SIGTERM:
            term_sent.set()
            return
        original_signal_group(process, signum)

    monkeypatch.setattr(process_module, "_signal_group", signal_group)
    try:
        closing = asyncio.create_task(tasks.close())
        await term_sent.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        assert all(not _group_exists(record.pid) for record in tasks.records)
        assert await tasks.close() == ()
        assert len(task_ids) == 3
    finally:
        for record in tasks.records:
            if _group_exists(record.pid):
                os.killpg(record.pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_close_terminates_many_tasks_in_one_grace_period(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.25)
    task_ids = [
        await tasks.start(
            _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
            tmp_path,
        )
        for _ in range(20)
    ]

    started = asyncio.get_running_loop().time()
    await tasks.close()
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 1
    assert all(not _group_exists(tasks.records[index].pid) for index in range(len(task_ids)))


@pytest.mark.asyncio
async def test_background_approval_is_enforced(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "denied-session")
    denied = ToolRegistry(
        tmp_path,
        session_store=store,
        approval_store=store,
        approval_policy=ApprovalPolicy(
            always_deny={"run_background"},
            default=ApprovalDecision.ALLOW,
        ),
skill_catalog=SkillCatalog.empty(),
    )
    result = await denied.execute(
        ToolCall("deny", "run_background", {"command": "sleep 30"})
    )
    assert result["isError"] is True
    await denied.close()


@pytest.mark.asyncio
async def test_more_than_eight_background_tasks_run_concurrently(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry()
    gate = tmp_path / "release"
    command = f"while ! test -e {shlex.quote(str(gate))}; do sleep 0.01; done"
    task_ids = [await tasks.start(command, tmp_path) for _ in range(12)]
    assert len(task_ids) == 12
    for _ in range(1000):
        if tasks.running_count == 12:
            break
        await asyncio.sleep(0.01)
    assert tasks.running_count == 12

    try:
        gate.touch()
        await asyncio.gather(*(tasks.wait(task_id) for task_id, _ in task_ids))
        assert tasks.running_count == 0
    finally:
        gate.touch()
        await tasks.close()


@pytest.mark.asyncio
async def test_background_resume_marks_old_task_exited(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    first = ToolRegistry(store.cwd, session_store=store, skill_catalog=SkillCatalog.empty())
    started = await first.execute(
        ToolCall("start", "run_background", {"command": "sleep 30"})
    )
    task_id = started["structuredContent"]["task_id"]
    await first.close()

    resumed_store = ConversationStore(
        tmp_path / "sessions", session_id=store.session_id
    )
    resumed = ToolRegistry(resumed_store.cwd, session_store=resumed_store, skill_catalog=SkillCatalog.empty())
    output = await resumed.execute(
        ToolCall("output", "task_output", {"task_id": task_id})
    )
    assert output["structuredContent"]["running"] is False
    assert "previous session" in output["structuredContent"]["note"]
    await resumed.close()


def test_background_footer_segment_stays_visible_at_narrow_width() -> None:
    assert "bg 2" in format_status("fake", "offline", "idle", background_count=2).plain
    narrow = format_status(
        "fake",
        "offline",
        "idle",
        background_count=2,
        vim_state="NORMAL",
        width=30,
    ).plain
    assert "bg 2" in narrow
    assert "NORMAL" not in narrow


@pytest.mark.asyncio
async def test_task_input_approval_is_scoped_to_each_data_chunk(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "store")
    policy = ApprovalPolicy(
        always_allow={"task_input(safe*)"},
        always_deny={"task_input(unsafe*)"},
        default="ask",
        store=store,
    )
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        enforce_approvals=True,
        skill_catalog=SkillCatalog.empty(),
    )
    definition = registry.definitions_by_name["task_input"]
    assert definition.requires_approval is True
    assert definition.approval_subject == "data"
    assert policy.decide(
        registry.resolve_call("task_input", {"data": "safe input"})
    ) is ApprovalDecision.ALLOW
    assert policy.decide(
        registry.resolve_call("task_input", {"data": "unsafe input"})
    ) is ApprovalDecision.DENY
    denied = await registry.execute(
        ToolCall("deny-input", "task_input", {"task_id": "unknown", "data": "unsafe"})
    )
    assert denied["isError"] is True
    assert registry.denied_tools == ["task_input"]
    await registry.close()


def test_background_tools_are_discovered() -> None:
    registry = ToolRegistry(Path("."), skill_catalog=SkillCatalog.empty())
    assert {schema["name"] for schema in registry.schemas} >= {
        "run_background",
        "task_output",
        "task_kill",
        "task_input",
    }


@pytest.mark.parametrize(
    "payload",
    [b"[" * 65 + b"]" * 65, b"[" * 10_000 + b"]" * 10_000, b"\xff"],
    ids=["depth65", "depth10000", "binary"],
)
async def test_registry_setup_ignores_corrupt_background_state(
    tmp_path: Path, payload: bytes
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    path = store.session_dir / "background_tasks.json"
    path.write_bytes(payload)
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
    assert registry.background_tasks.running_count == 0
    assert path.read_bytes() == payload
    await registry.close()
