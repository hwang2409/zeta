"""Tests for session-scoped background process tools."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    TextContent,
    ToolCall,
    ToolSchema,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools._shared.process import (
    BackgroundTaskRegistry,
    _BackgroundRecord,
    _group_exists,
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
        Message(MessageRole.USER, [TextContent("before notification")])
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


def test_tui_presented_cancellation_remains_available_to_headless_consumer(
    tmp_path: Path,
) -> None:
    from zeta.agent.notifications import build_notification_system_message

    store = ConversationStore(tmp_path)
    notification = store.append_agent_notification(
        "macro:shutdown",
        child_session_path="/tmp/macro.log",
        description="/matrix",
        status="canceled",
        text="background macro canceled on session shutdown",
        background_metadata=("background_macro", "session_shutdown"),
    )
    store.mark_agent_notification_presented_to_tui(notification.id)

    message = build_notification_system_message(store)

    assert message is not None
    assert message.metadata["notifications"] == [
        {
            "notification_id": notification.id,
            **notification.data,
            "kind": "agent_completion",
        }
    ]


def test_legacy_notification_without_kind_loads_as_agent_completion(tmp_path: Path) -> None:
    # S7: a notification row persisted without a `kind` field is treated as an
    # agent_completion by every durable-notification consumer.
    from zeta.agent.notifications import (
        build_notification_system_message,
        notification_events,
    )

    store = ConversationStore(tmp_path, session_id="legacy")
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
    message = build_notification_system_message(store)
    assert message is not None
    assert message.metadata["notifications"][0]["kind"] == "agent_completion"
    events = list(notification_events(store))
    assert len(events) == 1
    assert events[0].data["kind"] == "agent_completion"


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
        await _collect(loop.run_turn("first"))
        await _collect(loop.run_turn("second"))

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
    await _collect(loop.run_turn("start"))
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
        await _collect(loop.run_turn("start"))
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
    assert first["output"].startswith("[output truncated; dropped 8 bytes]\n")
    assert "89ab" in first["output"]
    assert first["cursor"] == 12
    second = await tasks.output(task_id, since=first["cursor"])
    assert second["output"] == "cdef"
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
    assert result["output"] == "[output truncated; dropped 4 bytes]\nbc"
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
    assert first["structuredContent"]["cursor"] == 9_931
    assert len(first["structuredContent"]["output"]) == 9_931
    assert len(first["content"][0]["text"]) == 9_999

    second = await registry.execute(
        ToolCall(
            "output-2",
            "task_output",
            {"task_id": task_id, "since": first["structuredContent"]["cursor"]},
        )
    )
    assert second["structuredContent"]["cursor"] == 19_862
    assert len(second["structuredContent"]["output"]) == 9_931
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
    command = _python(
        "import time; from pathlib import Path; "
        f"gate = Path({str(gate)!r}); "
        "while not gate.exists(): time.sleep(0.01)"
    )
    task_ids = [await tasks.start(command, tmp_path) for _ in range(12)]
    assert len(task_ids) == 12
    for _ in range(1000):
        if tasks.running_count == 12:
            break
        await asyncio.sleep(0.01)
    assert tasks.running_count == 12

    gate.touch()
    await asyncio.gather(*(tasks.wait(task_id) for task_id, _ in task_ids))
    assert tasks.running_count == 0
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
    assert policy.decide("task_input", {"data": "safe input"}) is ApprovalDecision.ALLOW
    assert policy.decide("task_input", {"data": "unsafe input"}) is ApprovalDecision.DENY
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
