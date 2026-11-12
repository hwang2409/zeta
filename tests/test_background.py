"""Tests for session-scoped background process tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import pty
import shlex
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

import zeta.agent.pending_delivery as pending_delivery_module
import zeta.tools._shared.background_task_owner as owner_module
import zeta.tools._shared.process as process_module
from tests.support.fake_backend import (
    FakeBackend,
    PerActorFakeBackend,
    ScriptedTurn,
)
from tests.support.owned import await_or_fail
from tests.support.processes import (
    kill_group_if_present,
    read_pid_file,
    wait_for_path_command,
    wait_pid_exited,
)
from zeta.agent.background_owner import BackgroundAgentOwner
from zeta.agent.receipt import receipt_tool_result
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.server import event_contract
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools._shared import background_spawn as spawn_module
from zeta.tools._shared.background_log_writer import _OrderedLogWriter
from zeta.tools._shared.background_output_archive import _BackgroundOutputArchive
from zeta.tools._shared.background_spawn import _group_exists, _terminate_process
from zeta.tools._shared.background_task_types import TaskExitInput, _BackgroundRecord
from zeta.tools._shared.process import BackgroundTaskRegistry
from zeta.tui.render import format_status, render_agent_notification, render_event


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


def _renderable_plain(renderable: object, *, width: int = 80) -> str:
    output = StringIO()
    Console(file=output, width=width, force_terminal=True).print(renderable)
    return Text.from_ansi(output.getvalue()).plain


async def _wait_for_exit(registry: BackgroundTaskRegistry, task_id: str) -> None:
    await asyncio.wait_for(registry.wait(task_id), timeout=30)


@pytest.mark.asyncio
async def test_termination_does_not_signal_reused_descendant_pid(monkeypatch) -> None:
    snapshots = iter(({10: {42: "old"}}, {42: "new"}))
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 10
        returncode = None

        async def wait(self) -> None:
            await asyncio.sleep(0.01)
            self.returncode = -9

    def snapshot(_):
        return next(snapshots)

    monkeypatch.setattr(spawn_module, "_snapshot_descendants_sync", snapshot)
    monkeypatch.setattr(spawn_module, "_signal_group", lambda process, number: None)
    monkeypatch.setattr(spawn_module, "_signal_pids", lambda pids, number: signals.extend(
        (pid, number) for pid in pids
    ))

    await _terminate_process(Process(), 0.001)
    assert signals == [(42, signal.SIGTERM)]


def _task_notification(item: TaskExitInput) -> pending_delivery_module.NotificationInput:
    data: dict[str, object] = {
        "task_id": item.task_id,
        "headline": item.command,
        "exit_code": item.exit_code,
        "output_tail": item.output_tail,
        "background_owner": item.owner,
        "background_phase": item.phase,
    }
    if item.log_path is not None:
        data["log_path"] = item.log_path
    if item.note is not None:
        data["note"] = item.note
    return pending_delivery_module.NotificationInput(
        "task_exit",
        data,
        dedupe_key=item.task_id,
        tui_presented=item.tui_presented,
    )


def _notification_registry(
    store: ConversationStore,
    *,
    wake_sink=None,
) -> BackgroundTaskRegistry:
    flow = pending_delivery_module.StepDeliveryFlow(store, wake_sink=wake_sink)
    return BackgroundTaskRegistry(
        notification_sink=lambda item: flow.produce(_task_notification(item))
    )


@pytest.mark.asyncio
async def test_shutdown_snapshot_does_not_block_live_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    loop_proved_live = asyncio.Event()
    snapshot_thread_id: int | None = None
    snapshot_calls = 0

    def slow_snapshot(root_pids: Sequence[int]) -> dict[int, dict[int, str]]:
        nonlocal snapshot_calls, snapshot_thread_id
        snapshot_calls += 1
        if snapshot_calls == 1:
            snapshot_thread_id = threading.get_ident()
            snapshot_started.set()

            async def prove_loop_live() -> None:
                loop_proved_live.set()

            asyncio.run_coroutine_threadsafe(prove_loop_live(), loop).result(
                timeout=10
            )
            assert release_snapshot.wait(timeout=10), "snapshot release timed out"
        return {} if root_pids is None else {pid: {} for pid in root_pids}

    monkeypatch.setattr(spawn_module, "_snapshot_descendants_sync", slow_snapshot)
    registry = BackgroundTaskRegistry(term_grace=0.01)
    await registry.start("sleep 30", tmp_path)
    closing = asyncio.create_task(registry.close())
    try:
        assert await asyncio.to_thread(snapshot_started.wait, 10)
        await asyncio.wait_for(loop_proved_live.wait(), 10)
        assert snapshot_thread_id != loop_thread_id
        release_snapshot.set()
        await asyncio.wait_for(closing, 10)
    finally:
        release_snapshot.set()
        await asyncio.wait_for(asyncio.gather(closing, return_exceptions=True), 10)


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
    assert registry.background_tasks.task_start_generation == 0
    result = await registry.execute(
        ToolCall("start", tool_name, arguments),
        _background=background,
    )
    task_id = result["structuredContent"]["task_id"]
    assert registry.background_tasks.task_start_generation == 1
    await _wait_for_exit(registry.background_tasks, task_id)
    lifecycle = [notice for notice in notices if notice.task_id == task_id]
    assert [notice.phase for notice in lifecycle] == ["started", "natural_exit"]
    expected_owner = "background_macro" if background else "run_background"
    assert [notice.owner for notice in lifecycle] == [expected_owner, expected_owner]
    await registry.close()


@pytest.mark.asyncio
async def test_cancelled_wait_caller_does_not_cancel_registry_monitor(
    tmp_path: Path,
) -> None:
    registry = BackgroundTaskRegistry()
    task_id, _ = await registry.start("sleep 30", tmp_path)
    wait_started = asyncio.Event()

    async def wait_for_task() -> None:
        wait_started.set()
        await asyncio.wait_for(registry.wait(task_id), 10)

    waiter = asyncio.create_task(wait_for_task())
    try:
        await asyncio.wait_for(wait_started.wait(), 10)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 10)
        assert registry.snapshot()[0].running is True
    finally:
        await asyncio.wait_for(
            registry.kill(task_id, cause="test requested task stop"), 10
        )
        await asyncio.wait_for(registry.close(), 10)




async def _collect(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_background_exit_persists_task_notification(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session")
    wakes = 0

    def wake() -> None:
        nonlocal wakes
        wakes += 1

    registry = _notification_registry(store, wake_sink=lambda _source: wake())
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
async def test_registry_close_persists_specific_kill_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "session")
    wakes = 0

    def wake() -> None:
        nonlocal wakes
        wakes += 1

    flow = pending_delivery_module.StepDeliveryFlow(
        store, wake_sink=lambda _source: wake()
    )
    tools = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )
    registry = tools.background_tasks
    registry.set_notification_sink(
        lambda item: flow.produce(_task_notification(item))
    )
    task_id, _ = await registry.start(_python("import time; time.sleep(30)"), Path.cwd())
    terminate_processes = process_module._terminate_processes

    async def assert_cause_before_signal(
        processes, term_grace, *, finish, after_signal=None
    ):
        rows = json.loads((store.session_dir / "background_tasks.json").read_text())
        row = next(row for row in rows if row["task_id"] == task_id)
        assert row["running"] is True
        assert row["terminal_phase"] == "session_shutdown"
        assert row["note"] == "tool registry closed"
        return await terminate_processes(
            processes, term_grace, finish=finish, after_signal=after_signal
        )

    # test-contract: seam inspect the durable record immediately before signaling
    monkeypatch.setattr(
        process_module, "_terminate_processes", assert_cause_before_signal
    )
    try:
        registry.begin_shutdown("tool registry closed")
        killed = await asyncio.wait_for(tools.close(), 10)

        assert task_id in killed
        notifications = store.agent_notifications(pending_only=False)
        assert len(notifications) == 1
        assert notifications[0].data["task_id"] == task_id
        assert notifications[0].data["background_phase"] == "session_shutdown"
        assert notifications[0].data["note"] == "tool registry closed"
        assert wakes == 1
    finally:
        await asyncio.wait_for(tools.close(), 10)
        store.close()


def _closed_session_kill_phase_and_line(
    root_dir: Path, session_id: str
) -> tuple[str, str]:
    """Read the persisted kill phase and the rendered task receipt line.

    Reopens the already-closed session so the assertions run against the
    durable record and the real TUI render path, not in-memory state.
    """
    rows = json.loads((root_dir / session_id / "background_tasks.json").read_text())
    phases = {row["terminal_phase"] for row in rows}
    assert phases == {rows[0]["terminal_phase"]}, phases
    reopened = ConversationStore(root_dir, session_id=session_id)
    try:
        receipts = [
            entry
            for entry in reopened.agent_notifications(pending_only=False)
            if entry.data.get("kind") == "task_exited"
        ]
        assert len(receipts) == 1, receipts
        rendered = render_agent_notification(
            StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=receipts[0].data)
        )
        assert rendered is not None
        line = _renderable_plain(rendered).strip()
    finally:
        reopened.close()
    return phases.pop(), line


_SESSION_EXIT_LINE = "background tasks killed on session exit"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["tui_exit", "serve_close", "finish_gate_kill"])
async def test_session_close_kill_phase_by_surface(
    tmp_path: Path, surface: str
) -> None:
    """Session close stores session_shutdown and renders the exit line.

    ctrl-d, /new, and serve close all tear down through close_session ->
    AgentLoop.close; #1193 regressed those paths to phase="task_kill", which
    hid the session-exit line (render.py) and kept the wrong recovery phase.
    The finish-gate case still kills tasks as task_kill and renders the plain
    receipt, so this guards both phases through the public interfaces.
    """
    sleep_30 = _python("import time; time.sleep(30)")
    if surface == "serve_close":
        from zeta.server.runtime import ServerRuntime

        runtime = ServerRuntime(tmp_path, provider="codex")
        await runtime.create_session()
        loop = runtime.loop
        assert loop is not None
        root_dir, session_id = loop.store.root_dir, loop.store.session_id
        await loop.tool_registry.background_tasks.start(sleep_30, tmp_path)
        await runtime.close()
    else:
        from zeta.runtime.cleanup import close_session

        store = ConversationStore(tmp_path / "session")
        loop = AgentLoop(
            FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
        )
        await loop.activate()
        root_dir, session_id = store.root_dir, store.session_id
        await loop.tool_registry.background_tasks.start(sleep_30, tmp_path)
        if surface == "finish_gate_kill":
            # The finish gate marks remaining tasks as an explicit kill before
            # the loop closes; that must stay task_kill, not session_shutdown.
            loop.tool_registry.background_tasks.begin_shutdown(
                "finish-gate task safety bound expired", phase="task_kill"
            )
        await close_session(loop)

    phase, line = _closed_session_kill_phase_and_line(root_dir, session_id)
    if surface == "finish_gate_kill":
        assert phase == "task_kill"
        assert _SESSION_EXIT_LINE not in line
        assert "exited" in line
    else:
        assert phase == "session_shutdown"
        assert _SESSION_EXIT_LINE in line


@pytest.mark.asyncio
async def test_task_exit_recovered_after_restart_exactly_once(tmp_path: Path) -> None:
    # S5: a previously-running row with no existing task_exited notification
    # yields exactly one recovery notification on the normal production path,
    # and repeated resumes never duplicate it.
    from tests.support.fake_backend import FakeBackend
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
    await loop1.activate()
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
    await loop2.activate()
    assert len(task_exits(store2)) == 1
    await loop2.close()
    store2.close()


@pytest.mark.asyncio
async def test_failed_run_background_start_uses_tool_error_as_canonical_result(tmp_path: Path) -> None:
    # A failed run_background start is rendered by its error tool card; do not
    # persist a second task_exited receipt.
    store = ConversationStore(tmp_path / "session")
    registry = _notification_registry(store)
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


async def _cancel_gate(owner, entered, release, *, name: str) -> None:
    lifecycle = owner.lifecycle
    assert lifecycle is not None
    await await_or_fail(entered.wait(), what=f"{name} entered")
    lifecycle.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await await_or_fail(owner.wait(), what=f"{name} cancellation completion")


async def _cancel_at_gate(owner, install, *, name: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        entered.set()
        await await_or_fail(release.wait(), what=f"{name} release")

    await install(hold)
    await _cancel_gate(owner, entered, release, name=name)


async def _cancel_process_wait(registry, owner, monkeypatch) -> None:
    process = owner.process
    assert process is not None
    original = process.wait

    async def install(hold):
        async def wait():
            await hold()
            return await original()
        monkeypatch.setattr(process, "wait", wait)

    await _cancel_at_gate(owner, install, name="process-wait")


async def _cancel_writer_close(registry, owner, monkeypatch) -> None:
    original = _OrderedLogWriter.close

    async def install(hold):
        async def close(writer):
            await hold()
            return await original(writer)
        monkeypatch.setattr(_OrderedLogWriter, "close", close)

    await _cancel_at_gate(owner, install, name="writer-close")


async def _cancel_archive(registry, owner, monkeypatch) -> None:
    tasks = registry.background_tasks
    archive = tasks._archive
    assert archive is not None
    original_append = archive.append
    append_entered = asyncio.Event()
    release_append = threading.Event()
    loop = asyncio.get_running_loop()
    source_fds: list[int] = []

    def append(task_id: str, source_fd: int, length: int) -> None:
        source_fds.append(source_fd)
        loop.call_soon_threadsafe(append_entered.set)
        assert release_append.wait(timeout=10)
        original_append(task_id, source_fd, length)

    # test-contract: seam cancel the real cleanup task inside the real archive append.
    monkeypatch.setattr(archive, "append", append)
    try:
        await await_or_fail(append_entered.wait(), what="archive append entered")
        lifecycle = owner.lifecycle
        assert lifecycle is not None
        cleanup = lifecycle.get_stack()[-1].f_locals["cleanup"]
        cleanup.cancel()
        cancellation_delivered = asyncio.Event()
        loop.call_soon(cancellation_delivered.set)
        await await_or_fail(
            cancellation_delivered.wait(), what="archive cancellation delivered"
        )

        os.fstat(source_fds[0])
        assert not owner.terminal_durable.is_set()
        assert archive.pread(owner.record.task_id, 0, owner.record.total_bytes) == b""

        release_append.set()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(owner.wait(), what="archive cancellation completion")

        assert owner.terminal_durable.is_set()
        assert archive.pread(
            owner.record.task_id, 0, owner.record.total_bytes
        ) == bytes(owner.record.output)
    finally:
        release_append.set()


async def _cancel_persist(registry, owner, monkeypatch) -> None:
    owner_registry = registry.background_tasks
    original = owner_registry._commit_transition

    async def install(hold):
        async def commit(*args, **kwargs):
            await hold()
            return await original(*args, **kwargs)
        # test-contract: seam gate terminal persistence at the owner await
        monkeypatch.setattr(owner_registry, "_commit_transition", commit)

    await _cancel_at_gate(owner, install, name="persist")


async def _cancel_evidence(registry, owner, monkeypatch) -> None:
    owner_registry = registry.background_tasks
    original = owner_registry._notification_sink

    async def install(hold):
        async def sink(item):
            await hold()
            return await original(item)
        # test-contract: seam gate evidence delivery at the owner await
        monkeypatch.setattr(owner_registry, "_notification_sink", sink)

    await _cancel_at_gate(owner, install, name="evidence")


async def _cancel_notice(registry, owner, monkeypatch) -> None:
    entered = asyncio.Event()
    owner_registry = registry.background_tasks
    original = owner_registry._notice

    def notice(value):
        entered.set()
        lifecycle = owner.lifecycle
        assert lifecycle is not None
        lifecycle.cancel()
        original(value)

    # test-contract: seam gate the synchronous lifecycle notice
    monkeypatch.setattr(owner_registry, "_notice", notice)
    await await_or_fail(entered.wait(), what="notice entered")
    with pytest.raises(asyncio.CancelledError):
        await await_or_fail(owner.wait(), what="notice cancellation completion")


_CANCEL_ROW_HANDLERS = {
    "process-wait": _cancel_process_wait,
    "ordered-writer-close": _cancel_writer_close,
    "archive": _cancel_archive,
    "persist": _cancel_persist,
    "evidence-submit": _cancel_evidence,
    "notice": _cancel_notice,
}


async def _cancel_close_at(
    registry, store, task_id: str, row: str, install
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    await install(entered, release)
    closing = asyncio.create_task(registry.close())
    try:
        await await_or_fail(entered.wait(), what=f"{row} entered")
        closing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(closing, what=f"{row} cancellation completion")
        await await_or_fail(registry.close(), what=f"{row} close completion")
        rows = json.loads(
            (store.session_dir / "background_tasks.json").read_text()
        )
        assert rows and all(not item["running"] for item in rows)
        assert len(store.agent_notifications(pending_only=False)) == 1
        assert (await registry.execute(
            ToolCall("output", "task_output", {"task_id": task_id})
        ))["structuredContent"]["running"] is False
    finally:
        release.set()
        if not closing.done():
            closing.cancel()
            await asyncio.wait_for(
                asyncio.gather(closing, return_exceptions=True), 10
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        "1-natural-exit-wins-kill-race",
        "2-agent-kill-wins",
        "3-external-cause",
        "terminal-notify-follows-publish",
        "natural-exit-success",
        "persist-failure-retries-same-candidate",
        "duplicate-input",
        "durable-receipt-dedupe-after-eviction",
        "live-close-defers-receipt",
        "restart-recovers-running-row",
        "suppressed-task-stays-suppressed",
        "legacy-pending-recovers-once",
        "cancel-process-wait",
        "cancel-reader-drain",
        "cancel-live-group-probe",
        "cancel-ordered-writer-close",
        "cancel-archive",
        "cancel-persist",
        "cancel-evidence-submit",
        "cancel-notice",
        "cancel-close-snapshot",
        "cancel-close-finish-owners",
        "cancel-close-terminate",
        "cancel-close-wait-until-durable",
        "cancel-close-aggregate-notice",
        "cancel-close-coordinator",
        "archive-failure-recovers-once",
        "restart-durable-terminal-notify-true",
        "restart-durable-terminal-notify-false",
        "two-task-close-ordering",
        "two-task-close-cancellation",
        "persistent-terminal-write-failure-close",
        "crash-before-aggregate",
        "group-exit-after-positive-probe",
    ],
)
async def test_background_task_owner_state_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    async with contextlib.AsyncExitStack() as stack:
        store = ConversationStore(tmp_path / row)
        stack.callback(store.close)
        registry = ToolRegistry(
            tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
        )
        stack.push_async_callback(registry.close)
        flow = pending_delivery_module.StepDeliveryFlow(store)
        registry.background_tasks.set_notification_sink(
            lambda item: flow.produce(_task_notification(item))
        )

        async def start_registered(command: str) -> str:
            result = await registry.execute(
                ToolCall("start", "run_background", {"command": command})
            )
            return result["structuredContent"]["task_id"]

        async def output_registered(task_id: str) -> dict[str, object]:
            result = await registry.execute(
                ToolCall("output", "task_output", {"task_id": task_id})
            )
            return result["structuredContent"]

        if row == "cancel-live-group-probe":
            child_ready = tmp_path / "detached-probe-child"
            probe_entered = asyncio.Event()
            probed_pids: list[int] = []
            original_group_exists = owner_module._group_exists

            def group_exists(pid: int) -> bool:
                exists = original_group_exists(pid)
                if exists and child_ready.exists():
                    probed_pids.append(pid)
                    probe_entered.set()
                return exists

            # test-contract: seam hold cancellation at the live-group probe await
            monkeypatch.setattr(owner_module, "_group_exists", group_exists)
            child = "import os,signal; print(os.getpid(), flush=True); signal.pause()"
            leader = (
                "import pathlib,subprocess,sys\n"
                f"child = subprocess.Popen([sys.executable, '-c', {child!r}], "
                "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
                f"pathlib.Path({str(child_ready)!r}).write_text(child.stdout.readline())\n"
            )
            task_id = await start_registered(_python(leader))
            owner = registry.background_tasks._owners[task_id]
            leader_pid = owner.record.pid
            child_pid = await read_pid_file(child_ready)
            try:
                await await_or_fail(probe_entered.wait(), what="live group probe")
                assert probed_pids == [leader_pid]
                lifecycle = owner.lifecycle
                assert lifecycle is not None
                lifecycle.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(
                        owner.wait(), what="probe cancellation completion"
                    )
                await wait_pid_exited(child_pid)
                assert (await output_registered(task_id))["running"] is False
                rows = json.loads(
                    (store.session_dir / "background_tasks.json").read_text()
                )
                assert rows and all(not item["running"] for item in rows)
            finally:
                kill_group_if_present(leader_pid)
            return

        if row.startswith("cancel-close-") and row != "cancel-close-coordinator":
            task_id = await start_registered(_python("import time; time.sleep(30)"))
            owner = registry.background_tasks._owners[task_id]

            async def install(entered, release):
                if row == "cancel-close-snapshot":
                    async def pending_start():
                        entered.set()
                        await await_or_fail(release.wait(), what="snapshot release")
                    pending = asyncio.create_task(pending_start())
                    registry.background_tasks._pending_starts.add(pending)
                    pending.add_done_callback(
                        registry.background_tasks._pending_starts.discard
                    )
                elif row == "cancel-close-finish-owners":
                    original = owner.finish_for_close
                    async def finish(cause):
                        entered.set()
                        await await_or_fail(release.wait(), what="finish release")
                        return await original(cause)
                    monkeypatch.setattr(owner, "finish_for_close", finish)
                elif row == "cancel-close-terminate":
                    original = registry.background_tasks._terminate_owned
                    async def terminate(*args, **kwargs):
                        entered.set()
                        await await_or_fail(release.wait(), what="terminate release")
                        return await original(*args, **kwargs)
                    # test-contract: seam gate the close terminator await.
                    monkeypatch.setattr(registry.background_tasks, "_terminate_owned", terminate)
                elif row == "cancel-close-wait-until-durable":
                    original = owner.wait_until_durable
                    async def durable():
                        entered.set()
                        await await_or_fail(release.wait(), what="durable release")
                        return await original()
                    monkeypatch.setattr(owner, "wait_until_durable", durable)
                else:
                    original = registry.background_tasks._notice
                    def notice(value):
                        if hasattr(value, "tasks"):
                            entered.set()
                            current = asyncio.current_task()
                            # test-contract: seam cancel at aggregate notice.
                            assert current is not None
                            current.cancel()
                        return original(value)
                    # test-contract: seam gate the aggregate notice await.
                    monkeypatch.setattr(registry.background_tasks, "_notice", notice)

            await _cancel_close_at(registry, store, task_id, row, install)
            return

        if row == "cancel-close-coordinator":
            task_id = await start_registered(_python("import time; time.sleep(30)"))
            closing = asyncio.create_task(registry.close())
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 10)
            await asyncio.wait_for(registry.close(), 10)
            assert (await output_registered(task_id))["running"] is False
            return

        if row == "cancel-process-wait":
            entered = asyncio.Event()
            release = asyncio.Event()
            original_wait = asyncio.subprocess.Process.wait

            async def gated_wait(process):
                entered.set()
                await await_or_fail(release.wait(), what="process-wait release")
                return await original_wait(process)

            monkeypatch.setattr(asyncio.subprocess.Process, "wait", gated_wait)
            task_id = await start_registered(_python("print('cancel-safe', flush=True)"))
            owner = registry.background_tasks._owners[task_id]
            await _cancel_gate(owner, entered, release, name="process-wait")
            status = await output_registered(task_id)
            assert status["running"] is False
            assert status["exit_code"] in {0, -15, -signal.SIGTERM, -signal.SIGKILL}
            return

        if row.startswith("cancel-"):
            if row == "cancel-reader-drain":
                reader_drain_entered = asyncio.Event()
                reader_drain_release = asyncio.Event()
                read_blocked = asyncio.Event()
                original_read_output = registry.background_tasks._read_output

                async def read_output(*args):
                    await asyncio.wait_for(read_blocked.wait(), 10)
                    return await original_read_output(*args)

                # test-contract: seam keep the reader pending so the drain race takes the gated branch
                monkeypatch.setattr(
                    registry.background_tasks, "_read_output", read_output
                )

                original_start = owner_module._BackgroundTaskOwner.start

                def gated_start(self) -> None:
                    original_wait = self._finish_output_drain.wait

                    async def wait() -> bool:
                        reader_drain_entered.set()
                        await await_or_fail(
                            reader_drain_release.wait(), what="reader-drain release"
                        )
                        return await original_wait()

                    # test-contract: seam install the drain gate before the owner runs
                    self._finish_output_drain.wait = wait
                    return original_start(self)

                monkeypatch.setattr(
                    owner_module._BackgroundTaskOwner, "start", gated_start
                )
            task_id = await start_registered(_python("print('cancel-safe', flush=True)"))
            owner = registry.background_tasks._owners[task_id]
            if row == "cancel-reader-drain":
                await _cancel_gate(
                    owner,
                    reader_drain_entered,
                    reader_drain_release,
                    name="reader-drain",
                )
            else:
                handler = _CANCEL_ROW_HANDLERS[row.removeprefix("cancel-")]
                await handler(registry, owner, monkeypatch)
            status = await output_registered(task_id)
            assert status["running"] is False
            assert status["exit_code"] in {0, -15, -signal.SIGTERM, -signal.SIGKILL}
            assert owner.record.terminal_phase in {"natural_exit", "task_kill", "session_shutdown"}
            return

        if row == "archive-failure-recovers-once":
            task_id = await start_registered(_python("print('archive-me')"))
            archive = registry.background_tasks._archive
            assert archive is not None
            original_append = archive.append
            failed = False

            def fail_once(*args, **kwargs):
                nonlocal failed
                if not failed:
                    failed = True
                    raise OSError("archive failed")
                return original_append(*args, **kwargs)

            monkeypatch.setattr(archive, "append", fail_once)
            with pytest.raises(OSError, match="archive failed"):
                await _wait_for_exit(registry.background_tasks, task_id)
            status = await output_registered(task_id)
            assert failed is True
            assert status["running"] is False
            assert "archive failed" in status["note"] or "recovery" in status["note"]
            return

        if row in {"restart-durable-terminal-notify-true", "restart-durable-terminal-notify-false"}:
            await registry.close()
            task_id = "durable-terminal"
            persisted = {
                "task_id": task_id, "command": "old command", "pid": 12345,
                "running": False, "exit_code": 0, "note": "done",
                "owner": "run_background",
                "notify_on_exit": row.endswith("true"),
            }
            (store.session_dir / "background_tasks.json").write_text(
                json.dumps([persisted]), encoding="utf-8"
            )
            resumed = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
            stack.push_async_callback(resumed.close)
            pending = resumed.background_tasks.set_notification_sink(
                lambda item: flow.produce(_task_notification(item))
            )
            for item in pending:
                await resumed.background_tasks.deliver_recovered_exit(item)
            receipts = store.agent_notifications(pending_only=False)
            assert bool(receipts) is row.endswith("true")
            return

        if row == "persistent-terminal-write-failure-close":
            events: list[str] = []
            original_notice = registry.background_tasks._notice
            registry.background_tasks._notice = lambda notice: (
                events.append("aggregate") if hasattr(notice, "tasks") else None,
                original_notice(notice),
            )[1]
            task_id = await start_registered(_python("import time; time.sleep(30)"))
            original_commit = registry.background_tasks._commit_transition

            async def fail_terminal_write(rows, publish, notice):
                if any(
                    item["task_id"] == task_id and not item["running"]
                    for item in rows
                ):
                    raise OSError("terminal write failed permanently")
                return await original_commit(rows, publish, notice)

            # test-contract: seam inject persistent terminal-write failure
            monkeypatch.setattr(
                registry.background_tasks, "_commit_transition", fail_terminal_write
            )
            with pytest.raises(OSError, match="terminal write failed permanently"):
                await asyncio.wait_for(registry.close(), 10)
            # test-contract: seam aggregate callback is the required internal absence check
            assert events == []
            assert store.agent_notifications(pending_only=False) == []
            # The owner is intentionally still running in storage after the
            # bounded failure; finish fixture cleanup without retrying close.
            registry.background_tasks._closed = True
            registry.background_tasks._closing = False
            return

        if row in {"two-task-close-ordering", "two-task-close-cancellation"}:
            events: list[str] = []
            task_ids: list[str] = []
            original_notice = registry.background_tasks._notice

            def record_aggregate(notice):
                if hasattr(notice, "tasks"):
                    rows = json.loads(
                        (store.session_dir / "background_tasks.json").read_text()
                    )
                    by_id = {item["task_id"]: item for item in rows}
                    assert all(by_id[task_id]["running"] is False for task_id in task_ids)
                    events.append("aggregate")
                return original_notice(notice)

            registry.background_tasks._notice = record_aggregate
            original_produce = flow.produce

            async def record_receipt(item):
                events.append("receipt")
                return await original_produce(item)

            monkeypatch.setattr(flow, "produce", record_receipt)
            task_ids.extend(
                [
                    await start_registered(_python("import time; time.sleep(30)")),
                    await start_registered(_python("import time; time.sleep(30)")),
                ]
            )
            if row.endswith("cancellation"):
                closing = asyncio.create_task(registry.close())
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, 10)
            await asyncio.wait_for(registry.close(), 10)
            assert events.count("aggregate") == 1
            assert events.count("receipt") == 2
            assert events.index("aggregate") < events.index("receipt")
            return

        if row == "crash-before-aggregate":
            eligible = [
                await start_registered(_python("import time; time.sleep(30)"))
                for _ in range(2)
            ]
            suppressed, _ = await registry.background_tasks.start(
                _python("import time; time.sleep(30)"),
                tmp_path,
                notify_on_exit=False,
                owner="background_macro",
            )
            original_notice = registry.background_tasks._notice

            def crash_aggregate(notice):
                if hasattr(notice, "tasks"):
                    raise RuntimeError("aggregate crash")
                original_notice(notice)

            registry.background_tasks._notice = crash_aggregate
            registry.background_tasks.begin_shutdown("session shutdown")
            with pytest.raises(RuntimeError, match="aggregate crash"):
                await registry.close()
            registry.background_tasks._notice = original_notice
            persisted = json.loads(
                (store.session_dir / "background_tasks.json").read_text()
            )
            by_id = {item["task_id"]: item for item in persisted}
            assert all(by_id[task_id]["running"] is False for task_id in eligible)
            assert all(
                by_id[task_id]["terminal_phase"] == "session_shutdown"
                and "session" in by_id[task_id]["note"]
                for task_id in eligible
            )
            assert by_id[suppressed]["running"] is False
            assert by_id[suppressed]["notify_on_exit"] is False
            assert store.agent_notifications(pending_only=False) == []

            store.close()
            resumed_store = ConversationStore(
                tmp_path / row, session_id=store.session_id
            )
            resumed_loop = AgentLoop(
                FakeBackend([]),
                resumed_store,
                max_turns=1,
                skill_catalog=SkillCatalog.empty(),
            )
            resumed_notices = []
            resumed_loop.tool_registry.background_tasks.set_notice_sink(
                resumed_notices.append
            )
            assert len(resumed_loop._pending_task_recovery) == 2
            await resumed_loop.activate()
            receipts = [
                entry
                for entry in resumed_store.agent_notifications(pending_only=False)
                if entry.data.get("kind") == "task_exited"
            ]
            assert {entry.data["task_id"] for entry in receipts} == set(eligible)
            assert all(
                entry.data["background_phase"] == "session_shutdown"
                and "session" in entry.data["note"]
                for entry in receipts
            )
            assert not any(
                getattr(notice, "phase", None) == "session_shutdown"
                for notice in resumed_notices
            )
            flow = pending_delivery_module.StepDeliveryFlow(resumed_store)
            await flow.prepare_request("terminal_fallback", [], None)
            await flow.acknowledge_active()
            for receipt in receipts:
                resumed_store.append_compaction_marker(
                    "task exit receipt evicted",
                    receipt.seq,
                    receipt.seq,
                    kind="evict",
                    view=[
                        {
                            "seq": receipt.seq,
                            "message": Message(
                                MessageRole.ASSISTANT,
                                [TextContent("[task exit receipt evicted]")],
                                metadata={
                                    "context_evicted": True,
                                    "eviction_receipt": "assistant",
                                    "source_seq": receipt.seq,
                                },
                            ).to_dict(),
                        }
                    ],
                )
            assembler = ContextAssembler(
                resumed_store,
                token_budget=1000,
                retained_tail=1,
                token_counter=lambda _message: 1,
            )
            await assembler.assemble_context(
                force=True, bypass_eviction_hysteresis=True
            )
            assert all(
                any(
                    entry.type == "compaction"
                    and entry.data["source_seq_start"] <= receipt.seq
                    <= entry.data["source_seq_end"]
                    for entry in resumed_store.replay()
                )
                for receipt in receipts
            )
            await resumed_loop.close()
            resumed_store.close()

            second_store = ConversationStore(
                tmp_path / row, session_id=store.session_id
            )
            second_loop = AgentLoop(
                FakeBackend([]),
                second_store,
                max_turns=1,
                skill_catalog=SkillCatalog.empty(),
            )
            await second_loop.activate()
            assert len(
                [
                    entry
                    for entry in second_store.agent_notifications(pending_only=False)
                    if entry.data.get("kind") == "task_exited"
                ]
            ) == 2
            await second_loop.close()
            second_store.close()
            return

        if row == "group-exit-after-positive-probe":
            task_id = await start_registered(_python("print('positive-probe')"))
            await _wait_for_exit(registry.background_tasks, task_id)
            assert (await output_registered(task_id))["running"] is False
            return

        if row in {
            "restart-recovers-running-row",
            "suppressed-task-stays-suppressed",
            "legacy-pending-recovers-once",
        }:
            await registry.close()
            task_id = "persisted-task"
            persisted: dict[str, object] = {
                "task_id": task_id,
                "command": "old command",
                "pid": 12345,
                "running": row != "legacy-pending-recovers-once",
                "exit_code": None,
                "note": "old terminal",
                "owner": "run_background",
            }
            if row == "suppressed-task-stays-suppressed":
                persisted["notify_on_exit"] = False
            elif row == "restart-recovers-running-row":
                persisted["notify_on_exit"] = True
            else:
                persisted["receipt_pending"] = True
            (store.session_dir / "background_tasks.json").write_text(
                json.dumps([persisted]), encoding="utf-8"
            )
            store.close()
            resumed_store = ConversationStore(
                tmp_path / row, session_id=store.session_id
            )
            resumed_loop = AgentLoop(
                FakeBackend([]),
                resumed_store,
                max_turns=1,
                skill_catalog=SkillCatalog.empty(),
            )
            try:
                await resumed_loop.activate()
                receipts = resumed_store.agent_notifications(pending_only=False)
                if row == "suppressed-task-stays-suppressed":
                    assert receipts == []
                else:
                    assert len(receipts) == 1
                    assert receipts[0].data["task_id"] == task_id
                    await resumed_loop.activate()
                    assert len(
                        resumed_store.agent_notifications(pending_only=False)
                    ) == 1
            finally:
                await resumed_loop.close()
                resumed_store.close()
            return

        if row in {"duplicate-input", "durable-receipt-dedupe-after-eviction"}:
            item = TaskExitInput(
                task_id="dedupe-task",
                command="printf done",
                exit_code=0,
                note="exited with code 0",
                owner="run_background",
                phase="natural_exit",
            )
            await flow.produce(_task_notification(item))
            await flow.produce(_task_notification(item))
            assert len(store.agent_notifications(pending_only=False)) == 1
            if row == "duplicate-input":
                return
            await flow.prepare_request("terminal_fallback", [], None)
            await flow.acknowledge_active()
            receipt = store.agent_notifications(pending_only=False)[0]
            store.append_compaction_marker(
                "receipt evicted",
                receipt.seq,
                receipt.seq,
                kind="evict",
                view=[
                    {
                        "seq": receipt.seq,
                        "message": Message(
                            MessageRole.ASSISTANT,
                            [TextContent("[task exit receipt evicted]")],
                            metadata={
                                "context_evicted": True,
                                "eviction_receipt": "assistant",
                                "source_seq": receipt.seq,
                            },
                        ).to_dict(),
                    }
                ],
            )
            assembler = ContextAssembler(
                store,
                token_budget=1000,
                retained_tail=1,
                token_counter=lambda _message: 1,
            )
            await assembler.assemble_context(
                force=True, bypass_eviction_hysteresis=True
            )
            marker = next(
                entry
                for entry in store.replay()
                if entry.type == "compaction"
                and entry.data["source_seq_start"] <= receipt.seq
                <= entry.data["source_seq_end"]
            )
            assert marker.data["source_seq_start"] <= receipt.seq
            await flow.produce(_task_notification(item))
            assert len(store.agent_notifications(pending_only=False)) == 1
            return

        if row == "terminal-notify-follows-publish":
            task_id = await start_registered(_python("print('done')"))
            owner = registry.background_tasks._owners[task_id]
            competitor: asyncio.Task[object] | None = None
            receipts: list[TaskExitInput] = []

            async def receipt(item):
                # test-contract: seam prove durable publication precedes notification sink.
                assert registry.background_tasks._records[task_id].running is False
                receipts.append(item)
                await flow.produce(_task_notification(item))

            def notice(_notice):
                nonlocal competitor
                competitor = asyncio.create_task(
                    flow.produce(
                        pending_delivery_module.NotificationInput(
                            "task_exit",
                            {
                                "task_id": "competing-task",
                                "headline": "competing task",
                                "exit_code": 0,
                                "output_tail": "",
                                "background_owner": "run_background",
                                "background_phase": "natural_exit",
                            },
                            dedupe_key="competing-task",
                        )
                    )
                )

            # test-contract: seam schedule a competing public producer from the synchronous notice callback.
            registry.background_tasks.set_notice_sink(notice)
            # test-contract: seam observe the notification sink ordering.
            monkeypatch.setattr(registry.background_tasks, "_notification_sink", receipt)
            await await_or_fail(owner.wait(), what="terminal ordering")
            assert owner.record.running is False
            assert len(receipts) == 1
            assert competitor is not None
            await await_or_fail(competitor, what="competing notification producer")
            notifications = store.agent_notifications(pending_only=False)
            task_ids = [entry.data["task_id"] for entry in notifications]
            assert task_ids.count(task_id) == 1
            assert task_ids.index(task_id) < task_ids.index("competing-task")
            return

        if row == "persist-failure-retries-same-candidate":
            task_id = await start_registered(_python("print('done')"))
            original_commit = registry.background_tasks._commit_transition
            failed = False

            async def fail_terminal_once(rows, publish, notice):
                nonlocal failed
                if not failed and any(item["task_id"] == task_id and not item["running"] for item in rows):
                    failed = True
                    raise OSError("terminal write failed")
                return await original_commit(rows, publish, notice)

            # test-contract: seam fail the terminal persistence after process exit
            monkeypatch.setattr(
                registry.background_tasks, "_commit_transition", fail_terminal_once
            )
            await _wait_for_exit(registry.background_tasks, task_id)
            status = await output_registered(task_id)
            assert failed is True
            assert status["running"] is False
            assert status["exit_code"] == 0
            assert len(store.agent_notifications(pending_only=False)) == 1
            return

        if row == "natural-exit-success":
            task_id = await start_registered(_python("print('done')"))
            await _wait_for_exit(registry.background_tasks, task_id)
            status = await output_registered(task_id)
            assert status["running"] is False
            assert status["exit_code"] == 0
            assert len(store.agent_notifications(pending_only=False)) == 1
            return

        if row == "live-close-defers-receipt":
            receipt_entered = asyncio.Event()
            release_receipt = asyncio.Event()
            produce = flow.produce

            async def gated_close_receipt(item):
                receipt_entered.set()
                await asyncio.wait_for(release_receipt.wait(), 10)
                return await produce(item)

            monkeypatch.setattr(flow, "produce", gated_close_receipt)
            task_id = await start_registered(_python("import time; time.sleep(30)"))
            closing = asyncio.create_task(registry.close())
            try:
                await asyncio.wait_for(receipt_entered.wait(), 10)
                assert not closing.done()
                persisted = json.loads(
                    (store.session_dir / "background_tasks.json").read_text()
                )
                task_row = next(item for item in persisted if item["task_id"] == task_id)
                assert task_row["running"] is False
                assert store.agent_notifications(pending_only=False) == []
            finally:
                release_receipt.set()
            await asyncio.wait_for(closing, 10)
            assert len(store.agent_notifications(pending_only=False)) == 1
            return

        if row == "1-natural-exit-wins-kill-race":
            task_id = await start_registered(
                _python("import sys; sys.stdin.read(1); raise SystemExit(7)")
            )
            termination_entered = asyncio.Event()
            release_termination = asyncio.Event()
            terminator_name = (
                "_terminate_processes"
                if hasattr(process_module, "_terminate_processes")
                else "_terminate_process"
            )
            terminator = getattr(process_module, terminator_name)

            async def gated_terminator(*args, **kwargs):
                termination_entered.set()
                await asyncio.wait_for(release_termination.wait(), 10)
                return await terminator(*args, **kwargs)

            # test-contract: seam hold the shared group terminator while the process exits
            monkeypatch.setattr(process_module, terminator_name, gated_terminator)
            killing = asyncio.create_task(
                registry.execute(ToolCall("kill", "task_kill", {"task_id": task_id}))
            )
            try:
                await asyncio.wait_for(termination_entered.wait(), 10)
                await asyncio.wait_for(
                    registry.background_tasks.input(task_id, "x", eof=True), 10
                )
                terminal = await asyncio.wait_for(
                    registry.background_tasks.wait(task_id), 10
                )
                assert terminal["exit_code"] == 7
            finally:
                release_termination.set()
            result = await asyncio.wait_for(killing, 10)
            assert result["content"][0]["text"] == (
                f"background task {task_id} exited, code 7"
            )
            outcome = result["structuredContent"]
            assert outcome.get("killed_by_this_call") is False
            assert outcome.get("cause") is None
            assert outcome.get("exit_code") == 7
            return

        task_id = await start_registered(_python("import time; time.sleep(30)"))
        if row == "3-external-cause":
            cause = "task safety deadline expired"
            outcome = await asyncio.wait_for(
                registry.background_tasks.kill(task_id, cause=cause), 10
            )
            assert outcome.get("killed_by_this_call") is True
            assert outcome.get("cause") == cause
            assert outcome.get("status") == "killed"
            assert outcome.get("exit_code") is not None
            return

        call = ToolCall("kill", "task_kill", {"task_id": task_id})
        result = await asyncio.wait_for(registry.execute(call), 10)
        cause = "task_kill tool requested by the agent"
        outcome = result["structuredContent"]
        assert outcome.get("killed_by_this_call") is True
        assert outcome.get("cause") == cause
        assert outcome.get("status") == "killed"
        event = StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=receipt_tool_result(call.id, result),
        )
        rendered = render_event(event)
        assert rendered is not None
        card = _renderable_plain(rendered, width=120)
        assert "killed by this call" in card
        assert cause in card
        projected = event_contract.tool_end(event)
        assert "killed by this call" in projected["tool_result"]["content"]
        assert projected["tool_result"]["structured_content"]["cause"] == cause


_ARCHIVE_NOTE = "background output archive failed; recovery will retry the retained log"


def _fail_archive_append(tasks: BackgroundTaskRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("forced archive failure")

    assert tasks._archive is not None
    # test-contract: seam inject archive failure through the owner's real archive call
    monkeypatch.setattr(tasks._archive, "append", fail)


@pytest.mark.asyncio
async def test_kill_cause_leads_archive_note_published_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "session")
    tools = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
    tasks = tools.background_tasks
    flow = pending_delivery_module.StepDeliveryFlow(store)
    tasks.set_notification_sink(lambda item: flow.produce(_task_notification(item)))
    cause = "test kill after archive failure"
    archive_noted = asyncio.Event()
    claimed = asyncio.Event()
    release_terminal = asyncio.Event()
    original_finish = owner_module._BackgroundTaskOwner._finish_terminal
    original_claim = owner_module._BackgroundTaskOwner._claim_termination

    async def gated_finish(self, exit_code):
        archive_noted.set()
        await await_or_fail(release_terminal.wait(), what="terminal release")
        return await original_finish(self, exit_code)

    async def signalled_claim(self, *args, **kwargs):
        result = await original_claim(self, *args, **kwargs)
        claimed.set()
        return result

    _fail_archive_append(tasks, monkeypatch)
    # test-contract: seam hold the owner between archive-note publication and terminal commit
    monkeypatch.setattr(owner_module._BackgroundTaskOwner, "_finish_terminal", gated_finish)
    # test-contract: seam signal once the concurrent kill has published its cause
    monkeypatch.setattr(owner_module._BackgroundTaskOwner, "_claim_termination", signalled_claim)
    killer: asyncio.Task[dict[str, object]] | None = None
    try:
        task_id, _ = await tasks.start("true", tmp_path)
        await await_or_fail(archive_noted.wait(), what="archive note published")
        assert tasks._owners[task_id].record.note == _ARCHIVE_NOTE
        killer = asyncio.create_task(tasks.kill(task_id, cause=cause))
        await await_or_fail(claimed.wait(), what="kill cause published")
        release_terminal.set()
        with pytest.raises(OSError, match="forced archive failure"):
            await await_or_fail(killer, what="kill completion")
        rows = json.loads((store.session_dir / "background_tasks.json").read_text())
        receipts = [
            entry
            for entry in store.agent_notifications(pending_only=False)
            if entry.data.get("kind") == "task_exited"
        ]
        assert rows[0]["note"] == f"{cause}; {_ARCHIVE_NOTE}"
        assert len(receipts) == 1
        assert receipts[0].data["note"] == f"{cause}; {_ARCHIVE_NOTE}"
        # the precedence state is runtime-only: the real stored row never carries it
        assert "note_priority" not in rows[0]
    finally:
        release_terminal.set()
        if killer is not None:
            await asyncio.wait_for(asyncio.gather(killer, return_exceptions=True), 10)
        await asyncio.wait_for(tools.close(), 10)
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["archive", "kill", "shutdown", "cutoff"])
async def test_terminal_note_writer_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    store = ConversationStore(tmp_path / "session")
    tools = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
    tasks = tools.background_tasks
    cause = f"table {writer} cause"
    try:
        if writer == "archive":
            _fail_archive_append(tasks, monkeypatch)
            task_id, _ = await tasks.start("true", tmp_path)
            with pytest.raises(OSError, match="forced archive failure"):
                await await_or_fail(tasks.wait(task_id), what="archive failure exit")
            expected = _ARCHIVE_NOTE
        elif writer == "kill":
            task_id, _ = await tasks.start("sleep 30", tmp_path)
            await await_or_fail(tasks.kill(task_id, cause=cause), what="kill")
            expected = cause
        else:
            if writer == "cutoff":
                held = asyncio.Event()

                async def held_output(*args: object, **kwargs: object) -> None:
                    await asyncio.wait_for(held.wait(), 10)

                # test-contract: seam keep task output open so close must cut it off
                monkeypatch.setattr(tasks, "_read_output", held_output)
                monkeypatch.setattr(process_module, "BACKGROUND_OUTPUT_DRAIN_TIMEOUT", 0.0)
            task_id, _ = await tasks.start("sleep 30", tmp_path)
            tasks.begin_shutdown(cause)
            await await_or_fail(tasks.close(), what="shutdown")
            expected = cause if writer == "shutdown" else f"{cause}; {owner_module.OUTPUT_CUTOFF_NOTE}"
        rows = json.loads((store.session_dir / "background_tasks.json").read_text())
        assert tasks.snapshot()[0].note == expected
        assert rows[0]["note"] == expected
    finally:
        await asyncio.wait_for(tools.close(), 10)
        store.close()


def test_background_note_has_one_writer() -> None:
    import ast

    root = Path(owner_module.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        allowed = {
            id(node)
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef) and fn.name == "_set_terminal_note"
            for node in ast.walk(fn)
        }
        for node in ast.walk(tree):
            if id(node) in allowed:
                continue
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
                else []
            )
            direct = any(isinstance(t, ast.Attribute) and t.attr == "note" for t in targets)
            via_setattr = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "note"
            )
            if direct or via_setattr:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        f"direct background note writes at {offenders}; route through _set_terminal_note "
        "so a kill or shutdown cause keeps precedence over archive and cutoff notes"
    )


@pytest.mark.asyncio
async def test_task_exit_recovers_original_receipt_after_store_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.fake_backend import FakeBackend

    store = ConversationStore(tmp_path / "session")
    tools = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )
    flow = pending_delivery_module.StepDeliveryFlow(store)
    tools.background_tasks.set_notification_sink(
        lambda item: flow.produce(_task_notification(item))
    )
    task_id, _ = await tools.background_tasks.start("sleep 30", tmp_path)
    cause = "test closed-store kill cause"
    _fail_archive_append(tools.background_tasks, monkeypatch)
    try:
        store.close()
        with pytest.raises(OSError, match="forced archive failure"):
            await tools.background_tasks.kill(
                task_id, cause=cause, tui_presented=True
            )
    finally:
        await asyncio.wait_for(tools.close(), 10)

    archived_rows = json.loads(
        (store.session_dir / "background_tasks.json").read_text()
    )

    resumed_store = ConversationStore(
        tmp_path / "session", session_id=store.session_id
    )
    loop = AgentLoop(
        FakeBackend([]),
        resumed_store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        await loop.activate()
        receipts = [
            entry
            for entry in resumed_store.agent_notifications(pending_only=False)
            if entry.data.get("kind") == "task_exited"
        ]
        assert len(receipts) == 1
        assert archived_rows[0]["notify_on_exit"] is True
        assert archived_rows[0]["receipt_pending"] is True
        assert archived_rows[0]["note"] == f"{cause}; {_ARCHIVE_NOTE}"
        assert archived_rows[0]["terminal_phase"] == "task_kill"
        assert archived_rows[0]["tui_presented"] is True
        assert receipts[0].data["task_id"] == task_id
        assert receipts[0].data["note"] == f"{cause}; {_ARCHIVE_NOTE}"
        assert receipts[0].data["background_phase"] == "task_kill"
        assert resumed_store.is_agent_notification_presented_to_tui(receipts[0].id)
        rows = json.loads(
            (resumed_store.session_dir / "background_tasks.json").read_text()
        )
        assert rows[0]["receipt_pending"] is False
    finally:
        await loop.close()
        resumed_store.close()


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


@pytest.mark.asyncio
async def test_tui_presented_notification_remains_available_to_notification_wake(
    tmp_path: Path,
) -> None:
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

    flow = pending_delivery_module.StepDeliveryFlow(store)
    flow.initialize_resident_edges()
    prepared = await flow.prepare_request("idle_start", [], None)
    message = prepared.messages[-1]

    assert message.metadata["notifications"] == [
        {
            "notification_id": notification.id,
            **notification.data,
            "kind": "agent_completion",
        }
    ]


@pytest.mark.asyncio
async def test_legacy_notification_without_kind_is_agent_completion_in_wake_payload(
    tmp_path: Path,
) -> None:
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

    flow = pending_delivery_module.StepDeliveryFlow(store)
    flow.initialize_resident_edges()
    prepared = await flow.prepare_request("idle_start", [], None)
    message = prepared.messages[-1]

    assert message.metadata["notifications"][0]["kind"] == "agent_completion"


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
    unknown = store.append_child_idle_notification(
        "future-child",
        child_session_path="agents/future",
        description="future notification",
        text="heads up",
    )
    assert unknown is not None
    persisted = store.path.read_text(encoding="utf-8").replace(
        '"kind":"child_idle"', '"kind":"monitor_alert"', 1
    )
    store.path.write_text(persisted, encoding="utf-8")

    # A fresh store on the same session forces load + validation of every row.
    reloaded = ConversationStore(tmp_path, session_id="kinds")
    kinds = [
        entry.data.get("kind", "agent_completion")
        for entry in reloaded.replay()
        if entry.type == "notification"
    ]
    assert kinds == ["task_exited", "agent_completion", "monitor_alert"]
    assert len(reloaded.agent_notifications(pending_only=False)) == 3


def test_child_question_notification_renders_question() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_question",
                "child_instance_id": "root:208",
                "question_id": "b2517c6f0fbb448e86dbb8b34fe0fbc0",
                "question": "should we deploy this change?",
                "description": "child question",
                "status": "pending",
                "text": "child root:208 asks: should we deploy this change?",
            },
        )
    )

    assert rendered is not None
    plain = _renderable_plain(rendered)
    assert "should we deploy this change?" in plain
    assert "unavailable" not in plain


def test_child_idle_notification_renders_readable_report() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_idle",
                "child_instance_id": "root:203",
                "description": "finish PR #529\nwith\t\x1b[2J a long report",
                "status": "child_idle",
                "text": "the child's report is not shown in the notice",
            },
        )
    )

    assert rendered is not None
    plain = _renderable_plain(rendered)
    assert "idle ← finish PR #529 with a long report :203 · idle" in plain
    assert "the child's report is not shown in the notice" in plain
    assert "\x1b" not in plain
    assert "\x9b" not in plain
    assert "\x07" not in plain


def test_child_idle_notification_without_description_shows_error_line() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui import theme
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_idle",
                "status": "child_idle",
                "text": "the child's report",
            },
        )
    )

    assert rendered is not None
    assert rendered.plain == "background agent notification unavailable"
    assert rendered.style == theme.ERROR


def test_child_question_withdrawn_notification_renders_text() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_question_withdrawn",
                "child_instance_id": "root:208",
                "question_id": "b2517c6f0fbb448e86dbb8b34fe0fbc0",
                "description": "child question withdrawn",
                "status": "withdrawn",
                "text": "question withdrawn: child finished",
            },
        )
    )

    assert rendered is not None
    plain = _renderable_plain(rendered)
    assert "withdraw ← child :208" in plain
    assert "b2517c6f0fbb448e86dbb8b34fe0fbc0" in plain
    assert "question withdrawn: child finished" not in plain

def test_child_question_notification_strips_terminal_controls() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_question",
                "child_instance_id": "root:208",
                "question_id": "question-1",
                "question": "first line\tsecond\x1b[2J\x9b31m\x07 words",
                "options": [],
            },
        )
    )

    assert rendered is not None
    plain = _renderable_plain(rendered)
    assert "\x1b" not in plain
    assert "\x9b" not in plain
    assert "\x07" not in plain
    assert "first line" in plain
    assert "second" in plain
    assert "words" in plain


def test_child_question_withdrawn_notification_ignores_legacy_text() -> None:
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    rendered = render_agent_notification(
        StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "kind": "child_question_withdrawn",
                "child_instance_id": "root:208",
                "question_id": "question-1",
                "text": "first line\tsecond\x1b[2J\x9b31m\x07 words",
            },
        )
    )

    assert rendered is not None
    plain = _renderable_plain(rendered)
    assert "withdraw ← child :208" in plain
    assert "question-1" in plain
    assert "first line" not in plain


def test_tui_renders_task_exit_and_ignores_unknown_kind() -> None:
    # S7: the TUI renders task_exited receipts and tolerates unknown kinds
    # without ever rendering them as agent completions.
    from zeta.protocol.types import StreamEventType
    from zeta.tui.render import render_agent_notification

    def render(data: dict) -> str:
        rendered = render_agent_notification(
            StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=data)
        )
        assert rendered is not None
        return _renderable_plain(rendered).strip()

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
async def test_failed_store_close_is_retained_for_retry(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store)
    original_close = store.close
    attempts = 0

    def flaky_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("close failed")
        original_close()

    store.close = flaky_close  # type: ignore[method-assign]
    owner.track_store(store)
    owner.mark_store_finished(store)
    owner.release_unused_stores()

    with pytest.raises(OSError, match="close failed"):
        await owner.wait_for_store_closes()
    assert store in owner._stores
    assert owner._store_close_tasks

    owner.release_unused_stores()
    await owner.wait_for_store_closes()
    assert store not in owner._stores
    with pytest.raises(OSError):
        os.fstat(store.directory_fd)


@pytest.mark.asyncio
async def test_unobserved_failed_store_close_is_not_retried(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store)
    original_close = store.close
    attempts = 0

    def flaky_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("close failed")
        original_close()

    store.close = flaky_close  # type: ignore[method-assign]
    owner.track_store(store)
    owner.mark_store_finished(store)
    owner.release_unused_stores()
    (task,) = tuple(owner._store_close_tasks)
    await asyncio.wait({task})

    # a later child completion releases stores before anyone observed the failure
    owner.release_unused_stores()
    assert attempts == 1
    with pytest.raises(OSError, match="close failed"):
        await owner.wait_for_store_closes()
    assert store in owner._stores


@pytest.mark.asyncio
async def test_canceled_store_close_waiter_does_not_cancel_shared_close(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store)
    original_close = store.close
    event_loop = asyncio.get_running_loop()
    close_entered = asyncio.Event()
    release_close = threading.Event()

    def gated_close() -> None:
        event_loop.call_soon_threadsafe(close_entered.set)
        release_close.wait(10)
        original_close()

    store.close = gated_close  # type: ignore[method-assign]
    owner.track_store(store)
    owner.mark_store_finished(store)
    owner.release_unused_stores()
    canceled_waiter = asyncio.create_task(owner.wait_for_store_closes())
    other_waiter = asyncio.create_task(owner.wait_for_store_closes())
    try:
        await asyncio.wait_for(close_entered.wait(), 10)
        # A canceled child turn stops waiting while the shared close still runs.
        canceled_waiter.cancel()
        await asyncio.wait({canceled_waiter}, timeout=10)
        release_close.set()
        await asyncio.wait_for(other_waiter, 10)
        assert store not in owner._stores
    finally:
        release_close.set()
        for waiter in (canceled_waiter, other_waiter):
            waiter.cancel()
        await asyncio.wait_for(
            asyncio.gather(canceled_waiter, other_waiter, return_exceptions=True),
            10,
        )
        await asyncio.wait_for(
            asyncio.gather(*tuple(owner._store_close_tasks), return_exceptions=True),
            10,
        )
        original_close()


@pytest.mark.asyncio
async def test_wait_for_store_closes_does_not_spin_before_done_callback(
    tmp_path: Path,
) -> None:
    class BoundedCloseTasks(set[asyncio.Task[None]]):
        checks = 0

        def __bool__(self) -> bool:
            self.checks += 1
            assert self.checks <= 3, "completed close task was gathered repeatedly"
            return len(self) > 0

    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store)
    release_close = asyncio.Event()

    async def gated_close() -> None:
        await release_close.wait()

    close_task = asyncio.create_task(gated_close())
    close_tasks = BoundedCloseTasks({close_task})
    owner._store_close_tasks = close_tasks
    close_task.add_done_callback(close_tasks.discard)

    async def wait_after_close_is_released() -> None:
        await release_close.wait()
        await owner.wait_for_store_closes()

    waiter = asyncio.create_task(wait_after_close_is_released())
    release_close.set()
    await asyncio.wait_for(waiter, timeout=1)
    store.close()


@pytest.mark.asyncio
async def test_background_agent_store_closes_before_root_shutdown(tmp_path: Path) -> None:
    call = ToolCall(
        "background",
        "agent",
        {"prompt": "child", "description": "background", "background": True},
    )
    backend = PerActorFakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ],
        {"child": [ScriptedTurn([TextContent("done")])]},
        parent_prompt="start",
        gated_child_prompts=set(),
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
    await asyncio.wait_for(owner.wait(), timeout=10)
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
    backend = PerActorFakeBackend(
        [ScriptedTurn(tool_calls=[foreground])],
        {
            "child": [ScriptedTurn(tool_calls=[grandchild])],
            "grandchild": [ScriptedTurn([TextContent("done")])],
        },
        parent_prompt="start",
        gated_child_prompts={"grandchild"},
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    tracked = []
    tracked_ready = asyncio.Event()
    owner = loop._background_owner
    track_stores = owner.track_store

    def track(child_store):
        tracked.append(child_store)
        track_stores(child_store)
        if len(tracked) == 2:
            tracked_ready.set()

    owner.track_store = track

    try:
        await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        await asyncio.wait_for(
            backend.child_started["grandchild"].wait(), timeout=10
        )
        await asyncio.wait_for(tracked_ready.wait(), timeout=10)
        os.fstat(tracked[0].directory_fd)

        backend.release_children.set()
        await asyncio.wait_for(owner.wait(), timeout=10)
        for child_store in tracked:
            with pytest.raises(OSError):
                os.fstat(child_store.directory_fd)
        assert store.directory_fd >= 0
    finally:
        backend.release_children.set()
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
async def test_background_stdin_is_owned_pipe_when_parent_stdin_is_a_tty(
    tmp_path: Path,
) -> None:
    tasks = BackgroundTaskRegistry()
    saved_stdin = os.dup(0)
    master_fd, slave_fd = pty.openpty()
    try:
        os.dup2(slave_fd, 0)
        task_id, _ = await tasks.start(
            _python(
                "import os, sys; print(os.isatty(0)); "
                "print(sys.stdin.buffer.read() == b'')"
            ),
            tmp_path,
        )
    finally:
        os.dup2(saved_stdin, 0)
        os.close(saved_stdin)
        os.close(master_fd)
        os.close(slave_fd)

    await tasks.input(task_id, eof=True)
    await _wait_for_exit(tasks, task_id)
    assert (await tasks.output(task_id))["output"] == "False\nTrue\n"
    await tasks.close()


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
    await asyncio.wait_for(
        asyncio.gather(*(tasks.input(task_id, part) for part in parts)), 10
    )
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
    try:
        with pytest.raises(ValueError, match="unknown background task"):
            await tasks.input("not-owned", "x")
        task_id, _ = await tasks.start("true", tmp_path)
        await _wait_for_exit(tasks, task_id)
        with pytest.raises(ValueError, match="not running"):
            await tasks.input(task_id, "x")
        task_id, _ = await tasks.start("sleep 30", tmp_path)
        writes = [tasks.input(task_id, "x" * 100) for _ in range(10)]
        await asyncio.wait_for(
            asyncio.gather(
                tasks.kill(task_id, cause="test requested task stop"),
                *writes,
                return_exceptions=True,
            ),
            10,
        )
        assert tasks.records[-1].stdin is None
    finally:
        await asyncio.wait_for(tasks.close(), 10)

    shutting_down = BackgroundTaskRegistry()
    try:
        shutdown_id, _ = await shutting_down.start("sleep 30", tmp_path)
        shutdown_writes = [
            shutting_down.input(shutdown_id, "x" * 100) for _ in range(10)
        ]
        await asyncio.wait_for(
            asyncio.gather(
                shutting_down.close(), *shutdown_writes, return_exceptions=True
            ),
            10,
        )
        assert shutting_down.records[-1].stdin is None
    finally:
        await asyncio.wait_for(shutting_down.close(), 10)


@pytest.mark.asyncio
async def test_task_output_reads_immediately_and_rejects_wait_argument(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    assert "wait_seconds" not in registry.definitions_by_name["task_output"].parameters[
        "properties"
    ]
    started = await registry.execute(
        ToolCall(
            "start",
            "run_background",
            {"command": _python("import sys\nsys.stdin.read()\nprint('done', end='')")},
        )
    )
    task_id = started["structuredContent"]["task_id"]
    try:
        immediate = await registry.execute(
            ToolCall("immediate", "task_output", {"task_id": task_id})
        )
        assert immediate["structuredContent"]["running"] is True
        invalid = await registry.execute(
            ToolCall("invalid", "task_output", {"task_id": task_id, "wait_seconds": 1})
        )
        assert invalid["isError"] is True
        assert "unexpected properties: wait_seconds" in invalid["content"][0]["text"]
    finally:
        try:
            await registry.execute(
                ToolCall("release", "task_input", {"task_id": task_id, "eof": True})
            )
        finally:
            await registry.close()


def test_task_output_old_transcript_arguments_replay(tmp_path: Path) -> None:
    store_dir = tmp_path / "sessions"
    with ConversationStore(store_dir, session_id="old-task-output") as store:
        store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [
                    ToolUseContent(
                        ToolCall(
                            "old-call",
                            "task_output",
                            {"task_id": "task-old", "wait_seconds": 5},
                        )
                    )
                ],
            )
        )
    with ConversationStore(store_dir, session_id="old-task-output") as replayed:
        call = replayed.messages()[0].content[0].tool_call
        assert call.arguments["wait_seconds"] == 5


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
async def test_wait_publishes_terminal_state_only_after_archive_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notices = []
    notifications = []

    async def notify(item: TaskExitInput) -> None:
        notifications.append(item)

    with ConversationStore(tmp_path / "sessions", session_id="archive-wait") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
            notice_sink=notices.append,
            notification_sink=notify,
        )
        release_archive = threading.Event()
        waiting = None
        try:
            archive = tasks._archive
            assert archive is not None
            real_append = archive.append
            archive_started = threading.Event()

            def blocked_append(task_id: str, source_fd: int, length: int) -> None:
                archive_started.set()
                assert release_archive.wait(timeout=10)
                real_append(task_id, source_fd, length)

            # test-contract: durability gates the archive commit before terminal publication
            monkeypatch.setattr(archive, "append", blocked_append)
            task_id, _ = await tasks.start("printf archived-output", tmp_path)
            assert await asyncio.to_thread(archive_started.wait, 10)

            wait_started = asyncio.Event()

            async def wait_for_terminal() -> dict[str, object]:
                wait_started.set()
                return await tasks.wait(task_id)

            waiting = asyncio.create_task(wait_for_terminal())
            await wait_started.wait()

            persisted = json.loads(
                (store.session_dir / "background_tasks.json").read_text(encoding="utf-8")
            )
            assert not waiting.done()
            assert tasks.snapshot()[0].running is True
            assert persisted[0]["running"] is True
            assert task_id not in tasks._archived_lengths
            assert [notice.phase for notice in notices] == ["started"]
            assert notifications == []
            process = tasks.records[0].process
            assert process is not None
            assert process.returncode == 0

            release_archive.set()
            result = await asyncio.wait_for(waiting, timeout=10)

            persisted = json.loads(
                (store.session_dir / "background_tasks.json").read_text(encoding="utf-8")
            )
            assert result == {"task_id": task_id, "running": False, "exit_code": 0}
            assert tasks.snapshot()[0].running is False
            assert persisted[0]["running"] is False
            assert tasks._archived_lengths[task_id] == len("archived-output")
            assert [notice.phase for notice in notices] == ["started", "natural_exit"]
            assert [item.task_id for item in notifications] == [task_id]
            assert (await tasks.output(task_id, since=0))["output"] == "archived-output"
        finally:
            release_archive.set()
            if waiting is not None:
                await asyncio.gather(waiting, return_exceptions=True)
            await tasks.close()


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
async def test_output_beyond_ring_recoverable_active_and_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ConversationStore(tmp_path / "sessions", session_id="concurrent-output") as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=7,
            call_limit=5,
        )
        try:
            payloads = (b"alpha-0123456789", b"beta-ABCDEFGHIJ", b"gamma-uvwxyz")
            task_ids = []
            for _ in payloads:
                task_id, _ = await tasks.start("cat", tmp_path)
                task_ids.append(task_id)

            output_ready = {task_id: asyncio.Event() for task_id in task_ids}
            original_append = tasks._append

            def signal_output(record: _BackgroundRecord, chunk: bytes) -> None:
                original_append(record, chunk)
                output_ready[record.task_id].set()

            # test-contract: seam BackgroundTaskRegistry has no public output-ready event
            monkeypatch.setattr(tasks, "_append", signal_output)

            async def read_exact(task_id: str, length: int) -> bytes:
                cursor = 0
                output = bytearray()
                while cursor < length:
                    result = await tasks.output(task_id, since=cursor)
                    output.extend(result["output"].encode())
                    assert result["cursor"] > cursor
                    cursor = result["cursor"]
                return bytes(output)

            await asyncio.gather(
                *(
                    tasks.input(task_id, payload[:8].decode())
                    for task_id, payload in zip(task_ids, payloads, strict=True)
                )
            )
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in output_ready.values())),
                timeout=10,
            )
            active = await asyncio.gather(
                *(read_exact(task_id, 8) for task_id in task_ids)
            )
            assert active == [payload[:8] for payload in payloads]

            await asyncio.gather(
                *(
                    tasks.input(task_id, payload[8:].decode(), eof=True)
                    for task_id, payload in zip(task_ids, payloads, strict=True)
                )
            )
            await asyncio.gather(
                *(_wait_for_exit(tasks, task_id) for task_id in task_ids)
            )
            finished = await asyncio.gather(
                *(
                    read_exact(task_id, len(payload))
                    for task_id, payload in zip(task_ids, payloads, strict=True)
                )
            )
            assert finished == list(payloads)
        finally:
            await tasks.close()


def test_many_finished_tasks_hold_bounded_descriptors(tmp_path: Path) -> None:
    script = r'''
import asyncio, json, os, resource, sys
from pathlib import Path
from zeta.core.store import ConversationStore
from zeta.storage_format import advance, open_writer
from zeta.tools._shared.process import BackgroundTaskRegistry

async def main(root: Path, lease) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, hard), hard))
    with ConversationStore(root / "sessions", session_id="fd-bound", _storage_lease=lease) as store:
        tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            output_limit=4,
            storage_lease=lease,
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

root = Path(sys.argv[1])
advance(root, expected_epoch=0)
lease = open_writer(root)
try:
    asyncio.run(main(root, lease))
finally:
    lease.close()
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
    opened = await asyncio.to_thread(lambda: manager.create(provider="codex", model="fake", cwd=tmp_path))
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
    write_entered = threading.Event()
    release_write = threading.Event()
    write_finished = threading.Event()

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
            write_entered.set()
            try:
                assert release_write.wait(timeout=10), "event loop could not release log write"
                return self._handle.write(data)
            finally:
                write_finished.set()

        def flush(self) -> None:
            self._handle.flush()

        def close(self) -> None:
            self._handle.close()

    monkeypatch.setattr(
        tasks, "_open_task_log", lambda task_id: SlowLog(original_open_log(task_id))
    )
    try:
        task_id, _ = await tasks.start("printf 0123456789abcdef", tmp_path)
        assert await asyncio.to_thread(write_entered.wait, 10)
        # Reaching here while the writer is blocked proves loop progress;
        # scheduler latency on a loaded CI runner is irrelevant.
        assert not write_finished.is_set()
        release_write.set()
        await _wait_for_exit(tasks, task_id)

        cursor = 0
        for expected in ("0123", "4567", "89ab", "cdef"):
            result = await tasks.output(task_id, since=cursor)
            assert result["output"] == expected
            cursor = result["cursor"]
        assert cursor == 16
    finally:
        release_write.set()
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
    ready = tmp_path / "detached-ready"
    terminated = tmp_path / "detached-terminated"
    child = (
        "import os,pathlib,signal,time; "
        f"ready=pathlib.Path({str(ready)!r}); "
        f"terminated=pathlib.Path({str(terminated)!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(terminated.write_text(str(os.getpid())), exit())[1]); "
        "ready.write_text(str(os.getpid())); "
        "time.sleep(300)"
    )
    tasks = BackgroundTaskRegistry()
    task_id, pid = await tasks.start(
        _python(
            "import subprocess,sys; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        ),
        tmp_path,
    )
    child_pid = await read_pid_file(ready)
    await wait_pid_exited(pid)
    assert (await tasks.output(task_id))["running"] is True
    assert _group_exists(pid)

    killing = asyncio.create_task(tasks.kill(task_id, cause="test requested task stop"))
    try:
        assert await read_pid_file(terminated) == child_pid
        result = await await_or_fail(killing, what="detached group kill completion")
        assert result["running"] is False
        assert not _group_exists(pid)
    finally:
        if not killing.done():
            os.kill(child_pid, signal.SIGKILL)
        await await_or_fail(
            asyncio.gather(killing, return_exceptions=True),
            what="detached group cleanup",
        )
        await tasks.close()




@pytest.mark.asyncio
async def test_background_kill_escalates_for_term_ignoring_process(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.03)
    task_id, _ = await tasks.start(
        _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
        tmp_path,
    )
    result = await tasks.kill(task_id, cause="test requested task stop")
    assert result["running"] is False
    assert result["exit_code"] in {-signal.SIGTERM, -signal.SIGKILL}
    assert not _group_exists(tasks.records[0].pid)
    await tasks.close()


@pytest.mark.asyncio
async def test_task_kill_cancellation_still_kills_escaped_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(child_pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    )
    tasks = BackgroundTaskRegistry(term_grace=0.03)
    release_second_snapshot = threading.Event()
    child_pid = None
    killing = None
    try:
        task_id, _ = await tasks.start(
            _python(
                "import os,subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
                "preexec_fn=os.setsid, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL); time.sleep(30)"
            ),
            tmp_path,
        )
        child_pid = await read_pid_file(child_pid_file)
        second_snapshot_started = threading.Event()
        original_snapshot = spawn_module._snapshot_descendants_sync
        snapshot_calls = 0

        def blocked_second_snapshot(root_pids: Sequence[int]):
            nonlocal snapshot_calls
            snapshot_calls += 1
            if snapshot_calls == 2:
                second_snapshot_started.set()
                assert release_second_snapshot.wait(timeout=10)
            return original_snapshot(root_pids)

        # test-contract: seam BackgroundTaskRegistry has no public descendant-snapshot event
        monkeypatch.setattr(
            spawn_module, "_snapshot_descendants_sync", blocked_second_snapshot
        )
        killing = asyncio.create_task(
            tasks.kill(task_id, cause="test requested task stop")
        )
        assert await asyncio.to_thread(second_snapshot_started.wait, 10)
        killing.cancel()
        release_second_snapshot.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(killing, 10)
        await wait_pid_exited(child_pid)
    finally:
        release_second_snapshot.set()
        if killing is not None:
            await asyncio.wait_for(
                asyncio.gather(killing, return_exceptions=True), 10
            )
        if child_pid is not None:
            kill_group_if_present(child_pid)
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
        await asyncio.wait_for(release_shutdown.wait(), 10)
        await original_close_stdin(record)

    monkeypatch.setattr(tasks, "_close_stdin", pause_close_stdin)
    closing = asyncio.create_task(tasks.close())
    try:
        await asyncio.wait_for(shutdown_started.wait(), 10)
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
        await asyncio.wait_for(closing, 10)
        assert all(not _group_exists(record.pid) for record in tasks.records)
    finally:
        release_shutdown.set()
        await asyncio.wait_for(
            asyncio.gather(closing, return_exceptions=True), 10
        )
        for record in tasks.records:
            kill_group_if_present(record.pid)




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
    original_signal_group = spawn_module._signal_group

    def signal_group(process: object, signum: signal.Signals) -> None:
        if signum == signal.SIGTERM:
            term_sent.set()
            return
        original_signal_group(process, signum)

    monkeypatch.setattr(spawn_module, "_signal_group", signal_group)
    try:
        closing = asyncio.create_task(tasks.close())
        await asyncio.wait_for(term_sent.wait(), 10)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 10)

        assert all(not _group_exists(record.pid) for record in tasks.records)
        assert await tasks.close() == ()
        assert len(task_ids) == 3
    finally:
        for record in tasks.records:
            kill_group_if_present(record.pid)


@pytest.mark.asyncio
async def test_close_terminates_many_tasks_in_one_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    termination_batches: list[int] = []
    # test-contract: complexity no public process-termination batch seam exists
    terminate_processes = process_module._terminate_processes

    async def record_batch(processes, term_grace, *, finish, after_signal=None):
        termination_batches.append(len(processes))
        return await terminate_processes(
            processes, term_grace, finish=finish, after_signal=after_signal
        )

    # test-contract: complexity shutdown must expose one bounded termination batch
    monkeypatch.setattr(process_module, "_terminate_processes", record_batch)
    tasks = BackgroundTaskRegistry(term_grace=0.25)
    task_ids = [
        await tasks.start(
            _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
            tmp_path,
        )
        for _ in range(20)
    ]

    await tasks.close()

    # test-contract: complexity shutdown cost is one batch, not one grace per task
    assert termination_batches == [len(task_ids)]
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
    try:
        command = wait_for_path_command(gate)
        task_ids = [
            await asyncio.wait_for(tasks.start(command, tmp_path), 10)
            for _ in range(12)
        ]
        assert len(task_ids) == 12
        assert tasks.running_count == 12

        gate.touch()
        await asyncio.wait_for(
            asyncio.gather(*(tasks.wait(task_id) for task_id, _ in task_ids)), 10
        )
        assert tasks.running_count == 0
    finally:
        gate.touch()
        await asyncio.wait_for(tasks.close(), 10)


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
    assert output["structuredContent"]["note"] == "tool registry closed"
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


@pytest.mark.asyncio
async def test_explicit_finish_task_safety_deadline_names_cause_for_every_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent import explicit_finish

    deadline = asyncio.Event()
    sleep_started = asyncio.Event()

    async def injected_deadline(timeout: float) -> None:
        assert timeout == 7
        sleep_started.set()
        await asyncio.wait_for(deadline.wait(), 10)

    monkeypatch.setattr(
        explicit_finish, "asyncio", SimpleNamespace(sleep=injected_deadline)
    )
    wake = asyncio.Event()
    store = ConversationStore(tmp_path)
    registry = _notification_registry(store, wake_sink=lambda _source: wake.set())
    first_id, _ = await asyncio.wait_for(
        registry.start(_python("import time; time.sleep(60)"), tmp_path), 10
    )
    kill = asyncio.create_task(
        explicit_finish.kill_tasks_after_deadline(registry, 7)
    )
    await asyncio.wait_for(sleep_started.wait(), 10)
    second_id, _ = await asyncio.wait_for(
        registry.start(_python("import time; time.sleep(60)"), tmp_path), 10
    )

    deadline.set()
    killed = await asyncio.wait_for(kill, 30)
    await asyncio.wait_for(wake.wait(), 10)
    events = store.agent_notifications(pending_only=False)
    cause = (
        "explicit-finish child task safety deadline expired after 7 seconds"
    )
    assert killed == (first_id, second_id)
    assert [event.data["task_id"] for event in events] == [first_id, second_id]
    assert all(event.data["background_phase"] == "task_kill" for event in events)
    assert all(event.data["note"] == cause for event in events)
    await asyncio.wait_for(registry.close(), 10)
    store.close()


@pytest.mark.asyncio
async def test_close_without_running_tasks_runs_no_process_table_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an empty session exit must not block the loop on a `ps` subprocess."""

    def no_subprocess(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("session exit ran a process-table query with no tasks")

    monkeypatch.setattr(spawn_module.subprocess, "run", no_subprocess)
    registry = BackgroundTaskRegistry()
    assert not await registry.close()
