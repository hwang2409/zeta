"""Tests for session-scoped background process tools."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools._shared.process import BackgroundTaskRegistry, _group_exists
from zeta.tui.render import format_status


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


async def _wait_for_exit(registry: BackgroundTaskRegistry, task_id: str) -> None:
    await asyncio.wait_for(registry.wait(task_id), timeout=30)


async def _collect(events):
    return [event async for event in events]


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

    await _collect(loop.run_turn("first"))
    await _collect(loop.run_turn("second"))

    for child_store in child_stores:
        with pytest.raises(OSError):
            os.fstat(child_store.directory_fd)
    assert store.directory_fd >= 0
    await loop.close()


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
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[foreground]),
            ScriptedTurn(tool_calls=[grandchild]),
            ScriptedTurn([TextContent("done")], delay=0.2),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    tracked = []
    owner = loop._background_owner
    track_store = owner.track_store
    owner.track_store = lambda child_store: (tracked.append(child_store), track_store(child_store))

    await _collect(loop.run_turn("start"))
    for _ in range(100):
        if len(tracked) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(tracked) == 2
    parent_fd = tracked[0].directory_fd
    os.fstat(parent_fd)
    await asyncio.wait_for(owner.wait(), timeout=5)
    for child_store in tracked:
        with pytest.raises(OSError):
            os.fstat(child_store.directory_fd)
    assert store.directory_fd >= 0
    await loop.close()


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
async def test_background_approval_cap_and_session_cleanup(tmp_path: Path) -> None:
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

    tasks = BackgroundTaskRegistry(max_tasks=1)
    first, _ = await tasks.start("sleep 30", tmp_path)
    with pytest.raises(ValueError, match="limit reached"):
        await tasks.start("sleep 30", tmp_path)
    await tasks.close()
    assert (await tasks.output(first))["running"] is False


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
