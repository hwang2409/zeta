"""Tests for session-scoped background process tools."""

from __future__ import annotations

import asyncio
import shlex
import signal
import sys
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.tools._process import BackgroundTaskRegistry, _group_exists
from zeta.tui.render import format_status
from zeta.types import ToolCall


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


async def _wait_for_exit(registry: BackgroundTaskRegistry, task_id: str) -> None:
    for _ in range(100):
        status = await registry.output(task_id, since=0)
        if not status["running"]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("background task did not exit")


@pytest.mark.asyncio
async def test_background_start_poll_and_kill_round_trip(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
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
    registry = ToolRegistry(tmp_path, max_output_chars=10_000)
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
    first = ToolRegistry(store.cwd, session_store=store)
    started = await first.execute(
        ToolCall("start", "run_background", {"command": "sleep 30"})
    )
    task_id = started["structuredContent"]["task_id"]
    await first.close()

    resumed_store = ConversationStore(
        tmp_path / "sessions", session_id=store.session_id
    )
    resumed = ToolRegistry(resumed_store.cwd, session_store=resumed_store)
    output = await resumed.execute(
        ToolCall("output", "task_output", {"task_id": task_id})
    )
    assert output["structuredContent"]["running"] is False
    assert "previous session" in output["structuredContent"]["note"]
    await resumed.close()


def test_background_footer_segment_degrades_as_a_whole() -> None:
    assert "bg 2" in format_status("fake", "offline", "idle", background_count=2).plain
    narrow = format_status(
        "fake",
        "offline",
        "idle",
        background_count=2,
        vim_state="NORMAL",
        width=30,
    ).plain
    assert "bg 2" not in narrow


def test_background_tools_are_discovered() -> None:
    registry = ToolRegistry(Path("."))
    assert {schema["name"] for schema in registry.schemas} >= {
        "run_background",
        "task_output",
        "task_kill",
    }
