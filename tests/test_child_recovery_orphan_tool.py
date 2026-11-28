"""A bash call left running by a killed parent is stopped before its child resumes."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import multiprocessing
import os
import shlex
import signal
import subprocess
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.processes import (
    _pid_state,
    kill_group_if_present,
    owned_child_process,
    read_pid_file,
    wait_pid_exited,
)
from zeta.agent import runner
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.session import env_home
from zeta.core.store import ConversationStore
from zeta.core.store._async_writes import _RepairChildInterruptionCommand
from zeta.protocol.types import MessageOrigin, MessageRole, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

CALL_ID = "orphan-call"


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    # The suite shares one isolated home; give each test its own so record checks see only this test.
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))


def _backend(turns):
    backend = FakeBackend(turns)
    backend.model, backend.provider = "gpt-5.6-sol", "codex"
    return backend


def _parent_with_running_bash(root: str, cwd: str, command: str, pipe, stall_after_spawn: bool = False) -> None:
    """Run in a spawned process: start a background child whose bash call never returns."""

    store = ConversationStore(Path(root), cwd=Path(cwd))
    if stall_after_spawn:
        bash_tool = importlib.import_module("zeta.tools.bash")
        real_spawn = bash_tool.spawn_background

        async def stalled_spawn(*args, **kwargs):
            await real_spawn(*args, **kwargs)
            # The command already runs; zeta is killed before it gets back from the spawn.
            await asyncio.wait_for(asyncio.Event().wait(), 300)

        # test-contract: seam zeta dies after the command started but before the spawn call returned.
        bash_tool.spawn_background = stalled_spawn

    async def main() -> None:
        child = _backend([
            ScriptedTurn(tool_calls=[ToolCall(CALL_ID, "bash", {"command": command, "timeout": 300})]),
            ScriptedTurn([TextContent("never")], delay=300),
        ])
        runner.resolve_child_backend = lambda *a, **kw: (child, None)
        loop = AgentLoop(_backend([]), store, skill_catalog=SkillCatalog.empty())
        await runner.spawn_child(loop, agent_type="general", task="work", cwd=None)
        pipe.send(store.session_id)
        # The test kills this process with SIGKILL; the deadline only ends a forgotten helper.
        await asyncio.wait_for(asyncio.Event().wait(), 300)

    asyncio.run(main())


def _seam(monkeypatch, name: str, value) -> None:
    # Imported here so that a checkout without the record module still reaches the behavior assertions.
    try:
        module = importlib.import_module("zeta.tools._shared.tool_process_record")
    except ImportError:
        return
    monkeypatch.setattr(module, name, value)


def _alive(pid: int) -> bool:
    state = _pid_state(pid)
    return state is not None and not state.startswith("Z")


class _Recovered(FakeBackend):
    """Record whether the orphan pids were alive when the recovered child first called the model."""

    def __init__(self, pids: list[int]) -> None:
        super().__init__([ScriptedTurn([TextContent("checked")]), ScriptedTurn([TextContent("checked")])])
        self.model, self.provider = "gpt-5.6-sol", "codex"
        self.pids = pids
        self.alive_at_first_call: list[int] | None = None
        self.called = asyncio.Event()

    @owned_completion
    async def complete(self, messages, tool_schemas):
        if self.alive_at_first_call is None:
            self.alive_at_first_call = [pid for pid in self.pids if _alive(pid)]
            self.called.set()
        async for event in FakeBackend.complete.__wrapped__(self, messages, tool_schemas):
            yield event


def _call_results(backend: FakeBackend) -> list[str]:
    messages, _ = backend.calls[0]
    return [
        message.tool_result.content
        for message in messages
        if message.role is MessageRole.TOOL_RESULT and message.tool_result.tool_call_id == CALL_ID
    ]


async def _crash_with_running_bash(
    tmp_path: Path, body: str, *, pid_files: tuple[str, ...], stall_after_spawn: bool = False,
) -> tuple[str, list[int]]:
    """Kill a real parent with SIGKILL while its child's bash call runs; return session id and pids."""

    command = f"echo $$ > {shlex.quote(str(tmp_path / 'pid'))}; {body}"
    parent_pipe, child_pipe = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_parent_with_running_bash,
        args=(str(tmp_path / "sessions"), str(tmp_path), command, child_pipe, stall_after_spawn),
    )
    async with owned_child_process(process, endpoints=(parent_pipe, child_pipe)):
        process.start()
        child_pipe.close()
        if not await asyncio.to_thread(parent_pipe.poll, 30):
            pytest.fail("crashing parent did not start its child within 30 seconds")
        session_id = parent_pipe.recv()
        pids = [await read_pid_file(tmp_path / name, timeout=30) for name in pid_files]
        os.kill(process.pid, signal.SIGKILL)
        await asyncio.to_thread(process.join, 10)
        assert process.exitcode == -signal.SIGKILL
    return session_id, pids


@contextlib.contextmanager
def _owned_groups(pids: list[int]):
    groups = set()
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            groups.add(os.getpgid(pid))
    try:
        yield
    finally:
        for group in groups:
            kill_group_if_present(group)


async def _resume(
    tmp_path: Path, monkeypatch, session_id: str, pids: list[int], *, expect_call: bool = True,
) -> tuple[_Recovered, dict]:
    """Resume the crashed parent through AgentLoop.activate(); return the child backend and lifecycle."""

    recovered = _Recovered(pids)
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *a, **kw: (recovered, None))
    with ConversationStore(tmp_path / "sessions", session_id=session_id, cwd=tmp_path) as store:
        loop = AgentLoop(_backend([]), store, skill_catalog=SkillCatalog.empty())
        try:
            await await_or_fail(loop.activate(), what="resume crashed parent")
            if expect_call:
                await await_or_fail(recovered.called.wait(), what="recovered child called its model")
            (marker,) = store.agent_children().values()
            lifecycle, _ = read_child_lifecycle(store, Path(marker["child_session_path"]))
            assert isinstance(lifecycle, LoadedAgentLifecycle)
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")
    return recovered, lifecycle.value


def _records() -> list[Path]:
    return sorted((env_home() / "run" / "tool-processes").rglob("*.lock"))


@pytest.mark.asyncio
async def test_running_orphan_is_stopped_before_child_resumes(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(tmp_path, "exec sleep 120", pid_files=("pid",))
    with _owned_groups(pids):
        assert _alive(pids[0]), "the orphan must outlive the killed parent"
        recovered, _ = await _resume(tmp_path, monkeypatch, session_id, pids)
        assert recovered.alive_at_first_call == [], "the orphan still ran when the child resumed"
        (result,) = _call_results(recovered)
        assert result.startswith("tool execution canceled"), result
        assert "stopped its leftover processes" in result, result
        assert not _records(), "recovery left a record behind"


@pytest.mark.asyncio
async def test_orphan_of_a_crash_right_after_spawn_is_stopped(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(
        tmp_path, "exec sleep 120", pid_files=("pid",), stall_after_spawn=True,
    )
    with _owned_groups(pids):
        assert _alive(pids[0]), "the orphan must outlive the killed parent"
        _, lifecycle = await _resume(tmp_path, monkeypatch, session_id, pids, expect_call=False)
        assert "recovery_blocked" not in lifecycle, lifecycle["recovery_blocked"]
        assert not _alive(pids[0]), "recovery did not stop the orphan"


@pytest.mark.asyncio
async def test_orphan_descendant_after_leader_exit_is_stopped(tmp_path, monkeypatch):
    bgpid = shlex.quote(str(tmp_path / "bgpid"))
    session_id, (leader, background) = await _crash_with_running_bash(
        tmp_path, f"sleep 120 & echo $! > {bgpid}; wait", pid_files=("pid", "bgpid"),
    )
    with _owned_groups([leader, background]):
        os.kill(leader, signal.SIGKILL)
        await wait_pid_exited(leader)
        assert _alive(background), "only the background descendant should hold the lock now"
        recovered, _ = await _resume(tmp_path, monkeypatch, session_id, [background])
        assert recovered.alive_at_first_call == [], "the descendant still ran when the child resumed"
        (result,) = _call_results(recovered)
        assert "stopped its leftover processes" in result, result


async def _open_record_elsewhere(record: Path) -> asyncio.subprocess.Process:
    """Start an unrelated process in its own session that holds ``record`` open, like a file indexer."""
    opener = await asyncio.create_subprocess_exec(
        "sh", "-c", 'exec 9<"$0"; echo open; exec sleep 120', str(record),
        stdout=asyncio.subprocess.PIPE, start_new_session=True,
    )
    assert opener.stdout is not None
    assert await await_or_fail(opener.stdout.readline(), what="unrelated opener") == b"open\n"
    return opener


@pytest.mark.asyncio
async def test_exited_orphan_is_not_killed_and_child_is_told(tmp_path, monkeypatch):
    session_id, (pid,) = await _crash_with_running_bash(tmp_path, "sleep 2", pid_files=("pid",))
    await wait_pid_exited(pid)
    (record,) = _records()
    unrelated = await asyncio.create_subprocess_exec("sleep", "120", start_new_session=True)
    opener = await _open_record_elsewhere(record)
    try:
        # A stale pgid in the record must never be signalled; only the lock and its holders count.
        record.write_text(json.dumps({"call_id": CALL_ID, "pgid": unrelated.pid, "stopping": False}))
        signalled = []
        # test-contract: seam record any signal the stop step sends.
        _seam(monkeypatch, "_signal_groups", lambda groups, signum: signalled.append(groups))
        recovered, _ = await _resume(tmp_path, monkeypatch, session_id, [])
        (result,) = _call_results(recovered)
        assert result.startswith("tool execution canceled"), result
        assert "ended before recovery" in result, result
        assert signalled == [], "a process outside the tool tree was signalled"
        assert _alive(unrelated.pid), "an unrelated process group was signalled"
        assert not _records()
    finally:
        for process in (unrelated, opener):
            kill_group_if_present(process.pid)
            await await_or_fail(process.wait(), what="reap unrelated process")


@pytest.mark.asyncio
async def test_unrelated_holder_of_a_busy_record_is_not_signalled(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(tmp_path, "exec sleep 120", pid_files=("pid",))
    (record,) = _records()
    unrelated = await _open_record_elsewhere(record)
    try:
        with _owned_groups(pids):
            recovered, _ = await _resume(tmp_path, monkeypatch, session_id, pids)
            assert recovered.alive_at_first_call == [], "the orphan still ran when the child resumed"
            assert "stopped its leftover processes" in _call_results(recovered)[0]
            assert _alive(unrelated.pid), "a process outside the tool tree that holds the record was signalled"
    finally:
        kill_group_if_present(unrelated.pid)
        await await_or_fail(unrelated.wait(), what="reap unrelated opener")


@pytest.mark.asyncio
async def test_restart_during_stop_resumes_once_without_a_second_copy(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(tmp_path, "trap '' TERM; exec sleep 120", pid_files=("pid",))
    with _owned_groups(pids):
        original = ConversationStore._commit_storage_command

        async def crash_after_kill(store, command):
            if isinstance(command, _RepairChildInterruptionCommand):
                raise InterruptedError("crashed after the kill")
            return await original(store, command)

        with monkeypatch.context() as crash:
            # test-contract: seam crash between the kill and the durable canceled result.
            crash.setattr(ConversationStore, "_commit_storage_command", crash_after_kill)
            with contextlib.suppress(InterruptedError):
                await _resume(tmp_path, monkeypatch, session_id, pids, expect_call=False)
        assert not _alive(pids[0]), "the first recovery did not stop the orphan"
        recovered, _ = await _resume(tmp_path, monkeypatch, session_id, pids)
        assert recovered.alive_at_first_call == []
        (result,) = _call_results(recovered)
        assert "stopped its leftover processes" in result, result
        assert not _records()


@pytest.mark.asyncio
async def test_unstoppable_orphan_blocks_recovery(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(tmp_path, "exec sleep 120", pid_files=("pid",))
    with _owned_groups(pids):
        # test-contract: seam the holder scan finds nobody while the lock is busy (lsof missing, EPERM).
        _seam(monkeypatch, "_holders", lambda fd, path: [])
        recovered, lifecycle = await _resume(tmp_path, monkeypatch, session_id, pids, expect_call=False)
        assert lifecycle["state"] == "suspended"
        # On Linux /bin/sh forks bash instead of exec-ing it, so the group leader is the shell, not $$.
        assert f"process group {os.getpgid(pids[0])}" in lifecycle.get("recovery_blocked", ""), lifecycle
        assert recovered.calls == []
        assert _alive(pids[0])
        assert _records(), "the record must stay so the next resume tries again"


@pytest.mark.asyncio
async def test_normal_quit_leaves_no_record(tmp_path, monkeypatch):
    child = _backend([
        ScriptedTurn(tool_calls=[ToolCall(CALL_ID, "bash", {
            "command": f"echo $$ > {shlex.quote(str(tmp_path / 'pid'))}; exec sleep 120", "timeout": 300,
        })]),
        ScriptedTurn([TextContent("never")], delay=300),
    ])
    monkeypatch.setattr(runner, "resolve_child_backend", lambda *a, **kw: (child, None))
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(_backend([]), store, skill_catalog=SkillCatalog.empty())
        pid = None
        try:
            await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn")
            pid = await read_pid_file(tmp_path / "pid", timeout=30)
            assert _records()
        finally:
            await await_or_fail(loop.close(), what="quit normally")
            if pid is not None:
                kill_group_if_present(pid)
        assert not _alive(pid)
        assert not _records()


@pytest.mark.asyncio
async def test_normal_quit_with_unstoppable_record_blocks_recovery(tmp_path, monkeypatch):
    child = _backend([
        ScriptedTurn(tool_calls=[ToolCall(CALL_ID, "bash", {
            "command": f"echo $$ > {shlex.quote(str(tmp_path / 'pid'))}; exec sleep 120", "timeout": 300,
        })]),
        ScriptedTurn([TextContent("never")], delay=300),
    ])
    monkeypatch.setattr(runner, "resolve_child_backend", lambda *a, **kw: (child, None))
    resume_child = importlib.import_module("zeta.agent.resume_child")

    def unstoppable(session_dir, call_ids):
        raise resume_child.OrphanedToolProcessError("could not be stopped (process group 4242)")

    # test-contract: seam a record survives the in-process cancel and its holder cannot be stopped.
    monkeypatch.setattr(resume_child, "_stop_orphans", unstoppable)
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(_backend([]), store, skill_catalog=SkillCatalog.empty())
        pid = None
        try:
            await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn")
            pid = await read_pid_file(tmp_path / "pid", timeout=30)
        finally:
            await await_or_fail(loop.close(), what="quit normally")
            if pid is not None:
                kill_group_if_present(pid)
        (marker,) = store.agent_children().values()
        lifecycle, _ = read_child_lifecycle(store, Path(marker["child_session_path"]))
        assert isinstance(lifecycle, LoadedAgentLifecycle)
        assert lifecycle.value["state"] == "suspended", lifecycle.value
        assert "process group 4242" in lifecycle.value.get("recovery_blocked", ""), lifecycle.value


@pytest.mark.asyncio
async def test_root_session_bash_removes_its_record(tmp_path):
    listing = tmp_path / "records"
    records = shlex.quote(str(env_home() / "run" / "tool-processes"))
    backend = _backend([
        ScriptedTurn(tool_calls=[
            ToolCall("root-done", "bash", {"command": f"ls -R {records} > {shlex.quote(str(listing))}"}),
            ToolCall("root-timeout", "bash", {"command": "exec sleep 120", "timeout": 0.5}),
        ]),
        ScriptedTurn([TextContent("done")]),
    ])
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(backend, store, max_turns=2, skill_catalog=SkillCatalog.empty())
        try:
            async for _ in loop.run_turn("go", origin=MessageOrigin.USER):
                pass
        finally:
            await await_or_fail(loop.close(), what="close root loop")
    messages, _ = backend.calls[-1]
    results = [m.tool_result for m in messages if m.role is MessageRole.TOOL_RESULT]
    assert [r.tool_call_id for r in results] == ["root-done", "root-timeout"], results
    assert "timed out" in results[1].content, results[1].content
    assert ".lock" in listing.read_text(), "a root-session call must hold a record while it runs"
    assert not _records(), "a finished or timed-out root-session call left its record behind"


@pytest.mark.asyncio
async def test_bash_call_runs_when_its_record_cannot_be_written(tmp_path, monkeypatch):
    module = importlib.import_module("zeta.tools._shared.tool_process_record")
    real_write = module._write

    def failing_pgid_write(fd, value):
        if value["pgid"] is not None:
            raise OSError("disk full")
        real_write(fd, value)

    # test-contract: seam the record write of the process group fails.
    monkeypatch.setattr(module, "_write", failing_pgid_write)
    backend = _backend([
        ScriptedTurn(tool_calls=[ToolCall("root-echo", "bash", {"command": "echo ran"})]),
        ScriptedTurn([TextContent("done")]),
    ])
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(backend, store, max_turns=2, skill_catalog=SkillCatalog.empty())
        try:
            async for _ in loop.run_turn("go", origin=MessageOrigin.USER):
                pass
        finally:
            await await_or_fail(loop.close(), what="close root loop")
    messages, _ = backend.calls[-1]
    (result,) = [m.tool_result for m in messages if m.role is MessageRole.TOOL_RESULT]
    assert "ran" in result.content, result.content
    assert not _records()


def test_holder_scan_finds_lsof_outside_path(tmp_path, monkeypatch):
    module = importlib.import_module("zeta.tools._shared.tool_process_record")
    if os.path.isdir("/proc/self/fd") or not any(os.access(p, os.X_OK) for p in ("/usr/sbin/lsof", "/usr/bin/lsof")):
        pytest.skip("the lsof scan runs only without /proc and with lsof installed")
    record = tmp_path / "record"
    record.write_text("")
    with open(record) as holder_file, subprocess.Popen(["sleep", "120"], stdin=holder_file) as holder:
        try:
            monkeypatch.setenv("PATH", "")
            with open(record) as own:
                assert holder.pid in module._holders(own.fileno(), record)
        finally:
            holder.kill()


def test_holder_that_exits_between_lock_probe_and_scan_is_not_unstoppable(tmp_path, monkeypatch):
    module = importlib.import_module("zeta.tools._shared.tool_process_record")
    session_dir = tmp_path / "session"
    record = module.open_record(session_dir, CALL_ID)
    holder = subprocess.Popen(["sleep", "120"], pass_fds=(record.fd,), start_new_session=True)
    record.set_pgid(holder.pid)
    record.close_parent_fd()

    def holder_exits_then_scan(fd, path):
        # The last tool process ends after the busy lock probe and before the holder scan sees it.
        holder.kill()
        holder.wait(10)
        return []

    try:
        # test-contract: seam the last holder exits between the lock probe and the holder scan.
        monkeypatch.setattr(module, "_holders", holder_exits_then_scan)
        outcome = module.stop_orphan(session_dir, CALL_ID)
        assert outcome.kind == "exited", outcome
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(10)
