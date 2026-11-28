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
from zeta.protocol.types import MessageRole, TextContent, ToolCall
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


def _parent_with_running_bash(root: str, cwd: str, command: str, pipe) -> None:
    """Run in a spawned process: start a background child whose bash call never returns."""

    store = ConversationStore(Path(root), cwd=Path(cwd))

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


async def _crash_with_running_bash(tmp_path: Path, body: str, *, pid_files: tuple[str, ...]) -> tuple[str, list[int]]:
    """Kill a real parent with SIGKILL while its child's bash call runs; return session id and pids."""

    command = f"echo $$ > {shlex.quote(str(tmp_path / 'pid'))}; {body}"
    parent_pipe, child_pipe = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_parent_with_running_bash,
        args=(str(tmp_path / "sessions"), str(tmp_path), command, child_pipe),
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


@pytest.mark.asyncio
async def test_exited_orphan_is_not_killed_and_child_is_told(tmp_path, monkeypatch):
    session_id, (pid,) = await _crash_with_running_bash(tmp_path, "sleep 0.3", pid_files=("pid",))
    await wait_pid_exited(pid)
    unrelated = await asyncio.create_subprocess_exec("sleep", "120", start_new_session=True)
    try:
        # A stale pgid in the record must never be signalled; only the lock and its holders count.
        for record in _records():
            record.write_text(json.dumps({"call_id": CALL_ID, "pgid": unrelated.pid, "stopping": False}))
        signalled = []
        # test-contract: seam record any signal the stop step sends.
        _seam(monkeypatch, "_signal_groups", lambda groups, signum: signalled.append(groups))
        recovered, _ = await _resume(tmp_path, monkeypatch, session_id, [])
        (result,) = _call_results(recovered)
        assert result.startswith("tool execution canceled"), result
        assert "ended before recovery" in result, result
        assert signalled == []
        assert _alive(unrelated.pid), "an unrelated process group was signalled"
        assert not _records()
    finally:
        kill_group_if_present(unrelated.pid)
        await await_or_fail(unrelated.wait(), what="reap unrelated sleep")


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
        assert f"process group {pids[0]}" in lifecycle.get("recovery_blocked", ""), lifecycle
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
