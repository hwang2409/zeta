"""A bash call left running by a killed parent is stopped before its child resumes."""

from __future__ import annotations

import asyncio
import contextlib
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
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageRole, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

CALL_ID = "orphan-call"


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
        await asyncio.sleep(300)  # the test kills this process with SIGKILL

    asyncio.run(main())


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


async def _resume(tmp_path: Path, monkeypatch, session_id: str, pids: list[int]) -> _Recovered:
    recovered = _Recovered(pids)
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *a, **kw: (recovered, None))
    with ConversationStore(tmp_path / "sessions", session_id=session_id, cwd=tmp_path) as store:
        loop = AgentLoop(_backend([]), store, skill_catalog=SkillCatalog.empty())
        try:
            await await_or_fail(loop.activate(), what="resume crashed parent")
            await await_or_fail(recovered.called.wait(), what="recovered child called its model")
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")
    return recovered


@pytest.mark.asyncio
async def test_running_orphan_is_stopped_before_child_resumes(tmp_path, monkeypatch):
    session_id, pids = await _crash_with_running_bash(tmp_path, "exec sleep 120", pid_files=("pid",))
    with _owned_groups(pids):
        assert _alive(pids[0]), "the orphan must outlive the killed parent"
        recovered = await _resume(tmp_path, monkeypatch, session_id, pids)
        assert recovered.alive_at_first_call == [], "the orphan still ran when the child resumed"
        (result,) = _call_results(recovered)
        assert result.startswith("tool execution canceled"), result
        assert "stopped its leftover processes" in result, result
