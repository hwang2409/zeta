from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.processes import kill_group_if_present
from zeta.agent import runner, turn_limit
from zeta.agent.presets import AgentPreset
from zeta.agent.recovery_lease import activate_agent_children
from zeta.core import agent_progress, agent_state
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import StreamEvent, StreamEventType, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tui.render import render_event


def _loop(store):
    backend = FakeBackend([])
    backend.model = "gpt-5.6-sol"
    backend.provider = "codex"
    return AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())


@pytest.mark.asyncio
async def test_shutdown_joins_marker_registration_before_suspending(tmp_path, monkeypatch):
    committing = asyncio.Event()
    closed_admission = asyncio.Event()
    release = asyncio.Event()
    from zeta.core.store._async_writes import _RegisterAgentChildCommand

    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = _loop(store)
        commit = store._commit_storage_command
        close = loop._background_owner.close_children

        async def gated_commit(command):
            if isinstance(command, _RegisterAgentChildCommand):
                committing.set()
                await await_or_fail(release.wait(), what="release marker commit")
            return await commit(command)

        async def close_children():
            closed_admission.set()
            await close()

        # test-contract: seam gate the real durable registration.
        monkeypatch.setattr(store, "_commit_storage_command", gated_commit)
        monkeypatch.setattr(loop._background_owner, "close_children", close_children)
        spawn = asyncio.create_task(runner.spawn_child(loop, agent_type="general", task="work", cwd=None))
        shutting_down = None
        try:
            await await_or_fail(committing.wait(), what="registration reached durable commit")
            shutting_down = asyncio.create_task(loop.close())
            await await_or_fail(closed_admission.wait(), what="shutdown is waiting for registration")
            assert not shutting_down.done()
            assert not store.agent_children()
            release.set()
            await await_or_fail(shutting_down, what="shutdown joined suspension")
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(spawn, what="setup stopped before starting")
            (run_id, marker), = store.agent_children().items()
            lifecycle, _ = read_child_lifecycle(store, Path(marker["child_session_path"]))
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            assert lifecycle.value["state"] == "suspended"
            assert lifecycle.value["suspension_reason"] == "normal_exit"
            assert marker["resume"]["account_pins"] == "pending"
            assert not store.agent_completion_notifications_by_child().get(run_id)
            assert not loop.backend.calls
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(spawn, return_exceptions=True), 10)
            if shutting_down is not None:
                await asyncio.wait_for(asyncio.gather(shutting_down, return_exceptions=True), 10)
            else:
                await await_or_fail(loop.close(), what="close runtime")


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before-notice", "after-notice"])
async def test_recovery_preserves_activated_episode_across_notice_interruption(tmp_path, monkeypatch, boundary):
    from zeta.core.store._async_writes import _AppendChildRecoveredCommand

    def backend(loop, model, **kwargs):
        assert model in (None, "gpt-5.6-sol")
        return loop.backend, None

    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", backend)
    running = asyncio.Event()
    keep_running = asyncio.Event()
    prompts = []

    async def consume(child, prompt, **kwargs):
        prompts.append(prompt)
        running.set()
        await await_or_fail(keep_running.wait(), what="active child waits for shutdown")

    monkeypatch.setattr(runner, "consume_child", consume)
    monkeypatch.setattr(runner, "consume_run", consume)
    root = tmp_path / "sessions"
    with ConversationStore(root, cwd=tmp_path) as store:
        loop = _loop(store)
        session_id = store.session_id
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn child")
            await await_or_fail(running.wait(), what="original child executed")
        finally:
            await await_or_fail(loop.close(), what="suspend child")
        marker = store.agent_children()[run_id]
        assert marker["resume"]["account_pins"] == "selected"

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        original = store._commit_storage_command

        async def interrupt_notice(command):
            if isinstance(command, _AppendChildRecoveredCommand):
                # test-contract: seam force interruption after runtime installation.
                assert loop._background_owner.owns_running(run_id)
                if boundary == "after-notice":
                    await original(command)
                raise InterruptedError("interrupted at recovery notice")
            return await original(command)

        try:
            with monkeypatch.context() as crash:
                # test-contract: seam interrupt the two durable notice boundaries.
                crash.setattr(store, "_commit_storage_command", interrupt_notice)
                with pytest.raises(InterruptedError, match="interrupted at recovery notice"):
                    await await_or_fail(activate_agent_children(loop), what="force crash boundary")
            assert prompts == ["work"]
            assert not store.agent_completion_notifications_by_child()
        finally:
            await await_or_fail(loop.close(), what="release failed reconstruction")

    running.clear()
    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        try:
            await await_or_fail(activate_agent_children(loop), what="retry reconstruction")
            await await_or_fail(running.wait(), what="recovered child executed")
            assert len(prompts) == 2
            assert "suspended" in prompts[-1]
            assert store.agent_children()[run_id]["child_session_path"] == marker["child_session_path"]
            notices = [row for row in store.agent_notifications(pending_only=False) if row.data.get("kind") == "child_recovered"]
            assert [row.data["logical_id"] for row in notices] == [
                f"child-recovered:{run_id}:1", f"child-recovered:{run_id}:2",
            ]
            assert not store.agent_completion_notifications_by_child()
            output = StringIO()
            Console(file=output, width=100).print(render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=notices[0].data)))
            assert "recovered" in output.getvalue()
        finally:
            await await_or_fail(loop.close(), what="close recovered runtime")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pending", "selected", "unavailable", "auxiliary-missing", "active-missing"])
async def test_resume_reconstructs_saved_model_accounts_and_lifetime_budget(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace

    from zeta.accounts.session import SessionAccountContext
    from zeta.accounts.store import AccountStore
    from zeta.agent.presets import GENERAL_PRESET
    from zeta.agent.resume_descriptor import preset_snapshot
    from zeta.core.session_metadata import SessionMetadata
    from zeta.protocol.types import ToolCall

    accounts = {"codex": "codex:saved"}
    restarts = []
    entered = asyncio.Event()
    release = asyncio.Event()
    provider = "ollama" if phase == "auxiliary-missing" else "codex"
    model = "qwen3:4b" if provider == "ollama" else "gpt-5.6-luna"

    class Accounts:
        def child_snapshot(self, provider):
            restarts.append(("select", provider))
            return SimpleNamespace(snapshot=lambda: accounts)

        def restore_child_snapshot(self, provider, pins):
            restarts.append(("restore", provider, pins))
            return SimpleNamespace(snapshot=lambda: pins)

    def resolve(loop, requested, *, pinned_context):
        assert requested == model
        assert pinned_context.snapshot() == ({} if phase == "auxiliary-missing" else accounts)
        if phase == "unavailable":
            return None, "model unavailable"
        return loop.backend, None

    async def consume(child, prompt, **kwargs):
        control = child.child_turn_control
        restarts.append(("execute", control.used, control.limit, prompt))
        entered.set()
        await await_or_fail(release.wait(), what="release resumed child")

    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", resolve)
    monkeypatch.setattr(runner, "consume_child", consume)
    root = tmp_path / "sessions"
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        run_id = f"{session_id}:1"
        with ConversationStore(store.session_dir / "agents", session_id="1") as child:
            child.start_agent_lifecycle(handle=run_id, started_at="2026-11-08T00:00:00+00:00", depth=1, agent_type="general", description="saved work")
            path = child.session_dir
        store.register_agent_child(
            ToolCall("saved-call", "agent", {"model": model, "max_turns": 4}),
            child_session_path=str(path), child_instance_id=run_id, description="saved work",
            background=True, agent_type="general", turns_used=3,
            resume={
                "preset": preset_snapshot(GENERAL_PRESET), "requested_model": model,
                "model": model, "provider": provider, "requested_max_turns": 4,
                "effective_max_turns": 4, "depth": 1, "cwd": str(tmp_path),
                "description": "saved work", "background": True, "follow_up": False,
                "explicit_finish": False, "account_pins": "pending" if phase == "pending" else "selected",
                "selected_accounts": {} if phase == "pending" else accounts,
            },
        )

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        loop.backend.provider = provider
        date = "2026-11-08T00:00:00+00:00"
        metadata = SessionMetadata(1, session_id, date, date, provider, model, str(tmp_path), 8, 200_000)
        loop.account_context = (
            SessionAccountContext(tmp_path, AccountStore(tmp_path / "accounts-home"), None, metadata)
            if phase in {"auxiliary-missing", "active-missing"} else Accounts()
        )
        try:
            before, _ = read_child_lifecycle(store, path)
            assert isinstance(before, LoadedAgentLifecycle)
            assert before.value["state"] == "suspended"
            await await_or_fail(activate_agent_children(loop), what="resume saved descriptor")
            if phase not in {"auxiliary-missing", "active-missing"}:
                assert restarts.pop(0) == (("select", "codex") if phase == "pending" else ("restore", "codex", accounts))
            descriptor = store.agent_children()[run_id]["resume"]
            assert (descriptor["account_pins"], descriptor["selected_accounts"]) == ("selected", accounts)
            if phase in {"unavailable", "active-missing"}:
                blocked, _ = read_child_lifecycle(store, path)
                assert isinstance(blocked, LoadedAgentLifecycle)
                assert blocked.value["state"] == "suspended"
                assert blocked.value["recovery_blocked"] == "saved model or account is unavailable"
                assert not entered.is_set()
                assert not store.agent_notifications(pending_only=False)
            else:
                resumed, _ = read_child_lifecycle(store, path)
                assert isinstance(resumed, LoadedAgentLifecycle)
                assert "recovery_blocked" not in resumed.value
                await await_or_fail(entered.wait(), what="resumed child executes")
                assert restarts[0][:3] == ("execute", 3, 4)
                assert "parent crashed" in restarts[0][3]
        finally:
            await await_or_fail(loop.close(), what="close reconstructed runtime")
            release.set()


class _ObservedBackend(FakeBackend):
    """Scripted backend that reports each provider call index before it streams."""

    def __init__(self, turns, on_call):
        super().__init__(turns)
        self.model, self.provider = "gpt-5.6-sol", "codex"
        self._on_call = on_call

    @owned_completion
    async def complete(self, messages, tool_schemas):
        self._on_call(len(self.calls))
        async for event in FakeBackend.complete.__wrapped__(self, messages, tool_schemas):
            yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["normal-exit", "crash"])
async def test_recovered_child_stats_cover_every_episode(tmp_path, monkeypatch, exit_mode):
    clock, pid = [1000.0], [1]
    _fake_clock_and_pid(monkeypatch, clock, pid)
    (tmp_path / "note.txt").write_text("note\n")
    read = ScriptedTurn(tool_calls=[ToolCall("read-1", "read", {"path": str(tmp_path / "note.txt")})])
    blocked = asyncio.Event()
    first = _ObservedBackend([read, ScriptedTurn([TextContent("never")], delay=60)],
                             lambda index: (clock.__setitem__(0, clock[0] + (30.0, 20.0)[index]), index == 1 and blocked.set()))
    root = tmp_path / "sessions"
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        loop = AgentLoop(first, store, skill_catalog=SkillCatalog.empty())
        try:
            run_id = await await_or_fail(
                runner.spawn_child(loop, agent_type="general", task="work", cwd=None),
                what="spawn child",
            )
            await await_or_fail(blocked.wait(), what="first episode used a tool and is blocked")
            clock[0] += 50.0
            if exit_mode == "crash":
                # A copy taken mid-run is what a killed process leaves on disk.
                shutil.copytree(root, tmp_path / "crashed")
        finally:
            await await_or_fail(loop.close(), what="close first parent")
    if exit_mode == "crash":
        # Markers hold absolute child paths, so the crash image replaces the original tree.
        shutil.rmtree(root)
        shutil.move(tmp_path / "crashed", root)

    pid[0] = 2
    second = _ObservedBackend([
        ScriptedTurn(tool_calls=[ToolCall("read-2", "read", {"path": str(tmp_path / "note.txt")})]),
        ScriptedTurn([TextContent("done")]),
    ], lambda index: clock.__setitem__(0, clock[0] + 10.0))
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *a, **kw: (second, None))
    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        try:
            await await_or_fail(activate_agent_children(loop), what="recover child")
            await await_or_fail(loop._background_owner.wait(), what="recovered child completes")
            receipt = store.agent_completion_notifications_by_child()[run_id]
            lifecycle, _ = read_child_lifecycle(store, Path(receipt.data["child_session_path"]))
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            value = lifecycle.value
            # Episode 1 runs 30 s + 20 s before the restart, then 50 s more; episode 2 runs 20 s.
            # A crash keeps episode 1 up to its last durable progress (30 s) and loses the tail.
            seconds = 120.0 if exit_mode == "normal-exit" else 50.0
            stats = receipt.data["stats"]
            assert (value["state"], value["tool_calls"], value["turns_used"]) == ("completed", 2, 3), value
            assert (stats["tool_calls"], stats["turns_used"]) == (2, 3), receipt.data
            # test-contract: latency the reported child duration is the contract, read from an injected clock
            assert (value["elapsed"], stats["elapsed"]) == (seconds, seconds), "elapsed must cover every episode"
            assert f"3 turns · {seconds:.1f}s · 2 tool calls" in receipt.data["text"], receipt.data["text"]
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")


@pytest.mark.asyncio
@pytest.mark.parametrize("epoch", [5, 12])
async def test_crashed_parent_below_recovery_epoch_cancels_running_child(tmp_path, monkeypatch, epoch):
    # Pin the home manifest before first open so the observed epoch is exact.
    (tmp_path / "storage-format.json").write_text(json.dumps({"writer_epoch": epoch}) + "\n")
    running = asyncio.Event()
    first = _ObservedBackend([ScriptedTurn([TextContent("never")], delay=60)], lambda index: running.set())
    root = tmp_path / "sessions"
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        assert (store._storage_lease.observed_epoch, store.child_recovery_enabled) == (epoch, False)
        loop = AgentLoop(first, store, skill_catalog=SkillCatalog.empty())
        try:
            run_id = await await_or_fail(
                runner.spawn_child(loop, agent_type="general", task="work", cwd=None),
                what="spawn child",
            )
            await await_or_fail(running.wait(), what="background child is mid-turn")
            assert not _progress_savers(), "below the recovery epoch a running child saves no progress"
            # A copy taken mid-run is what a killed process leaves on disk.
            shutil.copytree(root, tmp_path / "crashed")
        finally:
            await await_or_fail(loop.close(), what="close first parent")
    shutil.rmtree(root)
    shutil.move(tmp_path / "crashed", root)

    dispatched = []
    resolve = lambda *a, **kw: dispatched.append(a) or (FakeBackend([]), None)
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", resolve)
    monkeypatch.setattr(runner, "resolve_child_backend", resolve)
    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        try:
            await await_or_fail(loop.activate(), what="resume parent")
            await await_or_fail(loop._background_owner.wait(), what="no recovered child left running")
            assert dispatched == [], "a home below the recovery epoch must not re-dispatch the child"
            receipt = store.agent_completion_notifications_by_child()[run_id]
            lifecycle, _ = read_child_lifecycle(store, Path(receipt.data["child_session_path"]))
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            assert (receipt.data["status"], lifecycle.value["state"]) == ("canceled", "canceled"), receipt.data
            assert "session exited" in receipt.data["text"], receipt.data["text"]
            assert len(first.calls) == 1 and store.agent_children() == {}
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")


def _fake_clock_and_pid(monkeypatch, clock, pid):
    class _Os:
        def __getattr__(self, name):
            return getattr(os, name)

        def getpid(self):
            return pid[0]

    # test-contract: seam a fake monotonic clock and a new owner pid per process.
    monkeypatch.setattr(agent_state, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(agent_state, "os", _Os())


def _fast_progress_saves(monkeypatch):
    """Run the progress saver every millisecond; return a waiter for more saver wakeups."""

    ticks = [0, 0, asyncio.Event()]
    live = agent_progress.live_agent_lifecycle

    def counted(path):
        ticks[0] += 1
        if ticks[0] >= ticks[1]:
            ticks[2].set()
        return live(path)

    # test-contract: seam inject the save interval and count saver wakeups (raising=False reaches base's behavior).
    monkeypatch.setattr(turn_limit, "_PROGRESS_SAVE_SECONDS", 0.001, raising=False)
    monkeypatch.setattr(turn_limit, "live_agent_lifecycle", counted, raising=False)

    async def more_ticks(count):
        ticks[1], ticks[2] = ticks[0] + count, asyncio.Event()
        await await_or_fail(ticks[2].wait(), what=f"{count} saver wakeups")

    return more_ticks


def _record_lifecycle_writes(monkeypatch):
    writes = []
    write = ConversationStore._write_agent_lifecycle

    def recorded(self):
        write(self)
        writes.append(self._agent_lifecycle["state"])

    # test-contract: durability count every durable lifecycle write.
    monkeypatch.setattr(ConversationStore, "_write_agent_lifecycle", recorded)
    return writes


def _progress_savers():
    return [task for task in asyncio.all_tasks() if "save_progress_while_running" in repr(task.get_coro())]


def _child_lifecycle_file(store):
    (marker,) = store.agent_children().values()
    return Path(marker["child_session_path"]) / "agent_lifecycle.json"


def _saved_lifecycle(path):
    return json.loads(path.read_text())


async def _saved_progress(path, expected):
    """Wait, best effort, until the running child saved ``(elapsed, tool_calls)`` to disk."""

    def saved():
        value = _saved_lifecycle(path)
        return value["elapsed"], value["tool_calls"]

    # The tool call is counted when the runner consumes its start event, which can trail the
    # bash process start; a save in between has the new elapsed and the old count.
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(10):
            while saved() != expected:
                await asyncio.sleep(0)
    return saved()


async def _wait_for_file(path):
    async with asyncio.timeout(10):
        while not await asyncio.to_thread(path.exists):
            await asyncio.sleep(0)


def _kill_bash(pid_file):
    if pid_file.exists() and pid_file.read_text().strip():
        kill_group_if_present(int(pid_file.read_text()))


@pytest.mark.asyncio
async def test_crash_during_long_tool_call_keeps_saved_progress(tmp_path, monkeypatch):
    clock, pid = [1000.0], [1]
    _fake_clock_and_pid(monkeypatch, clock, pid)
    _fast_progress_saves(monkeypatch)
    writes = _record_lifecycle_writes(monkeypatch)
    pid_file = tmp_path / "bash.pid"
    (tmp_path / "note.txt").write_text("note\n")
    bash = ToolCall("bash-1", "bash", {"command": f"echo $$ > {pid_file}; exec sleep 300", "timeout": 300})
    first = _ObservedBackend([ScriptedTurn(tool_calls=[bash])], lambda index: clock.__setitem__(0, clock[0] + 30.0))
    root = tmp_path / "sessions"
    try:
        with ConversationStore(root, cwd=tmp_path) as store:
            session_id = store.session_id
            loop = AgentLoop(first, store, skill_catalog=SkillCatalog.empty())
            try:
                run_id = await await_or_fail(
                    runner.spawn_child(loop, agent_type="general", task="work", cwd=None),
                    what="spawn child",
                )
                await _wait_for_file(pid_file)
                lifecycle_file = _child_lifecycle_file(store)
                clock[0] += 40.0
                # 30 s model call + 40 s inside the bash call, and the started call counts.
                # Asserted after recovery, so a run without saves reaches the recovered totals first.
                saved_mid_call = await _saved_progress(lifecycle_file, (70.0, 1))
                # A copy taken mid-run is what a killed process leaves on disk.
                shutil.copytree(root, tmp_path / "crashed")
            finally:
                await await_or_fail(loop.close(), what="close first parent")
            assert not _progress_savers(), "shutdown must join the progress saver"
            assert writes[-1] == "suspended", writes
            # test-contract: latency the saved child duration is the contract, read from an injected clock
            assert _saved_lifecycle(lifecycle_file)["elapsed"] == 70.0
    finally:
        _kill_bash(pid_file)
    shutil.rmtree(root)
    shutil.move(tmp_path / "crashed", root)

    pid[0] = 2
    second = _ObservedBackend([
        ScriptedTurn(tool_calls=[ToolCall("read-2", "read", {"path": str(tmp_path / "note.txt")})]),
        ScriptedTurn([TextContent("done")]),
    ], lambda index: clock.__setitem__(0, clock[0] + 10.0))
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *a, **kw: (second, None))
    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        try:
            await await_or_fail(loop.activate(), what="recover child")
            await await_or_fail(loop._background_owner.wait(), what="recovered child completes")
            receipt = store.agent_completion_notifications_by_child()[run_id]
            lifecycle, _ = read_child_lifecycle(store, Path(receipt.data["child_session_path"]))
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            stats = receipt.data["stats"]
            # Episode 1 saved 70 s and the bash call; episode 2 adds 20 s and the read call.
            # test-contract: latency the reported child duration is the contract, read from an injected clock
            assert (lifecycle.value["elapsed"], stats["elapsed"]) == (90.0, 90.0), "elapsed must keep the long step"
            assert (lifecycle.value["tool_calls"], stats["tool_calls"]) == (2, 2), receipt.data
            assert "90.0s · 2 tool calls" in receipt.data["text"], receipt.data["text"]
            assert saved_mid_call == (70.0, 1), "progress inside a long step must be saved"
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["completed", "idle"])
async def test_progress_saves_stop_when_the_long_step_ends(tmp_path, monkeypatch, ending):
    clock, pid = [1000.0], [1]
    _fake_clock_and_pid(monkeypatch, clock, pid)
    more_ticks = _fast_progress_saves(monkeypatch)
    writes = _record_lifecycle_writes(monkeypatch)
    pid_file, go = tmp_path / "bash.pid", tmp_path / "go"
    command = f"echo $$ > {pid_file}; while [ ! -e {go} ]; do sleep 0.01; done"
    backend = _ObservedBackend(
        [ScriptedTurn(tool_calls=[ToolCall("bash-1", "bash", {"command": command, "timeout": 300})]),
         *[ScriptedTurn([TextContent("done")])] * 2],
        lambda index: clock.__setitem__(0, clock[0] + {0: 30.0, 1: 10.0}.get(index, 0.0)),
    )
    explicit = AgentPreset(name="explicit-test", preamble="", selection_guidance="test only", tool_names=None,
                           accepts_follow_ups=True, explicit_finish=True)
    agent_type, catalog = ("explicit-test", AgentCatalog((explicit,))) if ending == "idle" else ("general", None)
    try:
        with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
            loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty(), agent_catalog=catalog)
            loop.explicit_finish = ending == "idle"
            try:
                await await_or_fail(
                    runner.spawn_child(loop, agent_type=agent_type, task="work", cwd=None), what="spawn child",
                )
                await _wait_for_file(pid_file)
                lifecycle_file = _child_lifecycle_file(store)
                clock[0] += 40.0
                assert await _saved_progress(lifecycle_file, (70.0, 1)) == (70.0, 1), "progress inside a long step must be saved"
                await asyncio.to_thread(go.touch)
                if ending == "completed":
                    await await_or_fail(loop._background_owner.wait(), what="child completes")
                    assert _saved_lifecycle(lifecycle_file)["state"] == "completed"
                    assert not _progress_savers(), "a finished run must join its progress saver"
                else:
                    # The finish gate nudges an idle explicit-finish child once (turn 3) before it rests.
                    async with asyncio.timeout(10):
                        while (writes[-1], len(backend.calls)) != ("idle", 3):
                            await asyncio.sleep(0)
                    assert len(_progress_savers()) == 1
                    # A save checked just before the child went idle may still land; the saver awaits it before its next tick.
                    await more_ticks(2)
                    saved = len(writes)
                    await more_ticks(20)
                    assert len(writes) == saved, f"an idle child must not write progress: {writes[saved:]}"
                    # test-contract: latency the saved child duration is the contract, read from an injected clock
                    assert _saved_lifecycle(lifecycle_file)["elapsed"] == 80.0
            finally:
                await await_or_fail(loop.close(), what="close parent")
    finally:
        _kill_bash(pid_file)


@pytest.mark.asyncio
async def test_tool_call_counted_while_progress_saves_is_kept(tmp_path, monkeypatch):
    more_ticks = _fast_progress_saves(monkeypatch)
    reached, release = threading.Event(), threading.Event()
    live = agent_state.live_agent_lifecycle

    def held(path):
        value = live(path)
        if threading.current_thread() is not threading.main_thread() and not release.is_set():
            reached.set()
            release.wait(10)
        return value

    # test-contract: seam hold the save's worker thread after it read live progress (the bad order).
    monkeypatch.setattr(agent_state, "live_agent_lifecycle", held)
    with ConversationStore(tmp_path / "child") as store:
        store.start_agent_lifecycle(
            handle="child-1", started_at="2026-01-01T00:00:00+00:00", depth=1, agent_type="worker", description="work",
        )
        *_, save_progress = turn_limit.make_child_progress_callbacks(
            loop=None, child_store=store, child_marker_key="child-1", child_tool_call_id="call", recoverable=True,
        )
        saver = asyncio.create_task(save_progress())
        try:
            assert await await_or_fail(asyncio.to_thread(reached.wait, 10), what="a save reads live progress")
            store.update_agent_lifecycle(tool_calls=1)  # the runner counts a tool call while the save runs
            release.set()
            await more_ticks(5)
        finally:
            release.set()
            saver.cancel()
            await await_or_fail(asyncio.gather(saver, return_exceptions=True), what="join the progress saver")
        saved = json.loads(store.agent_lifecycle_path.read_text())
        assert (saved["state"], saved["tool_calls"]) == ("running", 1), "a save must not drop a tool call counted meanwhile"


@pytest.mark.asyncio
async def test_progress_save_queued_behind_a_rest_does_not_write(tmp_path, monkeypatch):
    from zeta.agent.run_consumer import _transition_idle
    from zeta.core.store._async_writes import _ChildActivityCommand

    _fast_progress_saves(monkeypatch)
    writes = _record_lifecycle_writes(monkeypatch)
    with ConversationStore(tmp_path / "child") as store:
        store.start_agent_lifecycle(
            handle="child-1", started_at="2026-01-01T00:00:00+00:00", depth=1, agent_type="worker", description="work",
        )
        commit, queued, release, saved = store._commit_storage_command, asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def held(command):
            queued.set()
            await await_or_fail(release.wait(), what="release the held save")
            try:
                return await commit(command)
            finally:
                saved.set()

        # test-contract: seam hold the saver's first save after its running check (the bad order).
        monkeypatch.setattr(store, "_commit_storage_command", held)
        *_, save_progress = turn_limit.make_child_progress_callbacks(
            loop=None, child_store=store, child_marker_key="child-1", child_tool_call_id="call", recoverable=True,
        )
        saver = asyncio.create_task(save_progress())
        try:
            await await_or_fail(queued.wait(), what="the saver submits a save while running")
            # test-contract: seam later saves and the idle transition commit normally.
            monkeypatch.setattr(store, "_commit_storage_command", commit)
            del writes[:]
            _transition_idle(store)  # the child rests while the save waits
            release.set()
            await await_or_fail(saved.wait(), what="the held save lands")
            assert writes == [], f"a progress save queued behind a rest must not write: {writes}"
            # test-contract: seam commit the idle transition the way the run consumer does.
            await await_or_fail(store._commit_storage_command(_ChildActivityCommand(state="idle")), what="commit idle")
        finally:
            release.set()
            saver.cancel()
            await await_or_fail(asyncio.gather(saver, return_exceptions=True), what="join the progress saver")
        saved_lifecycle = json.loads(store.agent_lifecycle_path.read_text())
        assert writes == ["idle"] and saved_lifecycle["idle_since"], "the idle transition writes the rest with idle_since"


@pytest.mark.asyncio
async def test_canceled_progress_save_keeps_tool_call_counted_meanwhile(tmp_path, monkeypatch):
    _fast_progress_saves(monkeypatch)
    reached, release = threading.Event(), threading.Event()
    live = agent_state.live_agent_lifecycle

    def held(path):
        value = live(path)
        if threading.current_thread() is not threading.main_thread() and not release.is_set():
            reached.set()
            release.wait(10)
        return value

    # test-contract: seam hold the save's worker thread after it read live progress (the bad order).
    monkeypatch.setattr(agent_state, "live_agent_lifecycle", held)
    with ConversationStore(tmp_path / "child") as store:
        store.start_agent_lifecycle(
            handle="child-1", started_at="2026-01-01T00:00:00+00:00", depth=1, agent_type="worker", description="work",
        )
        *_, save_progress = turn_limit.make_child_progress_callbacks(
            loop=None, child_store=store, child_marker_key="child-1", child_tool_call_id="call", recoverable=True,
        )
        saver = asyncio.create_task(save_progress())
        try:
            assert await await_or_fail(asyncio.to_thread(reached.wait, 10), what="a save reads live progress")
            store.update_agent_lifecycle(tool_calls=1)  # the runner counts a tool call while the save runs
            saver.cancel()  # the run ends while the save is in flight
            await asyncio.sleep(0)
            release.set()
            await await_or_fail(asyncio.gather(saver, return_exceptions=True), what="join the progress saver")
        finally:
            release.set()
            saver.cancel()
            await await_or_fail(asyncio.gather(saver, return_exceptions=True), what="join the progress saver")
        assert saver.cancelled()
        current = agent_progress.live_agent_lifecycle(store.agent_lifecycle_path)
        assert current["tool_calls"] == 1, "a canceled save must not leave its older snapshot published"


@pytest.mark.asyncio
async def test_failed_progress_save_logs_and_next_interval_saves(tmp_path, monkeypatch, caplog):
    _fast_progress_saves(monkeypatch)
    with ConversationStore(tmp_path / "child") as store:
        store.start_agent_lifecycle(
            handle="child-1", started_at="2026-01-01T00:00:00+00:00", depth=1, agent_type="worker", description="work",
        )
        commit, saved = store._commit_storage_command, asyncio.Event()
        calls = []

        async def flaky(command):
            calls.append(command)
            if len(calls) == 1:
                raise OSError("disk full")
            result = await commit(command)
            saved.set()
            return result

        # test-contract: seam fail the first progress save, then save normally.
        monkeypatch.setattr(store, "_commit_storage_command", flaky)
        *_, save_progress = turn_limit.make_child_progress_callbacks(
            loop=None, child_store=store, child_marker_key="child-1", child_tool_call_id="call", recoverable=True,
        )
        saver = asyncio.create_task(save_progress())
        try:
            with caplog.at_level("WARNING", logger="zeta.agent.turn_limit"):
                await await_or_fail(saved.wait(), what="a later interval saves after a failed save")
        finally:
            saver.cancel()
            await await_or_fail(asyncio.gather(saver, return_exceptions=True), what="join the progress saver")
        (record,) = [r for r in caplog.records if r.name == "zeta.agent.turn_limit"]
        assert record.levelname == "WARNING" and "disk full" in str(record.exc_info[1]), record
