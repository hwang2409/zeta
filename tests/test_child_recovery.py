from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.agent import runner
from zeta.agent.recovery_lease import activate_agent_children
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
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
    import os
    import shutil
    from types import SimpleNamespace

    from tests.support.fake_backend import ScriptedTurn
    from zeta.core import agent_state
    from zeta.protocol.types import TextContent, ToolCall

    clock = [1000.0]
    pid = [1]

    class _Os:
        def __getattr__(self, name):
            return getattr(os, name)

        def getpid(self):
            return pid[0]

    # test-contract: seam a fake monotonic clock and a new owner pid per process.
    monkeypatch.setattr(agent_state, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(agent_state, "os", _Os())
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
