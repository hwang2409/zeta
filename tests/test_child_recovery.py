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
@pytest.mark.parametrize("boundary", ["install", "notice"])
async def test_recovery_reuses_episode_after_failure_before_activation(tmp_path, monkeypatch, boundary):
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
                if boundary == "notice":
                    await original(command)
                raise InterruptedError("interrupted before activation")
            return await original(command)

        try:
            with monkeypatch.context() as crash:
                # test-contract: seam interrupt the two durable notice boundaries.
                crash.setattr(store, "_commit_storage_command", interrupt_notice)
                with pytest.raises(InterruptedError, match="interrupted before activation"):
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
            notice, = [row for row in store.agent_notifications(pending_only=False) if row.data.get("kind") == "child_recovered"]
            assert notice.data["logical_id"] == f"child-recovered:{run_id}:1"
            assert not store.agent_completion_notifications_by_child()
            output = StringIO()
            Console(file=output, width=100).print(render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=notice.data)))
            assert "recovered" in output.getvalue()
        finally:
            await await_or_fail(loop.close(), what="close recovered runtime")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pending", "selected", "unavailable"])
async def test_resume_reconstructs_saved_model_accounts_and_lifetime_budget(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace

    from zeta.agent.presets import GENERAL_PRESET
    from zeta.agent.resume_descriptor import preset_snapshot
    from zeta.protocol.types import ToolCall

    accounts = {"codex": "codex:saved"}
    restarts = []
    entered = asyncio.Event()
    release = asyncio.Event()
    model = "gpt-5.6-luna"

    class Accounts:
        def child_snapshot(self, provider):
            restarts.append(("select", provider))
            return SimpleNamespace(snapshot=lambda: accounts)

        def restore_child_snapshot(self, provider, pins):
            restarts.append(("restore", provider, pins))
            return SimpleNamespace(snapshot=lambda: pins)

    def resolve(loop, requested, *, pinned_context):
        assert requested == model
        assert pinned_context.snapshot() == accounts
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
                "model": model, "provider": "codex", "requested_max_turns": 4,
                "effective_max_turns": 4, "depth": 1, "cwd": str(tmp_path),
                "description": "saved work", "background": True, "follow_up": False,
                "explicit_finish": False, "account_pins": "pending" if phase == "pending" else "selected",
                "selected_accounts": {} if phase == "pending" else accounts,
            },
        )

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store)
        loop.account_context = Accounts()
        try:
            before, _ = read_child_lifecycle(store, path)
            assert isinstance(before, LoadedAgentLifecycle)
            assert before.value["state"] == "suspended"
            await await_or_fail(activate_agent_children(loop), what="resume saved descriptor")
            assert restarts[0] == (("select", "codex") if phase == "pending" else ("restore", "codex", accounts))
            descriptor = store.agent_children()[run_id]["resume"]
            assert (descriptor["account_pins"], descriptor["selected_accounts"]) == ("selected", accounts)
            if phase == "unavailable":
                blocked, _ = read_child_lifecycle(store, path)
                assert isinstance(blocked, LoadedAgentLifecycle)
                assert blocked.value["state"] == "suspended"
                assert "unavailable" in blocked.value["recovery_blocked"]
                assert not entered.is_set()
                assert not store.agent_notifications(pending_only=False)
            else:
                await await_or_fail(entered.wait(), what="resumed child executes")
                assert restarts[1][:3] == ("execute", 3, 4)
                assert "parent crashed" in restarts[1][3]
        finally:
            await await_or_fail(loop.close(), what="close reconstructed runtime")
            release.set()
