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
                assert loop._background_owner.owns_running(run_id)
                if boundary == "notice":
                    await original(command)
                raise InterruptedError("interrupted before activation")
            return await original(command)

        try:
            with monkeypatch.context() as crash:
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
