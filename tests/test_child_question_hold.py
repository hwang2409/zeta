from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.agent import runner
from zeta.agent.pending_delivery import send_to_run
from zeta.agent.recovery_lease import activate_agent_children
from zeta.core.agent_progress import register_child_run_sink
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


def _loop(store, monkeypatch, turns, *, limit=None):
    backend = FakeBackend(turns)
    backend.model = "gpt-5.6-sol"
    backend.provider = "codex"
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    loop.explicit_finish = True
    catalog = loop.tool_registry.agent_catalog
    monkeypatch.setattr(loop.tool_registry, "agent_catalog", replace(catalog, agents=tuple(
        replace(preset, explicit_finish=True, max_turns=limit) if preset.name == "run" else preset
        for preset in catalog.agents
    )))
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *a, **kw: (backend, None))
    return loop


def _finish(call_id):
    return ToolCall(call_id, "finish", {"status": "done", "summary": "approved"})


ASK = ToolCall("ask", "ask_parent", {"question": "may i finish?"})


@pytest.fixture
async def observe(monkeypatch):
    from zeta.agent.conversation_channel import ConversationChannel

    reached = asyncio.Event()
    wait = ConversationChannel.wait

    async def waiting(channel, run_id):
        reached.set()
        await wait(channel, run_id)

    # test-contract: observe the resident child reaching the idle wait edge.
    monkeypatch.setattr(ConversationChannel, "wait", waiting)
    ended = []

    def sink(kind, run_id, row, number, store):
        if kind == "ended":
            ended.append(run_id)
            owner.call_soon_threadsafe(reached.set)

    owner = asyncio.get_running_loop()
    unregister = register_child_run_sink(sink)
    try:
        yield reached, ended
    finally:
        unregister()


def _cards(store):
    return [row for row in store.agent_notifications() if row.data.get("kind") == "child_question"]


async def _answer(loop, run_id, card):
    owner = loop._background_owner
    return await send_to_run(loop.store, run_id, f"Re {card.data['question_id']}: approved",
                             owner=owner, channel=owner.conversation_channel)


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True], ids=["late-answer", "last-turn-restart"])
async def test_pending_question_precedes_finish(tmp_path, monkeypatch, restart, observe):
    root = tmp_path / "sessions"
    first = ScriptedTurn(tool_calls=[ASK, _finish("early")])
    idle = ScriptedTurn([TextContent("waiting for the answer")])
    done = ScriptedTurn(tool_calls=[_finish("settled")])
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        loop = _loop(store, monkeypatch, [first] if restart else [first, idle, idle, done],
                     limit=1 if restart else None)
        reached, ended = observe
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="run", task="work", cwd=None),
                                         what="start asking child")
            await await_or_fail(reached.wait(), what="child becomes idle or ends")
            assert not ended, "unanswered question must hold the child, not end it"
            marker = store.agent_children()[run_id]
            path = Path(marker["child_session_path"])
            state, _ = read_child_lifecycle(store, path)
            assert isinstance(state, LoadedAgentLifecycle)
            assert state.value["state"] == "idle", "unanswered question must hold the child, not end it"
            card, = _cards(store)
            assert not ended
            assert not (path / "turn-limit-handoff.md").exists()
            if not restart:
                assert await _answer(loop, run_id, card) is None
                await await_or_fail(loop._background_owner.wait(), what="one terminal answer finish")
                assert len(ended) == 1
                assert not _cards(store)
        finally:
            await await_or_fail(loop.close(), what="close original parent")
    if not restart:
        return
    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store, monkeypatch, [done], limit=1)
        try:
            await await_or_fail(activate_agent_children(loop), what="resume held child")
            assert not loop.backend.calls
            card, = _cards(store)
            assert await _answer(loop, run_id, card) is None
            await await_or_fail(loop._background_owner.wait(), what="settlement ends same run")
            assert len(loop.backend.calls) == len(ended) == 1
            state, _ = read_child_lifecycle(store, path)
            assert isinstance(state, LoadedAgentLifecycle)
            assert state.value["turns_used"] == 2
            assert not _cards(store)
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")


@pytest.mark.asyncio
async def test_restart_repairs_answer_before_provider_replay(tmp_path, monkeypatch, observe):
    root = tmp_path / "sessions"
    idle = ScriptedTurn([TextContent("waiting for the answer")])
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        loop = _loop(store, monkeypatch, [ScriptedTurn(tool_calls=[ASK]), idle, idle])
        reached, _ = observe
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="run", task="work", cwd=None),
                                         what="start asking child")
            await await_or_fail(reached.wait(), what="child waits for an answer")
            card, = _cards(store)
            channel = loop._background_owner.conversation_channel
            route = channel.produce_to_live_parent

            async def interrupt(child_id, build):
                if build(store).kind == "question_answer_ack":
                    raise InterruptedError("ancestor answer commit interrupted")
                return await route(child_id, build)

            # test-contract: cut the answer across two real stores.
            with monkeypatch.context() as cut:
                cut.setattr(channel, "produce_to_live_parent", interrupt)
                with pytest.raises(InterruptedError, match="ancestor answer commit interrupted"):
                    await _answer(loop, run_id, card)
        finally:
            await await_or_fail(loop.close(), what="release original owner")

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store, monkeypatch, [ScriptedTurn(tool_calls=[_finish("settled")])])
        try:
            await await_or_fail(activate_agent_children(loop), what="reconcile answered question on restart")
            assert not _cards(store), "restart must settle the ancestor card before replay"
            await await_or_fail(loop._background_owner.wait(), what="replay accepted answer")
            assert len(loop.backend.calls) == 1
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")
