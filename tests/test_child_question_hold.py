from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.agent import run_consumer, runner
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
@pytest.mark.parametrize("mode", ["late-answer", "last-turn-restart", "charged-restart"])
async def test_pending_question_precedes_finish(tmp_path, monkeypatch, mode, observe):
    restart = mode != "late-answer"
    root = tmp_path / "sessions"
    first = ScriptedTurn(tool_calls=[ASK, _finish("early")])
    done = ScriptedTurn(tool_calls=[_finish("settled")])
    with ConversationStore(root, cwd=tmp_path) as store:
        session_id = store.session_id
        loop = _loop(store, monkeypatch, [first] if restart else [first, done],
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
                assert len(loop.backend.calls) == 1, "refused finish must not dispatch again before the answer"
                assert await _answer(loop, run_id, card) is None
                await await_or_fail(loop._background_owner.wait(), what="one terminal answer finish")
                assert len(loop.backend.calls) == 2
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
            if mode == "charged-restart":
                charged = asyncio.Event()

                async def stop_before_dispatch(*args):
                    charged.set()
                    await await_or_fail(asyncio.Event().wait(), what="crash before settlement dispatch")

                # test-contract: stop after the durable charge but before provider dispatch.
                with monkeypatch.context() as cut:
                    # test-contract: durability interrupt the charged attempt before dispatch.
                    cut.setattr(run_consumer, "_transition_running", stop_before_dispatch)
                    # test-contract: seam send through the resident child delivery flow.
                    assert await _answer(loop, run_id, card) is None
                    await await_or_fail(charged.wait(), what="charge before dispatch")
                    await await_or_fail(loop.close(), what="release charged child without dispatch")
                assert not loop.backend.calls
            else:
                assert await _answer(loop, run_id, card) is None
                await await_or_fail(loop._background_owner.wait(), what="settlement ends same run")
                assert len(loop.backend.calls) == len(ended) == 1
                state, _ = read_child_lifecycle(store, path)
                assert isinstance(state, LoadedAgentLifecycle)
                assert state.value["turns_used"] == 2
                assert not _cards(store)
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")
    if mode == "charged-restart":
        with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
            loop = _loop(store, monkeypatch, [done], limit=1)
            try:
                await await_or_fail(activate_agent_children(loop), what="recover charged settlement")
                await await_or_fail(loop._background_owner.wait(), what="exhausted handoff without replay")
                assert not loop.backend.calls, "a charged settlement must not dispatch again"
                assert len(ended) == 1
                state, _ = read_child_lifecycle(store, path)
                assert isinstance(state, LoadedAgentLifecycle)
                assert state.value["turns_used"] == 2
                assert (path / "turn-limit-handoff.md").is_file()
                assert not _cards(store)
            finally:
                await await_or_fail(loop.close(), what="close parent after exhausted handoff")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["answer-restart", "answer-resident", "withdraw-restart"])
async def test_reconciles_question_before_replay(tmp_path, monkeypatch, observe, mode):
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
                if build(store).kind == ("child_question_close" if mode == "withdraw-restart" else "question_answer_ack"):
                    raise InterruptedError("ancestor answer commit interrupted")
                return await route(child_id, build)

            # test-contract: cut the answer across two real stores.
            with monkeypatch.context() as cut:
                cut.setattr(channel, "produce_to_live_parent", interrupt)
                if mode == "withdraw-restart":
                    channel_owner = loop._background_owner
                    transition = getattr(channel_owner, "run_transition", lambda _: None)(run_id)
                    if transition is not None:
                        with pytest.raises(InterruptedError, match="ancestor answer commit interrupted"):
                            await transition.withdraw_questions()
                else:
                    with pytest.raises(InterruptedError, match="ancestor answer commit interrupted"):
                        await _answer(loop, run_id, card)
            if mode == "answer-resident":
                transition = getattr(loop._background_owner, "run_transition", lambda _: None)(run_id)
                if transition is not None:
                    await transition.repair_before_replay()
                assert not _cards(store), "resident repair must remove the stale ancestor card"
                rows = store.active_branch_head_sequence()
                await transition.repair_before_replay()
                assert store.active_branch_head_sequence() == rows, "resident repair must not append a second ack"
                return
        finally:
            await await_or_fail(loop.close(), what="release original owner")

    with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
        loop = _loop(store, monkeypatch, [ScriptedTurn(tool_calls=[_finish("settled")])])
        try:
            await await_or_fail(activate_agent_children(loop), what="reconcile answered question on restart")
            assert not _cards(store), "restart must settle the ancestor card before replay"
            if mode == "withdraw-restart":
                assert not loop.backend.calls
                assert any(row.data.get("kind") == "child_question_withdrawn" for row in store.agent_notifications())
                return
            await await_or_fail(loop._background_owner.wait(), what="replay accepted answer")
            assert len(loop.backend.calls) == 1
        finally:
            await await_or_fail(loop.close(), what="close resumed parent")


def _tui_text(event):
    from io import StringIO

    from rich.console import Console

    from zeta.tui.render import render_event

    output = StringIO()
    Console(file=output, width=100, color_system=None).print(render_event(event))
    return output.getvalue()


def _serve_notices(store):
    from zeta.server.handlers.history_projection import _history_entry_rows

    return [row["notification"] for entry in store.agent_notifications(pending_only=False)
            for row in _history_entry_rows(entry, {})]


@pytest.mark.asyncio
async def test_cancel_closes_acknowledged_question(tmp_path, monkeypatch, observe):
    from zeta.protocol.types import StreamEvent, StreamEventType

    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = _loop(store, monkeypatch, [ScriptedTurn(tool_calls=[ASK, _finish("early")])])
        reached, _ = observe
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="run", task="work", cwd=None),
                                         what="start asking child")
            await await_or_fail(reached.wait(), what="hold for answer")
            card, = _cards(store)
            assert "pending" in _tui_text(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=card.data))
            store.acknowledge_agent_notification(card.id)
            canceled = await loop.tool_registry.execute(ToolCall("cancel", "agent_cancel", {"handle": run_id}))
            assert not canceled.get("isError"), canceled
            await await_or_fail(loop._background_owner.wait(), what="finish cancellation")
            assert store.agent_completion_notifications_by_child()[run_id].data["status"] == "canceled"
            close_cards = [row for row in store.agent_notifications(pending_only=False)
                           if row.data.get("kind") == "child_question_withdrawn"]
            assert len(close_cards) == 1, "cancel must close the acknowledged question card once"
            close, = close_cards
            assert close.data["question_id"] == card.data["question_id"]
            assert "withdrawn" in _tui_text(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=close.data))
            assert [row["status"] for row in _serve_notices(store)
                    if row.get("kind") == "child_question_withdrawn"] == ["withdrawn"]
        finally:
            await await_or_fail(loop.close(), what="close parent")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["separate", "queued-together", "at-limit"])
async def test_held_child_delivers_follow_ups_and_answer(tmp_path, monkeypatch, observe, mode):
    first = ScriptedTurn(tool_calls=[ASK, _finish("early")])
    follow_up = ScriptedTurn(tool_calls=[_finish("still-early")])
    done = ScriptedTurn(tool_calls=[_finish("settled")])
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        limit = mode == "at-limit"
        loop = _loop(store, monkeypatch, [first, done] if limit else [first, follow_up, done],
                     limit=1 if limit else None)
        reached, ended = observe
        release = asyncio.Event()
        checkpoint = asyncio.Event()
        channel = loop._background_owner.conversation_channel
        if mode != "separate":
            wait = channel.wait

            async def hold_wake(run_id):
                await wait(run_id)
                await await_or_fail(release.wait(), what="queue both follow-ups before delivering either")

            # test-contract: force both real deliveries to be queued before the child wakes.
            monkeypatch.setattr(channel, "wait", hold_wake)
        if mode == "queued-together":
            consume = runner.consume_child

            async def observe_delivery(*args, **kwargs):
                result = await consume(*args, **kwargs)
                if kwargs.get("notification_turn"):
                    checkpoint.set()
                return result

            # test-contract: observe whether the real follow-up turn reached the model.
            monkeypatch.setattr(runner, "consume_child", observe_delivery)
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="run", task="work", cwd=None),
                                         what="start asking child")
            await await_or_fail(reached.wait(), what="hold for answer")
            card, = _cards(store)
            owner = loop._background_owner
            reached.clear()
            assert await send_to_run(store, run_id, "plain follow-up", owner=owner,
                                     channel=owner.conversation_channel) is None
            if mode == "separate":
                await await_or_fail(reached.wait(), what="return to hold after follow-up")
                assert len(loop.backend.calls) == 2, "plain delivery must wake a child held for an answer"
                assert not ended and _cards(store) == [card]
            assert await _answer(loop, run_id, card) is None
            release.set()
            if mode == "queued-together":
                await await_or_fail(checkpoint.wait(), what="process queued follow-ups")
                assert len(loop.backend.calls) >= 2, "a second queued prompt must not defer the first provider call"
            await await_or_fail(owner.wait(), what="finish after answer")
            assert len(loop.backend.calls) == (2 if limit else 3)
            assert b"plain follow-up" in loop.backend.request_bytes[1], "queued follow-up must reach the model"
            assert loop.backend.request_bytes[-1].count(f"Re {card.data['question_id']}: approved".encode()) == 1
            assert ended == [run_id]
            receipts = [row for row in store.agent_notifications(pending_only=False)
                        if row.data.get("kind") == "agent_completion"]
            assert [row.data["status"] for row in receipts] == ["completed"]
        finally:
            release.set()
            await await_or_fail(loop.close(), what="close parent")


@pytest.mark.asyncio
async def test_last_turn_finish_card_matches_durable_status(tmp_path, monkeypatch, observe):
    from zeta.agent.turn_limit import project_notification_data
    from zeta.protocol.types import StreamEvent, StreamEventType
    from zeta.server.event_contract import tool_end

    first = ScriptedTurn(tool_calls=[ASK, _finish("early")])
    done = ScriptedTurn(tool_calls=[_finish("settled")])
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = _loop(store, monkeypatch, [first, done], limit=2)
        events = []
        loop.set_background_event_sink(events.append)
        reached, _ = observe
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="run", task="work", cwd=None),
                                         what="start asking child")
            await await_or_fail(reached.wait(), what="hold for answer")
            card, = _cards(store)
            assert await _answer(loop, run_id, card) is None
            await await_or_fail(loop._background_owner.wait(), what="finish on last allowed turn")
            receipt = store.agent_completion_notifications_by_child()[run_id]
            assert receipt.data["status"] == "completed"
            terminal, = [event for event in events if event.type is StreamEventType.TOOL_EXECUTION_END
                         and event.data.get("notification_id") == receipt.id]
            rendered = _tui_text(terminal)
            assert "completed" in rendered and "turn_limit" not in rendered, "TUI finish card must match durable completion"
            assert tool_end(terminal)["tool_result"]["structured_content"].get("status") != "turn_limit", (
                "serve tool_end must not project a turn limit over accepted finish"
            )
            notice = StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=project_notification_data(receipt.data))
            rendered = _tui_text(notice)
            assert "completed" in rendered and "turn_limit" not in rendered
            assert [row["status"] for row in _serve_notices(store)
                    if row.get("kind") == "agent_completion"] == ["completed"]
        finally:
            await await_or_fail(loop.close(), what="close parent")
