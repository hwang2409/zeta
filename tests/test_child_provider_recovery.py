from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.test_completion_owner import _AttemptBackend, _ManualClock, _SequenceAttempt
from zeta.agent import runner
from zeta.core.agent_progress import register_child_run_sink
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType, ToolCall
from zeta.providers.completion import AttemptSignal, TransportActivity
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["event-data", "transient", "committed", "terminal"])
async def test_provider_exhaustion_preserves_only_replay_safe_background_work(tmp_path, monkeypatch, failure):
    clock = _ManualClock()

    async def advance(delay):
        clock.advance(clock.now() + delay)

    def signal(event):
        return AttemptSignal(TransportActivity.MODEL_PROGRESS, event)

    error = signal(StreamEvent(
        StreamEventType.ERROR,
        error=ErrorInfo("server_error" if failure != "event-data" and failure != "terminal" else "upstream", "offline"),
        data={"retryable": True, "retry_after": 1} if failure == "event-data" else {"retry_after": 1},
    ))
    events = [error]
    if failure == "committed":
        events.insert(0, signal(StreamEvent(StreamEventType.MESSAGE_UPDATE, tool_call=ToolCall("write", "write", {}))))
    backend = _AttemptBackend([_SequenceAttempt(*events) for _ in range(5)], clock, wait=advance)
    backend.provider = "codex"
    backend.model = "gpt-5.6-sol"
    # test-contract: inject transport attempts while retaining the real completion and child owners.
    monkeypatch.setattr(runner, "resolve_child_backend", lambda *_a, **_kw: (backend, None))
    reached = asyncio.Event()
    ended = []
    event_loop = asyncio.get_running_loop()

    def observe(kind, run_id, row, _number, _store):
        if kind == "ended":
            ended.append(run_id)
            event_loop.call_soon_threadsafe(reached.set)
        elif kind == "progress" and row["status"] == "suspended":
            event_loop.call_soon_threadsafe(reached.set)

    unregister = register_child_run_sink(observe)
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
        try:
            run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="start child")
            await await_or_fail(reached.wait(), what="child suspends or ends")
            path = Path(store.session_dir) / "agents" / run_id.rsplit(":", 1)[-1]
            state, _ = read_child_lifecycle(store, path)
            assert isinstance(state, LoadedAgentLifecycle)
            if failure in {"committed", "terminal"}:
                assert backend.created == 1
                assert state.value["state"] == "failed"
                assert ended == [run_id]
            else:
                assert backend.created == 5
                assert state.value["state"] == "suspended", "replay-safe provider exhaustion must suspend, not end the child"
                assert state.value["suspension_reason"] == "provider_unavailable"
                assert state.value["turns_used"] == 0
                assert not ended
        finally:
            unregister()
            await await_or_fail(loop.close(), what="close child tree")
