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


class _RecoveryBackend(_AttemptBackend):
    provider = "codex"
    model = "gpt-5.6-sol"

    def __init__(self, clock):
        super().__init__([], clock)
        self.requests = []
        self.offline = True

    def create_attempt(self, messages, tool_schemas, token=None):
        self.requests.append(messages)
        if self.offline:
            event = StreamEvent(StreamEventType.ERROR, error=ErrorInfo("server_error", "offline"), data={"retry_after": 1})
            return _SequenceAttempt(AttemptSignal(TransportActivity.MODEL_PROGRESS, event))
        from zeta.protocol.types import Message, MessageRole, TextContent

        message = Message(MessageRole.ASSISTANT, [TextContent("done")])
        return _SequenceAttempt(AttemptSignal(TransportActivity.MODEL_PROGRESS, StreamEvent(StreamEventType.MESSAGE_END, message=message)))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["false-positive", "send", "cancel", "cancel-suspension", "cancel-activation", "restart"])
async def test_retained_provider_episode_controls_replay(tmp_path, monkeypatch, scenario):
    from datetime import UTC, datetime, timedelta
    from io import StringIO

    from rich.console import Console

    from zeta.agent.pending_delivery import send_to_run
    from zeta.agent.recovery_lease import activate_agent_children
    from zeta.core.store._async_writes import (
        _AppendChildRecoveredCommand,
        _ChildSuspensionCommand,
    )
    from zeta.tui.render import render_event

    clock = _ManualClock()
    epoch = datetime(2026, 11, 27, tzinfo=UTC)
    backend = _RecoveryBackend(clock)
    monkeypatch.setattr(runner, "resolve_child_backend", lambda *_a, **_kw: (backend, None))
    monkeypatch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *_a, **_kw: (backend, None))
    notices = asyncio.Queue()
    ended = asyncio.Queue()
    event_loop = asyncio.get_running_loop()

    def observe(kind, run_id, row, revision, child_store):
        if kind == "ended":
            event_loop.call_soon_threadsafe(ended.put_nowait, (run_id, row))
            event_loop.call_soon_threadsafe(notices.put_nowait, ("ended", row))

    commit_entered = asyncio.Event()
    commit_release = asyncio.Event()
    probe_entered = asyncio.Event()
    probe_release = asyncio.Event()
    unregister = register_child_run_sink(observe)
    commit = ConversationStore._commit_storage_command

    async def publish_commit(store, command):
        if isinstance(command, _ChildSuspensionCommand) and (
            scenario == "cancel-suspension" and command.provider_retry is not None
            or scenario == "cancel-activation" and command.activate is not None
        ):
            commit_entered.set()
            if scenario == "cancel-suspension":
                notices.put_nowait(("suspending", None))
            await await_or_fail(commit_release.wait(), what="release the child lifecycle write")
        result = await commit(store, command)
        if isinstance(command, _AppendChildRecoveredCommand) and getattr(command, "suspension", None) is not None:
            notices.put_nowait(("suspended", result[0].data))
        return result

    # test-contract: seam gate the durable write to control cancellation order.
    monkeypatch.setattr(ConversationStore, "_commit_storage_command", publish_commit)

    async def suspended():
        kind, row = await await_or_fail(notices.get(), what="durable suspend notice or terminal receipt")
        assert kind == "suspended", "provider exhaustion must retain a sendable, cancellable live run"
        return row

    def lifecycle(store, run_id):
        path = store.session_dir / "agents" / run_id.rsplit(":", 1)[-1]
        loaded, _ = read_child_lifecycle(store, path)
        assert isinstance(loaded, LoadedAgentLifecycle)
        return loaded.value

    root = tmp_path / "sessions"
    try:
        with ConversationStore(root, cwd=tmp_path) as store:
            session_id = store.session_id
            loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
            tree = loop._background_owner
            if (recovery := getattr(tree, "provider_recovery", None)) is not None:
                from zeta.providers.recovery_probe import Readiness

                async def probe(_backend):
                    probe_entered.set()
                    await await_or_fail(probe_release.wait(), what="release the readiness probe")
                    return Readiness.READY

                # test-contract: control time and network only, not the transition owner.
                recovery.clock = clock
                recovery.wall = lambda: epoch + timedelta(seconds=clock.now())
                recovery._probe = probe
            try:
                run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn retained child")
                if scenario == "cancel-suspension":
                    kind, _ = await await_or_fail(notices.get(), what="suspension or terminal write")
                    assert kind == "suspending", "cancel must be possible during a nonterminal suspension write"
                    transition = tree.run_transition(run_id)
                    assert transition is not None
                    request = transition.request_cancel()
                    commit_release.set()
                    assert await await_or_fail(request.confirmation, what="cancel during suspension")
                    _, row = await await_or_fail(ended.get(), what="cancel writes one terminal receipt")
                    assert row["status"] == "canceled"
                    assert len(backend.requests) == 5
                    assert ended.empty()
                    return
                notice = await suspended()
                state = lifecycle(store, run_id)
                assert state["turns_used"] == 0
                assert state["next_retry_at"] == (epoch + timedelta(seconds=5)).isoformat()
                deadline = state["retry_deadline_at"]
                output = StringIO()
                Console(file=output, width=120).print(render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=notice)))
                assert "suspended: provider unavailable, retrying" in output.getvalue()
                assert state["next_retry_at"] in output.getvalue()
                if scenario != "restart":
                    clock.advance(5)
                    await await_or_fail(probe_entered.wait(), what="one account probe begins")
                    transition = tree.run_transition(run_id)
                    assert transition is not None
                    if scenario == "false-positive":
                        probe_release.set()
                        await suspended()
                        second = lifecycle(store, run_id)
                        assert second["retry_deadline_at"] == deadline
                        assert second["next_retry_at"] == (epoch + timedelta(seconds=15)).isoformat()
                        assert second["suspension_episode"] == state["suspension_episode"]
                        assert second["turns_used"] == 0
                        assert len(backend.requests) == 10
                        assert backend.requests[0] == backend.requests[5]
                        assert len([row for row in store.agent_notifications(pending_only=False) if row.data.get("kind") == "child_suspended"]) == 1
                    else:
                        if scenario == "send":
                            backend.offline = False
                            assert await send_to_run(store, run_id, "queued", owner=tree, channel=tree.conversation_channel) is None
                            probe_release.set()
                        else:
                            if scenario == "cancel-activation":
                                probe_release.set()
                                await await_or_fail(commit_entered.wait(), what="enter durable activation")
                            request = transition.request_cancel()
                            commit_release.set()
                            probe_release.set()
                            assert await await_or_fail(request.confirmation, what="cancel wins")
                        _, terminal = await await_or_fail(ended.get(), what="one terminal child fact")
                        assert terminal["status"] == ("completed" if scenario == "send" else "canceled")
                        assert len(backend.requests) == (7 if scenario == "send" else 5)
                        if scenario == "send":
                            assert backend.requests[0] == backend.requests[5]
                            assert terminal["turns_used"] == (2 if scenario == "send" else 1)
                        assert ended.empty()
            finally:
                commit_release.set()
                probe_release.set()
                await await_or_fail(loop.close(), what="close suspended or resumed tree")
        if scenario == "restart":
            backend.offline = False
            with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
                loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
                try:
                    assert lifecycle(store, run_id)["retry_deadline_at"] == deadline
                    await await_or_fail(activate_agent_children(loop), what="explicit parent resume")
                    _, terminal = await await_or_fail(ended.get(), what="restored child completes")
                    assert terminal["turns_used"] == (2 if scenario == "send" else 1)
                    assert lifecycle(store, run_id)["suspension_episode"] == state["suspension_episode"]
                    assert "retry_deadline_at" not in lifecycle(store, run_id)
                finally:
                    await await_or_fail(loop.close(), what="close restored tree")
    finally:
        unregister()


@pytest.mark.asyncio
async def test_provider_probe_fans_out_per_account_then_stops_at_original_deadline(tmp_path):
    from datetime import UTC, datetime, timedelta

    clock = _ManualClock()
    wall = datetime(2026, 11, 27, tzinfo=UTC)
    actions = asyncio.Queue()
    probe_ticks = []
    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(_RecoveryBackend(clock), store, skill_catalog=SkillCatalog.empty())
        try:
            recovery = getattr(loop._background_owner, "provider_recovery", None)
            assert recovery is not None, "a live tree must share one account probe schedule"
            from zeta.providers.recovery_probe import Readiness

            async def probe(account):
                probe_ticks.append((account, clock.now()))
                return Readiness.READY if account == "second" else Readiness.RETRY

            recovery.clock = clock
            recovery.wall = lambda: wall + timedelta(seconds=clock.now())
            recovery._probe = probe
            deadline = (wall + timedelta(hours=1)).isoformat()
            for account in ("first", "second"):
                for i in range(20):
                    run = f"{account}:{i}"

                    async def activate(schedule, run=run):
                        actions.put_nowait((run, "activate", schedule.probe_at))

                    async def update(schedule, stopped, run=run):
                        actions.put_nowait((run, "stop" if stopped else "wait", schedule.probe_at))

                    recovery.subscribe(account, run, account, deadline=deadline,
                                       schedule=recovery.schedule(account, None), activate=activate, update=update)
            t = 0
            delays = [5, 10, 20, 40, 80, 160, *([300] * 10)]
            for step, delay in enumerate(delays):
                t += delay
                clock.advance(t)
                changed = [await await_or_fail(actions.get(), what="account tick notification") for _ in range(40 if step == 0 else 20)]
                assert probe_ticks.count(("first", t)) == 1
                assert probe_ticks.count(("second", t)) == (1 if step == 0 else 0)
                assert len({row[0] for row in changed}) == len(changed)
                assert sum(row[1] == "activate" for row in changed) == (20 if step == 0 else 0)
            clock.advance(3600)
            stopped = [await await_or_fail(actions.get(), what="account retry cap") for _ in range(20)]
            assert len(probe_ticks) == len(delays) + 1
            assert all(row[1] == "stop" for row in stopped)
            for account in ("first", "second"):
                for i in range(20):
                    recovery.replay_progress(account, f"{account}:{i}")
                assert recovery.schedule(account, None).step == 0, "zero subscribers must delete the prior schedule"
            assert actions.empty()
        finally:
            await await_or_fail(loop.close(), what="close tree and coordinator")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["claude", "codex", "ollama"])
async def test_provider_readiness_uses_context_free_generation(tmp_path, provider):
    import json
    from datetime import UTC, datetime, timedelta

    import httpx

    from tests.test_codex import message_stream, sse, store_for
    from zeta.providers import recovery_probe
    from zeta.providers.anthropic import AnthropicApiKeyCredential, AnthropicBackend
    from zeta.providers.codex import CodexBackend
    from zeta.providers.ollama import OllamaBackend

    requests = []
    exhausted = True

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            reset = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
            return httpx.Response(200, json={"rate_limit": {"primary_window": {
                "used_percent": 100 if exhausted else 0, "limit_window_seconds": 3600,
                "reset_after_seconds": 3600, "reset_at": reset,
            }}})
        if provider == "codex":
            return httpx.Response(200, text=sse(message_stream()))
        if provider == "claude":
            return httpx.Response(200, text=(
                'data: {"type":"message_start","message":{"id":"probe","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n\n'
            ))
        return httpx.Response(200, text='{"message":{"role":"assistant","content":"x"},"done":false}\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if provider == "codex":
            backend = CodexBackend(client=client, token_store=store_for(tmp_path / "codex.json"))
        elif provider == "claude":
            backend = AnthropicBackend(client=client, token_store=AnthropicApiKeyCredential("fixture"))
        else:
            backend = OllamaBackend(client=client)
        try:
            if provider == "codex":
                assert await recovery_probe.probe_readiness(backend) is recovery_probe.Readiness.RETRY
                assert [request.method for request in requests] == ["GET"]
                exhausted = False
            assert await recovery_probe.probe_readiness(backend) is recovery_probe.Readiness.READY
            assert [request.method for request in requests] == (["GET", "GET", "POST"] if provider == "codex" else ["POST"])
            payload = json.loads(requests[-1].content)
            assert payload["model"] == backend.model
            assert not payload.get("tools") and not payload.get("system")
            if provider == "codex":
                assert payload["instructions"] == "Reply with one character."
                assert [part["text"] for entry in payload["input"] for part in entry["content"]] == ["x"]
                assert "max_output_tokens" not in payload
            else:
                assert payload["messages"] == [{"role": "user", "content": "x"}]
                assert (payload["max_tokens"] if provider == "claude" else payload["options"]["num_predict"]) == 1
        finally:
            await backend.aclose()
