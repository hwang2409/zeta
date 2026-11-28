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
@pytest.mark.parametrize("scenario", ["false-positive", "send", "restart", "shutdown-suspension", "activation-crash"])
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

    commit_release = asyncio.Event()
    quiesced = asyncio.Event()
    close_release = asyncio.Event()
    probe_entered = asyncio.Event()
    probe_release = asyncio.Event()
    unregister = register_child_run_sink(observe)
    commit = ConversationStore._commit_storage_command

    crash_before_notice = scenario == "activation-crash"

    async def publish_commit(store, command):
        nonlocal crash_before_notice
        if crash_before_notice and isinstance(command, _AppendChildRecoveredCommand) and command.suspension is None:
            crash_before_notice = False
            assert lifecycle(store, command.run_id)["state"] == "running", "crash gap starts after activation commits"
            raise OSError("injected loss before notice append")
        if scenario == "shutdown-suspension" and isinstance(command, _ChildSuspensionCommand) and command.provider_retry is not None:
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
                if scenario == "shutdown-suspension":
                    close_recovery = recovery.close

                    async def gate_close():
                        quiesced.set()
                        await await_or_fail(close_release.wait(), what="release coordinator shutdown")
                        await close_recovery()

                    monkeypatch.setattr(recovery, "close", gate_close)
            try:
                run_id = await await_or_fail(runner.spawn_child(loop, agent_type="general", task="work", cwd=None), what="spawn retained child")
                if scenario == "shutdown-suspension":
                    kind, _ = await await_or_fail(notices.get(), what="suspension write or terminal receipt")
                    assert kind == "suspending", "shutdown must be possible during suspension"
                    closing = asyncio.create_task(loop.close())
                    try:
                        await await_or_fail(quiesced.wait(), what="tree closed admission")
                        commit_release.set()
                        await await_or_fail(tree.wait(), what="child settles while coordinator is closing")
                        assert ended.empty(), "normal shutdown must not publish a terminal receipt"
                        state = lifecycle(store, run_id)
                        assert state["state"] == "suspended"
                        deadline = state["retry_deadline_at"]
                    finally:
                        commit_release.set()
                        close_release.set()
                        await await_or_fail(closing, what="finish closing tree")
                else:
                    notice = await suspended()
                    state = lifecycle(store, run_id)
                    assert state["turns_used"] == 0
                    assert state["next_retry_at"] == (epoch + timedelta(seconds=5)).isoformat()
                    deadline = state["retry_deadline_at"]
                    output = StringIO()
                    Console(file=output, width=120).print(render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=notice)))
                    assert "suspended: provider unavailable, retrying" in output.getvalue()
                    assert state["next_retry_at"] in output.getvalue()
                    if scenario == "activation-crash":
                        with pytest.raises(OSError, match="injected loss before notice append"):
                            await await_or_fail(send_to_run(store, run_id, "queued", owner=tree, channel=tree.conversation_channel), what="interrupt after activation commit")
                        assert not any(row.data.get("kind") == "child_recovered" for row in store.agent_notifications(pending_only=False))
                    elif scenario != "restart":
                        clock.advance(5)
                        await await_or_fail(probe_entered.wait(), what="one account probe begins")
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
                            backend.offline = False
                            assert await send_to_run(store, run_id, "queued", owner=tree, channel=tree.conversation_channel) is None
                            probe_release.set()
                            _, terminal = await await_or_fail(ended.get(), what="one terminal child fact")
                            assert terminal["status"] == "completed"
                            assert terminal["turns_used"] == 2
                            assert len(backend.requests) == 7
                            assert backend.requests[0] == backend.requests[5]
                            assert ended.empty()
            finally:
                commit_release.set()
                probe_release.set()
                await await_or_fail(loop.close(), what="close suspended or resumed tree")
        if scenario == "activation-crash":
            with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
                loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
                try:
                    # The missing notice must be repaired even if this resume cannot activate.
                    with monkeypatch.context() as patch:
                        patch.setattr("zeta.agent.resume_child.resolve_child_backend", lambda *_a, **_kw: (None, "offline"))
                        await await_or_fail(activate_agent_children(loop), what="repair notice with no backend")
                    assert lifecycle(store, run_id)["state"] == "suspended"
                    repaired = [row for row in store.agent_notifications(pending_only=False) if row.data.get("kind") == "child_recovered"]
                    assert [row.data["episode"] for row in repaired] == [state["suspension_episode"]]
                finally:
                    await await_or_fail(loop.close(), what="close blocked recovery")
        if scenario in {"restart", "shutdown-suspension", "activation-crash"}:
            backend.offline = False
            with ConversationStore(root, session_id=session_id, cwd=tmp_path) as store:
                loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
                try:
                    assert lifecycle(store, run_id)["retry_deadline_at"] == deadline
                    await await_or_fail(activate_agent_children(loop), what="explicit parent resume")
                    _, terminal = await await_or_fail(ended.get(), what="restored child completes")
                    assert terminal["turns_used"] == 1
                    if scenario == "activation-crash":
                        recovered = [row for row in store.agent_notifications(pending_only=False) if row.data.get("kind") == "child_recovered"]
                        assert [row.id for row in recovered] == [repaired[0].id]
                    assert lifecycle(store, run_id)["suspension_episode"] == state["suspension_episode"]
                    assert "retry_deadline_at" not in lifecycle(store, run_id)
                finally:
                    await await_or_fail(loop.close(), what="close restored tree")
    finally:
        unregister()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,write", [
    ("running", None),
    ("provider_suspending", "suspend-state"),
    ("provider_suspending", "suspend-notice"),
    ("provider_suspended", None),
    ("activating", "activate-notice"),
    ("activating", "activate-state"),
])
@pytest.mark.parametrize("event", ["send", "cancel", "shutdown", "probe_success", "probe_failure", "deadline", "finish"])
async def test_provider_owner_transition_table(tmp_path, monkeypatch, phase, write, event):
    from datetime import UTC, datetime

    from zeta.agent.background_lifecycle import finish_background_child
    from zeta.agent.child_run_owner import ChildRunTransitionOwner
    from zeta.agent.pending_delivery import StepDeliveryFlow
    from zeta.core.store._async_writes import (
        _AppendChildRecoveredCommand,
        _ChildSuspensionCommand,
    )
    from zeta.protocol.types import ToolResult
    from zeta.providers.recovery_schedule import RecoverySchedule

    backend = _RecoveryBackend(_ManualClock())
    entered, release = asyncio.Event(), asyncio.Event()
    subscribed, replayed, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    invoked, finish_reached = asyncio.Event(), asyncio.Event()
    enqueued, activations, receipts = [], [], []
    tasks = []

    def start(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    async def noop(*_args):
        return ()

    def result(text, error, _status):
        return {"content": [{"text": text}], "isError": error}

    async def execute():
        await await_or_fail(finish.wait(), what="execution completes")
        return result("done", False, "completed")

    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as parent:
        loop = AgentLoop(backend, parent, skill_catalog=SkillCatalog.empty())
        tree = loop._background_owner
        recovery = tree.provider_recovery
        recovery.clock = _ManualClock()
        now = datetime(2026, 11, 27, tzinfo=UTC)
        recovery.wall = lambda: now
        schedule = RecoverySchedule("2026-11-27T00:00:15+00:00", 1)
        retry = RecoverySchedule("2026-11-27T00:00:35+00:00", 2)

        def subscribe(*_args, **_kwargs):
            subscribed.set()

        # test-contract: stop the external scheduler; deliver probe events through the owner.
        monkeypatch.setattr(recovery, "subscribe", subscribe)
        with ConversationStore(parent.session_dir / "agents", session_id="1", cwd=tmp_path) as child:
            run_id = f"{parent.session_id}:1"
            flow = StepDeliveryFlow(child)
            tree.conversation_channel.register_delivery_flow(run_id, flow)
            owner = ChildRunTransitionOwner(child, "tool", background=True)
            owner.run_id = run_id
            owner.bind_provider_recovery(loop, backend, "work")
            owner.attach_setup(tree, start(noop()), noop)
            execution = start(execute())
            owner.attach_runtime(execution, execution.cancel, noop)
            commit = ConversationStore._commit_storage_command

            async def gate(store, command):
                edge = None
                if isinstance(command, _AppendChildRecoveredCommand):
                    edge = "suspend-notice" if command.suspension is not None else "activate-notice"
                elif isinstance(command, _ChildSuspensionCommand):
                    if command.activate is not None:
                        edge = "activate-state"
                    elif command.provider_retry is not None:
                        edge = "suspend-state"
                if edge is not None and edge == write and not entered.is_set():
                    entered.set()
                    await await_or_fail(release.wait(), what=f"release {edge}")
                if edge == "activate-notice":
                    state = child.agent_lifecycle()
                    assert state is not None and state["state"] == "running", "recovery notice must follow durable activation"
                value = await commit(store, command)
                if edge == "activate-state" and value is not None and value["state"] == "running":
                    activations.append(value)
                return value

            # test-contract: seam gate real lifecycle and notice writes on both sides of each handoff.
            monkeypatch.setattr(ConversationStore, "_commit_storage_command", gate)

            async def suspend_for_provider():
                await owner.provider_suspend("server_error", 0)
                replayed.set()

            async def enqueue():
                enqueued.append("message")

            async def dispatch():
                invoked.set()
                if event == "send":
                    await owner.send(enqueue)
                elif event == "cancel":
                    request = owner.request_cancel()
                    assert await await_or_fail(request.confirmation, what="cancel wins")
                elif event == "shutdown":
                    owner.quiesce()
                    if phase == "provider_suspending":
                        # Let the child observe admission closed before suspend can acquire the lock.
                        await await_or_fail(asyncio.gather(waiting, return_exceptions=True), what="suspension exits")
                        execution.cancel()
                        await await_or_fail(finalizer, what="shutdown settles before suspension join")
                    await owner.suspend()
                elif event == "finish":
                    finish.set()
                    await await_or_fail(asyncio.shield(execution), what="execution reaches terminal finish")
                    finish_reached.set()
                    await await_or_fail(finalizer, what="terminal finish joins the current write")
                else:
                    await owner.provider_probe_result(1, event, retry)

            try:
                await owner.start(lambda: child.start_agent_lifecycle(
                    handle=run_id, started_at=now.isoformat(), depth=1, agent_type="general", description="work",
                ))
                finalizer = start(finish_background_child(
                    child_task=execution, child_store=child, parent_store=parent, tool_call=ToolCall("tool", "agent", {}),
                    child_instance_id=run_id, child_path=str(child.session_dir), description="work", child_turns=lambda: 0,
                    build_result=result, validate_result=lambda row, key: ToolResult(key, row["content"][0]["text"], row["isError"]),
                    publish_event=receipts.append, cleanup=noop, close_child=noop, error_message=str, background_owner=tree,
                    run_owner=owner,
                ))
                waiting = start(suspend_for_provider()) if phase != "running" else None
                if phase in {"provider_suspended", "activating"}:
                    await await_or_fail(subscribed.wait(), what="durable suspension subscribed")
                activating = start(owner.provider_probe_result(1, "probe_success", schedule)) if phase == "activating" else None
                if write is not None:
                    await await_or_fail(entered.wait(), what=f"enter {write}")
                dispatched = start(dispatch())
                await await_or_fail(invoked.wait(), what=f"admit {event}")
                if event == "finish":
                    await await_or_fail(finish_reached.wait(), what="finish reaches the gated phase")
                release.set()
                await await_or_fail(dispatched, what=f"settle {event}")
                if activating is not None:
                    await await_or_fail(activating, what="join initial activation")

                if event not in {"shutdown", "cancel", "finish"}:
                    should_replay = (
                        phase == "activating" or phase == "provider_suspended" and event in {"send", "probe_success"}
                        or phase == "provider_suspending" and event == "send"
                    )
                    if should_replay:
                        await await_or_fail(replayed.wait(), what="activation wakes the child")
                        assert len(activations) == 1
                        assert activations[0]["current_step"] == "recovering interrupted work"
                    else:
                        assert not replayed.is_set()
                        assert not activations
                    assert bool(enqueued) == (event == "send")
                    state = child.agent_lifecycle()
                    assert state is not None
                    if phase == "provider_suspended" and event == "deadline":
                        assert state["retry_state"] == "stopped"
                        assert "next_retry_at" not in state
                    elif phase == "provider_suspended" and event == "probe_failure":
                        assert state["next_retry_at"] == retry.probe_at
                    if phase == "provider_suspending" and event == "send":
                        assert not subscribed.is_set()
                    assert not receipts
                    finish.set()

                await await_or_fail(finalizer, what="join terminal publication or restart preservation")
                state = child.agent_lifecycle()
                assert state is not None
                committed = len(activations)
                recovered = [row for row in parent.agent_notifications(pending_only=False) if row.data.get("kind") == "child_recovered"]
                assert len(recovered) == committed <= 1
                if phase == "activating" and event in {"shutdown", "cancel", "finish"}:
                    assert committed == (write == "activate-notice")
                if event == "shutdown":
                    assert state["state"] == "suspended"
                    assert state.get("finished_at") is None
                    assert not receipts, "normal shutdown must not publish a terminal receipt"
                    assert not replayed.is_set()
                else:
                    assert state["state"] == ("canceled" if event == "cancel" else "completed")
                    assert "retrying" not in state["current_step"]
                    assert len(receipts) == 1
                    assert not await owner.cancel(), "a second terminal transition must lose"
                    if event in {"cancel", "finish"}:
                        assert not replayed.is_set()
            finally:
                release.set()
                finish.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await await_or_fail(asyncio.gather(*tasks, return_exceptions=True), what="join matrix tasks")
                await tree.conversation_channel.unregister_delivery_flow(run_id)
                await await_or_fail(loop.close(), what="close parent")


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
    from zeta.accounts.usage import read_usage
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
                credentials = backend.token_store
                assert read_usage(credentials.store, credentials.account).known_exhausted()
                exhausted = False
            assert await recovery_probe.probe_readiness(backend) is recovery_probe.Readiness.READY
            assert [request.method for request in requests] == (["GET", "GET", "POST"] if provider == "codex" else ["POST"])
            if provider == "codex":
                assert not read_usage(credentials.store, credentials.account).known_exhausted()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture",
    [
        "anthropic_messages_429_rate_limit_error.json",
        "anthropic_messages_400_extra_usage.json",
        "codex_responses_429_usage_limit_reached.json",
    ],
)
async def test_provider_readiness_keeps_waiting_on_an_exhausted_account(tmp_path, fixture):
    import json

    import httpx

    from tests.support.credentials import named_account_credentials
    from tests.test_codex import store_for
    from zeta.oauth import OAuthTokens
    from zeta.providers import recovery_probe
    from zeta.providers.anthropic import AnthropicBackend
    from zeta.providers.codex import CodexBackend

    payload = json.loads((Path(__file__).parent / "fixtures/usage" / fixture).read_text())
    meta = payload.pop("_fixture")
    requests = []

    def handler(request):
        requests.append(request.method)
        if request.method == "GET":
            # the usage pre-check cannot tell; only the generation reply shows the exhausted plan
            return httpx.Response(503)
        return httpx.Response(meta["http_status"], headers=meta.get("headers", {}), json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if fixture.startswith("codex"):
            backend = CodexBackend(client=client, token_store=store_for(tmp_path / "codex.json"))
        else:
            credentials = named_account_credentials(tmp_path / "home", "anthropic", "first")
            credentials.save(OAuthTokens("token-first", "refresh", 4_000_000_000))
            backend = AnthropicBackend(
                client=client, token_store=credentials, base_url="https://test.invalid/v1/messages"
            )
        try:
            result = await recovery_probe.probe_readiness(backend)
        finally:
            await backend.aclose()
    assert requests[-1] == "POST", "the probe must reach the generation request"
    assert result is recovery_probe.Readiness.RETRY, "an exhausted account is still waiting, not a stop"
