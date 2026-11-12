"""child_run_events: terminal routes and delegated approvals over real ``zeta serve``.

Every test runs a real ``zeta serve`` over a unix socket and an authenticated websocket
with a real store and the real agent runtime. A wire reducer counts lifecycle facts by
``(run_id, sequence)``: the joined durable baseline plus the live continuation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.core.checkpoints import load_session_json
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
FEATURES = ("child_run_history", "child_run_events", "child_runs")
TERMINAL = {"completed", "failed", "canceled"}


class Actors:
    """One scripted backend per actor, chosen by the actor's first prompt.

    A gated actor holds its first model call until the test releases it; a failing
    actor's model call raises, so its run ends ``failed``.
    """

    def __init__(
        self,
        scripts: dict[str, list[ScriptedTurn]],
        *,
        gate: Collection[str] = (),
        fail: Collection[str] = (),
    ) -> None:
        self.backends = {prompt: FakeBackend(turns) for prompt, turns in scripts.items()}
        self.entered = {prompt: asyncio.Event() for prompt in gate}
        self.release = {prompt: asyncio.Event() for prompt in gate}
        self.fail = set(fail)

    def actor(self, messages: Sequence[Message]) -> str:
        for message in messages:
            if message.role is MessageRole.USER:
                for block in message.content:
                    if isinstance(block, TextContent) and block.text in self.backends:
                        return block.text
        raise AssertionError("no scripted actor for this model call")

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any) -> Any:
        prompt = self.actor(messages)
        if prompt in self.entered and not self.entered[prompt].is_set():
            self.entered[prompt].set()
            await await_or_fail(self.release[prompt].wait(), what=f"release {prompt}")
        if prompt in self.fail:
            raise RuntimeError(f"{prompt} backend failed")
        stream = FakeBackend.complete.__wrapped__(self.backends[prompt], messages, tool_schemas)
        async for event in stream:
            yield event


@dataclass
class Serve:
    server: ZetaServer
    transport: Transport
    controller: ReplayClient
    session_id: str
    clients: list[ReplayClient] = field(default_factory=list)
    next_id: int = 100

    def request_id(self) -> int:
        self.next_id += 1
        return self.next_id

    @property
    def store(self) -> Any:
        return self.server.runtime.opened.store

    async def observer(self) -> ReplayClient:
        client = await connect_replay(self.server, self.transport)
        self.clients.append(client)
        hello = await negotiate(client, observer=True, extra=FEATURES, protocol_version="1.2")
        assert "child_run_events" in hello["result"]["capabilities"]["features"], hello
        return client

    async def join(self, client: ReplayClient, parent: str | None) -> dict[str, Any]:
        """Join the scope where ``parent``'s children report: the top or one run."""
        method, params = (
            ("child_run_events", {"session_id": self.session_id, "run_id": parent})
            if parent is not None
            else ("session_history_and_subscribe", {"session_id": self.session_id})
        )
        reply = await client.request(self.request_id(), method, params)
        assert "error" not in reply, reply
        return reply["result"]

    async def drain(self, client: ReplayClient) -> list[dict[str, Any]]:
        """Every event the client received before a later status response."""
        status = self.request_id()
        await client.send(status, "status", {})
        events: list[dict[str, Any]] = []
        while (frame := await client.read()).get("id") != status:
            if "params" in frame:
                events.append(frame["params"])
        return events


async def _open(tmp_path: Path, transport: Transport, backend: Any) -> Serve:
    server = replay_server(tmp_path, transport, lambda provider, model, home: (backend, model or "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    await negotiate(controller, extra=FEATURES, protocol_version="1.2")
    created = await controller.request(2, "new_session", {"provider": "codex"})
    return Serve(server, transport, controller, created["result"]["session"]["session_id"])


async def _close(serve: Serve) -> None:
    for client in (serve.controller, *serve.clients):
        await client.close()
    await serve.server.close()


def _lifecycle_path(home_session_dir: Path, run_id: str) -> Path:
    path = home_session_dir
    for number in run_id.split(":")[1:]:
        path = path / "agents" / number
    return path / "agent_lifecycle.json"


@dataclass
class EndProbe:
    """The lifecycle owner's ``child_ended`` publications, with the run's durable
    lifecycle read from disk at the moment of publication."""

    session_dir: Path
    ended: list[tuple[str, int, dict[str, Any]]] = field(default_factory=list)

    def sink(
        self,
        kind: str,
        run_id: str,
        row: dict[str, object],
        number: int,
        store: object | None,
    ) -> None:
        del store
        if kind == "ended":
            durable = load_session_json(_lifecycle_path(self.session_dir, run_id))
            self.ended.append((run_id, number, durable))

    def for_run(self, run_id: str) -> list[tuple[int, dict[str, Any]]]:
        return [(number, durable) for run, number, durable in self.ended if run == run_id]


async def _until(client: ReplayClient, name: str, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every event in wire order up to and including the first ``name`` (for ``run_id``)."""
    events: list[dict[str, Any]] = []
    while True:
        frame = await await_or_fail(client.read(), what=f"wait for {name}")
        params = frame.get("params")
        if not isinstance(params, dict):
            continue
        events.append(params)
        if params.get("event") == name and run_id in {None, params.get("run_id")}:
            return events


def _lifecycle(events: list[dict[str, Any]], name: str, run_id: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == name and event.get("run_id") == run_id]


def _reduced(events: list[dict[str, Any]], name: str, run_id: str) -> set[tuple[str, int]]:
    return {(event["run_id"], event["sequence"]) for event in _lifecycle(events, name, run_id)}


def _agent(call_id: str, prompt: str, background: bool) -> ToolCall:
    arguments: dict[str, object] = {"prompt": prompt, "description": prompt}
    if background:
        arguments["background"] = True
    return ToolCall(call_id, "agent", arguments)


def _text(text: str) -> ScriptedTurn:
    return ScriptedTurn([TextContent(text)])


# --- live terminal routes --------------------------------------------------------


@dataclass(frozen=True)
class LiveRoute:
    background: bool
    nested: bool
    outcome: str

    @property
    def name(self) -> str:
        mode = "background" if self.background else "foreground"
        return f"{mode}-{'nested' if self.nested else 'direct'}-live-{self.outcome}"

    @property
    def target_prompt(self) -> str:
        return "grand-work" if self.nested else "child-work"

    def actors(self) -> Actors:
        target = _agent("agent-2" if self.nested else "agent-1", self.target_prompt, self.background)
        target_turns = [_text("target done"), _text("target noted")]
        if self.nested:
            scripts = {
                "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
                "child-work": [ScriptedTurn(tool_calls=[target]), _text("child done"), _text("child noted")],
                "grand-work": target_turns,
            }
        else:
            scripts = {
                "delegate": [ScriptedTurn(tool_calls=[target]), _text("parent done"), _text("parent noted")],
                "child-work": target_turns,
            }
        gate = {"child-work", self.target_prompt}
        fail = {self.target_prompt} if self.outcome == "failed" else set()
        return Actors(scripts, gate=gate, fail=fail)


LIVE_ROUTES = [
    LiveRoute(False, False, "completed"),
    LiveRoute(False, False, "canceled"),
    LiveRoute(False, True, "completed"),
    LiveRoute(False, True, "failed"),
    LiveRoute(True, False, "completed"),
    LiveRoute(True, False, "failed"),
    LiveRoute(True, True, "completed"),
]


def _receipt_entry(store: Any, route: LiveRoute, run_id: str, call_id: str) -> Any:
    """The immediate parent's durable terminal receipt: a tool result or a notification."""
    matches = []
    for entry in store.entries:
        if route.background:
            if (
                entry.type == "notification"
                and entry.data.get("child_instance_id") == run_id
                and not entry.data.get("display_only")
            ):
                matches.append(entry)
        elif entry.type == "message":
            result = entry.data["message"].get("tool_result")
            if isinstance(result, dict) and result.get("tool_call_id") == call_id:
                matches.append(entry)
    assert len(matches) == 1, matches
    return matches[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("route", LIVE_ROUTES, ids=[route.name for route in LIVE_ROUTES])
async def test_live_terminal_route_publishes_one_child_ended_after_the_receipt(
    tmp_path: Path, transport: Transport, route: LiveRoute
) -> None:
    backend = route.actors()
    serve = await _open(tmp_path, transport, backend)
    from zeta.core.agent_progress import register_child_run_sink

    probe = EndProbe(serve.store.session_dir)
    unregister = register_child_run_sink(probe.sink)
    try:
        session_id = serve.session_id
        direct = f"{session_id}:1"
        target = f"{direct}:1" if route.nested else direct
        parent = direct if route.nested else None
        top = await serve.observer()
        top_stream = (await serve.join(top, None))["replay"]["stream_id"]
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        await await_or_fail(backend.entered["child-work"].wait(), what="direct child started")
        watcher = top
        if route.nested:
            watcher = await serve.observer()
            await serve.join(watcher, direct)
            backend.release["child-work"].set()
            await await_or_fail(backend.entered["grand-work"].wait(), what="nested child started")
        before_end = await serve.drain(watcher)
        # 1. the parent discovers the run by its durable start row; 2. no end yet
        started = _lifecycle(before_end, "child_started", target)
        assert len(started) == 1, before_end
        # the live start row carries the one-owner parent_run_id: null for a direct
        # child, the parent run's id for a nested child
        assert started[0]["child"]["parent_run_id"] == parent, started[0]
        if route.nested:
            child_started_at = before_end.index(started[0])
            agent_started_at = next(
                index
                for index, event in enumerate(before_end)
                if index > child_started_at and event.get("event") == "agent_start"
            )
            assert child_started_at < agent_started_at, (
                "nested child_started must precede its forwarded agent_start"
            )
        assert _lifecycle(before_end, "child_ended", target) == []
        assert probe.for_run(target) == []
        if route.outcome == "canceled":
            aborted = await serve.controller.request(serve.request_id(), "abort", {})
            assert aborted["result"]["aborted"] is True, aborted
        backend.release_all()
        settled = "turn_aborted" if route.outcome == "canceled" else "agent_end"
        live = [*before_end, *await _until(watcher, "child_ended", target)]
        top_live = live if watcher is top else []
        if route.nested:
            top_live += await _until(top, "child_ended", direct)
        top_live += await _until(top, settled)
        live += await serve.drain(watcher)
        if watcher is not top:
            top_live += await serve.drain(top)

        # 3. exactly one end for the run on its immediate parent's stream
        ends = _lifecycle(live, "child_ended", target)
        assert len(ends) == 1, ends
        end = ends[0]
        assert end["run_id"] == target
        # 4. the row is terminal and its sequence is the immediate-parent receipt
        assert end["child"]["status"] == route.outcome, end
        assert "finished_at" in end["child"], end
        assert end["child"]["parent_run_id"] == parent, end
        receipt_store = serve.store
        child_store = None
        if route.nested:
            from zeta.core.store import ConversationStore

            child_store = ConversationStore(
                serve.store.session_dir / "agents", session_id="1",
                _storage_lease=serve.store._storage_lease,
            )
            receipt_store = child_store
        try:
            receipt = _receipt_entry(
                receipt_store, route, target, "agent-2" if route.nested else "agent-1"
            )
        finally:
            if child_store is not None:
                child_store.close()
        assert end["sequence"] == receipt.seq, (end, receipt)
        # the lifecycle was terminal on disk when the receipt published the end
        published = probe.for_run(target)
        assert [number for number, _durable in published] == [receipt.seq]
        assert published[0][1]["state"] in TERMINAL and published[0][1].get("finished_at")
        # 5. a nested run's root display mirror is not a second parent event
        if route.nested:
            assert _lifecycle(top_live, "child_ended", target) == []
            assert _lifecycle(top_live, "child_started", target) == []

        # 7. reconnect: one durable end row, and a new stream from zero
        later = await serve.observer()
        rejoined = await serve.join(later, parent)
        assert rejoined["replay"]["stream_id"] != top_stream
        assert rejoined["replay"]["history_seq"] == 0
        history = rejoined["messages"]
        assert _reduced(history, "child_started", target) == {(target, started[0]["sequence"])}
        assert _reduced(history, "child_ended", target) == {(target, receipt.seq)}
        [ended_row] = _lifecycle(history, "child_ended", target)
        assert ended_row["id"] == f"{receipt.id}:child_ended"
        assert ended_row["child"]["status"] == route.outcome
        # both durable history rows carry the one-owner parent_run_id
        [started_row] = _lifecycle(history, "child_started", target)
        assert started_row["child"]["parent_run_id"] == parent, started_row
        assert ended_row["child"]["parent_run_id"] == parent, ended_row
        after = await serve.drain(later)
        assert _lifecycle(after, "child_ended", target) == []
        assert all(event["seq"] == index for index, event in enumerate(after, 1)), after
        # the reducer over baseline plus continuation holds one fact on both streams
        assert _reduced(live, "child_ended", target) | _reduced(history, "child_ended", target) == {
            (target, receipt.seq)
        }
        if route.nested:
            top_history = (await serve.join(later, None))["messages"]
            assert _lifecycle(top_history, "child_ended", target) == []
        # a child_run_events join of the ended run the live projection no longer
        # holds reads the fallback row from disk; it carries the same parent_run_id
        fallback = await serve.observer()
        fallback_child = (await serve.join(fallback, target))["child"]
        assert fallback_child["run_id"] == target, fallback_child
        assert fallback_child["parent_run_id"] == parent, fallback_child
    finally:
        unregister()
        backend.release_all()
        await _close(serve)


# --- recovered terminal routes ---------------------------------------------------


@dataclass(frozen=True)
class RecoveredRoute:
    background: bool
    nested: bool
    # canceled: the run was still running at the crash; completed: its lifecycle was
    # final but the parent had no receipt; existing: the receipt was installed but the
    # marker was not cleared, so recovery must find it and publish nothing; mirrored:
    # a nested run's root display mirror was written but not its parent's receipt
    state: str

    @property
    def name(self) -> str:
        mode = "background" if self.background else "foreground"
        return f"{mode}-{'nested' if self.nested else 'direct'}-recovered-{self.state}"

    @property
    def outcome(self) -> str:
        return "canceled" if self.state == "canceled" else "completed"


RECOVERED_ROUTES = [
    RecoveredRoute(False, False, "canceled"),
    RecoveredRoute(False, False, "completed"),
    RecoveredRoute(False, False, "existing"),
    RecoveredRoute(False, True, "canceled"),
    RecoveredRoute(False, True, "completed"),
    RecoveredRoute(True, False, "canceled"),
    RecoveredRoute(True, False, "completed"),
    RecoveredRoute(True, False, "existing"),
    RecoveredRoute(True, True, "canceled"),
    RecoveredRoute(True, True, "completed"),
    RecoveredRoute(True, True, "mirrored"),
]

NOW = "2026-01-01T00:00:00+00:00"


def _seed_child(
    parent: Any, call_id: str, prompt: str, run_id: str, parent_run_id: str | None,
    *, background: bool, depth: int,
) -> Any:
    """Durable state the runner writes before a child's first turn."""
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import ToolUseContent

    call = _agent(call_id, prompt, background)
    parent.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    number = parent.allocate_agent_index()
    child = ConversationStore(parent.session_dir / "agents", session_id=str(number), cwd=parent.cwd)
    child.mark_agent_parent(call_id, agent_type="general")
    parent.register_agent_child(
        call, child_session_path=str(child.session_dir), description=prompt,
        agent_type="general", background=background, child_instance_id=run_id,
    )
    parent.append_child_run_started(run_id, parent_run_id, "general", prompt, NOW)
    child.start_agent_lifecycle(
        handle=run_id, started_at=NOW, depth=depth, agent_type="general", description=prompt,
    )
    return child


def _install_receipt(parent: Any, child: Any, route: RecoveredRoute, run_id: str, call_id: str) -> None:
    """The receipt a live run installed just before the process exited."""
    from zeta.protocol.types import ToolResult

    child.finish_agent_lifecycle("completed", final_result="target done")
    if route.background:
        parent.append_agent_notification(
            run_id, child_session_path=str(child.session_dir), description="target",
            status="completed", text="target done",
        )
        return
    parent.append_message(
        Message(
            MessageRole.TOOL_RESULT, [TextContent("target done")],
            tool_result=ToolResult(
                call_id, "target done",
                structured_content={"status": "completed", "child_instance_id": run_id},
            ),
        )
    )


def _seed_crashed_session(home: Path, route: RecoveredRoute) -> tuple[str, Path]:
    from zeta.core.session import SessionManager

    opened = SessionManager(home).create(provider="codex", model="offline", cwd=home)
    top = opened.store
    session_id = top.session_id
    direct = f"{session_id}:1"
    stores = [top]
    try:
        child = _seed_child(
            top, "agent-1", "child-work", direct, None,
            background=route.background and not route.nested, depth=1,
        )
        stores.append(child)
        parent, target_store, target, call_id = top, child, direct, "agent-1"
        if route.nested:
            target = f"{direct}:1"
            target_store = _seed_child(
                child, "agent-2", "grand-work", target, direct,
                background=route.background, depth=2,
            )
            stores.append(target_store)
            parent, call_id = child, "agent-2"
        if route.state == "completed":
            target_store.finish_agent_lifecycle("completed", final_result="target done")
        elif route.state == "existing":
            _install_receipt(parent, target_store, route, target, call_id)
        elif route.state == "mirrored":
            target_store.finish_agent_lifecycle("completed", final_result="target done")
            top.append_agent_notification(
                target, child_session_path=str(target_store.session_dir), description="target",
                status="completed", text="target done", display_only=True,
            )
    finally:
        for store in reversed(stores):
            store.close()
    return session_id, top.session_dir


def _receipts(store: Any, run_id: str) -> list[Any]:
    found = []
    for entry in store.entries:
        if entry.type == "notification":
            if entry.data.get("child_instance_id") == run_id and not entry.data.get("display_only"):
                found.append(entry)
        elif entry.type == "message":
            result = entry.data["message"].get("tool_result")
            structured = result.get("structured_content") if isinstance(result, dict) else None
            if isinstance(structured, dict) and structured.get("child_instance_id") == run_id:
                found.append(entry)
    return found


async def _resumed(tmp_path: Path, transport: Transport, session_id: str) -> Serve:
    server = replay_server(tmp_path, transport, lambda provider, model, home: (FakeBackend([]), model or "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    serve = Serve(server, transport, controller, session_id)
    try:
        await negotiate(controller, extra=FEATURES, protocol_version="1.2")
        resumed = await controller.request(2, "resume", {"session_id": session_id})
        assert "error" not in resumed, resumed
    except BaseException:
        await _close(serve)
        raise
    return serve


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("route", RECOVERED_ROUTES, ids=[route.name for route in RECOVERED_ROUTES])
async def test_recovered_terminal_route_presents_one_durable_child_ended(
    tmp_path: Path, transport: Transport, route: RecoveredRoute
) -> None:
    session_id, session_dir = await asyncio.to_thread(_seed_crashed_session, tmp_path, route)
    direct = f"{session_id}:1"
    target = f"{direct}:1" if route.nested else direct
    parent = direct if route.nested else None
    from zeta.core.agent_progress import register_child_run_sink

    probe = EndProbe(session_dir)
    unregister = register_child_run_sink(probe.sink)
    serve = None
    try:
        for attempt in range(2):
            # the second resume is a second recovery process over the same home
            serve = await _resumed(tmp_path, transport, session_id)
            parent_store = serve.store
            child_store = None
            if route.nested:
                from zeta.core.store import ConversationStore

                child_store = ConversationStore(
                    serve.store.session_dir / "agents", session_id="1",
                    _storage_lease=serve.store._storage_lease,
                )
                parent_store = child_store
            try:
                receipts = _receipts(parent_store, target)
            finally:
                if child_store is not None:
                    child_store.close()
            assert len(receipts) == 1, receipts
            receipt = receipts[0]
            # a run recovery finishes publishes once, and only after its lifecycle is
            # final on disk; an existing receipt and a later recovery publish nothing
            published = probe.for_run(target)
            expected = [] if route.state == "existing" or attempt else [receipt.seq]
            assert [number for number, _durable in published] == expected, published
            for _number, durable in published:
                assert durable["state"] in TERMINAL and durable.get("finished_at"), durable
            watcher = await serve.observer()
            joined = await serve.join(watcher, parent)
            history = joined["messages"]
            assert len(_lifecycle(history, "child_started", target)) == 1, history
            assert _reduced(history, "child_ended", target) == {(target, receipt.seq)}
            [ended_row] = _lifecycle(history, "child_ended", target)
            assert ended_row["id"] == f"{receipt.id}:child_ended"
            assert ended_row["child"]["status"] == route.outcome, ended_row
            assert "finished_at" in ended_row["child"]
            assert joined["replay"]["history_seq"] == 0
            live = await serve.drain(watcher)
            assert _lifecycle(live, "child_ended", target) == []
            if route.nested:
                top_history = (await serve.join(watcher, None))["messages"]
                assert _lifecycle(top_history, "child_ended", target) == []
            await _close(serve)
            serve = None
            probe.ended.clear()
    finally:
        unregister()
        if serve is not None:
            await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("nested", (False, True), ids=("direct", "nested"))
async def test_recovery_that_loses_the_receipt_race_publishes_nothing(
    tmp_path: Path, transport: Transport, nested: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second recovery process installs the receipt between this one's check and its
    atomic append; the append reports not-created, so this process publishes nothing."""
    from zeta.core.store import ConversationStore

    route = RecoveredRoute(True, nested, "canceled")
    session_id, session_dir = await asyncio.to_thread(_seed_crashed_session, tmp_path, route)
    target = f"{session_id}:1:1" if nested else f"{session_id}:1"
    raced: list[bool] = []
    original = ConversationStore.append_agent_notification_if_absent

    def racing(self: Any, child_instance_id: str, **kwargs: Any) -> Any:
        if child_instance_id == target and not kwargs.get("display_only") and not raced:
            # the other process's receipt commits first
            self.append_agent_notification(child_instance_id, **kwargs)
            entry, created = original(self, child_instance_id, **kwargs)
            raced.append(created)
            return entry, created
        return original(self, child_instance_id, **kwargs)

    # test-contract: seam force the other writer's commit inside the check-then-append window
    monkeypatch.setattr(ConversationStore, "append_agent_notification_if_absent", racing)
    from zeta.core.agent_progress import register_child_run_sink

    probe = EndProbe(session_dir)
    unregister = register_child_run_sink(probe.sink)
    serve = None
    try:
        serve = await _resumed(tmp_path, transport, session_id)
        assert raced == [False]
        assert probe.for_run(target) == []
        watcher = await serve.observer()
        history = (await serve.join(watcher, f"{session_id}:1" if nested else None))["messages"]
        ended = _lifecycle(history, "child_ended", target)
        assert len(ended) == 1, history
        assert ended[0]["child"]["status"] == "canceled"
    finally:
        unregister()
        if serve is not None:
            await _close(serve)


# --- delegated child-view approval round trip -------------------------------------


def _consecutive(events: list[dict[str, Any]], stream_id: str) -> None:
    assert [event["stream_id"] for event in events] == [stream_id] * len(events), events
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1)), events


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("decision", ("approve", "deny"))
@pytest.mark.parametrize("nested", (False, True), ids=("direct", "nested"))
async def test_delegated_child_approval_round_trip_on_parent_and_child_streams(
    tmp_path: Path, transport: Transport, decision: str, nested: bool
) -> None:
    target_file = tmp_path / "approved.txt"
    write = ToolCall("call-write", "write", {"path": str(target_file), "content": "written"})
    target_prompt = "grand-work" if nested else "child-work"
    target_turns = [ScriptedTurn(tool_calls=[write]), _text("target done")]
    if nested:
        scripts = {
            "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
            "child-work": [ScriptedTurn(tool_calls=[_agent("agent-2", "grand-work", False)]), _text("child done")],
            "grand-work": target_turns,
        }
    else:
        scripts = {
            "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
            "child-work": target_turns,
        }
    backend = Actors(scripts, gate={"child-work", target_prompt})
    serve = await _open(tmp_path, transport, backend)
    try:
        session_id = serve.session_id
        direct = f"{session_id}:1"
        target = f"{direct}:1" if nested else direct
        top = await serve.observer()
        top_stream = (await serve.join(top, None))["replay"]["stream_id"]
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        await await_or_fail(backend.entered["child-work"].wait(), what="direct child started")
        parent, parent_stream = top, top_stream
        if nested:
            parent = await serve.observer()
            parent_stream = (await serve.join(parent, direct))["replay"]["stream_id"]
            backend.release["child-work"].set()
            await await_or_fail(backend.entered["grand-work"].wait(), what="nested child started")
        child = await serve.observer()
        joined = await child.request(
            serve.request_id(), "child_run_events", {"session_id": session_id, "run_id": target}
        )
        # a nested run is joinable by its run id, with its own current row
        assert joined.get("result", {}).get("child", {}).get("run_id") == target, joined
        child_stream = joined["result"]["replay"]["stream_id"]
        backend.release_all()

        parent_events = await _until(parent, "approval_request")
        parent_request = parent_events[-1]
        request_id = parent_request["request_id"]
        # the child copy is admitted with the parent copy, so it is queued by now
        child_events = await serve.drain(child)
        child_requests = [e for e in child_events if e["event"] == "approval_request"]
        assert [e["request_id"] for e in child_requests] == [request_id], child_events
        child_request = child_requests[0]
        assert (child_request["stream_id"], parent_request["stream_id"]) == (child_stream, parent_stream)
        status = await serve.controller.request(serve.request_id(), "status", {})
        assert [item["request_id"] for item in status["result"]["pending_approvals"]] == [request_id]
        # observers can see both streams but cannot settle the approval
        for method in ("approve", "deny"):
            refused = await child.request(serve.request_id(), method, {"request_id": request_id})
            assert refused["error"]["code"] == -32008, refused
        settled = await serve.controller.request(serve.request_id(), decision, {"request_id": request_id})
        assert "error" not in settled, settled

        parent_events += await _until(parent, "child_ended", target)
        await await_or_fail(top.event("agent_end"), what="session settles")
        child_events += await serve.drain(child)
        parent_events += await serve.drain(parent)
        status = await serve.controller.request(serve.request_id(), "status", {})
        assert status["result"]["pending_approvals"] == []

        for events, stream_id in ((child_events, child_stream), (parent_events, parent_stream)):
            # 1/5: one request and one end on each stream, each with its own seq run
            requests = [e for e in events if e["event"] == "approval_request" and e["request_id"] == request_id]
            ends = [e for e in events if e["event"] == "approval_end" and e["request_id"] == request_id]
            assert (len(requests), len(ends)) == (1, 1), events
            assert requests[0]["seq"] < ends[0]["seq"]
            _consecutive(events, stream_id)
        # 3: approve runs the gated tool once; deny settles without running it
        tool_ends = [
            e for e in child_events
            if e["event"] == "tool_end" and e["tool_result"]["tool_call_id"] == "call-write"
        ]
        assert len(tool_ends) == 1, child_events
        if decision == "approve":
            assert target_file.exists(), "the approved tool did not run"
            assert target_file.read_text() == "written"
            assert tool_ends[0]["tool_result"]["is_error"] is False, tool_ends
        else:
            assert not target_file.exists()
            assert tool_ends[0]["tool_result"]["is_error"] is True, tool_ends
    finally:
        backend.release_all()
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_late_delegated_approval_end_after_the_run_ends_holds_the_child_stream(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delegated ``approval_end`` admitted after the target run already ended must
    not end the child stream with ``session_replaced``: an ended run's rows are frozen
    and cannot renumber. This forces deny -> the target's ``child_ended`` (its store is
    popped) -> the late ``approval_end`` admit, the ordering that races in the
    round-trip test above.
    """
    from zeta.server.approval_lifecycle import ApprovalLifecycle

    entered, release = asyncio.Event(), asyncio.Event()
    original = ApprovalLifecycle._append_delegated_history

    async def gated(self: Any, record: Any, payload: Any) -> Any:
        # test-contract: hold the delegated copy's approval_end until the run has ended.
        # persist_durably has already woken the target's tool, so the run proceeds while
        # this settlement waits; holding it does not block the run from ending.
        if isinstance(record.target.key, tuple):
            entered.set()
            await await_or_fail(release.wait(), what="release delegated approval_end")
        return await original(self, record, payload)

    monkeypatch.setattr(ApprovalLifecycle, "_append_delegated_history", gated)  # test-contract: seam force the late delegated approval_end to admit after the run ends

    target_file = tmp_path / "approved.txt"
    write = ToolCall("call-write", "write", {"path": str(target_file), "content": "written"})
    scripts = {
        "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
        "child-work": [ScriptedTurn(tool_calls=[_agent("agent-2", "grand-work", False)]), _text("child done")],
        "grand-work": [ScriptedTurn(tool_calls=[write]), _text("target done")],
    }
    backend = Actors(scripts, gate={"child-work", "grand-work"})
    serve = await _open(tmp_path, transport, backend)
    try:
        session_id = serve.session_id
        direct = f"{session_id}:1"
        target = f"{direct}:1"
        top = await serve.observer()
        await serve.join(top, None)
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        await await_or_fail(backend.entered["child-work"].wait(), what="direct child started")
        # the direct child's scope is where the target run's child_ended reports
        parent = await serve.observer()
        await serve.join(parent, direct)
        backend.release["child-work"].set()
        await await_or_fail(backend.entered["grand-work"].wait(), what="nested child started")
        child = await serve.observer()
        joined = await child.request(
            serve.request_id(), "child_run_events", {"session_id": session_id, "run_id": target}
        )
        assert joined.get("result", {}).get("child", {}).get("run_id") == target, joined
        child_stream = joined["result"]["replay"]["stream_id"]
        backend.release_all()

        parent_events = await _until(parent, "approval_request")
        request_id = parent_events[-1]["request_id"]
        # deny persists the decision and wakes the target's tool, then holds the delegated
        # approval_end at its history write; send without awaiting so the run can end first
        deny_id = serve.request_id()
        await serve.controller.send(deny_id, "deny", {"request_id": request_id})
        await await_or_fail(entered.wait(), what="delegated approval_end reached its write")
        # the target run ends and its store is popped before the held approval_end admits
        await _until(parent, "child_ended", target)
        release.set()
        settled = await serve.controller.response(deny_id)
        assert "error" not in settled, settled
        await await_or_fail(top.event("agent_end"), what="session settles")

        child_events = await serve.drain(child)
        stream_ended = [e for e in child_events if e["event"] == "replay_stream_ended"]
        ends = [e for e in child_events if e["event"] == "approval_end" and e["request_id"] == request_id]
        # the ended run's rows are frozen, so the late approval_end holds the stream
        assert stream_ended == [], child_events
        assert [e["stream_id"] for e in ends] == [child_stream], child_events
    finally:
        release.set()
        backend.release_all()
        await _close(serve)


# --- existing paths this feature changed (each fails on the base 7cc04469) ---------


async def _plain_observer(serve: Serve, features: tuple[str, ...]) -> ReplayClient:
    client = await connect_replay(serve.server, serve.transport)
    serve.clients.append(client)
    await negotiate(client, observer=True, extra=features, protocol_version="1.2")
    return client


@dataclass(frozen=True)
class CancelRoute:
    background: bool
    nested: bool
    live: bool

    @property
    def name(self) -> str:
        mode = "background" if self.background else "foreground"
        where = "live" if self.live else "recovered"
        return f"{mode}-{'nested' if self.nested else 'direct'}-{where}-canceled"


CANCEL_ROUTES = [
    CancelRoute(False, False, True),
    CancelRoute(False, False, False),
    CancelRoute(False, True, False),
    CancelRoute(True, False, False),
    CancelRoute(True, True, False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("route", CANCEL_ROUTES, ids=[route.name for route in CANCEL_ROUTES])
async def test_parent_receipt_installs_only_after_the_canceled_lifecycle_is_final(
    tmp_path: Path, transport: Transport, route: CancelRoute, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canceled run's lifecycle is final on disk before its parent installs the receipt,
    so no reader sees an installed terminal receipt next to a running lifecycle."""
    from zeta.core.store import ConversationStore

    target_call = "agent-2" if route.nested else "agent-1"
    installed: list[tuple[str | None, object]] = []
    state: dict[str, Any] = {}

    def observe(store: Any) -> None:
        durable = load_session_json(_lifecycle_path(state["session_dir"], state["target"]))
        installed.append((durable.get("state"), durable.get("finished_at")))

    append_message = ConversationStore.append_message

    def receipt_message(self: Any, message: Message, *args: Any, **kwargs: Any) -> Any:
        entry = append_message(self, message, *args, **kwargs)
        result = message.tool_result
        if result is not None and result.tool_call_id == target_call and "target" in state:
            observe(self)
        return entry

    append_if_absent = ConversationStore.append_agent_notification_if_absent

    def receipt_notification(self: Any, child_instance_id: str, **kwargs: Any) -> Any:
        entry, created = append_if_absent(self, child_instance_id, **kwargs)
        if created and child_instance_id == state.get("target") and not kwargs.get("display_only"):
            observe(self)
        return entry, created

    # test-contract: seam read the durable lifecycle the moment the parent receipt installs
    monkeypatch.setattr(ConversationStore, "append_message", receipt_message)
    monkeypatch.setattr(ConversationStore, "append_agent_notification_if_absent", receipt_notification)
    serve = None
    backend = None
    try:
        if route.live:
            mark_canceled = ConversationStore.mark_agent_canceled_async
            skipped: list[str] = []

            async def cleanup_did_not_finalize(self: Any, parent_tool_call_id: str) -> None:
                # the runner's cancel cleanup finalizes through the async storage
                # command; refuse it once (as when its terminal fence refuses) so
                # only the receipt owner's synchronous finalize can make the
                # lifecycle terminal before the receipt installs. the receipt owner
                # uses the sync mark_agent_canceled, which this seam never touches.
                if parent_tool_call_id == "agent-1" and not skipped:
                    skipped.append(parent_tool_call_id)
                    return
                await mark_canceled(self, parent_tool_call_id)

            # test-contract: seam the runner's cancel cleanup does not finalize the lifecycle
            monkeypatch.setattr(ConversationStore, "mark_agent_canceled_async", cleanup_did_not_finalize)
            backend = Actors(
                {
                    "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
                    "child-work": [_text("child done")],
                },
                gate={"child-work"},
            )
            serve = await _open(tmp_path, transport, backend)
            state.update(session_dir=serve.store.session_dir, target=f"{serve.session_id}:1")
            top = await _plain_observer(serve, ())
            await serve.join(top, None)
            await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
            await await_or_fail(backend.entered["child-work"].wait(), what="child started")
            aborted = await serve.controller.request(serve.request_id(), "abort", {})
            assert aborted["result"]["aborted"] is True, aborted
            backend.release_all()
            await _until(top, "turn_aborted")
            assert skipped == ["agent-1"]
        else:
            seeded = RecoveredRoute(route.background, route.nested, "canceled")
            session_id, session_dir = await asyncio.to_thread(_seed_crashed_session, tmp_path, seeded)
            target = f"{session_id}:1:1" if route.nested else f"{session_id}:1"
            state.update(session_dir=session_dir, target=target)
            serve = await _resumed(tmp_path, transport, session_id)
        assert len(installed) == 1, installed
        lifecycle_state, finished_at = installed[0]
        assert (lifecycle_state, finished_at is not None) == ("canceled", True), installed
        if not route.nested and not route.background:
            # the served history shows that receipt once
            watcher = await _plain_observer(serve, ())
            history = (await serve.join(watcher, None))["messages"]
            receipts = [
                row for row in history
                if (row.get("tool_result") or {}).get("tool_call_id") == target_call
            ]
            assert len(receipts) == 1, history
    finally:
        if backend is not None:
            backend.release_all()
        if serve is not None:
            await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_child_lifecycle_rows_reach_only_child_run_events_connections(
    tmp_path: Path, transport: Transport
) -> None:
    backend = Actors(
        {
            "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
            "child-work": [_text("child done")],
        }
    )
    serve = await _open(tmp_path, transport, backend)
    try:
        run_id = f"{serve.session_id}:1"
        plain = await _plain_observer(serve, ())
        child = await _plain_observer(serve, FEATURES)
        await serve.join(plain, None)
        await serve.join(child, None)
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        child_live = await _until(child, "agent_end")
        plain_live = await _until(plain, "agent_end")
        # live: only the child_run_events connection receives lifecycle events
        assert [e["event"] for e in child_live if e["event"].startswith("child_")][:1] == ["child_started"]
        assert len(_lifecycle(child_live, "child_ended", run_id)) == 1, child_live
        assert [e for e in plain_live if e["event"].startswith("child_")] == []
        # history: only the child_run_events connection sees the durable rows
        later_plain = await _plain_observer(serve, ())
        later_child = await _plain_observer(serve, FEATURES)
        plain_history = (await serve.join(later_plain, None))["messages"]
        child_history = (await serve.join(later_child, None))["messages"]
        assert [row.get("event") for row in child_history if str(row.get("event")).startswith("child_")] == [
            "child_started", "child_ended",
        ], child_history
        assert [row for row in plain_history if str(row.get("event")).startswith("child_")] == []
        assert len(plain_history) == len(child_history) - 2
    finally:
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_nested_delegated_approval_reaches_the_controller_and_resumes(
    tmp_path: Path, transport: Transport
) -> None:
    """A nested child's approval keeps its own run id up the chain, so serve matches
    it to the request the nested child registered instead of denying it."""
    target_file = tmp_path / "approved.txt"
    write = ToolCall("call-write", "write", {"path": str(target_file), "content": "written"})
    backend = Actors(
        {
            "delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", False)]), _text("parent done")],
            "child-work": [ScriptedTurn(tool_calls=[_agent("agent-2", "grand-work", False)]), _text("child done")],
            "grand-work": [ScriptedTurn(tool_calls=[write]), _text("target done")],
        }
    )
    serve = await _open(tmp_path, transport, backend)
    try:
        top = await _plain_observer(serve, ())
        await serve.join(top, None)
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        event: dict[str, Any] = {}
        while event.get("event") not in {"approval_request", "error"}:
            event = await top.event()
        assert event["event"] == "approval_request", event
        assert event["agent_instance_id"] == f"{serve.session_id}:1:1", event
        settled = await serve.controller.request(serve.request_id(), "approve", {"request_id": event["request_id"]})
        assert "error" not in settled, settled
        await _until(top, "agent_end")
        assert target_file.exists(), "the approved nested tool did not run"
        assert target_file.read_text() == "written"
    finally:
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("background", (False, True), ids=("foreground", "background"))
async def test_child_setup_failure_ends_once_after_its_failed_lifecycle(
    tmp_path: Path, transport: Transport, background: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child whose loop cannot be built is finalized ``failed`` before its receipt."""
    from zeta.agent import runner
    from zeta.core.agent_progress import register_child_run_sink

    async def broken_setup(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("child setup failed")

    # test-contract: seam the child loop cannot be built after its start row is durable
    monkeypatch.setattr(runner, "_build_child_loop", broken_setup)
    backend = Actors(
        {"delegate": [ScriptedTurn(tool_calls=[_agent("agent-1", "child-work", background)]), _text("parent done")]}
    )
    serve = await _open(tmp_path, transport, backend)
    probe = EndProbe(serve.store.session_dir)
    unregister = register_child_run_sink(probe.sink)
    try:
        run_id = f"{serve.session_id}:1"
        top = await serve.observer()
        await serve.join(top, None)
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        live = await _until(top, "agent_end")
        live += await serve.drain(top)
        assert len(_lifecycle(live, "child_started", run_id)) == 1, live
        ends = _lifecycle(live, "child_ended", run_id)
        assert len(ends) == 1, live
        assert ends[0]["child"]["status"] == "failed" and "finished_at" in ends[0]["child"]
        [receipt] = [
            entry for entry in serve.store.entries
            if entry.type == "message"
            and (entry.data["message"].get("tool_result") or {}).get("tool_call_id") == "agent-1"
        ]
        assert ends[0]["sequence"] == receipt.seq
        published = probe.for_run(run_id)
        assert [number for number, _durable in published] == [receipt.seq]
        assert published[0][1]["state"] == "failed" and published[0][1].get("finished_at")
    finally:
        unregister()
        await _close(serve)


# --- cancellation inside the background terminal transition ----------------------

TRANSITION_AWAITS = (
    ("store", "finalize_background_agent"),
    ("channel", "close_questions"),
    ("channel", "drain_terminal"),
    ("channel", "publish_completion"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("owner,method", TRANSITION_AWAITS, ids=[name for _owner, name in TRANSITION_AWAITS])
async def test_cancel_at_each_background_transition_await_still_publishes_one_consistent_end(
    tmp_path: Path, transport: Transport, owner: str, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design 9.1: the finalize/adopt/receipt transition is shielded from the watcher's cancel.

    The watcher is canceled while the transition waits at one of its awaits. The
    lifecycle still becomes terminal, the receipt installs once, and exactly one
    ``child_ended`` reaches the wire after the durable lifecycle is terminal.
    """
    from zeta.agent.conversation_channel import ConversationChannel
    from zeta.core.agent_progress import register_child_run_sink
    from zeta.core.store import ConversationStore

    entered, release = asyncio.Event(), asyncio.Event()
    holder = ConversationStore if owner == "store" else ConversationChannel
    original = getattr(holder, method)

    async def gated(self: Any, *args: Any, **kwargs: Any) -> Any:
        current = asyncio.current_task()
        if current is not None and current.get_coro().__qualname__.endswith("commit_adopt_publish"):
            entered.set()
            await await_or_fail(release.wait(), what=f"release {method}")
        return await original(self, *args, **kwargs)

    # test-contract: seam hold the background terminal transition at one of its awaits
    monkeypatch.setattr(holder, method, gated)
    route = LiveRoute(True, False, "completed")
    backend = route.actors()
    serve = await _open(tmp_path, transport, backend)
    probe = EndProbe(serve.store.session_dir)
    unregister = register_child_run_sink(probe.sink)
    try:
        target = f"{serve.session_id}:1"
        top = await serve.observer()
        await serve.join(top, None)
        await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
        await await_or_fail(backend.entered["child-work"].wait(), what="background child started")
        backend.release_all()
        await await_or_fail(entered.wait(), what=f"transition reached {method}")
        [watcher] = [
            task for task in asyncio.all_tasks()
            if task.get_coro().__qualname__.endswith("finish_background")
        ]
        # the cancel is delivered to the watcher before the held transition resumes
        watcher.cancel()
        release.set()
        await await_or_fail(asyncio.wait({watcher}), what="background watcher ended")
        assert watcher.cancelled(), "the deferred cancel was not re-raised"
        loop = asyncio.get_running_loop()
        hop = loop.create_future()
        loop.call_soon_threadsafe(hop.set_result, None)
        await await_or_fail(hop, what="lifecycle callbacks applied")
        live = await serve.drain(top)

        ends = _lifecycle(live, "child_ended", target)
        assert len(ends) == 1, live
        assert ends[0]["child"]["status"] == "completed" and "finished_at" in ends[0]["child"]
        receipt = _receipt_entry(serve.store, route, target, "agent-1")
        assert ends[0]["sequence"] == receipt.seq
        published = probe.for_run(target)
        assert [number for number, _durable in published] == [receipt.seq]
        assert published[0][1]["state"] == "completed" and published[0][1].get("finished_at")
        durable = load_session_json(_lifecycle_path(serve.store.session_dir, target))
        assert durable["state"] == "completed"
    finally:
        unregister()
        release.set()
        backend.release_all()
        await _close(serve)
