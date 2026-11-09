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
from zeta.core.agent_progress import register_child_run_sink
from zeta.core.checkpoints import load_session_json
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
FEATURES = ("child_run_history", "child_run_events")
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
        if parent is None:
            reply = await client.request(
                self.request_id(), "session_history_and_subscribe", {"session_id": self.session_id}
            )
        else:
            reply = await client.request(
                self.request_id(), "child_run_events", {"session_id": self.session_id, "run_id": parent}
            )
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

    def sink(self, kind: str, run_id: str, row: dict[str, object], number: int) -> None:
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
    finally:
        unregister()
        backend.release_all()
        await _close(serve)
