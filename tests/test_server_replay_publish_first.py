"""Publish-first frames reach each replay stream once: history xor live.

``assistant_message`` and the ``user_message`` echo of a send without a delivery
id are published before their history row installs. A replay stream's copy waits
until the row is found (then it carries the row's seq) or until the turn ends
(then the row will never install, and the copy is live-only). Non-replay
clients keep today's timing.

Every test runs a real ``zeta serve`` over a unix socket and a websocket with a
real store; each hold is a gate with a deadline.
"""
from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.branch_reads import resident_entries
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_replay_table import _frames_before_status
from zeta.model_input import ModelInputEnvelope
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer
from zeta.server.protocol import EventDescriptor

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
FEATURES = ("delivery_id", "user_message_event", "user_message_delivery_id", "model_input_ids")
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


class Gates:
    """Named holds: a held step reports it was reached, then waits for release."""

    def __init__(self) -> None:
        self._gates: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}

    def _pair(self, name: str) -> tuple[asyncio.Event, asyncio.Event]:
        return self._gates.setdefault(name, (asyncio.Event(), asyncio.Event()))

    async def hold(self, name: str) -> None:
        entered, release = self._pair(name)
        entered.set()
        await await_or_fail(release.wait(), what=f"release {name}")

    async def reached(self, name: str) -> None:
        await await_or_fail(self._pair(name)[0].wait(), what=f"{name} reached")

    def release(self, name: str) -> None:
        self._pair(name)[1].set()

    def release_all(self) -> None:
        for _entered, release in self._gates.values():
            release.set()


class StreamBackend:
    """One turn of scripted provider events; a str step holds on that gate."""

    def __init__(self, gates: Gates, steps: list[StreamEvent | str]) -> None:
        self.gates = gates
        self.steps = steps

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any) -> Any:
        del messages, tool_schemas
        for step in self.steps:
            if isinstance(step, str):
                await self.gates.hold(step)
            else:
                yield step


def _end(text: str, **data: Any) -> StreamEvent:
    return StreamEvent(
        StreamEventType.MESSAGE_END,
        message=Message(MessageRole.ASSISTANT, [TextContent(text)]),
        data=data,
    )


@dataclass
class Serve:
    server: ZetaServer
    transport: Transport
    controller: ReplayClient
    session_id: str
    gates: Gates
    clients: list[ReplayClient] = field(default_factory=list)
    next_id: int = 100

    def request_id(self) -> int:
        self.next_id += 1
        return self.next_id

    @property
    def loop_owner(self) -> Any:
        return self.server.runtime.loop

    @property
    def store(self) -> Any:
        return self.server.runtime.opened.store

    async def observer(self) -> ReplayClient:
        client = await connect_replay(self.server, self.transport)
        self.clients.append(client)
        await negotiate(client, observer=True, extra=FEATURES, protocol_version="1.2")
        return client

    async def join(self, client: ReplayClient) -> list[dict[str, Any]]:
        response = await client.request(
            self.request_id(), "session_history_and_subscribe", {"session_id": self.session_id}
        )
        assert "error" not in response, response
        return response["result"]["messages"]

    async def joined(self) -> tuple[ReplayClient, list[dict[str, Any]]]:
        client = await self.observer()
        return client, await self.join(client)

    async def live(self, client: ReplayClient) -> list[dict[str, Any]]:
        """Every event the client got before a later status response."""
        status = self.request_id()
        await client.send(status, "status", {})
        frames: list[dict[str, Any]] = []
        while (frame := await client.read()).get("id") != status:
            if "params" in frame:
                frames.append(frame["params"])
        return frames

    async def close(self) -> None:
        self.gates.release_all()
        for client in (self.controller, *self.clients):
            await client.close()
        await self.server.close()


async def _serve(tmp_path: Path, transport: Transport, backend: Any, gates: Gates) -> Serve:
    server = replay_server(tmp_path, transport, lambda provider, model, home: (backend, model or "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    # a non-replay controller: its frames keep today's timing
    await negotiate(controller, replay=False, extra=FEATURES, protocol_version="1.2")
    created = await controller.request(2, "new_session", {"provider": "codex"})
    return Serve(server, transport, controller, created["result"]["session"]["session_id"], gates)


async def _event(client: ReplayClient, name: str) -> dict[str, Any]:
    return await client.event(name)


def _frame_texts(frame: dict[str, Any]) -> list[str]:
    if frame["event"] == "user_message":
        return [frame.get("text", "")]
    if frame["event"] == "assistant_message":
        return [block.get("text", "") for block in frame["message"].get("content", [])]
    return []


def _count(history: list[dict[str, Any]], frames: list[dict[str, Any]], marker: str) -> int:
    """How often one effect appears in a join's history plus its live frames."""
    texts = [
        block.get("text") or ""
        for row in history
        for block in row.get("content") or []
        if isinstance(block, dict)
    ]
    texts += [text for frame in frames for text in _frame_texts(frame)]
    return sum(marker in text for text in texts)


async def _counts(
    serve: Serve, joins: dict[str, tuple[ReplayClient, list[dict[str, Any]]]], marker: str
) -> dict[str, int]:
    return {
        name: _count(history, await serve.live(client), marker)
        for name, (client, history) in joins.items()
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_config_changed_replay_observers_are_feature_filtered(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _assistant_backend)
    await server.start()
    controller = await connect_replay(server, transport)
    without_feature = await connect_replay(server, transport)
    with_feature = await connect_replay(server, transport)
    without_delivery = await connect_replay(server, transport)
    try:
        await negotiate(
            controller,
            extra=("live_config_changed", "memory_updated", "user_message_event", "delivery_id", "user_message_delivery_id"),
        )
        session_id = (await controller.request(2, "new_session", {"provider": "codex"}))["result"]["session"]["session_id"]
        await negotiate(without_feature, observer=True)
        await negotiate(
            with_feature,
            observer=True,
            extra=("live_config_changed", "memory_updated", "user_message_event", "delivery_id", "user_message_delivery_id"),
        )
        await negotiate(without_delivery, observer=True, extra=("user_message_event",))
        with_joined = None
        for client in (without_feature, with_feature, without_delivery):
            joined = await client.request(2, "session_history_and_subscribe", {"session_id": session_id})
            if client is with_feature:
                with_joined = joined
            assert joined["result"]["replay"]["history_seq"] == 0

        profile = tmp_path / "agents" / "specialist.md"
        profile.parent.mkdir(parents=True)
        profile.write_text(
            "---\nname: specialist\ndescription: focused specialist\n---\nprofile body\n",
            encoding="utf-8",
        )
        assert server.runtime.loop is not None
        prepared: list[StreamEvent] = []
        server.runtime.loop.set_background_event_sink(prepared.append)
        await server.runtime.loop.refresh_live_config()
        assert len(prepared) == 1
        await server.control._event(prepared[0], background=True)

        config = await with_feature.event("config_changed")
        assert with_joined is not None
        assert config["stream_id"] == with_joined["result"]["replay"]["stream_id"]
        assert config["seq"] == 1
        frames = await _frames_before_status(without_feature, 3)
        assert not [
            frame for frame in frames
            if frame.get("params", {}).get("event") == "config_changed"
        ]

        await server.control._notify(
            EventDescriptor.TURN_ABORTED, session_id, data={"reason": "test"}
        )
        common_with = await with_feature.event("turn_aborted")
        common_without = await without_feature.event("turn_aborted")
        assert common_with["seq"] == 2
        assert common_without["seq"] == 1

        server.control._publish_memory_notice(session_id, "memory-marker")
        memory = await with_feature.event("memory_updated")
        assert memory["seq"] == 3
        assert memory["message"] == "memory-marker"
        frames = await _frames_before_status(without_feature, 4)
        assert not [
            frame for frame in frames
            if frame.get("params", {}).get("event") == "memory_updated"
        ]

        await server.control.user_message(
            "user-marker",
            "default",
            frozenset({"user_message_event", "delivery_id", "user_message_delivery_id"}),
            delivery_id="delivery-marker",
        )
        user = await with_feature.event("user_message")
        assert user["seq"] == 4
        assert user["delivery_id"] == "delivery-marker"
        # user_message rows are in every replay client's history, so a replay
        # client receives the live echo even without user_message_event; only the
        # delivery_id field stays per-connection.
        user_without_feature = await without_feature.event("user_message")
        assert user_without_feature["seq"] == 2
        assert "delivery_id" not in user_without_feature
        user_without_delivery = await without_delivery.event("user_message")
        assert user_without_delivery["seq"] == 2
        assert "delivery_id" not in user_without_delivery
    finally:
        await without_delivery.close()
        await with_feature.close()
        await without_feature.close()
        await controller.close()
        await server.close()


# --- moved in from the frontier proof (scratch/replay-frontier-proof@14af66a2) ---


def _assistant_backend(provider: str, model: str | None, home: Path) -> tuple[Any, str]:
    del provider, home
    return FakeBackend([ScriptedTurn(content=[TextContent("assistant-marker")])]), model or "offline"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_join_between_assistant_event_and_its_row_sees_it_once(tmp_path: Path, transport: Any) -> None:
    server = replay_server(tmp_path, transport, _assistant_backend)
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    release, entered, appended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        await negotiate(controller)
        session_id = (await controller.request(2, "new_session", {"provider": "codex"}))["result"]["session"]["session_id"]
        await negotiate(observer, observer=True)
        loop_owner = server.runtime.loop
        original = loop_owner._append_turn_message_with_approvals

        async def gated(message: Any, approvals: Any) -> Any:
            if getattr(message.role, "value", message.role) == "assistant":
                entered.set()
                await await_or_fail(release.wait(), what="release assistant row append")
                result = await original(message, approvals)
                appended.set()
                return result
            return await original(message, approvals)

        # test-contract: seam hold the assistant row append after its MESSAGE_END event was published
        loop_owner._append_turn_message_with_approvals = gated
        await controller.send(3, "send", {"text": "hello"})
        await await_or_fail(entered.wait(), what="assistant event published, row not installed")
        joined = await observer.request(4, "session_history_and_subscribe", {"session_id": session_id})
        history = [
            block.get("text")
            for row in joined["result"]["messages"]
            for block in row.get("content", [])
            if isinstance(block, dict)
        ]
        release.set()
        await await_or_fail(appended.wait(), what="assistant row installed")
        # the replay copy goes out at the next admission after the row
        await await_or_fail(controller.event("agent_end"), what="turn end")
        frames = await _frames_before_status(observer, 5)
        live = [
            block.get("text")
            for f in frames
            if f.get("params", {}).get("event") == "assistant_message"
            for block in f["params"]["message"].get("content", [])
        ]
        assert (history + live).count("assistant-marker") == 1, f"history={history} live={live}"
    finally:
        release.set()
        await observer.close()
        await controller.close()
        await server.close()


def _hold_assistant_row(serve: Serve, gates: Gates) -> None:
    original = serve.loop_owner._append_turn_message_with_approvals

    async def gated(message: Any, approvals: Any) -> Any:
        if message.role is MessageRole.ASSISTANT:
            await gates.hold("assistant_row")
        return await original(message, approvals)

    # test-contract: seam hold the assistant row append after its MESSAGE_END event was published
    serve.loop_owner._append_turn_message_with_approvals = gated


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_non_replay_controller_gets_assistant_message_before_its_row(
    tmp_path: Path, transport: Transport
) -> None:
    gates = Gates()
    serve = await _serve(tmp_path, transport, FakeBackend([ScriptedTurn([TextContent("timing-marker")])]), gates)
    try:
        _hold_assistant_row(serve, gates)
        await serve.controller.request(serve.request_id(), "send", {"text": "hello"})
        await gates.reached("assistant_row")
        # today's timing: a non-replay client's copy does not wait for the row
        early = await _event(serve.controller, "assistant_message")
        assert _frame_texts(early) == ["timing-marker"]
        window = await serve.joined()
        gates.release("assistant_row")
        await _event(serve.controller, "agent_end")
        assert await _counts(serve, {"window": window}, "timing-marker") == {"window": 1}
    finally:
        await serve.close()


# --- the user echo of a send without a delivery id ---


async def _send_plain(serve: Serve) -> str:
    await serve.controller.request(serve.request_id(), "send", {"text": "echo-marker"})
    return "echo-marker"


async def _send_images(serve: Serve) -> str:
    image = {"name": "shot.png", "mime_type": "image/png", "data": base64.b64encode(PNG).decode()}
    response = await serve.controller.request(
        serve.request_id(),
        "send_images",
        {"session_id": serve.session_id, "text": "image-marker", "images": [image]},
    )
    assert response["result"]["accepted"] is True, response
    return "image-marker"


async def _send_skill(serve: Serve) -> str:
    """T7b: a skill expansion's row has origin skill_expansion, not user."""
    input_id = serve.server.control._model_inputs.register(
        serve.session_id,
        ModelInputEnvelope("skill-marker expanded body", "skill-marker", MessageOrigin.SKILL_EXPANSION),
    )
    response = await serve.controller.request(serve.request_id(), "send", {"input_id": input_id})
    assert response["result"]["accepted"] is True, response
    return "skill-marker"


SENDS: dict[str, Callable[[Serve], Any]] = {
    "send": _send_plain,
    "send_images": _send_images,
    "skill": _send_skill,
}


def _hold_user_row(serve: Serve, gates: Gates) -> None:
    """Hold the user row's append, then hold MCP startup after it installed."""
    loop_owner = serve.loop_owner
    append = loop_owner._append_turn_message
    ensure_mcp = loop_owner._ensure_mcp_servers

    async def held_append(message: Any, *args: Any, **kwargs: Any) -> Any:
        if message.role is MessageRole.USER:
            await gates.hold("user_row")
        return await append(message, *args, **kwargs)

    async def held_mcp() -> Any:
        await gates.hold("mcp")
        return await ensure_mcp()

    # test-contract: seam hold the user row append after its echo was published
    loop_owner._append_turn_message = held_append
    # test-contract: seam hold the turn between the user row's install and AGENT_START
    loop_owner._ensure_mcp_servers = held_mcp


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("route", tuple(SENDS))
async def test_join_while_user_row_is_held_sees_echo_once(
    tmp_path: Path, transport: Transport, route: str
) -> None:
    gates = Gates()
    serve = await _serve(tmp_path, transport, FakeBackend([ScriptedTurn([TextContent("ok")])]), gates)
    try:
        joins = {"before": await serve.joined()}
        _hold_user_row(serve, gates)
        marker = await SENDS[route](serve)
        await gates.reached("user_row")
        # today's timing for the non-replay controller
        assert marker in (await _event(serve.controller, "user_message"))["text"]
        joins["before_row"] = await serve.joined()
        gates.release("user_row")
        await gates.reached("mcp")
        joins["after_row"] = await serve.joined()
        gates.release("mcp")
        await _event(serve.controller, "agent_end")
        joins["after_turn"] = await serve.joined()
        frames = {name: await serve.live(client) for name, (client, _history) in joins.items()}
        counts = {name: _count(joins[name][1], frames[name], marker) for name in joins}
        assert counts == {"before": 1, "before_row": 1, "after_row": 1, "after_turn": 1}
        # latency bound: a replay stream's echo goes out with AGENT_START, not later
        kinds = [frame["event"] for frame in frames["before"]]
        assert kinds.index("user_message") < kinds.index("agent_start"), kinds
    finally:
        await serve.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_turn_canceled_before_its_first_step_closes_its_echo(
    tmp_path: Path, transport: Transport
) -> None:
    """A turn task canceled before it runs has no pump end; its end closes the echo."""
    gates = Gates()
    serve = await _serve(tmp_path, transport, FakeBackend([ScriptedTurn([TextContent("ok")])]), gates)
    control = serve.server.control
    start_turn = control.start_turn
    try:
        joins = {"before": await serve.joined()}

        def canceled_start(*args: Any, **kwargs: Any) -> None:
            start_turn(*args, **kwargs)
            assert control._turn_task is not None
            control._turn_task.cancel()

        # test-contract: seam cancel the turn task before its first step
        control.start_turn = canceled_start  # type: ignore[method-assign]
        await serve.controller.request(serve.request_id(), "send", {"text": "lost-marker"})
        assert control._turn_task is not None
        await await_or_fail(asyncio.wait({control._turn_task}), what="canceled turn task")
        control.start_turn = start_turn  # type: ignore[method-assign]
        joins["after_cancel"] = await serve.joined()
        _hold_user_row(serve, gates)
        await serve.controller.request(serve.request_id(), "send", {"text": "kept-marker"})
        await gates.reached("user_row")
        joins["before_row"] = await serve.joined()
        gates.release("user_row")
        gates.release("mcp")
        await _event(serve.controller, "agent_end")
        frames = {name: await serve.live(client) for name, (client, _history) in joins.items()}
        lost = {name: _count(joins[name][1], frames[name], "lost-marker") for name in joins}
        kept = {name: _count(joins[name][1], frames[name], "kept-marker") for name in joins}
        # the canceled send's row never installs: live-only, closed when its task ended
        assert lost == {"before": 1, "after_cancel": 0, "before_row": 0}
        assert kept == {"before": 1, "after_cancel": 1, "before_row": 1}
    finally:
        control.start_turn = start_turn  # type: ignore[method-assign]
        await serve.close()


# --- assistant rows: failed, superseded, never installed ---


def _hold_terminal_admission(serve: Serve, gates: Gates) -> None:
    """Hold the turn after its last row and before the pump's end."""
    store = serve.store
    admit = store.admit_step_delivery_terminal

    async def held(intent: Any) -> Any:
        if intent == "foreground_end":
            await gates.hold("terminal")
        return await admit(intent)

    # test-contract: seam hold the foreground terminal admission after the turn's last row
    store.admit_step_delivery_terminal = held


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_failed_assistant_row_reaches_each_stream_once(tmp_path: Path, transport: Transport) -> None:
    """A truncated reply whose stream then fails installs a ``failed`` row."""
    gates = Gates()
    backend = StreamBackend(gates, [_end("failed-marker", truncated=True), "after_end"])
    serve = await _serve(tmp_path, transport, backend, gates)
    try:
        joins = {"before": await serve.joined()}
        _hold_terminal_admission(serve, gates)
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await gates.reached("after_end")
        joins["before_row"] = await serve.joined()
        gates.release("after_end")
        await gates.reached("terminal")
        states = [
            entry.data["message"].get("metadata", {}).get("response_state")
            for entry in resident_entries(serve.store)
            if entry.type == "message" and entry.data["message"]["role"] == "assistant"
        ]
        assert states == ["failed"], states
        joins["after_row"] = await serve.joined()
        gates.release("terminal")
        await _event(serve.controller, "agent_end")
        assert await _counts(serve, joins, "failed-marker") == {
            "before": 1, "before_row": 1, "after_row": 1
        }
    finally:
        await serve.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_second_assistant_message_before_the_first_row_supersedes_it(
    tmp_path: Path, transport: Transport
) -> None:
    """A truncated reply is reset and replaced; only the replacement's row installs."""
    gates = Gates()
    backend = StreamBackend(
        gates,
        [
            _end("first-marker", truncated=True),
            StreamEvent(StreamEventType.ASSISTANT_RESET),
            _end("second-marker"),
        ],
    )
    serve = await _serve(tmp_path, transport, backend, gates)
    try:
        joins = {"before": await serve.joined()}
        _hold_assistant_row(serve, gates)
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await gates.reached("assistant_row")
        joins["before_row"] = await serve.joined()
        gates.release("assistant_row")
        await _event(serve.controller, "agent_end")
        joins["after_turn"] = await serve.joined()
        frames = {name: await serve.live(client) for name, (client, _history) in joins.items()}
        first = {name: _count(joins[name][1], frames[name], "first-marker") for name in joins}
        second = {name: _count(joins[name][1], frames[name], "second-marker") for name in joins}
        # the superseded copy is live-only and provisional: no later joiner sees it
        assert first == {"before": 1, "before_row": 0, "after_turn": 0}
        assert second == {"before": 1, "before_row": 1, "after_turn": 1}
    finally:
        await serve.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_assistant_row_that_never_installs_is_sent_live_at_turn_end(
    tmp_path: Path, transport: Transport
) -> None:
    """C5: an aborted reply whose partial row fails to persist has no row."""
    gates = Gates()
    backend = StreamBackend(gates, [_end("orphan-marker", truncated=True), "after_end"])
    serve = await _serve(tmp_path, transport, backend, gates)
    try:
        joins = {"before": await serve.joined()}
        persisted = threading.Event()

        def failing(*args: Any, **kwargs: Any) -> None:
            persisted.set()
            raise OSError("disk failed")

        # test-contract: seam the aborted partial row fails to persist, so it never installs
        serve.loop_owner._persist_partial_with_cancelled_tools = failing
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await gates.reached("after_end")
        joins["window"] = await serve.joined()
        with pytest.warns(RuntimeWarning, match="failed to persist partial state"):
            aborted = await serve.controller.request(serve.request_id(), "abort", {})
        assert aborted["result"]["aborted"] is True, aborted
        assert persisted.is_set()
        joins["after_turn"] = await serve.joined()
        assert await _counts(serve, joins, "orphan-marker") == {
            "before": 1, "window": 1, "after_turn": 0
        }
    finally:
        await serve.close()


def _assistant_count(history: list[dict[str, Any]], frames: list[dict[str, Any]]) -> int:
    """Assistant replies in a join's history plus its live frames, text or not."""
    rows = sum(row.get("role") == "assistant" for row in history)
    return rows + sum(frame["event"] == "assistant_message" for frame in frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_empty_failed_assistant_row_releases_the_held_reply_once(
    tmp_path: Path, transport: Transport
) -> None:
    """An empty reply whose stream fails installs an empty ``failed`` row; it still releases."""
    gates = Gates()
    empty = StreamEvent(
        StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, []), data={"truncated": True}
    )
    backend = StreamBackend(gates, [empty, "after_end"])
    serve = await _serve(tmp_path, transport, backend, gates)
    try:
        joins = {"before": await serve.joined()}
        _hold_terminal_admission(serve, gates)
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await gates.reached("after_end")
        joins["before_row"] = await serve.joined()
        gates.release("after_end")
        await gates.reached("terminal")
        rows = [
            entry.data["message"]
            for entry in resident_entries(serve.store)
            if entry.type == "message" and entry.data["message"]["role"] == "assistant"
        ]
        assert [(row.get("metadata", {}).get("response_state"), row["content"]) for row in rows] == [
            ("failed", [])
        ], rows
        joins["after_row"] = await serve.joined()
        gates.release("terminal")
        await _event(serve.controller, "agent_end")
        counts = {
            name: _assistant_count(history, await serve.live(client))
            for name, (client, history) in joins.items()
        }
        assert counts == {"before": 1, "before_row": 1, "after_row": 1}
    finally:
        await serve.close()


async def _until_agent_end(client: ReplayClient) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        frame = await await_or_fail(client.read(), what="controller frame")
        if "params" in frame:
            frames.append(frame["params"])
            if frame["params"].get("event") == "agent_end":
                return frames


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_replay_controller_without_a_stream_gets_publish_first_frames_plainly(
    tmp_path: Path, transport: Transport
) -> None:
    """A controller that negotiated replay but never joined, or whose stream ended, still gets them."""
    backend = FakeBackend([ScriptedTurn([TextContent("reply-one")]), ScriptedTurn([TextContent("reply-two")])])
    server = replay_server(tmp_path, transport, lambda provider, model, home: (backend, model or "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    try:
        await negotiate(controller, extra=FEATURES, protocol_version="1.2")
        created = await controller.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        await controller.request(3, "send", {"text": "echo-one"})
        never_joined = await _until_agent_end(controller)
        joined = await controller.request(4, "session_history_and_subscribe", {"session_id": session_id})
        assert "error" not in joined, joined
        # a rejected rejoin ends the stream and starts none
        rejected = await controller.request(5, "session_history_and_subscribe", {"session_id": "missing"})
        assert "error" in rejected, rejected
        await controller.request(6, "send", {"text": "echo-two"})
        after_end = await _until_agent_end(controller)
        ended = [frame.get("reason") for frame in after_end if frame["event"] == "replay_stream_ended"]
        assert ended == ["rejoined"], after_end
        for frames, echo, reply in (
            (never_joined, "echo-one", "reply-one"),
            (after_end, "echo-two", "reply-two"),
        ):
            kept = [frame for frame in frames if frame["event"] in {"user_message", "assistant_message"}]
            texts = [text for frame in kept for text in _frame_texts(frame)]
            assert (texts.count(echo), texts.count(reply)) == (1, 1), texts
            assert all("stream_id" not in frame and "seq" not in frame for frame in kept), kept
    finally:
        await controller.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_pressure_eviction_keeps_a_held_reply_for_a_rejoin_before_its_row(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evicted stream's rows still install, so its held reply must outlive the eviction."""
    monkeypatch.setattr("zeta.server.session_replay.REPLAY_SCOPE_MAX_BYTES", 4096)
    gates = Gates()
    big = StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="x" * 8192)
    # a truncated reply keeps the provider stream open; its failed row installs at the end
    backend = StreamBackend(gates, [_end("kept-marker", truncated=True), "joined", big, "evicted"])
    serve = await _serve(tmp_path, transport, backend, gates)
    try:
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await gates.reached("joined")
        client, _history = await serve.joined()
        gates.release("joined")
        # the large frame passes the scope's byte bound while the reply is held
        await gates.reached("evicted")
        history = await serve.join(client)
        gates.release("evicted")
        await _event(serve.controller, "agent_end")
        frames = await serve.live(client)
        ended = [frame.get("reason") for frame in frames if frame["event"] == "replay_stream_ended"]
        assert ended == ["evicted"], frames
        assert _count(history, frames, "kept-marker") == 1
    finally:
        await serve.close()
