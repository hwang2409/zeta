"""A join inside the window between a row's install and its live frame sees it once.

Each test holds one real publisher after its history row is in the store, joins a
replay stream in that window, releases the publisher, and counts the effect in
the join's history plus the stream's live frames. Real ``zeta serve`` over a unix
socket and a websocket, real store; every hold waits with a deadline.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, PerActorFakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.protocol.types import StreamEventType, TextContent, ToolCall
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
FEATURES = ("delivery_id", "user_message_event", "user_message_delivery_id")


class Window:
    """Real serve with a controller and replay observers that join on demand."""

    def __init__(self, server: ZetaServer, transport: Transport) -> None:
        self.server = server
        self.transport = transport
        self.clients: list[ReplayClient] = []
        self.session_id = ""
        self.next_id = 100
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    def request_id(self) -> int:
        self.next_id += 1
        return self.next_id

    async def start(self) -> ReplayClient:
        await self.server.start()
        controller = await connect_replay(self.server, self.transport)
        self.clients.append(controller)
        await negotiate(controller, replay=False, extra=FEATURES, protocol_version="1.2")
        created = await controller.request(2, "new_session", {"provider": "codex"})
        self.session_id = created["result"]["session"]["session_id"]
        return controller

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

    async def held(self) -> None:
        """Run inside a gated publisher: report the window, then wait for release."""
        self.entered.set()
        await await_or_fail(self.release.wait(), what="release held publisher")

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
        self.release.set()
        for client in self.clients:
            await client.close()
        await self.server.close()


def _backend(backend: Any) -> Callable[..., tuple[Any, str]]:
    return lambda provider, model, home: (backend, model or "offline")


async def _event(client: ReplayClient, name: str, **match: Any) -> dict[str, Any]:
    while True:
        params = await client.event(name)
        if all(params.get(key) == value for key, value in match.items()):
            return params


def _first_seq(frames: list[dict[str, Any]]) -> int | None:
    return next((frame["seq"] for frame in frames if "seq" in frame), None)


def _read_call(tmp_path: Path, call_id: str) -> ToolCall:
    target = tmp_path / f"{call_id}.txt"
    target.write_text("data", encoding="utf-8")
    return ToolCall(call_id, "read", {"path": str(target)})


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("mode", ("idle", "after_newer_row", "during_join"))
async def test_join_while_send_echo_is_held_sees_message_once(
    tmp_path: Path, transport: Transport, mode: str
) -> None:
    window = Window(
        replay_server(tmp_path, transport, _backend(FakeBackend([ScriptedTurn([TextContent("ok")])]))),
        transport,
    )
    try:
        controller = await window.start()
        observer = await window.observer()
        original = window.server.control.user_message

        async def held(*args: Any, **kwargs: Any) -> None:
            await window.held()
            await original(*args, **kwargs)

        # test-contract: seam hold the echo after its delivery row installed
        window.server.control.user_message = held  # type: ignore[method-assign]
        send = asyncio.create_task(
            controller.request(3, "send", {"text": "marker-text", "delivery_id": "d-1"})
        )
        await await_or_fail(window.entered.wait(), what="delivery row installed")
        if mode == "after_newer_row":
            window.server.runtime.opened.store.append_task_notification(
                task_id="newer", command="true", exit_code=0
            )
        if mode == "during_join":
            connections = window.server.connections
            assert connections is not None
            read = connections.read_replay_history

            async def read_then_release_echo(*args: Any) -> Any:
                result = await read(*args)
                # the echo is admitted while this join is still pending
                window.release.set()
                await await_or_fail(send, what="send response")
                return result

            # test-contract: seam admit the echo between the join's history read and its end
            connections.read_replay_history = read_then_release_echo  # type: ignore[method-assign]
        history = await window.join(observer)
        window.release.set()
        await await_or_fail(send, what="send response")
        await _event(controller, "agent_end")
        frames = await window.live(observer)
        live = [frame.get("text") for frame in frames if frame["event"] == "user_message"]
        texts = [
            block.get("text")
            for row in history
            for block in row.get("content", [])
            if isinstance(block, dict)
        ]
        assert (texts + live).count("marker-text") == 1, f"history={texts} live={live}"
        # a frame the join's history already holds spends no seq
        assert _first_seq(frames) == 1, frames
    finally:
        await window.close()


def _background_backend(tmp_path: Path) -> PerActorFakeBackend:
    agent = ToolCall(
        "agent-bg", "agent", {"prompt": "child-work", "description": "child", "background": True}
    )
    return PerActorFakeBackend(
        [
            ScriptedTurn(tool_calls=[agent]),
            ScriptedTurn([TextContent("parent done")]),
            ScriptedTurn([TextContent("noted")]),
        ],
        {
            "child-work": [
                ScriptedTurn(tool_calls=[_read_call(tmp_path, "call-child")]),
                ScriptedTurn([TextContent("child done")]),
            ]
        },
        parent_prompt="delegate",
    )


def _terminal(frames: list[dict[str, Any]], notification_id: str) -> int:
    return sum(
        frame["event"] == "tool_end"
        and (frame.get("data") or {}).get("notification_id") == notification_id
        for frame in frames
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_background_terminal_card_reaches_each_stream_once(
    tmp_path: Path, transport: Transport
) -> None:
    window = Window(replay_server(tmp_path, transport, _backend(_background_backend(tmp_path))), transport)
    try:
        controller = await window.start()
        early = await window.observer()
        late = await window.observer()
        await window.join(early)
        control = window.server.control
        original = control._event
        held_ids: list[str] = []

        async def gated(event: Any, **kwargs: Any) -> None:
            if event.type is StreamEventType.TOOL_EXECUTION_END and "notification_id" in event.data:
                held_ids.append(event.data["notification_id"])
                await window.held()
            await original(event, **kwargs)

        # test-contract: seam hold the terminal card after its notification row installed
        control._event = gated  # type: ignore[method-assign]
        await controller.request(3, "send", {"text": "delegate"})
        request = await _event(controller, "approval_request", delegated=True)
        await controller.request(4, "approve", {"request_id": request["request_id"]})
        await await_or_fail(window.entered.wait(), what="child completion row installed")
        history = await window.join(late)
        window.release.set()
        notification_id = held_ids[0]
        while _terminal([await _event(controller, "tool_end")], notification_id) == 0:
            pass
        early_frames = await window.live(early)
        late_frames = await window.live(late)
        late_count = sum(row["id"] == notification_id for row in history) + _terminal(
            late_frames, notification_id
        )
        early_count = _terminal(early_frames, notification_id)
        assert early_count == 1, f"live terminal cards on a stream joined before: {early_count}"
        assert late_count == 1, f"history+live terminal cards for one completion: {late_count}"
    finally:
        await window.close()


def _approval_count(history: list[dict[str, Any]], frames: list[dict[str, Any]], request_id: str) -> int:
    return sum(
        row.get("event") == "approval_request" and row.get("request_id") == request_id for row in history
    ) + sum(
        frame["event"] == "approval_request" and frame.get("request_id") == request_id for frame in frames
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_join_while_delegated_approval_request_is_held_sees_it_once(
    tmp_path: Path, transport: Transport
) -> None:
    window = Window(replay_server(tmp_path, transport, _backend(_background_backend(tmp_path))), transport)
    try:
        controller = await window.start()
        observer = await window.observer()
        store = window.server.runtime.opened.store
        original = store.append_approval_history_async

        async def gated(payload: Any) -> Any:
            entry = await original(payload)
            if payload.get("event") == "approval_request":
                await window.held()
            return entry

        # test-contract: seam hold the request admission after its approval_history row installed
        store.append_approval_history_async = gated
        await controller.request(3, "send", {"text": "delegate"})
        await await_or_fail(window.entered.wait(), what="approval_history row installed")
        history = await window.join(observer)
        window.release.set()
        request = await _event(controller, "approval_request", delegated=True)
        frames = await window.live(observer)
        count = _approval_count(history, frames, request["request_id"])
        assert count == 1, f"history+live approval_request for one request: {count}"
    finally:
        await window.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_join_while_approval_request_is_held_sees_it_once(
    tmp_path: Path, transport: Transport
) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_read_call(tmp_path, "call-fg")]), ScriptedTurn([TextContent("done")])]
    )
    window = Window(replay_server(tmp_path, transport, _backend(backend)), transport)
    try:
        controller = await window.start()
        observer = await window.observer()
        control = window.server.control
        original = control._start_approval

        async def gated(*args: Any, **kwargs: Any) -> None:
            await window.held()
            await original(*args, **kwargs)

        # test-contract: seam hold the request admission after its assistant row installed
        control._start_approval = gated  # type: ignore[method-assign]
        await controller.request(3, "send", {"text": "run"})
        await await_or_fail(window.entered.wait(), what="assistant row installed")
        history = await window.join(observer)
        window.release.set()
        request = await _event(controller, "approval_request")
        frames = await window.live(observer)
        count = _approval_count(history, frames, request["request_id"])
        assert count == 1, f"history+live approval_request for one request: {count}"
    finally:
        await window.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_join_while_receipt_is_held_sees_it_once(tmp_path: Path, transport: Transport) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("ok")])])
    window = Window(replay_server(tmp_path, transport, _backend(backend)), transport)
    try:
        controller = await window.start()
        observer = await window.observer()
        notification = window.server.runtime.opened.store.append_agent_notification(
            "child-done", child_session_path="/tmp/child", description="child", status="completed", text="done"
        )
        control = window.server.control
        original = control._event

        async def gated(event: Any, **kwargs: Any) -> None:
            if event.type is StreamEventType.AGENT_NOTIFICATION:
                await window.held()
            await original(event, **kwargs)

        # test-contract: seam hold the receipt after the turn acknowledged its notification
        control._event = gated  # type: ignore[method-assign]
        await controller.request(3, "send", {"text": "go"})
        await await_or_fail(window.entered.wait(), what="receipt ready")
        history = await window.join(observer)
        window.release.set()
        await _event(controller, "sub_agent_receipt")
        frames = await window.live(observer)
        count = sum(row["id"] == notification.id for row in history) + sum(
            frame["event"] == "sub_agent_receipt"
            and (frame.get("data") or {}).get("notification_id") == notification.id
            for frame in frames
        )
        assert count == 1, f"history+live receipts for one notification: {count}"
    finally:
        await window.close()
