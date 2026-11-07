"""Real-transport coverage for bounded per-connection serve output."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

import pytest

from tests import test_server_websocket as websocket
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import TIMEOUT, _socket_path
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.server import ZetaServer
from zeta.server import connections as connections_module
from zeta.server import server as server_module
from zeta.server.protocol import ErrorDescriptor, EventDescriptor
from zeta.server.server import _Client

Transport = Literal["unix", "websocket"]
Role = Literal["controller", "observer"]


async def _connect(
    server: ZetaServer,
    transport: Transport,
    role: Role,
    *,
    start: bool = True,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if start:
        await server.start()
    if transport == "unix":
        return await asyncio.open_unix_connection(str(server.socket_path))
    assert server._websocket is not None
    token = (
        server._websocket.controller_token
        if role == "controller"
        else server._websocket.observer_token
    )
    reader, writer, status, _headers = await websocket._connect(
        server, token=None, protocol_token=token
    )
    assert status == 101
    return reader, writer


async def _send_json(
    writer: asyncio.StreamWriter, transport: Transport, value: object
) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    writer.write(
        payload + b"\n" if transport == "unix" else websocket._masked_frame(payload)
    )
    await writer.drain()


async def _read_json(
    reader: asyncio.StreamReader, transport: Transport
) -> dict[str, Any]:
    if transport == "unix":
        payload = await asyncio.wait_for(reader.readline(), TIMEOUT)
        assert payload
    else:
        opcode, payload = await websocket._read_frame(reader)
        assert opcode == 1
    return json.loads(payload)


async def _rpc(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    transport: Transport,
    request_id: int,
    method: str,
    params: dict[str, object] | None = None,
) -> dict[str, Any]:
    await _send_json(
        writer,
        transport,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        },
    )
    while True:
        frame = await _read_json(reader, transport)
        if frame.get("id") == request_id:
            return frame


async def _event(
    reader: asyncio.StreamReader, transport: Transport, name: str
) -> dict[str, Any]:
    while True:
        frame = await _read_json(reader, transport)
        params = frame.get("params", {})
        if params.get("event") == name:
            return params


async def _read_or_close(
    reader: asyncio.StreamReader, transport: Transport
) -> dict[str, Any] | None:
    if transport == "unix":
        payload = await asyncio.wait_for(reader.readline(), TIMEOUT)
        return json.loads(payload) if payload else None
    try:
        opcode, payload = await websocket._read_frame(reader)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    if opcode == 8:
        return None
    assert opcode == 1
    return json.loads(payload)


async def _read_until_close(
    reader: asyncio.StreamReader, transport: Transport
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        if transport == "unix":
            payload = await asyncio.wait_for(reader.readline(), TIMEOUT)
            if not payload:
                return frames
        else:
            try:
                opcode, payload = await websocket._read_frame(reader)
            except (asyncio.IncompleteReadError, ConnectionError):
                return frames
            if opcode == 8:
                return frames
            assert opcode == 1
        frames.append(json.loads(payload))


async def _ready_connection(
    tmp_path: Path, transport: Transport, role: Role
) -> tuple[
    ZetaServer,
    asyncio.StreamReader,
    asyncio.StreamWriter,
    Any,
]:
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
    )
    reader, writer = await _connect(server, transport, role)
    params: dict[str, object] = {"protocol_version": "1.2", "role": role}
    if role == "observer":
        params["features"] = ["serve_observers"]
    await _send_json(
        writer,
        transport,
        {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": params},
    )
    assert (await _read_json(reader, transport))["id"] == 1
    state = next(
        state
        for state in server.connections._connections.values()
        if state.role == role
    )
    return server, reader, writer, state


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_unregistered_descriptors_do_not_break_live_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
) -> None:
    server, reader, writer, state = await _ready_connection(
        tmp_path, transport, "controller"
    )
    try:
        invalid_event = await asyncio.wait_for(
            asyncio.gather(
                server.connections.publish_controller("unregistered", "session"),
                return_exceptions=True,
            ),
            10,
        )
        assert invalid_event == [False]

        assert await server.connections.publish_controller(
            EventDescriptor.ASSISTANT_DELTA, "session", delta="still-open"
        )
        event = await _read_json(reader, transport)
        assert event["params"]["delta"] == "still-open"

        client = state.connection
        original_dispatch = client._dispatch

        async def fail_dispatch(
            request_id: str | int | None, method: str, params: dict[str, Any]
        ) -> object:
            if method == "ping":
                raise RuntimeError("injected failure")
            return await original_dispatch(request_id, method, params)

        # test-contract: seam force dispatch failure through the real transport
        monkeypatch.setattr(client, "_dispatch", fail_dispatch)
        with monkeypatch.context() as context:
            invalid_descriptors = type(
                "Invalid", (), {"INTERNAL_ERROR": "unregistered"}
            )
            context.setattr(server_module, "ErrorDescriptor", invalid_descriptors)
            await _send_json(
                writer,
                transport,
                {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
            )
            error = await _read_json(reader, transport)
        assert error["id"] == 2
        assert error["error"]["code"] == ErrorDescriptor.INTERNAL_ERROR.code

        await _send_json(
            writer,
            transport,
            {"jsonrpc": "2.0", "id": 3, "method": "status", "params": {}},
        )
        assert (await _read_json(reader, transport))["id"] == 3
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


def _retained(output: Any) -> tuple[int, int]:
    frames = len(output.queue) + (output.current_frame is not None)
    frame_count = (
        output.retained_frames if hasattr(output, "retained_frames") else frames
    )
    byte_count = (
        output.retained_bytes
        if hasattr(output, "retained_bytes")
        else output.queued_bytes
    )
    return frame_count, byte_count


def _offer(server: ZetaServer, state: Any, role: Role, label: str) -> Awaitable[bool]:
    if role == "controller":
        return server.connections.publish_controller(
            EventDescriptor.ASSISTANT_DELTA, "session", delta=label
        )
    return server.connections.send(
        state.connection,
        state.connection.codec.notification(
            EventDescriptor.ASSISTANT_DELTA, "session", delta=label
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
@pytest.mark.parametrize("role", ["controller", "observer"])
@pytest.mark.parametrize("bound", ["frames", "bytes"])
async def test_ready_output_overflow_uses_shared_bound_and_reserved_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
    role: Role,
    bound: str,
) -> None:
    server, reader, writer, state = await _ready_connection(tmp_path, transport, role)
    output = state.output
    assert output is not None
    control = server.connections._overflow_frame(role)
    payload = state.connection.codec.notification
    byte_limit = (
        len(payload(EventDescriptor.ASSISTANT_DELTA, "session", delta="head"))
        + len(payload(EventDescriptor.ASSISTANT_DELTA, "session", delta="tail"))
        + len(control)
    )
    monkeypatch.setattr(
        connections_module,
        "OUTPUT_MAX_FRAMES",
        3 if bound == "frames" else 256,
        raising=False,
    )
    monkeypatch.setattr(
        connections_module,
        "OUTPUT_MAX_BYTES",
        16_384 if bound == "frames" else byte_limit,
        raising=False,
    )
    head_started = asyncio.Event()
    release_head = asyncio.Event()
    original_write = _Client._write_physical

    async def gated_write(client: _Client, payload: bytes) -> bool:
        decoded = json.loads(payload)
        if decoded.get("params", {}).get("delta") == "head":
            head_started.set()
            await release_head.wait()
        return await original_write(client, payload)

    # test-contract: seam stalls the real transport at a deterministic write
    monkeypatch.setattr(_Client, "_write_physical", gated_write)
    try:
        head = asyncio.create_task(_offer(server, state, role, "head"))
        await head_started.wait()
        tail_started = asyncio.Event()

        async def offer_tail() -> bool:
            tail_started.set()
            return await _offer(server, state, role, "tail")

        tail = asyncio.create_task(offer_tail())
        await tail_started.wait()
        assert _retained(output)[0] == 2
        trigger_started = asyncio.Event()

        async def offer_trigger() -> bool:
            trigger_started.set()
            return await _offer(server, state, role, "trigger")

        trigger = asyncio.create_task(offer_trigger())
        await trigger_started.wait()
        later_started = asyncio.Event()

        async def offer_later() -> bool:
            later_started.set()
            return await _offer(server, state, role, "later")

        later = asyncio.create_task(offer_later())
        await later_started.wait()
        release_head.set()
        results = await asyncio.gather(
            head, tail, trigger, later, return_exceptions=True
        )

        first = await _read_json(reader, transport)
        assert first["params"]["delta"] == "head"
        second = await _read_json(reader, transport)
        assert second == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32011, "message": f"{role} output overflow"},
        }
        assert results == [True, False, False, False]
        retained = _retained(output)
        assert retained[0] <= connections_module.OUTPUT_MAX_FRAMES
        assert retained[1] <= connections_module.OUTPUT_MAX_BYTES
        assert await _read_until_close(reader, transport) == []
        assert _retained(output) == (0, 0)
    finally:
        release_head.set()
        writer.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_controller_overflow_does_not_cancel_active_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
) -> None:
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_FRAMES", 3, raising=False)
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_BYTES", 16_384, raising=False)
    backend = FakeBackend([ScriptedTurn([TextContent("completed")])])
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
        backend_factory=lambda provider, model, home: (
            backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(server, transport, "controller")
    head_started = asyncio.Event()
    release_head = asyncio.Event()
    controller_released = asyncio.Event()
    turn_event_queued = asyncio.Event()
    turn_finalized = asyncio.Event()
    outcomes: list[str] = []
    finalized_turn_ids: list[object] = []
    original_write = _Client._write_physical
    original_release = server.connections.release
    original_enqueue = server.connections._enqueue
    original_finalize = server.control.deliveries.finalize_turn

    async def gated_write(client: _Client, payload: bytes) -> bool:
        frame = json.loads(payload)
        if frame.get("params", {}).get("delta") == "head":
            head_started.set()
            await release_head.wait()
        return await original_write(client, payload)

    def release(connection: Any) -> None:
        if server.connections.is_controller(connection):
            controller_released.set()
        original_release(connection)

    def observe_enqueue(output: Any, frame: Any, *, wake: bool = True) -> None:
        original_enqueue(output, frame, wake=wake)
        value = json.loads(frame.payload)
        if value.get("params", {}).get("event") == "agent_start":
            turn_event_queued.set()

    async def record_finalize(
        session_id: str,
        state: Any,
        *args: Any,
        schedule_wake: bool,
    ) -> None:
        turn_id, outcome = args if len(args) == 2 else (None, args[0])
        finalized_turn_ids.append(turn_id)
        outcomes.append(str(outcome))
        await original_finalize(session_id, state, *args, schedule_wake=schedule_wake)
        turn_finalized.set()

    # test-contract: seam stalls the real transport at a deterministic write
    monkeypatch.setattr(_Client, "_write_physical", gated_write)
    monkeypatch.setattr(server.connections, "release", release)
    # test-contract: seam identifies when the active turn publication is queued
    monkeypatch.setattr(server.connections, "_enqueue", observe_enqueue)
    monkeypatch.setattr(server.control.deliveries, "finalize_turn", record_finalize)
    try:
        await _send_json(
            writer,
            transport,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {
                    "protocol_version": "1.2",
                    "role": "controller",
                    "features": ["served_turn_end"],
                },
            },
        )
        assert (await _read_json(reader, transport))["id"] == 1
        await _send_json(
            writer,
            transport,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": {"provider": "codex"},
            },
        )
        created = await _read_json(reader, transport)
        session_id = created["result"]["session"]["session_id"]

        head = asyncio.create_task(
            server.connections.publish_controller(
                EventDescriptor.ASSISTANT_DELTA, session_id, delta="head"
            )
        )
        await asyncio.wait_for(head_started.wait(), TIMEOUT)
        server.control.start_turn("finish despite output overload")
        await asyncio.wait_for(turn_event_queued.wait(), TIMEOUT)
        trigger_started = asyncio.Event()

        async def trigger_overflow() -> bool:
            trigger_started.set()
            return await server.connections.publish_controller(
                EventDescriptor.ASSISTANT_DELTA, session_id, delta="trigger"
            )

        trigger = asyncio.create_task(trigger_overflow())
        await trigger_started.wait()
        assert trigger.done() and await trigger is False
        release_head.set()
        frames = await _read_until_close(reader, transport)
        assert frames == [
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "event": EventDescriptor.ASSISTANT_DELTA,
                    "session_id": session_id,
                    "delta": "head",
                },
            },
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32011,
                    "message": "controller output overflow",
                },
            },
        ]
        assert await head is True
        await asyncio.wait_for(turn_finalized.wait(), TIMEOUT)
        assert outcomes == ["disconnected"]
        await asyncio.wait_for(controller_released.wait(), TIMEOUT)

        reconnect_reader, reconnect_writer = await _connect(
            server, transport, "controller", start=False
        )
        try:
            await _send_json(
                reconnect_writer,
                transport,
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "hello",
                    "params": {
                        "protocol_version": "1.2",
                        "role": "controller",
                        "features": ["served_turn_end"],
                    },
                },
            )
            assert (await _read_json(reconnect_reader, transport))["id"] == 4
            await _send_json(
                reconnect_writer,
                transport,
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "status",
                    "params": {},
                },
            )
            status = (await _read_json(reconnect_reader, transport))["result"]
            assert status["session"]["session_id"] == session_id
            assert status["state"] == "idle"
            assert status["last_turn_outcome"] == {
                "turn_id": finalized_turn_ids[0],
                "reason": "disconnect",
            }
            await _send_json(
                reconnect_writer,
                transport,
                {
                    "jsonrpc": "2.0",
                    "id": 51,
                    "method": "status",
                    "params": {},
                },
            )
            repeated_status = (await _read_json(reconnect_reader, transport))["result"]
            assert repeated_status["last_turn_outcome"] == status["last_turn_outcome"]
            await _send_json(
                reconnect_writer,
                transport,
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "session_history",
                    "params": {"session_id": session_id},
                },
            )
            history = (await _read_json(reconnect_reader, transport))["result"][
                "messages"
            ]
            served = [row for row in history if row.get("event") == "served_turn_end"]
            assert len(served) == 1
            durable_turn_id = served[0]["data"].get("turn_id")
            assert isinstance(durable_turn_id, str)
            assert durable_turn_id == finalized_turn_ids[0]
            assert served[0]["data"]["reason"] == "disconnect"
        finally:
            reconnect_writer.close()
    finally:
        release_head.set()
        writer.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_active_flush_overflow_sends_control_then_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
) -> None:
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_FRAMES", 2, raising=False)
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_BYTES", 16_384, raising=False)
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
    )
    reader, writer = await _connect(server, transport, "controller")
    hello_started = asyncio.Event()
    release_hello = asyncio.Event()
    original_write = _Client._write_physical

    async def gated_write(client: _Client, payload: bytes) -> bool:
        if json.loads(payload).get("id") == 1:
            hello_started.set()
            await release_hello.wait()
        return await original_write(client, payload)

    # test-contract: seam stalls the real transport during the hello flush
    monkeypatch.setattr(_Client, "_write_physical", gated_write)
    try:
        await _send_json(
            writer,
            transport,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "hello",
                "params": {"protocol_version": "1.2"},
            },
        )
        await hello_started.wait()
        state = next(iter(server.connections._connections.values()))
        output = state.output
        assert output is not None
        results = [
            await server.connections.publish_controller(
                EventDescriptor.ASSISTANT_DELTA, "session", delta=f"tail-{index}"
            )
            for index in range(257)
        ]
        later = await server.connections.publish_controller(
            EventDescriptor.ASSISTANT_DELTA, "session", delta="later"
        )

        await _send_json(
            writer,
            transport,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "list_sessions",
                "params": {},
            },
        )
        release_hello.set()
        assert (await _read_json(reader, transport))["id"] == 1
        error = await _read_json(reader, transport)
        assert error == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32011, "message": "controller output overflow"},
        }
        assert results[0] is False
        assert all(result is False for result in results)
        assert later is False
        assert _retained(output)[0] <= 2
        assert _retained(output)[1] <= 16_384
        assert await _read_until_close(reader, transport) == []
        assert _retained(output) == (0, 0)
    finally:
        release_hello.set()
        writer.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_overflow_deadline_closes_without_promising_error_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
) -> None:
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_FRAMES", 2, raising=False)
    monkeypatch.setattr(connections_module, "OUTPUT_MAX_BYTES", 16_384, raising=False)
    deadline_armed = asyncio.Event()
    release_deadline = asyncio.Event()
    deadline_fired = asyncio.Event()
    deadline_calls = 0

    async def controlled_deadline() -> None:
        nonlocal deadline_calls
        deadline_calls += 1
        deadline_armed.set()
        await release_deadline.wait()
        deadline_fired.set()

    # test-contract: seam fires the production deadline without a wall clock
    monkeypatch.setattr(
        connections_module,
        "_overflow_deadline_wait",
        controlled_deadline,
        raising=False,
    )
    server, reader, writer, state = await _ready_connection(
        tmp_path, transport, "controller"
    )
    output = state.output
    assert output is not None
    head_started = asyncio.Event()
    release_head = asyncio.Event()
    original_write = _Client._write_physical

    async def gated_write(client: _Client, payload: bytes) -> bool:
        decoded = json.loads(payload)
        if decoded.get("params", {}).get("delta") == "head":
            head_started.set()
            await release_head.wait()
        return await original_write(client, payload)

    # test-contract: seam stalls the real transport until the deadline fires
    monkeypatch.setattr(_Client, "_write_physical", gated_write)
    tasks: list[asyncio.Task[bool]] = []
    try:
        head = asyncio.create_task(_offer(server, state, "controller", "head"))
        tasks.append(head)
        await head_started.wait()

        async def started_offer(label: str, started: asyncio.Event) -> bool:
            started.set()
            return await _offer(server, state, "controller", label)

        trigger_started = asyncio.Event()
        trigger = asyncio.create_task(started_offer("trigger", trigger_started))
        tasks.append(trigger)
        await trigger_started.wait()
        later_started = asyncio.Event()
        later = asyncio.create_task(started_offer("later", later_started))
        tasks.append(later)
        await later_started.wait()

        if trigger.done():
            await deadline_armed.wait()
            deadline_task = output.overflow_deadline_task
            assert deadline_task is not None
            release_deadline.set()
            await deadline_fired.wait()
            await deadline_task
        else:
            release_head.set()

        peer_result = await _read_or_close(reader, transport)
        assert peer_result is None, (
            f"peer received output after overflow: {peer_result}"
        )
        assert deadline_calls == 1
        assert trigger.done() and await trigger is False
        assert later.done() and await later is False
        await asyncio.gather(head, return_exceptions=True)
        assert _retained(output) == (0, 0)
    finally:
        release_deadline.set()
        release_head.set()
        writer.close()
        await server.close()
        for task in tasks:
            task.cancel()
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), 10
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_status_outcome_follows_active_branch_and_connection_feature(
    tmp_path: Path,
    transport: Transport,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first result")]),
            ScriptedTurn([TextContent("fork result")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    controller_reader, controller_writer = await _connect(
        server, transport, "controller"
    )
    observer_writer: asyncio.StreamWriter | None = None
    legacy_writer: asyncio.StreamWriter | None = None
    try:
        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "controller",
                "features": ["served_turn_end"],
            },
        )
        created = await _rpc(
            controller_reader,
            controller_writer,
            transport,
            2,
            "new_session",
            {"provider": "codex"},
        )
        session_id = created["result"]["session"]["session_id"]

        observer_reader, observer_writer = await _connect(
            server, transport, "observer", start=False
        )
        await _rpc(
            observer_reader,
            observer_writer,
            transport,
            3,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "served_turn_end"],
            },
        )
        legacy_reader, legacy_writer = await _connect(
            server, transport, "observer", start=False
        )
        await _rpc(
            legacy_reader,
            legacy_writer,
            transport,
            4,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers"],
            },
        )

        empty = await _rpc(controller_reader, controller_writer, transport, 5, "status")
        assert "last_turn_outcome" in empty["result"]
        assert empty["result"]["last_turn_outcome"] is None
        legacy_empty = await _rpc(legacy_reader, legacy_writer, transport, 6, "status")
        assert "last_turn_outcome" not in legacy_empty["result"]

        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            7,
            "send",
            {"text": "first"},
        )
        started = await _event(controller_reader, transport, "turn_start")
        first_turn_id = started["data"]["turn_id"]
        await _event(controller_reader, transport, "agent_end")
        first_status = (
            await _rpc(controller_reader, controller_writer, transport, 8, "status")
        )["result"]
        first_outcome = {"turn_id": first_turn_id, "reason": "completed"}
        assert first_status["last_turn_outcome"] == first_outcome
        observer_status = (
            await _rpc(observer_reader, observer_writer, transport, 9, "status")
        )["result"]
        assert observer_status["last_turn_outcome"] == first_outcome
        legacy_status = (
            await _rpc(legacy_reader, legacy_writer, transport, 10, "status")
        )["result"]
        assert "last_turn_outcome" not in legacy_status

        history = await _rpc(
            controller_reader,
            controller_writer,
            transport,
            11,
            "session_history",
            {"session_id": session_id},
        )
        user_id = next(
            row["id"]
            for row in history["result"]["messages"]
            if row.get("role") == "user"
        )
        tree = await _rpc(
            controller_reader,
            controller_writer,
            transport,
            12,
            "session_tree",
            {"session_id": session_id},
        )
        original_leaf = tree["result"]["branches"][0]["id"]
        forked = await _rpc(
            controller_reader,
            controller_writer,
            transport,
            13,
            "fork_message",
            {"session_id": session_id, "message_id": user_id},
        )
        fork_leaf = next(
            branch["id"] for branch in forked["result"]["branches"] if branch["current"]
        )
        fork_status = (
            await _rpc(controller_reader, controller_writer, transport, 14, "status")
        )["result"]
        assert fork_status["last_turn_outcome"] is None

        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            15,
            "switch_branch",
            {"session_id": session_id, "head_id": original_leaf},
        )
        restored = (
            await _rpc(controller_reader, controller_writer, transport, 16, "status")
        )["result"]
        assert restored["last_turn_outcome"] == first_outcome
        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            17,
            "switch_branch",
            {"session_id": session_id, "head_id": fork_leaf},
        )
        selected_fork = (
            await _rpc(controller_reader, controller_writer, transport, 18, "status")
        )["result"]
        assert selected_fork["last_turn_outcome"] is None

        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            19,
            "send",
            {"text": "fork"},
        )
        fork_started = await _event(controller_reader, transport, "turn_start")
        await _event(controller_reader, transport, "agent_end")
        fork_outcome = (
            await _rpc(controller_reader, controller_writer, transport, 20, "status")
        )["result"]["last_turn_outcome"]
        assert fork_outcome == {
            "turn_id": fork_started["data"]["turn_id"],
            "reason": "completed",
        }
        assert fork_outcome != first_outcome

        observer_writer.close()
        await observer_writer.wait_closed()
        observer_reader, observer_writer = await _connect(
            server, transport, "observer", start=False
        )
        await _rpc(
            observer_reader,
            observer_writer,
            transport,
            21,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "served_turn_end"],
            },
        )
        reconnected = (
            await _rpc(observer_reader, observer_writer, transport, 22, "status")
        )["result"]
        assert reconnected["last_turn_outcome"] == fork_outcome
    finally:
        controller_writer.close()
        if observer_writer is not None:
            observer_writer.close()
        if legacy_writer is not None:
            legacy_writer.close()
        await server.close()


@pytest.mark.asyncio
async def test_status_projects_foreign_turn_end_after_resident_append(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        provider="codex",
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )
    reader, writer = await _connect(server, "unix", "controller")
    foreign: ConversationStore | None = None
    try:
        await _rpc(
            reader,
            writer,
            "unix",
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "controller",
                "features": ["served_turn_end"],
            },
        )
        created = await _rpc(
            reader,
            writer,
            "unix",
            2,
            "new_session",
            {"provider": "codex"},
        )
        session_id = created["result"]["session"]["session_id"]
        assert server.runtime.opened is not None
        resident = server.runtime.opened.store
        foreign = ConversationStore(resident.root_dir, session_id=session_id)
        await foreign.append_turn_end_async("1" * 32, "error")

        resident.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("resident append")])
        )

        status = (await _rpc(reader, writer, "unix", 3, "status"))["result"]
        assert status["last_turn_outcome"] == {
            "turn_id": "1" * 32,
            "reason": "error",
        }
    finally:
        if foreign is not None:
            foreign.close()
        writer.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_observer_status_rejects_session_swap_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        port=None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    controller_reader, controller_writer = await _connect(
        server, transport, "controller"
    )
    observer_writer: asyncio.StreamWriter | None = None
    stale: asyncio.Task[dict[str, Any]] | None = None
    release_projection = threading.Event()
    try:
        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "controller",
                "features": ["served_turn_end"],
            },
        )
        session_a = (
            await _rpc(
                controller_reader,
                controller_writer,
                transport,
                2,
                "new_session",
                {"provider": "codex"},
            )
        )["result"]["session"]["session_id"]
        await _rpc(
            controller_reader,
            controller_writer,
            transport,
            3,
            "send",
            {"text": "finish"},
        )
        await _event(controller_reader, transport, "agent_end")

        observer_reader, observer_writer = await _connect(
            server, transport, "observer", start=False
        )
        await _rpc(
            observer_reader,
            observer_writer,
            transport,
            4,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "served_turn_end"],
            },
        )
        observed_a = (
            await _rpc(observer_reader, observer_writer, transport, 5, "status")
        )["result"]
        assert observed_a["session"]["session_id"] == session_a
        assert observed_a["last_turn_outcome"]["reason"] == "completed"

        projection_started = threading.Event()
        original_project_status = connections_module.project_status

        def gated_project_status(target: object, **kwargs: object) -> dict[str, Any]:
            projection_started.set()
            assert release_projection.wait(timeout=10), (
                "status projection was not released"
            )
            return original_project_status(target, **kwargs)

        # test-contract: seam pauses the off-loop projection after capture
        monkeypatch.setattr(connections_module, "project_status", gated_project_status)
        stale = asyncio.create_task(
            _rpc(observer_reader, observer_writer, transport, 6, "status")
        )
        try:
            projection_ready = await asyncio.wait_for(
                asyncio.to_thread(projection_started.wait, 10), TIMEOUT
            )
        except TimeoutError:
            release_projection.set()
            stale.cancel()
            await asyncio.wait_for(
                asyncio.gather(stale, return_exceptions=True), 10
            )
            raise
        if not projection_ready:
            release_projection.set()
            stale.cancel()
            await asyncio.wait_for(
                asyncio.gather(stale, return_exceptions=True), 10
            )
            pytest.fail("status projection did not start")
        session_b = (
            await _rpc(
                controller_reader,
                controller_writer,
                transport,
                7,
                "new_session",
                {"provider": "codex"},
            )
        )["result"]["session"]["session_id"]
        release_projection.set()
        stale_result = await asyncio.wait_for(stale, 10)
        assert stale_result["error"]["code"] == -32003

        fresh = (await _rpc(observer_reader, observer_writer, transport, 8, "status"))[
            "result"
        ]
        assert fresh["session"]["session_id"] == session_b
        assert fresh["last_turn_outcome"] is None
    finally:
        release_projection.set()
        if stale is not None:
            stale.cancel()
            await asyncio.wait_for(
                asyncio.gather(stale, return_exceptions=True), 10
            )
        controller_writer.close()
        if observer_writer is not None:
            observer_writer.close()
        await server.close()
