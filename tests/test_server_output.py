"""Real-transport coverage for bounded per-connection serve output."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

import pytest

from tests import test_server_websocket as websocket
from tests.test_server import TIMEOUT, _socket_path
from zeta.server import ZetaServer
from zeta.server import connections as connections_module
from zeta.server.server import _Client

Transport = Literal["unix", "websocket"]
Role = Literal["controller", "observer"]


async def _connect(
    server: ZetaServer, transport: Transport, role: Role
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
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
            "assistant_delta", "session", delta=label
        )
    return server.connections.send(
        state.connection,
        state.connection.codec.notification("assistant_delta", "session", delta=label),
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
        len(payload("assistant_delta", "session", delta="head"))
        + len(payload("assistant_delta", "session", delta="tail"))
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
        assert results[2:] == [False, False]
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
                "assistant_delta", "session", delta=f"tail-{index}"
            )
            for index in range(257)
        ]
        later = await server.connections.publish_controller(
            "assistant_delta", "session", delta="later"
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
        await asyncio.gather(*tasks, return_exceptions=True)
