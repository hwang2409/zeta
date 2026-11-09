from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_server_websocket as websocket
from tests.support.serve_close import close_serve
from tests.support.server_client import event, request
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer

TIMEOUT = 30
Transport = str


async def _connect_transport(
    server: ZetaServer, transport: Transport
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    await asyncio.wait_for(server.start(), TIMEOUT)
    if transport == "unix":
        return await asyncio.wait_for(
            asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
        )
    assert server._websocket is not None
    reader, writer, status, _ = await websocket._connect(
        server, token=None, protocol_token=server._websocket.controller_token
    )
    assert status == 101
    return reader, writer


async def _send_transport(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    transport: Transport,
    value: dict[str, object],
) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    writer.write(payload + b"\n" if transport == "unix" else websocket._masked_frame(payload))
    await asyncio.wait_for(writer.drain(), TIMEOUT)


# test-contract: transport websocket frames need the raw frame decoder; unix uses the same helper for both transports.
async def _read_transport(
    reader: asyncio.StreamReader, transport: Transport
) -> dict[str, object]:
    if transport == "unix":
        # test-contract: transport raw unix socket reader pairs with websocket frames.
        line = await asyncio.wait_for(reader.readline(), TIMEOUT)
        assert line
        return json.loads(line)
    opcode, payload = await asyncio.wait_for(websocket._read_frame(reader), TIMEOUT)
    assert opcode == 1
    return json.loads(payload)


async def _rpc_transport(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    transport: Transport,
    request_id: str,
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    await _send_transport(
        reader,
        writer,
        transport,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    while True:
        frame = await _read_transport(reader, transport)
        if frame.get("id") == request_id:
            return frame


async def _event_transport(
    reader: asyncio.StreamReader, transport: Transport, name: str
) -> dict[str, object]:
    while True:
        frame = await _read_transport(reader, transport)
        if frame.get("params", {}).get("event") == name:  # type: ignore[union-attr]
            return frame



async def _connect(server: ZetaServer) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    await asyncio.wait_for(server.start(), TIMEOUT)
    assert server.port is not None
    return await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
    )


@pytest.mark.asyncio
async def test_run_work_lists_and_cancels_real_task_and_publishes_todo(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        hello = (
            await request(
                reader,
                writer,
                "hello",
                "hello",
                {"protocol_version": "1.2", "features": ["run_work"]},
            )
        )[-1]
        session = (
            await request(
                reader, writer, "new", "new_session", {"provider": "codex"}
            )
        )[-1]["result"]["session"]
        session_id = session["session_id"]
        registry = server.runtime.loop.tool_registry.background_tasks
        command = (
            f"{shlex.quote(sys.executable)} -c "
            + shlex.quote("import sys; sys.stdin.buffer.read()")
        )
        task_id, _pid = await registry.start(command, tmp_path)
        todo = [{"content": "finish transport proof", "status": "in_progress"}]
        server.runtime.opened.store.set_todo_items(todo)

        list_response = (
            await request(
                reader,
                writer,
                "list",
                "run_work_list",
                {"session_id": session_id},
            )
        )[-1]
        assert isinstance(list_response.get("result"), dict), (
            f"run_work_list returned no snapshot: {list_response}"
        )
        listed = list_response["result"]
        assert listed["todo"] == todo
        assert listed["tasks"][0]["id"] == task_id
        assert listed["tasks"][0]["status"] == "running"
        assert listed["tasks"][0]["cancelable"] is True
        assert listed["tasks"][0]["elapsed_seconds"] >= 0
        assert "run_work" in hello["result"]["capabilities"]["features"]

        started = await event(reader, "background_task_updated")
        assert started["task"] == {
            "id": task_id,
            "status": "starting",
            "command": command,
            "elapsed_seconds": started["task"]["elapsed_seconds"],
            "cancelable": True,
        }
        assert started["task"]["elapsed_seconds"] >= 0
        assert (await event(reader, "todo_updated"))["todo"] == todo

        canceled = (
            await request(
                reader,
                writer,
                "cancel",
                "background_task_cancel",
                {"session_id": session_id, "task_id": task_id},
            )
        )[-1]["result"]["task"]
        assert canceled["id"] == task_id
        assert canceled["status"] == "killed"
        assert canceled["cancelable"] is False
        stopping = await event(reader, "background_task_updated")
        assert stopping["task"]["status"] == "stopping"
        terminal = await event(reader, "background_task_updated")
        assert terminal["task"] == canceled
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_todo_write_after_closed_server_loop_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    await request(
        reader,
        writer,
        "hello",
        "hello",
        {"protocol_version": "1.2", "features": ["run_work"]},
    )
    await request(reader, writer, "new", "new_session", {"provider": "codex"})
    store = server.runtime.opened.store
    # test-contract: seam this is the post-commit edge that can race server teardown
    todo_committed = store._todo_change_sink
    assert todo_committed is not None
    await close_serve(server, writer)

    notifications: list[object] = []

    async def record_notification(*args: object, **kwargs: object) -> None:
        notifications.append((args, kwargs))

    # test-contract: seam the test runner loop stays open, so inject the closed serve loop
    monkeypatch.setattr(server.control, "_notify", record_notification)
    closed_loop = asyncio.new_event_loop()
    exception_contexts: list[dict[str, object]] = []
    closed_loop.set_exception_handler(
        lambda _loop, context: exception_contexts.append(context)
    )
    closed_loop.close()
    server.connections = SimpleNamespace(loop=closed_loop)  # type: ignore[assignment]

    todo_committed([{"content": "late write", "status": "pending"}])

    assert notifications == []
    assert exception_contexts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_run_work_real_transports_and_all_task_statuses(
    tmp_path: Path, transport: Transport
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=short_socket_path(tmp_path), port=None,
        ws_port=0 if transport == "websocket" else None, provider="codex"
    )
    reader, writer = await _connect_transport(server, transport)
    try:
        hello = await _rpc_transport(
            reader, writer, transport, "hello", "hello",
            {"protocol_version": "1.2", "features": ["run_work"]},
        )
        assert "run_work" in hello["result"]["capabilities"]["features"]  # type: ignore[index]
        created = await _rpc_transport(
            reader, writer, transport, "new", "new_session", {"provider": "codex"}
        )
        session_id = created["result"]["session"]["session_id"]  # type: ignore[index]
        registry = server.runtime.loop.tool_registry.background_tasks
        command = f"{shlex.quote(sys.executable)} -c " + shlex.quote("import sys; sys.stdin.buffer.read()")
        task_id, _ = await registry.start(command, tmp_path)
        started = await _event_transport(reader, transport, "background_task_updated")
        assert started["params"]["task"]["status"] == "starting"  # type: ignore[index]
        listed = await _rpc_transport(reader, writer, transport, "list", "run_work_list", {"session_id": session_id})
        assert listed["result"]["tasks"][0]["status"] == "running"  # type: ignore[index]
        await _send_transport(reader, writer, transport, {
            "jsonrpc": "2.0", "id": "cancel", "method": "background_task_cancel",
            "params": {"session_id": session_id, "task_id": task_id},
        })
        stopping = None
        killed = None
        canceled = None
        while stopping is None or canceled is None:
            frame = await _read_transport(reader, transport)
            if frame.get("params", {}).get("event") == "background_task_updated":  # type: ignore[union-attr]
                status = frame["params"]["task"]["status"]  # type: ignore[index]
                if status == "stopping":
                    stopping = frame
                elif status == "killed":
                    killed = frame
            elif frame.get("id") == "cancel":
                canceled = frame
        assert stopping is not None
        assert stopping["params"]["task"]["status"] == "stopping"  # type: ignore[index]
        assert canceled["result"]["task"]["status"] == "killed"  # type: ignore[index]
        if killed is None:
            killed = await _event_transport(reader, transport, "background_task_updated")
        assert killed["params"]["task"]["status"] == "killed"  # type: ignore[index]
        for exit_code, expected in ((0, "exited"), (3, "failed")):
            command = f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit({exit_code})'"
            await registry.start(command, tmp_path)
            await _event_transport(reader, transport, "background_task_updated")
            terminal = await _event_transport(reader, transport, "background_task_updated")
            assert terminal["params"]["task"]["status"] == expected  # type: ignore[index]
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_run_work_events_are_suppressed_without_negotiation_or_for_observer(
    tmp_path: Path, transport: Transport
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=short_socket_path(tmp_path), port=None,
        ws_port=0 if transport == "websocket" else None, provider="codex"
    )
    controller_reader, controller_writer = await _connect_transport(server, transport)
    try:
        await _rpc_transport(controller_reader, controller_writer, transport, "hello", "hello", {"protocol_version": "1.2", "features": []})
        await _rpc_transport(controller_reader, controller_writer, transport, "new", "new_session", {"provider": "codex"})
        await server.runtime.loop.tool_registry.background_tasks.start("true", tmp_path)
        sentinel = await _rpc_transport(controller_reader, controller_writer, transport, "sentinel", "status", {})
        assert sentinel["id"] == "sentinel"
    finally:
        await close_serve(server, controller_writer)

    server = ZetaServer(
        home=tmp_path / "observer", socket_path=short_socket_path(tmp_path / "observer"), port=None,
        ws_port=0 if transport == "websocket" else None, provider="codex"
    )
    controller_reader, controller_writer = await _connect_transport(server, transport)
    try:
        await _rpc_transport(controller_reader, controller_writer, transport, "hello", "hello", {"protocol_version": "1.2", "features": ["run_work"]})
        await _rpc_transport(controller_reader, controller_writer, transport, "new", "new_session", {"provider": "codex"})
        if transport == "unix":
            observer_reader, observer_writer = await asyncio.open_unix_connection(str(server.socket_path))
        else:
            assert server._websocket is not None
            observer_reader, observer_writer, status, _ = await websocket._connect(server, token=None, protocol_token=server._websocket.observer_token)
            assert status == 101
        try:
            await _rpc_transport(observer_reader, observer_writer, transport, "observer-hello", "hello", {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers", "run_work"]})
            await server.runtime.loop.tool_registry.background_tasks.start("true", tmp_path)
            sentinel = await _rpc_transport(observer_reader, observer_writer, transport, "observer-status", "status", {})
            assert sentinel["id"] == "observer-status"
        finally:
            await close_serve(server, observer_writer)
    finally:
        if not controller_writer.is_closing():
            await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_run_work_requires_negotiated_feature(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await request(
            reader,
            writer,
            "hello",
            "hello",
            {"protocol_version": "1.2", "features": []},
        )
        response = (
            await request(
                reader,
                writer,
                "list",
                "run_work_list",
                {"session_id": "missing"},
            )
        )[-1]
        assert response["error"]["code"] == -32601
    finally:
        await close_serve(server, writer)
