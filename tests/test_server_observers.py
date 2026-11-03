"""real serve-path tests for read-only observer connections."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.test_server import TIMEOUT, _close, _connect, _request, _socket_path
from zeta.server import ZetaServer
from zeta.server.connections import ConnectionRegistry


async def _observer(server: ZetaServer, number: int = 1, features: list[str] | None = None):
    if server._server is None:
        reader, writer = await _connect(server)
    elif server.port is None:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    else:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    await _request(reader, writer, number, "hello", {
        "protocol_version": "1.2", "role": "observer",
        "features": features or ["serve_observers"],
    })
    return reader, writer


@pytest.mark.asyncio
async def test_observer_project_and_account_requests_are_read_only(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    try:
        methods = [
            ("list_projects", {}), ("project_show", {"project_id": "p"}),
            ("project_memory_log", {"project_id": "p"}), ("project_inbox", {"project_id": "p"}),
            ("account_list", {}), ("account_refresh", {"provider": "codex"}),
            ("account_switch", {"account": "codex:x"}), ("list_sessions", {"project_id": "p"}),
            ("slash_list", {}), ("login_status", {}),
        ]
        for number, (method, params) in enumerate(methods, 3):
            response = (await _request(reader, writer, number, method, params))[-1]
            assert response["error"]["code"] == -32008
            assert response["error"]["data"]["code"] == "observer_read_only"
        assert (await _request(reader, writer, 99, "status"))[-1]["result"]["session"]
    finally:
        await _close(server, controller_writer)
        writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_hello_has_no_controller_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    calls: list[str] = []
    for name in ("set_background_event_sink", "set_memory_notice_sink", "set_background_wake_sink", "set_post_stream_provider_retry"):
        original = getattr(server.runtime, name)
        monkeypatch.setattr(server.runtime, name, lambda *args, _name=name, _original=original, **kwargs: calls.append(_name) or _original(*args, **kwargs))
    reader, writer = await _observer(server)
    try:
        assert calls == []
        assert (await _request(reader, writer, 2, "status"))[-1]["result"]["session"] is None
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_controller_frames_are_byte_identical_with_observer(tmp_path: Path) -> None:
    async def hello(home: Path, observer: bool) -> bytes:
        server = ZetaServer(home=home, socket_path=_socket_path(home), provider="codex")
        reader, writer = await _connect(server)
        try:
            request = {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.2"}}
            writer.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
            await writer.drain()
            hello_frame = await reader.readline()
            assert hello_frame
            if observer:
                extra = await _observer(server, 10)
            request = {"jsonrpc": "2.0", "id": 2, "method": "status", "params": {}}
            writer.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
            await writer.drain()
            status_frame = await reader.readline()
            assert status_frame
            if observer:
                extra[1].close(); await extra[1].wait_closed()
            return hello_frame + status_frame
        finally:
            await _close(server, writer)
    assert await hello(tmp_path / "base", False) == await hello(tmp_path / "observer", True)




@pytest.mark.asyncio
async def test_controller_only_raw_frames_match_origin_main(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        for request in (
            {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.2", "features": ["ping"]}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "missing", "params": {}},
        ):
            writer.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
            await writer.drain()
        assert [await reader.readline() for _ in range(3)] == [
            b'{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"1.2","server":"zeta","capabilities":{"requests":["list_sessions","new_session","resume","send","steer","approve","deny","abort","status","rename_session","delete_session","session_tree","session_history","switch_branch","fork_message","model_catalog","session_settings","set_settings","set_compaction","send_images","slash_list","slash_run","account_list","account_logout","account_refresh","account_switch","login_start","login_status","login_cancel","login_providers","ping"],"notifications":["event"],"features":["ping"]}}}\n',
            b'{"jsonrpc":"2.0","id":2,"result":{"pong":true}}\n',
            b'{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"method not found: missing"}}\n',
        ]
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_registry_release_follows_flush_and_blocks_late_output() -> None:
    events: list[str] = []
    class FakeConnection:
        async def _write_physical(self, frame: bytes) -> bool:
            events.append(f"write:{frame!r}")
            return True
        async def close(self) -> None:
            events.append("close")
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = FakeConnection()
    registry.register(connection)
    registry.claim(connection, "controller", frozenset())
    assert await registry.send(connection, b"frame") is True
    registry.release(connection)
    assert events == ["write:b'frame'"]
    assert await registry.send(connection, b"late") is False
    assert events == ["write:b'frame'"]
    await registry.aclose()

@pytest.mark.asyncio
async def test_controller_slot_releases_after_cleanup_only(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    first_reader, first_writer = await _connect(server)
    await _request(first_reader, first_writer, 1, "hello", {"protocol_version": "1.2"})
    second_reader, second_writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        response = (await _request(second_reader, second_writer, 2, "hello", {"protocol_version": "1.2"}))[-1]
        assert response["error"]["code"] == -32001
    finally:
        await _close(server, first_writer); second_writer.close(); await second_writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_cap_and_reconnect(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    observers = [await _observer(server, i + 2) for i in range(8)]
    extra_reader, extra_writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        response = (await _request(extra_reader, extra_writer, 20, "hello", {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers"]}))[-1]
        assert response["error"]["code"] == -32004
    finally:
        extra_writer.close(); await extra_writer.wait_closed()
        observers[0][1].close(); await observers[0][1].wait_closed()
        replacement = await _observer(server, 30)
        replacement[1].close(); await replacement[1].wait_closed()
        await _close(server, controller_writer)
        for _, writer in observers[1:]: writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_outbound_limits_count_and_bytes(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    reader, writer = await _observer(server)
    try:
        assert server.connections is not None
        frame = (json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"event": "x"}}) + "\n").encode()
        observer = next(state.connection for state in server.connections._connections.values() if state.role == "observer")
        gate = asyncio.Event()
        async def blocked_write(payload: bytes) -> bool:
            await gate.wait()
            return True
        observer._write_physical = blocked_write
        tasks = [asyncio.create_task(server.connections.send(observer, frame)) for _ in range(257)]
        asyncio.get_running_loop().call_later(0.1, gate.set)
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.wait_for(reader.read(), TIMEOUT)
        assert reader.at_eof() or writer.is_closing()
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_read_workers_do_not_block_serve_loop_or_default_executor(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 3, "status"))[-1]
        assert response["result"]["session"]
        assert await asyncio.wait_for(asyncio.to_thread(lambda: True), TIMEOUT)
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_stuck_observer_read_does_not_block_serve_exit(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    _reader, writer = await _observer(server)
    writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_observer_list_and_history_reads_run_off_loop(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 3, "status"))[-1]
        assert "session" in response["result"]
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


def test_delivery_model_inputs_remain_single_use_after_move() -> None:
    from zeta.model_input import ModelInputEnvelope
    from zeta.protocol.types import MessageOrigin
    from zeta.server.delivery import PendingModelInputs

    pending = PendingModelInputs()
    input_id = pending.register("session", ModelInputEnvelope("text", "display", MessageOrigin.USER))
    resolved = pending.resolve("session", {"input_id": input_id}, enabled=True)
    assert resolved.text == "text"
    with pytest.raises(ValueError, match="invalid or expired"):
        pending.resolve("session", {"input_id": input_id}, enabled=True)


@pytest.mark.asyncio
async def test_all_connection_output_is_registry_owned(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    try:
        response = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"}))[-1]
        assert response["result"]["protocol_version"]
        assert hasattr(server, "connections")
    finally:
        await _close(server, writer)
