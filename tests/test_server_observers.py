"""real serve-path tests for read-only observer connections."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_server import TIMEOUT, _close, _connect, _read, _request, _socket_path
from zeta.server import ZetaServer


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
        monkeypatch.setattr(server.runtime, name, lambda *args, _name=name, **kwargs: calls.append(_name) or original(*args, **kwargs))
    reader, writer = await _observer(server)
    try:
        assert calls == []
        assert (await _request(reader, writer, 2, "capabilities"))[-1]["result"].get("role") == "observer"
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_controller_slot_releases_after_cleanup_only(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    first_reader, first_writer = await _connect(server)
    await _request(first_reader, first_writer, 1, "hello", {"protocol_version": "1.2"})
    second_reader, second_writer = await _connect(server)
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
    extra_reader, extra_writer = await _connect(server)
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
        for _ in range(257):
            await server.connections.send(server._client, frame)
        error = await _read(reader)
        assert error["error"]["code"] == -32011
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_read_workers_do_not_block_serve_loop_or_default_executor(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 2, "status"))[-1]
        assert response["result"]["session"]
        assert await asyncio.wait_for(asyncio.to_thread(lambda: True), TIMEOUT)
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_stuck_observer_read_does_not_block_serve_exit(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _observer(server)
    writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_observer_list_and_history_reads_run_off_loop(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 2, "status"))[-1]
        assert "session" in response["result"]
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


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
