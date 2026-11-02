"""real serve-path tests for read-only observer connections."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_server import (
    TIMEOUT,
    _close,
    _connect,
    _event,
    _read,
    _request,
    _socket_path,
)
from zeta.server import ZetaServer


def _entry(seq: int) -> Any:
    return SimpleNamespace(
        seq=seq,
        id=f"entry-{seq}",
        parent_id=None,
        lane="main",
        type="message",
        data={"message": {"role": "user", "content": [{"type": "text", "text": str(seq)}]}},
    )


async def _observer(
    server: ZetaServer,
    number: int = 1,
    features: list[str] | None = None,
):
    if server._server is None:
        reader, writer = await _connect(server)
    elif server.port is None:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    else:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    await _request(
        reader,
        writer,
        number,
        "hello",
        {"protocol_version": "1.2", "role": "observer", "features": features or ["serve_observers"]},
    )
    return reader, writer


async def _snapshot(reader, writer, number: int = 2) -> dict[str, Any]:
    return (await _request(reader, writer, number, "session_snapshot", {"limit": 128}))[-1]["result"]


@pytest.mark.asyncio
async def test_installed_rows_callback_races_observer_churn(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    stable_reader, stable_writer = await _observer(server, 10)
    await _snapshot(stable_reader, stable_writer)
    churn_reader, churn_writer = await _observer(server, 20)
    try:
        callback = server.runtime.opened.store.on_entries_installed
        assert callback is not None
        publish = asyncio.create_task(asyncio.to_thread(callback, (_entry(10), _entry(11))))
        churn_writer.close()
        await asyncio.wait_for(churn_writer.wait_closed(), TIMEOUT)
        await asyncio.wait_for(publish, TIMEOUT)
        row1 = await _read(stable_reader)
        row2 = await _read(stable_reader)
        assert [row1["params"]["row"]["seq"], row2["params"]["row"]["seq"]] == [10, 11]
        await _request(controller_reader, controller_writer, 3, "status")
    finally:
        await _close(server, controller_writer)
        stable_writer.close()
        await stable_writer.wait_closed()


@pytest.mark.asyncio
async def test_installed_rows_ingress_overflow_closes_observers_and_keeps_controller(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    observer_reader, observer_writer = await _observer(server, 10)
    await _snapshot(observer_reader, observer_writer)
    try:
        callback = server.runtime.opened.store.on_entries_installed
        assert callback is not None
        thread = threading.Thread(target=lambda: [callback((_entry(i),)) for i in range(257)])
        thread.start()
        thread.join(TIMEOUT)
        assert not thread.is_alive()
        overflow = await _read(observer_reader)
        assert overflow["error"]["code"] == -32011
        assert (await _request(controller_reader, controller_writer, 3, "status"))[-1]["result"]["session"]
    finally:
        await _close(server, controller_writer)
        observer_writer.close()
        await observer_writer.wait_closed()


@pytest.mark.asyncio
async def test_child_run_offer_filters_mixed_observer_features(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2", "features": ["child_run_events"]})
    first_reader, first_writer = await _observer(server, 10, ["serve_observers", "child_run_events"])
    second_reader, second_writer = await _observer(server, 20)
    try:
        assert server.connections is not None
        frame = (json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"event": "child_progress"}}) + "\n").encode()
        assert await server.connections.offer_live(frame, required_feature="child_run_events") is True
        assert (await _read(first_reader))["params"]["event"] == "child_progress"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_read(second_reader), 0.1)
        assert (await _read(controller_reader))["params"]["event"] == "child_progress"
    finally:
        await _close(server, controller_writer)
        first_writer.close(); second_writer.close()
        await first_writer.wait_closed(); await second_writer.wait_closed()


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
async def test_controller_event_order_is_unchanged_with_observer(tmp_path: Path) -> None:
    async def run(home: Path, with_observer: bool) -> list[str]:
        server = ZetaServer(home=home, socket_path=_socket_path(home), provider="codex")
        reader, writer = await _connect(server)
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        extra = None
        if with_observer:
            extra = await _observer(server)
            await _snapshot(*extra)
        try:
            frames = await _request(reader, writer, 3, "send", {"text": "order"})
            while not any(f.get("params", {}).get("event") == "turn_end" for f in frames):
                frames.append(await _read(reader))
            return [f.get("params", {}).get("event", "response") for f in frames]
        finally:
            await _close(server, writer)
            if extra:
                extra[1].close(); await extra[1].wait_closed()

    baseline = await run(tmp_path / "base", False)
    attached = await run(tmp_path / "attached", True)
    assert attached == baseline


@pytest.mark.asyncio
async def test_mid_turn_observer_snapshot_has_no_duplicate_rows(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    controller_writer.write((json.dumps({"jsonrpc": "2.0", "id": 3, "method": "send", "params": {"text": "mid"}}) + "\n").encode())
    await controller_writer.drain()
    reader, writer = await _observer(server)
    try:
        snapshot = await _snapshot(reader, writer)
        rows = list(snapshot["rows"])
        while True:
            try:
                frame = await asyncio.wait_for(_read(reader), 0.2)
            except asyncio.TimeoutError:
                break
            if frame.get("params", {}).get("event") == "transcript_row":
                rows.append(frame["params"]["row"])
        assert len({(row["seq"], row["entry_id"]) for row in rows}) == len(rows)
    finally:
        await _close(server, controller_writer)
        writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_unix_and_websocket_observers_receive_live_rows(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), ws_port=0, provider="codex")
    await server.start()
    controller_reader, controller_writer = await asyncio.open_unix_connection(str(server.socket_path))
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    unix_reader, unix_writer = await _observer(server, 10)
    await _snapshot(unix_reader, unix_writer)
    try:
        from tests.test_server_websocket import _connect as ws_connect, _read_json, _send_json
        ws_reader, ws_writer, status = await ws_connect(server, token=server._websocket.token)
        assert status == 101
        await _send_json(ws_writer, {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers"]}})
        assert (await _read_json(ws_reader))["result"]["server"] == "zeta"
        await _send_json(ws_writer, {"jsonrpc": "2.0", "id": 2, "method": "session_snapshot", "params": {}})
        await _read_json(ws_reader)
        frames = await _request(controller_reader, controller_writer, 3, "send", {"text": "live"})
        while not any(f.get("params", {}).get("event") == "turn_end" for f in frames):
            frames.append(await _read(controller_reader))
        async def next_row(reader, read):
            while True:
                frame = await asyncio.wait_for(read(reader), TIMEOUT)
                if frame.get("params", {}).get("event") == "transcript_row":
                    return frame
        assert (await next_row(unix_reader, _read))["params"]["event"] == "transcript_row"
        assert (await next_row(ws_reader, _read_json))["params"]["event"] == "transcript_row"
        ws_writer.close(); await ws_writer.wait_closed()
    finally:
        await _close(server, controller_writer)
        unix_writer.close(); await unix_writer.wait_closed()
