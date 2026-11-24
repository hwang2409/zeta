from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from zeta.protocol.types import TextContent
from zeta.server.runtime import ServerRuntime

LEGACY = ("approval_reconnect", "turn_reconnect")
MULTI = (*LEGACY, "multi_controller", "resume_attach")
REFUSED = {"code": -32001, "message": "another client is already connected"}


async def _hello_bytes(client: ReplayClient, *, session_id: str | None = None) -> bytes:
    """Read the first frame without re-encoding; an event before hello fails."""
    params = {"protocol_version": "1.1", "features": list(LEGACY)}
    if session_id is not None:
        params["session_id"] = session_id
    await client.send(1, "hello", params)
    # test-contract: transport compare the actual serve payload and frame order.
    if not client.websocket:
        return await await_or_fail(client.reader.readline(), what="legacy hello bytes")
    header, size = await await_or_fail(client.reader.readexactly(2), what="legacy websocket header")
    assert header == 0x81 and size < 128
    if size == 126:
        size = int.from_bytes(await await_or_fail(client.reader.readexactly(2), what="frame size"), "big")
    return await await_or_fail(client.reader.readexactly(size), what="legacy websocket payload")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", ("unattached", "idle", "work", "shutdown", "acquire-retire"))
@pytest.mark.asyncio
async def test_public_legacy_grace_lifetime(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TextContent("held")
    backend = GatedBackend([([text, HOLD], [text])])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    clients: list[ReplayClient] = []
    released: dict[object, asyncio.Event] = {}
    expire = asyncio.Event()
    retired = asyncio.Event()

    async def peer() -> ReplayClient:
        client = await await_or_fail(connect_replay(server, transport), what="connect grace peer")
        clients.append(client)
        return client

    async def leave(client: ReplayClient) -> None:
        disconnected = asyncio.Event()
        released[server.connections.clients()[-1]] = disconnected
        await await_or_fail(client.close(), what="close grace peer")
        await await_or_fail(disconnected.wait(), what="server released grace peer")

    async def deadline() -> None:
        await await_or_fail(expire.wait(), what="advance grace deadline")

    await await_or_fail(server.start(), what="start grace server")
    try:
        unregister = server.connections.unregister

        def disconnected(connection):
            unregister(connection)
            if connection in released:
                released[connection].set()

        # test-contract: seam observe the real transport's completed disconnect.
        monkeypatch.setattr(server.connections, "unregister", disconnected)
        legacy = await peer()
        original_hello = await _hello_bytes(legacy)
        assert "result" in json.loads(original_hello)
        empty_status = await legacy.request(2, "status", {})
        no_session = await legacy.request(5, "send", {"text": "no session"})
        assert "error" in no_session
        if row != "unattached":
            created = await legacy.request(3, "new_session", {})
            session_id = created["result"]["session"]["session_id"]
        if row in {"work", "shutdown"}:
            control = server.controllers.attached().control
            # test-contract: seam replace the real grace clock with an explicit event.
            monkeypatch.setattr(control, "_grace_deadline", deadline)
            assert "result" in await legacy.request(4, "send", {"text": "hold"})
            await await_or_fail(backend.held[0].wait(), what="turn reached backend")
        await leave(legacy)
        if row == "idle":
            legacy = await peer()
            assert await _hello_bytes(legacy, session_id=session_id) == original_hello
            assert (await legacy.request(2, "status", {}))["result"]["session"]["session_id"] == session_id
            await leave(legacy)
        multi = await peer()
        if row in {"work", "shutdown"}:
            assert (await negotiate(multi, replay=False, extra=MULTI)).get("error") == REFUSED
            if row == "shutdown":
                grace = control.reconnect_grace_task
                assert grace is not None and not grace.done()
                await await_or_fail(server.close(), what="shut down during legacy grace")
                assert await await_or_fail(multi._read_or_eof(), what="refused peer EOF") is None
                assert grace.done(), "shutdown left a grace task running"
                return
            legacy = await peer()
            assert await _hello_bytes(legacy, session_id=session_id) == original_hello
            status = (await legacy.request(2, "status", {}))["result"]
            assert (status["session"]["session_id"], status["state"]) == (session_id, "running")
            await leave(legacy)
            grace = control.reconnect_grace_task
            assert grace is not None
            expire.set()
            await await_or_fail(asyncio.shield(grace), what="settle expired legacy work")
            multi = await peer()

        accepted = await negotiate(multi, replay=False, extra=MULTI)
        assert "multi_controller" in accepted.get("result", {}).get("capabilities", {}).get("features", ()), accepted
        assert await multi.request(2, "status", {}) == empty_status
        assert await multi.request(5, "send", {"text": "no session"}) == no_session
        if row in {"idle", "work", "acquire-retire"}:
            resumed = await multi.request(3, "resume", {"session_id": session_id})
            assert resumed["result"]["session"]["session_id"] == session_id
            if row == "acquire-retire":
                close = ServerRuntime.close

                async def observe_retirement(runtime, *args, **kwargs):
                    was_target = runtime.opened is not None and runtime.session_id == session_id
                    await await_or_fail(close(runtime, *args, **kwargs), what="close resumed runtime")
                    if was_target:
                        retired.set()

                # test-contract: seam wait for real runtime retirement after zero-grace disconnect.
                monkeypatch.setattr(ServerRuntime, "close", observe_retirement)
                server.approval_reconnect_grace = 0
                await leave(multi)
                await await_or_fail(retired.wait(), what="retire multi-acquired legacy session")
                legacy = await peer()
                assert await _hello_bytes(legacy) == original_hello
                assert await legacy.request(2, "status", {}) == empty_status
                assert await legacy.request(5, "send", {"text": "no session"}) == no_session
            else:
                other = await peer()
                assert (await negotiate(other, replay=False)).get("error") == REFUSED
    finally:
        expire.set()
        backend.open_all()
        for client in clients:
            await await_or_fail(client.close(), what="close test peer")
        await await_or_fail(server.close(), what="close grace server")
