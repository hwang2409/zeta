"""A fresh multi-controller `hello` while only reconnect grace keeps multi mode.

Each row drops the only controller, leaving its idle session in grace, and then
connects through the real serve socket."""
from __future__ import annotations

import asyncio
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
from tests.test_server_replay_table import _backend_factory
from tests.test_server_session_park import _until
from zeta.server.session_controllers import _MultiGrace

MULTI = ("multi_controller",)
ATTACH = (*MULTI, "resume_attach")
CAPACITY = {"code": -32001, "message": "another client is already connected"}


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", ("hello_reclaim", "attach_reclaim", "cap", "legacy"))
@pytest.mark.asyncio
async def test_fresh_multi_hello_is_admitted_while_only_grace_holds_multi(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await await_or_fail(server.start(), what="start grace hello server")
    clients: list[ReplayClient] = []
    expire = asyncio.Event()

    async def connect() -> ReplayClient:
        client = await await_or_fail(connect_replay(server, transport), what="grace hello socket")
        clients.append(client)
        return client

    async def held_deadline() -> None:
        await await_or_fail(expire.wait(), what="grace deadline", timeout=60)

    try:
        owner = await connect()
        assert "result" in await negotiate(owner, replay=False, extra=ATTACH)
        session_id = (await owner.request(2, "new_session", {}))["result"]["session"]["session_id"]
        record = server.controllers.resident(session_id)
        assert record is not None
        # test-contract: seam an event in place of the wall-clock grace deadline.
        monkeypatch.setattr(record.control, "_grace_deadline", held_deadline)

        def graced() -> bool:
            return isinstance(server.controllers.owner_of(record), _MultiGrace)

        await await_or_fail(owner.close(), what="drop the only controller")
        await _until(lambda: graced() and not server.controllers._controller_members, "idle grace")
        assert server.controllers.controller_mode == "multi"

        if row == "legacy":
            refused = await negotiate(await connect(), replay=False)
            assert refused.get("error") == CAPACITY, "legacy hello crossed multi grace"
            assert graced()
            return

        fresh = await connect()
        admitted = await negotiate(fresh, replay=False, extra=MULTI)
        assert "result" in admitted, f"fresh multi hello refused during grace: {admitted}"

        if row == "cap":
            for _ in range(3):
                assert "result" in await negotiate(await connect(), replay=False, extra=MULTI)
            fifth = await negotiate(await connect(), replay=False, extra=MULTI)
            assert fifth.get("error") == CAPACITY, fifth
            assert graced(), "admitting fresh controllers ended the grace"
            return

        # The fresh controller cannot take the graced session without its exact id.
        for request_id, target in enumerate((session_id, session_id[:8]), 2):
            refused = await fresh.request(request_id, "resume", {"session_id": target})
            assert refused.get("error", {}).get("code") == -32014, refused
        assert graced(), "a refused resume ended the grace"
        status = await fresh.request(4, "status", {})
        assert (status["result"].get("session") or {}).get("session_id") != session_id, status

        reclaimer = await connect()
        if row == "hello_reclaim":
            hello = await reclaimer.request(1, "hello", {
                "protocol_version": "1.1", "features": list(MULTI), "session_id": session_id,
            })
            assert "result" in hello, hello
        else:
            assert "result" in await negotiate(reclaimer, replay=False, extra=ATTACH)
            resumed = await reclaimer.request(2, "resume", {"session_id": session_id})
            assert resumed.get("result", {}).get("session", {}).get("session_id") == session_id, resumed
        status = await reclaimer.request(5, "status", {})
        assert status["result"]["session"]["session_id"] == session_id
        assert server.controllers.resident(session_id) is record, "exact reclaim must keep the record"
    finally:
        expire.set()
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close grace hello server")
