"""A fresh multi-controller `hello` while only reconnect grace keeps multi mode.

Each row drops the only controller, leaving its idle session in grace, and then
connects through the real serve socket."""
from __future__ import annotations

import asyncio
import sys
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
from zeta.server.session_controllers import MAX_MULTI_CONTROLLERS, _MultiGrace

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
            assert "result" in await fresh.request(2, "new_session", {})
            for _ in range(MAX_MULTI_CONTROLLERS - 1):
                holder = await connect()
                assert "result" in await negotiate(holder, replay=False, extra=MULTI)
                assert "result" in await holder.request(2, "new_session", {})
            refused = await negotiate(await connect(), replay=False, extra=MULTI)
            assert refused.get("error") == CAPACITY, refused
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


_HOLD_LEASE = """
import sys
from zeta.core.session import SessionManager
with SessionManager(sys.argv[1]).open(sys.argv[2]).store:
    print("owned", flush=True)
    sys.stdin.read()
"""


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_list_sessions_marks_grace_session_resident(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await await_or_fail(server.start(), what="start resident list server")
    clients: list[ReplayClient] = []
    expire = asyncio.Event()
    holder: asyncio.subprocess.Process | None = None

    async def held_deadline() -> None:
        await await_or_fail(expire.wait(), what="grace deadline", timeout=60)

    try:
        owner = await await_or_fail(connect_replay(server, transport), what="owner socket")
        clients.append(owner)
        assert "result" in await negotiate(owner, replay=False, extra=MULTI)
        session_id = (await owner.request(2, "new_session", {}))["result"]["session"]["session_id"]
        record = server.controllers.resident(session_id)
        assert record is not None
        # test-contract: seam an event in place of the wall-clock grace deadline.
        monkeypatch.setattr(record.control, "_grace_deadline", held_deadline)
        await await_or_fail(owner.close(), what="clean close of the only controller")
        await _until(
            lambda: isinstance(server.controllers.owner_of(record), _MultiGrace), "idle grace"
        )

        lister = await await_or_fail(connect_replay(server, transport), what="lister socket")
        clients.append(lister)
        assert "result" in await negotiate(lister, replay=False, extra=MULTI)

        async def row(request_id: int) -> dict[str, object]:
            sessions = (await lister.request(request_id, "list_sessions", {}))["result"]["sessions"]
            return next(item for item in sessions if item["session_id"] == session_id)

        graced = await row(2)
        assert graced["in_use"] is True, graced
        assert graced.get("resident") is True, "a session in this serve's grace must be resident"

        expire.set()
        await await_or_fail(server.controllers.wait_closed(session_id), what="grace expiry closes")
        await _until(lambda: server.controllers.resident(session_id) is None, "record retired")
        freed = await row(3)
        assert freed["in_use"] is False, freed
        assert freed.get("resident") is False, "an expired grace session is not resident"

        holder = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _HOLD_LEASE, str(tmp_path), session_id,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        assert holder.stdout is not None
        line = await await_or_fail(holder.stdout.readline(), what="other process holds the lease")
        assert line.strip() == b"owned", line
        foreign = await row(4)
        assert foreign["in_use"] is True, foreign
        assert foreign.get("resident") is False, "a session held by another process is not resident"
    finally:
        expire.set()
        if holder is not None and holder.returncode is None:
            assert holder.stdin is not None
            holder.stdin.close()
            try:
                await asyncio.wait_for(holder.wait(), 10)
            except TimeoutError:
                holder.kill()
                await asyncio.wait_for(holder.wait(), 10)
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close resident list server")
