"""Multi-controller swap rows through the real serve socket: a swap parks or
refuses a busy session as legacy does, and is never a disconnect."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from tests.test_server_mcp_bindings import LocalMCP, allow_local_http
from tests.test_server_multi_controller_resume import _legacy_resident
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server.control import ServeControl
from zeta.server.delivery import DeliveryRequests
from zeta.server.runtime import ServerRuntime

PARK = ("multi_controller", "session_park", "resume_attach")
RECONNECT = (*PARK, "approval_reconnect", "turn_reconnect")
SETTINGS = {"provider": "codex", "model": "gpt-5.6-sol"}
HELD = ([TextContent("held"), HOLD], [TextContent("held")])
NEXT = ([TextContent("next")], [TextContent("next")])


async def _turn_end(client: ReplayClient, request_id: int, session_id: str) -> None:
    sent = await client.request(request_id, "send", {"text": f"turn {request_id}"})
    assert sent.get("result", {}).get("accepted") is True, sent
    while (end := await await_or_fail(client.event(match={"session_id": session_id}), what="turn end"))[
        "event"
    ] not in ("turn_end", "error"):
        pass
    assert end["event"] == "turn_end", f"the turn failed: {end}"
    await await_or_fail(client.event("agent_end", {"session_id": session_id}), what="agent end")


async def _hold(client: ReplayClient, backend: GatedBackend, request_id: int, gate: int) -> str:
    """Start a held turn in a new session and return its id."""
    created = await client.request(request_id, "new_session", SETTINGS)
    session_id = created["result"]["session"]["session_id"]
    assert "result" in await client.request(request_id + 1, "send", {"text": "hold"})
    await await_or_fail(backend.held[gate].wait(), what="the held turn")
    return session_id


async def _parked_rows(client: ReplayClient, request_id: int) -> list[tuple[str, str]]:
    status = await client.request(request_id, "status", {})
    return [(row["session_id"], row["state"]) for row in status["result"]["parked_sessions"]]


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_multi_park_without_reconnect(tmp_path: Path, transport: Transport) -> None:
    backend = GatedBackend([HELD, NEXT])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start park server")
    clients: list[ReplayClient] = []
    try:
        owner = await connect_replay(server, transport)
        other = await connect_replay(server, transport)
        clients += [owner, other]
        assert "result" in await negotiate(owner, replay=False, extra=PARK)
        assert "result" in await negotiate(other, replay=False, extra=PARK)
        parked = await _hold(owner, backend, 2, 0)
        swapped = await owner.request(4, "new_session", SETTINGS)
        current = swapped["result"]["session"]["session_id"]

        assert await _parked_rows(owner, 5) == [(parked, "running")], "the busy session must park"
        refused = await other.request(2, "resume", {"session_id": parked})
        assert refused.get("error", {}).get("code") == -32014, refused
        backend.open_all()
        closed = await await_or_fail(
            owner.event("parked_session_changed", {"session_id": parked, "data.state": "closed"}),
            what="parked session closes",
        )
        assert closed["data"]["last_turn_outcome"]["reason"] == "completed", closed
        await _turn_end(owner, 6, current)
    finally:
        backend.open_all()
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close park server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("close_fails", (False, True))
@pytest.mark.asyncio
async def test_multi_park_closed_releases_resident_slot(
    tmp_path: Path, transport: Transport, close_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sessionless record, the parked session, and the attached session fill the server.
    monkeypatch.setattr("zeta.server.session_controllers.MAX_RESIDENT_RECORDS", 3)
    backend = GatedBackend([HELD])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    closing, release = asyncio.Event(), asyncio.Event()
    owner = record = None
    try:
        await await_or_fail(server.start(), what="start resident server")
        owner = await connect_replay(server, transport)
        assert "result" in await negotiate(owner, replay=False, extra=PARK)
        parked = await _hold(owner, backend, 2, 0)
        assert "result" in await owner.request(4, "new_session", SETTINGS)
        record = server.parking.get(parked)
        assert record is not None
        close = record.runtime.close

        async def held_close() -> None:
            closing.set()
            await await_or_fail(release.wait(), what="release parked runtime close")
            await close()
            if close_fails:
                raise RuntimeError("injected close failure")

        # test-contract: seam hold cleanup so premature notification cannot race slot release.
        monkeypatch.setattr(record.runtime, "close", held_close)
        backend.open_all()
        await await_or_fail(closing.wait(), what="parked runtime starts closing")
        refused = await owner.request(5, "new_session", SETTINGS)
        assert refused.get("error", {}).get("data") == {"reason": "resident_limit"}, refused

        # The response fences notifications queued before the held close. If a
        # closed event already arrived, react while cleanup is still held.
        if not any(
            frame.get("params", {}).get("event") == "parked_session_changed"
            and frame["params"].get("session_id") == parked
            and frame["params"].get("data", {}).get("state") == "closed"
            for frame in owner.buffered
        ):
            release.set()
        await await_or_fail(
            owner.event("parked_session_changed", {"session_id": parked, "data.state": "closed"}),
            what="parked session closes",
        )
        opened = await owner.request(6, "new_session", SETTINGS)
        assert "result" in opened, f"closed must free a resident slot: {opened}"
    finally:
        release.set()
        backend.open_all()
        if record is not None and record.close_task is not None:
            # Let the server report the retire failure, not shutdown re-raise it.
            await await_or_fail(asyncio.wait((record.close_task,)), what="drain parked close")
        if owner is not None:
            await owner.close()
        await await_or_fail(server.close(), what="close resident server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_multi_park_outlives_grace_while_connected(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def expired_deadline(_control: ServeControl) -> None:
        return None  # injected clock: any grace deadline has already passed

    # test-contract: seam the grace deadline is the injected clock; no sleep as proof.
    monkeypatch.setattr(ServeControl, "_grace_deadline", expired_deadline)
    backend = GatedBackend([HELD, NEXT])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start park server")
    owner = None
    try:
        owner = await connect_replay(server, transport)
        assert "result" in await negotiate(owner, replay=False, extra=RECONNECT)
        parked = await _hold(owner, backend, 2, 0)
        swapped = await owner.request(4, "new_session", SETTINGS)
        current = swapped["result"]["session"]["session_id"]

        # the owner stays connected, so the expired grace deadline must not apply
        assert await _parked_rows(owner, 5) == [(parked, "running")], "the parked turn was cut"
        backend.open_all()
        closed = await await_or_fail(
            owner.event("parked_session_changed", {"session_id": parked, "data.state": "closed"}),
            what="parked session closes",
        )
        assert closed["data"]["last_turn_outcome"]["reason"] == "completed", closed
        await _turn_end(owner, 7, current)
    finally:
        backend.open_all()
        if owner is not None:
            await owner.close()
        await await_or_fail(server.close(), what="close park server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", ("new_session", "resume_disk", "resume_resident"))
@pytest.mark.asyncio
async def test_multi_busy_swap_without_park(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GatedBackend([HELD])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start swap server")
    owner = None
    try:
        target = None
        if row == "resume_resident":
            target = await _legacy_resident(server, transport, monkeypatch=monkeypatch)
        owner = await connect_replay(server, transport)
        assert "result" in await negotiate(owner, replay=False, extra=("multi_controller", "new_session_settings", "served_turn_end"))
        if row == "resume_disk":
            created = await owner.request(2, "new_session", SETTINGS)
            target = created["result"]["session"]["session_id"]
        busy = await _hold(owner, backend, 3, 0)
        request: tuple[str, dict[str, Any]] = (
            ("new_session", SETTINGS) if target is None else ("resume", {"session_id": target})
        )

        refused = await owner.request(5, *request)

        assert refused.get("error", {}).get("code") == -32004, refused
        status = await owner.request(6, "status", {})
        assert status["result"]["session"]["session_id"] == busy
        backend.open_all()
        await await_or_fail(owner.event("turn_end", {"session_id": busy}), what="busy turn end")
        final = await owner.request(7, "status", {})
        assert final["result"]["last_turn_outcome"]["reason"] == "completed", final
    finally:
        backend.open_all()
        if owner is not None:
            await owner.close()
        await await_or_fail(server.close(), what="close swap server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", ("headers", "servers"))
@pytest.mark.asyncio
async def test_idle_resume_changes_bindings(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    config = {"transport": "streamable-http", "url": f"http://127.0.0.1:{port}"}
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": config, "other": config}}))
    server_name = "page" if row == "headers" else "other"
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall("call-1", "page__echo", {})]),
        ScriptedTurn([TextContent("first")]),
        ScriptedTurn(tool_calls=[ToolCall("call-2", f"{server_name}__echo", {})]),
        ScriptedTurn([TextContent("second")]),
    ])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start binding server")
    client = None
    try:
        client = await connect_replay(server, transport)
        features = ("multi_controller", "new_session_settings", "session_mcp_bindings")
        assert "result" in await negotiate(client, replay=False, extra=features)
        old = {"server": "page", "headers": {"Authorization": "Bearer old"}}
        created = await client.request(2, "new_session", {
            **SETTINGS, "approval_mode": "allow", "mcp_bindings": [old],
        })
        session_id = created["result"]["session"]["session_id"]
        await _turn_end(client, 3, session_id)

        new = {"server": server_name, "headers": {"Authorization": "Bearer new"}}
        resumed = await client.request(4, "resume", {"session_id": session_id, "mcp_bindings": [new]})

        assert resumed.get("result", {}).get("session", {}).get("session_id") == session_id, resumed
        await _turn_end(client, 5, session_id)
        calls = [header for method, header in zip(mcp.methods, mcp.headers, strict=False) if method == "tools/call"]
        assert calls == ["Bearer old", "Bearer new"]
    finally:
        if client is not None:
            await client.close()
        await await_or_fail(server.close(), what="close binding server")
        await mcp.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_unhealthy_parked_reopen_sends_closed(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GatedBackend([HELD, NEXT])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start park server")
    owner = None
    try:
        owner = await connect_replay(server, transport)
        assert "result" in await negotiate(owner, replay=False, extra=RECONNECT)
        parked = await _hold(owner, backend, 2, 0)
        assert "result" in await owner.request(4, "new_session", SETTINGS)
        await await_or_fail(
            owner.event("parked_session_changed", {"session_id": parked}), what="session parks"
        )
        record = server.parking.get(parked)
        assert record is not None
        # stands in for failed approval storage on the parked session
        monkeypatch.setattr(record.control, "unhealthy", lambda: True)

        resumed = await owner.request(5, "resume", {"session_id": parked})

        assert resumed["result"]["session"]["session_id"] == parked, resumed
        rows = [
            frame["params"]["data"]["state"] for frame in owner.wire_received
            if frame.get("params", {}).get("event") == "parked_session_changed"
            and frame["params"]["session_id"] == parked
        ]
        assert rows[-1:] == ["closed"], f"the closed parked row was not sent: {rows}"
        await _turn_end(owner, 6, parked)
    finally:
        backend.open_all()
        if owner is not None:
            await owner.close()
        await await_or_fail(server.close(), what="close park server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", ("new_session", "resume_resident"))
@pytest.mark.asyncio
async def test_swap_refuses_pending_end_of_turn_step(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without session_park a swap reserves the outgoing session as legacy
    ``reserve_session_swap`` does: a turn's end step still running refuses it."""
    entered, gate, left = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = DeliveryRequests.continue_after_turn

    async def held_end_step(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await await_or_fail(gate.wait(), what="end-of-turn gate")
        await original(*args, **kwargs)
        left.set()

    backend = GatedBackend([NEXT, NEXT])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start swap server")
    owner = None
    try:
        target = None
        if row == "resume_resident":
            target = await _legacy_resident(server, transport, monkeypatch=monkeypatch)
        owner = await connect_replay(server, transport)
        assert "result" in await negotiate(owner, replay=False, extra=("multi_controller", "served_turn_end"))
        outgoing = (await owner.request(2, "new_session", SETTINGS))["result"]["session"]["session_id"]
        # test-contract: seam the finished turn's end step is held open by the test
        monkeypatch.setattr(DeliveryRequests, "continue_after_turn", held_end_step)
        await _turn_end(owner, 3, outgoing)
        await await_or_fail(entered.wait(), what="the end-of-turn step")
        request: tuple[str, dict[str, Any]] = (
            ("new_session", SETTINGS) if target is None else ("resume", {"session_id": target})
        )

        refused = await owner.request(4, *request)

        assert refused.get("error", {}).get("code") == -32004, f"the swap cut the end-of-turn step: {refused.get('error')}"
        gate.set()
        await await_or_fail(left.wait(), what="the end-of-turn step ends")
        await _turn_end(owner, 5, outgoing)
    finally:
        gate.set()
        if owner is not None:
            await owner.close()
        await await_or_fail(server.close(), what="close swap server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_concurrent_first_opens_without_park(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connections without a session share one unopened record: one connection's
    first open does not make another's refuse as busy."""
    entered, gate = asyncio.Event(), asyncio.Event()
    original = ServerRuntime.create_session
    calls: list[None] = []

    async def held_first(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        if len(calls) == 1:
            entered.set()
            await await_or_fail(gate.wait(), what="first open gate")
        return await original(*args, **kwargs)

    # test-contract: seam the first connection's setup await is held open by the test
    monkeypatch.setattr(ServerRuntime, "create_session", held_first)
    backend = GatedBackend([NEXT])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start open server")
    clients: list[ReplayClient] = []
    try:
        first = await connect_replay(server, transport)
        second = await connect_replay(server, transport)
        clients += [first, second]
        assert "result" in await negotiate(first, replay=False, extra=("multi_controller",))
        assert "result" in await negotiate(second, replay=False, extra=("multi_controller",))
        await first.send(2, "new_session", SETTINGS)
        await await_or_fail(entered.wait(), what="first open in setup")

        opened = await second.request(2, "new_session", SETTINGS)

        assert "result" in opened, f"a concurrent first open was refused: {opened.get('error')}"
        gate.set()
        assert "result" in await first.response(2)
        await _turn_end(second, 3, opened["result"]["session"]["session_id"])
    finally:
        gate.set()
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close open server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_failed_idle_rebind_keeps_old_bindings(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    config = {"transport": "streamable-http", "url": f"http://127.0.0.1:{port}"}
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": config, "other": config}}))
    def echo(call_id: str) -> ToolUseContent:
        return ToolUseContent(ToolCall(call_id, "page__echo", {}))

    backend = GatedBackend([
        ([], [echo("call-1")]), NEXT,
        ([TextContent("held"), HOLD], [TextContent("held"), echo("call-2")]), NEXT,
    ])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start binding server")
    client = None
    try:
        client = await connect_replay(server, transport)
        features = ("multi_controller", "resume_attach", "new_session_settings", "session_mcp_bindings")
        assert "result" in await negotiate(client, replay=False, extra=features)
        old = {"server": "page", "headers": {"Authorization": "Bearer old"}}
        created = await client.request(2, "new_session", {
            **SETTINGS, "approval_mode": "allow", "mcp_bindings": [old],
        })
        session_id = created["result"]["session"]["session_id"]
        await _turn_end(client, 3, session_id)
        resident = server.controllers.resident(session_id)
        assert resident is not None and resident.runtime.loop is not None

        async def failed_rebind(*_args: object) -> None:
            raise RuntimeError("rebind failed")

        # test-contract: seam the loop's rebind fails on the real resume path
        with monkeypatch.context() as patch:
            patch.setattr(resident.runtime.loop, "rebind_mcp", failed_rebind)
            new = {"server": "other", "headers": {"Authorization": "Bearer new"}}
            failed = await client.request(4, "resume", {"session_id": session_id, "mcp_bindings": [new]})
        assert "error" in failed, failed

        assert "result" in await client.request(5, "send", {"text": "hold"})
        await await_or_fail(backend.held[0].wait(), what="the held turn")
        attached = await client.request(6, "resume", {"session_id": session_id, "mcp_bindings": [old]})

        assert attached.get("result", {}).get("attached") == "live_turn", (
            f"the failed rebind changed the session's MCP servers: {attached}"
        )
        backend.open_all()
        await await_or_fail(client.event("agent_end", {"session_id": session_id}), what="held turn end")
        calls = [header for method, header in zip(mcp.methods, mcp.headers, strict=False) if method == "tools/call"]
        assert calls == ["Bearer old", "Bearer old"]
    finally:
        backend.open_all()
        if client is not None:
            await client.close()
        await await_or_fail(server.close(), what="close binding server")
        await mcp.close()
