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
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from zeta.protocol.types import TextContent
from zeta.server.runtime import ServerRuntime

FEATURES = ("multi_controller", "session_park", "resume_attach", "approval_reconnect", "turn_reconnect")


def _hello(features: tuple[str, ...], session_id: str | None = None) -> dict[str, object]:
    params: dict[str, object] = {"protocol_version": "1.1", "features": list(features)}
    if session_id is not None:
        params["session_id"] = session_id
    return params


def _controlled(session_id: str) -> dict[str, object]:
    return {
        "code": -32014,
        "message": "session is controlled by another client",
        "data": {"code": "session_controlled", "session_id": session_id},
    }


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    ("state", "route", "interrupt"),
    [
        (state, route, None)
        for state in ("attached", "parked", "grace", "parked_grace", "free")
        for route in ("hello", "resume", "storage_resume", "prefix_resume")
    ] + [
        ("grace", "resume", "controller_claimed"),
        ("grace", "resume", "controller_ready"),
        ("grace", "resume", "publish_session_grant_reset"),
        ("free", "storage_resume", "resume_session"),
    ],
)
@pytest.mark.asyncio
async def test_acquisition_owner_table(
    tmp_path: Path, transport: Transport, route: str, state: str, interrupt: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TextContent("held")
    backend = GatedBackend([([text, HOLD], [text])])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    clients: list[ReplayClient] = []
    expire = asyncio.Event()
    await await_or_fail(server.start(), what="start acquisition server")

    async def connect() -> ReplayClient:
        client = await await_or_fail(connect_replay(server, transport), what="acquisition socket")
        clients.append(client)
        return client

    try:
        owner, contender = await connect(), await connect()
        assert "result" in await negotiate(owner, replay=False, extra=FEATURES)
        features = tuple(feature for feature in FEATURES if route != "storage_resume" or feature != "resume_attach")
        if route != "hello":
            assert "result" in await negotiate(contender, replay=False, extra=features)
            before = (await contender.request(2, "new_session", {}))["result"]["session"]["session_id"]
        target = (await owner.request(2, "new_session", {}))["result"]["session"]["session_id"]
        if state in ("parked", "parked_grace"):
            assert "result" in await owner.request(3, "send", {"text": "hold"})
            await await_or_fail(backend.held[0].wait(), what="owner's turn is held")
            assert "result" in await owner.request(4, "new_session", {})
            await owner.event("parked_session_changed", {"session_id": target})
        if state in ("grace", "parked_grace", "free"):
            record = server.controllers.resident(target)
            # test-contract: seam locate the target control to gate its next await.
            assert record is not None
            peer = server.connections.clients()[0]
            disconnected = asyncio.Event()
            unregister = server.connections.unregister

            def mark_disconnected(connection):
                unregister(connection)
                if connection is peer:
                    disconnected.set()

            async def grace_deadline() -> None:
                await await_or_fail(expire.wait(), what="expire grace on demand")

            # test-contract: seam observe transport loss after semantic release begins; control the grace clock.
            monkeypatch.setattr(server.connections, "unregister", mark_disconnected)
            # test-contract: seam replace the grace clock with an event.
            monkeypatch.setattr(record.control, "_grace_deadline", grace_deadline)
            await await_or_fail(owner.close(), what="disconnect owner")
            await await_or_fail(disconnected.wait(), what="owner unregistered")
            if state == "free":
                closing = asyncio.Event()
                close = record.runtime.close

                async def mark_close() -> None:
                    closing.set()
                    await await_or_fail(close(), what="close expired runtime")

                # test-contract: seam observe grace expiry retiring the record before reopening it.
                monkeypatch.setattr(record.runtime, "close", mark_close)
                expire.set()
                await await_or_fail(closing.wait(), what="grace expiry starts runtime close")
                await await_or_fail(server.controllers.wait_closed(target), what="close expired target")

        if interrupt is not None:
            # test-contract: seam locate the target control to gate its next await.
            assert record is not None
            await _cancel_and_retry(server, connect, contender, record, target, interrupt, monkeypatch)
            return

        if route == "hello":
            response = await contender.request(3, "hello", _hello(features, target))
        else:
            response = await contender.request(3, "resume", {"session_id": target[:12] if route == "prefix_resume" else target})
        if state in ("attached", "parked"):
            assert response.get("error") == _controlled(target), response
            if route == "hello":
                again = await contender.request(4, "hello", _hello(features, target))
                assert again.get("error") == _controlled(target), again
                assert "result" in await contender.request(5, "hello", _hello(features))
            else:
                status = await contender.request(4, "status", {})
                assert status["result"]["session"]["session_id"] == before
            status = await owner.request(5, "status", {})
            if state == "parked":
                assert [row["session_id"] for row in status["result"]["parked_sessions"]] == [target]
                own_resume = await owner.request(6, "resume", {"session_id": target})
                assert own_resume["result"]["session"]["session_id"] == target
            else:
                assert status["result"]["session"]["session_id"] == target
        elif state == "free" and route == "hello":
            assert response.get("error", {}).get("code") == -32003, response
        elif state in ("grace", "parked_grace") and route in ("storage_resume", "prefix_resume"):
            assert response.get("error") == _controlled(target), response
            status = await contender.request(4, "status", {})
            assert status["result"]["session"]["session_id"] == before
        else:
            assert "result" in response, response
            status = await contender.request(4, "status", {})
            assert status["result"]["session"]["session_id"] == target
    finally:
        expire.set()
        backend.open_all()
        for client in clients:
            await await_or_fail(client.close(), what="close acquisition socket")
        await await_or_fail(server.close(), what="close acquisition server")


async def _cancel_and_retry(server, connect, contender, record, target, interrupt, monkeypatch) -> None:
    peer = server.connections.clients()[0]
    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seam = ServerRuntime if interrupt == "resume_session" else record.control
    original = getattr(seam, interrupt)

    async def hold(*args, **kwargs):
        # The claim has side effects before commit; cancel after it returns to the gate.
        result = await original(*args, **kwargs)
        task = asyncio.current_task()
        assert task is not None
        task.add_done_callback(lambda _: settled.set())
        entered.set()
        await await_or_fail(release.wait(), what=f"release {interrupt}")
        return result

    duplicate = await connect()
    assert "result" in await negotiate(duplicate, replay=False, extra=FEATURES)
    # test-contract: seam gate the real setup task after one acquisition await, then cancel its socket handler.
    with monkeypatch.context() as timing:
        timing.setattr(seam, interrupt, hold)
        try:
            await contender.send(3, "resume", {"session_id": target})
            await await_or_fail(entered.wait(), what=f"acquisition reaches {interrupt}")
            denied = await duplicate.request(2, "resume", {"session_id": target})
            assert denied.get("error") == _controlled(target), denied
            assert peer._run_task is not None
            peer._run_task.cancel()
            await await_or_fail(settled.wait(), what="cancelled setup and rollback settled")
        finally:
            release.set()

    retried = await duplicate.request(3, "resume", {"session_id": target})
    assert "result" in retried, retried
    assert retried["result"]["session"]["session_id"] == target
