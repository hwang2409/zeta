from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    classify,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.server import ZetaServer
from zeta.server import connections as connection_module
from zeta.server.protocol import EventDescriptor
from zeta.server.server import _Client

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
ROWS = (
    "success",
    "history_read_race",
    "physical_response_gate",
    "response_admission_failure",
    "response_write_failure",
    "scope_cap",
    "global_record_cap",
    "global_byte_cap",
    "new_session_replacement",
    "resume_replacement",
    "controller_handoff",
    "same_connection_rejoin",
    "connection_close",
    "concurrent_join",
    "unnegotiated",
    "mixed_connections",
    "publication_routes",
)


def _scope(value: str) -> Any:
    from zeta.server.session_replay import ReplayScopeId

    return ReplayScopeId.top(value)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


async def _started(tmp_path: Path, transport: Transport) -> tuple[ZetaServer, ReplayClient, str]:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    hello = await negotiate(client)
    assert "session_history_and_subscribe" in hello["result"]["capabilities"]["requests"]
    created = await client.request(2, "new_session", {"provider": "codex"})
    return server, client, created["result"]["session"]["session_id"]


async def _join(client: ReplayClient, session_id: str, request_id: int = 3) -> dict[str, Any]:
    response = await client.request(
        request_id, "session_history_and_subscribe", {"session_id": session_id}
    )
    assert "error" not in response, response
    replay = response["result"]["replay"]
    assert replay["history_seq"] == 0
    return response


async def _publish(server: ZetaServer, session_id: str, count: int = 1) -> None:
    for index in range(count):
        await await_or_fail(
            server.control._notify(
                EventDescriptor.TURN_START,
                session_id,
                data={"turn_id": f"turn-{index}"},
            ),
            what="publish replay event",
        )


async def _assert_events(
    client: ReplayClient, stream_id: str, count: int, *, start: int = 1
) -> list[dict[str, Any]]:
    events = [await client.event("turn_start") for _ in range(count)]
    assert [event["stream_id"] for event in events] == [stream_id] * count
    assert [event["seq"] for event in events] == list(range(start, start + count))
    return events


async def _capacity_returns(server: ZetaServer, client: ReplayClient, session_id: str) -> None:
    joined = await _join(client, session_id, 90)
    stream_id = joined["result"]["replay"]["stream_id"]
    for sequence in range(1, 301):
        await _publish(server, session_id)
        event = await client.event("turn_start")
        assert event["stream_id"] == stream_id
        assert event["seq"] == sequence


async def _success(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    server, client, session_id = await _started(tmp_path, transport)
    try:
        joined = await _join(client, session_id)
        assert classify(joined) == ("response", 3)
        assert isinstance(joined["result"]["messages"], list)
        stream_id = joined["result"]["replay"]["stream_id"]
        await _publish(server, session_id, 3)
        await _assert_events(client, stream_id, 3)
    finally:
        await client.close()
        await server.close()


async def _history_read_race(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    assert server.connections is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original = server.connections.run_read

    async def gated(connection: Any, callable: Callable[[], Any], *, session_id: str | None = None) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="release history read")
        return await original(connection, callable, session_id=session_id)

    monkeypatch.setattr(server.connections, "run_read", gated)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="history read entered")
        await _publish(server, session_id)
        release.set()
        response = await client.response(3)
        stream_id = response["result"]["replay"]["stream_id"]
        event = await client.event("turn_start")
        assert [classify(response), classify({"method": "event", "params": event})] == [
            ("response", 3),
            ("event", stream_id),
        ]
        assert event["seq"] == 1
    finally:
        release.set()
        await client.close()
        await server.close()


async def _physical_response_gate(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    entered, release = asyncio.Event(), asyncio.Event()
    original = _Client._write_physical

    async def gated(owner: _Client, payload: bytes) -> bool:
        if b'"id":3' in payload:
            entered.set()
            await await_or_fail(release.wait(), what="release physical join response")
        return await original(owner, payload)

    monkeypatch.setattr(_Client, "_write_physical", gated)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="physical join response entered")
        await _publish(server, session_id, 3)
        release.set()
        response = await client.response(3)
        stream_id = response["result"]["replay"]["stream_id"]
        events = await _assert_events(client, stream_id, 3)
        assert [classify(response), *(classify({"method": "event", "params": event}) for event in events)] == [
            ("response", 3),
            *(("event", stream_id) for _ in events),
        ]
    finally:
        release.set()
        await client.close()
        await server.close()


async def _response_admission_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    monkeypatch.setattr(connection_module, "OUTPUT_MAX_FRAMES", 1)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        failure = await client.read()
        assert failure == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32011,
                "message": "controller output overflow",
            },
        }
        assert server.connections is not None
        assert server.connections.replay.scope_for(server._client) is None
    finally:
        await client.close()
        await server.close()


async def _response_write_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    failed = asyncio.Event()
    fail_active = False
    original = _Client._write_physical

    async def fail_write(owner: _Client, payload: bytes) -> bool:
        if b'"id":3' in payload or (fail_active and b'"seq":1' in payload):
            failed.set()
            return False
        return await original(owner, payload)

    monkeypatch.setattr(_Client, "_write_physical", fail_write)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(failed.wait(), what="join response write failure")
        assert server.connections is not None and server._client is not None
        assert server.connections.replay.scope_for(server._client) is None
        assert not any(
            item.replay_scope is not None
            for item in server.connections._connections[server._client].output.queue
        )
        joined = await _join(client, session_id, 4)
        failed.clear()
        fail_active = True
        await _publish(server, session_id)
        await await_or_fail(failed.wait(), what="active replay write failure")
        assert server.connections.replay.scope_for(server._client) is None
        fail_active = False
        fresh = await _join(client, session_id, 5)
        assert fresh["result"]["replay"]["stream_id"] != joined["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        await _assert_events(client, fresh["result"]["replay"]["stream_id"], 1)
    finally:
        await client.close()
        await server.close()


async def _same_connection_rejoin(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    entered, release, activated = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = _Client._write_physical

    async def gated(owner: _Client, payload: bytes) -> bool:
        if b'"seq":1' in payload and not entered.is_set():
            entered.set()
            await await_or_fail(release.wait(), what="release old rejoin frame")
        return await original(owner, payload)

    monkeypatch.setattr(_Client, "_write_physical", gated)
    assert server.connections is not None
    original_activate = server.connections.replay.activate_join

    def activate(connection: Any) -> None:
        original_activate(connection)
        activated.set()

    monkeypatch.setattr(server.connections.replay, "activate_join", activate)
    try:
        first = await _join(client, session_id)
        activated.clear()
        first_stream = first["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        await await_or_fail(entered.wait(), what="old rejoin frame entered")
        await _publish(server, session_id)
        await client.send(4, "session_history_and_subscribe", {"session_id": session_id})
        assert server._client is not None
        await await_or_fail(activated.wait(), what="replacement stream activated")
        release.set()
        second = await client.response(4)
        second_stream = second["result"]["replay"]["stream_id"]
        assert second_stream != first_stream
        old_frames = [
            frame for frame in client.buffered
            if frame.get("params", {}).get("stream_id") == first_stream
        ]
        assert [frame["params"]["seq"] for frame in old_frames] == [1]
        client.buffered.clear()
        assert len(server.connections.replay._subscriptions) == 1
        assert not server.connections.replay._pending
        await _publish(server, session_id)
        await _assert_events(client, second_stream, 1)
    finally:
        release.set()
        await client.close()
        await server.close()


async def _connection_close(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server, client, session_id = await _started(tmp_path, transport)
    await _join(client, session_id)
    await client.close()
    assert server.connections is not None
    await await_or_fail(server.connections.run_blocking(lambda: None), what="connection release turn")
    assert not server.connections.replay._subscriptions
    assert server.connections.replay._records == 0
    replacement = await connect_replay(server, transport)
    try:
        await negotiate(replacement)
        await _capacity_returns(server, replacement, session_id)
    finally:
        await replacement.close()
        await server.close()


async def _scope_cap(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    joined = await _join(client, session_id)
    assert server.connections is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original_write = _Client._write_physical

    async def gated(owner: _Client, payload: bytes) -> bool:
        if b'"seq":1' in payload and not entered.is_set():
            entered.set()
            await await_or_fail(release.wait(), what="release scope-cap frame")
        return await original_write(owner, payload)

    monkeypatch.setattr(_Client, "_write_physical", gated)
    monkeypatch.setattr("zeta.server.session_replay.REPLAY_SCOPE_MAX_RECORDS", 2)
    unrelated = _scope("unrelated")
    assert server.connections.replay.reserve(unrelated, 1) is None
    try:
        await _publish(server, session_id)
        await await_or_fail(entered.wait(), what="first scope-cap frame")
        await _publish(server, session_id, 2)
        assert server.connections.replay.scope_for(server._client) is None
        assert unrelated in server.connections.replay.scopes()
        release.set()
        first = await client.event("turn_start")
        assert first["stream_id"] == joined["result"]["replay"]["stream_id"]
        assert first["seq"] == 1
        fresh = await _join(client, session_id, 4)
        await _publish(server, session_id)
        await _assert_events(client, fresh["result"]["replay"]["stream_id"], 1)
    finally:
        release.set()
        await client.close()
        await server.close()


async def _global_cap(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    *,
    by_bytes: bool,
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    joined = await _join(client, session_id)
    assert server.connections is not None
    replay = server.connections.replay
    incumbents = [_scope(f"incumbent-{index}") for index in range(4)]
    if by_bytes:
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_MAX_BYTES", 800)
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_SCOPE_MAX_BYTES", 220)
        for scope in incumbents:
            assert replay.reserve(scope, 200) is None
    else:
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_MAX_RECORDS", 16)
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_SCOPE_MAX_RECORDS", 4)
        for scope in incumbents:
            for _ in range(4):
                assert replay.reserve(scope, 1) is None
    await _publish(server, session_id)
    event = await client.event("turn_start")
    assert event["stream_id"] == joined["result"]["replay"]["stream_id"]
    assert event["seq"] == 1
    assert incumbents[0] not in replay.scopes()
    assert all(scope in replay.scopes() for scope in incumbents[1:])
    await _publish(server, session_id)
    await _assert_events(client, joined["result"]["replay"]["stream_id"], 1, start=2)
    await client.close()
    await server.close()


async def _global_record_cap(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _global_cap(tmp_path, transport, monkeypatch, by_bytes=False)


async def _global_byte_cap(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _global_cap(tmp_path, transport, monkeypatch, by_bytes=True)


async def _active_admission_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    joined = await _join(client, session_id)
    assert server.connections is not None
    original = server.connections._admit
    rejected = False

    def reject_once(state: Any, frame: bytes) -> bool:
        nonlocal rejected
        if not rejected and b'"seq":1' in frame:
            rejected = True
            return False
        return original(state, frame)

    monkeypatch.setattr(server.connections, "_admit", reject_once)
    try:
        await _publish(server, session_id)
        assert rejected
        assert not server.connections.replay._subscriptions
        assert server.connections.replay._records == 0
        fresh = await _join(client, session_id, 4)
        assert fresh["result"]["replay"]["stream_id"] != joined["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        await _assert_events(client, fresh["result"]["replay"]["stream_id"], 1)
    finally:
        await client.close()
        await server.close()


async def _replacement(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    server, controller, original_session = await _started(tmp_path, transport)
    target_session = original_session
    if method == "resume":
        replacement = await controller.request(5, "new_session", {"provider": "codex"})
        current_session = replacement["result"]["session"]["session_id"]
    else:
        current_session = original_session
    active = await connect_replay(server, transport)
    pending = await connect_replay(server, transport)
    await negotiate(active, observer=True)
    await negotiate(pending, observer=True)
    active_join = await _join(active, current_session)
    old_stream = active_join["result"]["replay"]["stream_id"]
    assert server.connections is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original_read = server.connections.run_read

    async def gated(connection: Any, callable: Callable[[], Any], *, session_id: str | None = None) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="release replacement history read")
        return await original_read(connection, callable, session_id=session_id)

    monkeypatch.setattr(server.connections, "run_read", gated)
    try:
        await pending.send(3, "session_history_and_subscribe", {"session_id": current_session})
        await await_or_fail(entered.wait(), what="pending replacement join")
        params = {"provider": "codex"} if method == "new_session" else {"session_id": target_session}
        replaced = await controller.request(6, method, params)
        new_session = replaced["result"]["session"]["session_id"]
        release.set()
        stale = await pending.response(3)
        assert stale["error"] == {
            "code": -32009,
            "message": "replay continuity was invalidated",
            "data": {"code": "stale_event_sequence"},
        }
        await _publish(server, new_session)
        assert not server.connections.replay._subscriptions
        replacement_event = await active.event("turn_start")
        assert "stream_id" not in replacement_event and "seq" not in replacement_event
        fresh = await _join(active, new_session, 4)
        fresh_stream = fresh["result"]["replay"]["stream_id"]
        assert fresh_stream != old_stream
        await _publish(server, new_session)
        await _assert_events(active, fresh_stream, 1)
    finally:
        release.set()
        await pending.close()
        await active.close()
        await controller.close()
        await server.close()


async def _new_session_replacement(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _replacement(tmp_path, transport, monkeypatch, "new_session")


async def _resume_replacement(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _replacement(tmp_path, transport, monkeypatch, "resume")


async def _concurrent_join(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server, client, session_id = await _started(tmp_path, transport)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await client.send(4, "session_history_and_subscribe", {"session_id": session_id})
        first = await client.response(3)
        second = await client.response(4)
        first_stream = first["result"]["replay"]["stream_id"]
        second_stream = second["result"]["replay"]["stream_id"]
        assert first_stream != second_stream
        assert server.connections is not None
        assert len(server.connections.replay._subscriptions) == 1
        assert not server.connections.replay._pending
        await _publish(server, session_id)
        await _assert_events(client, second_stream, 1)
    finally:
        await client.close()
        await server.close()


async def _controller_handoff(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, old, session_id = await _started(tmp_path, transport)
    joined = await _join(old, session_id)
    old_stream = joined["result"]["replay"]["stream_id"]
    assert server.connections is not None
    released = asyncio.Event()
    original_release = server.connections.release

    def release(connection: Any) -> None:
        original_release(connection)
        released.set()

    monkeypatch.setattr(server.connections, "release", release)
    await old.close()
    await await_or_fail(released.wait(), what="old controller released")
    replacement = await connect_replay(server, transport)
    try:
        await negotiate(replacement)
        assert not server.connections.replay._subscriptions
        await _publish(server, session_id)
        plain = await replacement.event("turn_start")
        assert "stream_id" not in plain and "seq" not in plain
        fresh = await _join(replacement, session_id, 3)
        fresh_stream = fresh["result"]["replay"]["stream_id"]
        assert fresh_stream != old_stream
        await _publish(server, session_id)
        await _assert_events(replacement, fresh_stream, 1)
    finally:
        await replacement.close()
        await server.close()


async def _publication_routes(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server, client, session_id = await _started(tmp_path, transport)
    joined = await _join(client, session_id)
    stream_id = joined["result"]["replay"]["stream_id"]
    ordinary = (
        EventDescriptor.ASSISTANT_DELTA,
        EventDescriptor.ASSISTANT_MESSAGE,
        EventDescriptor.TOOL_START,
        EventDescriptor.TOOL_END,
        EventDescriptor.APPROVAL_REQUEST,
        EventDescriptor.USER_MESSAGE,
        EventDescriptor.USER_MESSAGE,
        EventDescriptor.USAGE,
        EventDescriptor.ERROR,
        EventDescriptor.AGENT_END,
        EventDescriptor.SUB_AGENT_RECEIPT,
    )
    observed: list[dict[str, Any]] = []
    for index, descriptor in enumerate(ordinary, 1):
        await await_or_fail(
            server.control._notify(descriptor, session_id, data={"route": index}),
            what=f"publish {descriptor.value}",
        )
        observed.append(await client.event(descriptor.value))
    assert server.connections is not None
    admitted = server.connections.try_admit_controller(
        EventDescriptor.APPROVAL_END, session_id, data={"route": "settlement"}
    )
    assert admitted.kind == "admitted"
    observed.append(await client.event(EventDescriptor.APPROVAL_END.value))
    assert await server.connections.publish_controller(
        EventDescriptor.DELIVERY_STATUS, session_id, data={"route": "direct"}
    )
    observed.append(await client.event(EventDescriptor.DELIVERY_STATUS.value))
    await server.connections.publish_feature_event(
        "session_replay_seq",
        EventDescriptor.SESSION_GRANTS_CHANGED,
        session_id,
        data={"route": "grants"},
    )
    observed.append(await client.event(EventDescriptor.SESSION_GRANTS_CHANGED.value))
    assert [event["stream_id"] for event in observed] == [stream_id] * len(observed)
    assert [event["seq"] for event in observed] == list(range(1, len(observed) + 1))
    await client.close()
    await server.close()


async def _unnegotiated(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        hello = await negotiate(client, replay=False)
        assert "session_history_and_subscribe" not in hello["result"]["capabilities"]["requests"]
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        response = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
        assert response["error"] == {
            "code": -32601,
            "message": "session_history_and_subscribe requires the negotiated session_replay_seq feature",
        }
        await _publish(server, session_id)
        event = await client.event("turn_start")
        assert "stream_id" not in event and "seq" not in event
    finally:
        await client.close()
        await server.close()


async def _mixed_connections(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    server, replay_client, session_id = await _started(tmp_path, transport)
    plain = await connect_replay(server, transport)
    try:
        await negotiate(plain, replay=False, observer=True)
        joined = await _join(replay_client, session_id)
        stream_id = joined["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        sequenced = await replay_client.event("turn_start")
        ordinary = await plain.event("turn_start")
        assert sequenced["stream_id"] == stream_id and sequenced["seq"] == 1
        assert "stream_id" not in ordinary and "seq" not in ordinary
        assert {key: value for key, value in sequenced.items() if key not in {"stream_id", "seq"}} == ordinary
    finally:
        await plain.close()
        await replay_client.close()
        await server.close()


async def _not_implemented(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    del tmp_path, transport, monkeypatch
    pytest.fail("table row has not been implemented")


SCENARIOS: dict[str, Callable[[Path, Transport, pytest.MonkeyPatch], Awaitable[None]]] = {
    row: _not_implemented for row in ROWS
}
SCENARIOS.update(
    success=_success,
    history_read_race=_history_read_race,
    physical_response_gate=_physical_response_gate,
    response_admission_failure=_response_admission_failure,
    response_write_failure=_response_write_failure,
    scope_cap=_scope_cap,
    global_record_cap=_global_record_cap,
    global_byte_cap=_global_byte_cap,
    same_connection_rejoin=_same_connection_rejoin,
    connection_close=_connection_close,
    new_session_replacement=_new_session_replacement,
    resume_replacement=_resume_replacement,
    controller_handoff=_controller_handoff,
    concurrent_join=_concurrent_join,
    unnegotiated=_unnegotiated,
    mixed_connections=_mixed_connections,
    publication_routes=_publication_routes,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("row", ROWS)
async def test_session_replay_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport,
    row: str,
) -> None:
    await SCENARIOS[row](tmp_path, transport, monkeypatch)
