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
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.server import ZetaServer, ergonomics
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
    "active_write_failure",
    "active_admission_failure",
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
    "observer_without_join",
    "join_failure",
    "global_pressure_two_victims",
    "join_during_swap",
    "observer_write_failure",
    "old_generation_failure",
    "observer_settlement",
    "observer_controller_absent",
    "replay_only_backward_page",
    "controller_reconnect_keeps_observer",
    "approval_reconnect_rejoin",
    "global_cap_self_eviction",
    "stream_ended_unnegotiated",
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


def _ended(frame: dict[str, Any], session_id: str, stream_id: str, reason: str) -> bool:
    return frame.get("params") == {
        "event": "replay_stream_ended",
        "session_id": session_id,
        "stream_id": stream_id,
        "reason": reason,
    }


def _wire(frames: list[dict[str, Any]]) -> list[tuple[object, ...]]:
    """Reduce wire frames to (event, stream_id, seq or reason) for exact comparison."""
    return [
        (
            frame["params"]["event"],
            frame["params"].get("stream_id"),
            frame["params"].get("seq", frame["params"].get("reason")),
        )
        for frame in frames
        if "params" in frame
    ]


def _watch_release(server: ZetaServer, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Set the returned event when the server releases the current controller connection."""
    assert server.connections is not None
    connection, released = server._client, asyncio.Event()
    original = server.connections.release

    def release(target: Any) -> None:
        original(target)
        if target is connection:
            released.set()

    # test-contract: seam observe the real connection release after its transport closes
    monkeypatch.setattr(server.connections, "release", release)
    return released


async def _frames_before_status(client: ReplayClient, request_id: int) -> list[dict[str, Any]]:
    """Return every frame queued before a later status response on one output queue."""
    await client.send(request_id, "status", {})
    frames: list[dict[str, Any]] = []
    while classify(frame := await client.read()) != ("response", request_id):
        frames.append(frame)
    return frames


async def _capacity_returns(server: ZetaServer, client: ReplayClient, session_id: str) -> None:
    joined = await _join(client, session_id, 90)
    stream_id = joined["result"]["replay"]["stream_id"]
    for sequence in range(1, 301):
        await _publish(server, session_id)
        assert server.connections is not None and server._client is not None
        assert server.connections.replay.stream_id(server._client) is not None
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

    # test-contract: seam gate the real transport write after output admission
    monkeypatch.setattr(_Client, "_write_physical", gated)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="physical join response entered")
        assert server.connections is not None and server._client is not None
        assert server._client in server.connections.replay._subscriptions
        assert server._client not in server.connections.replay._pending
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
    assert server.connections is not None
    activated = False
    original_activate = server.connections.replay.activate_join

    def activate(connection: Any) -> None:
        nonlocal activated
        activated = True
        original_activate(connection)

    monkeypatch.setattr(server.connections.replay, "activate_join", activate)
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
        assert not activated, "pending tail was released before response admission"
        assert server.connections.replay.stream_id(server._client) is None
    finally:
        await client.close()
        await server.close()


async def _response_write_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    owner = server._client
    released = _watch_release(server, monkeypatch)
    failed = asyncio.Event()
    original = _Client._write_physical

    async def fail_write(owner: _Client, payload: bytes) -> bool:
        if b'"id":3' in payload:
            failed.set()
            return False
        return await original(owner, payload)

    # test-contract: seam force the real transport write to report failure
    monkeypatch.setattr(_Client, "_write_physical", fail_write)
    try:
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(failed.wait(), what="join response write failure")
        assert server.connections is not None
        assert server.connections.replay.stream_id(owner) is None
        state = server.connections._connections.get(owner)
        assert state is None or not any(item.replay_scope is not None for item in state.output.queue)
        frames = await client.closed(4)
        assert not any(frame.get("params", {}).get("event") == "replay_stream_ended" for frame in frames)
        await await_or_fail(released.wait(), what="failed connection released")
        replacement = await connect_replay(server, transport)
        try:
            await negotiate(replacement, observer=True)
            fresh = await _join(replacement, session_id, 5)
            await _publish(server, session_id)
            await _assert_events(replacement, fresh["result"]["replay"]["stream_id"], 1)
        finally:
            await replacement.close()
    finally:
        await client.close()
        await server.close()


async def _active_write_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    await _join(client, session_id)
    owner = server._client
    failed = asyncio.Event()
    original = _Client._write_physical

    async def fail_active(owner: _Client, payload: bytes) -> bool:
        if b'"seq":1' in payload:
            failed.set()
            return False
        return await original(owner, payload)

    # test-contract: seam force one active replay frame to fail physically
    monkeypatch.setattr(_Client, "_write_physical", fail_active)
    try:
        await _publish(server, session_id)
        await await_or_fail(failed.wait(), what="active replay write failure")
        assert server.connections is not None
        assert server.connections.replay.stream_id(owner) is None
        frames = await client.closed(4)
        assert all(frame["params"]["event"] != "replay_stream_ended" for frame in frames), "write failure sent an end event"
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

    # test-contract: seam hold one old physical frame across replacement admission
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
        assert _wire(old_frames) == [
            ("turn_start", first_stream, 1),
            ("replay_stream_ended", first_stream, "rejoined"),
        ], "rejoin did not end the old stream with exactly one event"
        assert _ended(old_frames[1], session_id, first_stream, "rejoined")
        client.buffered.clear()
        assert len(server.connections.replay._subscriptions) == 1
        assert not server.connections.replay._pending
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(client, 5)) == [("turn_start", second_stream, 1)]
    finally:
        release.set()
        await client.close()
        await server.close()


async def _connection_close(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    await _join(client, session_id)
    assert server.connections is not None and server._client is not None
    closed_connection = server._client
    released = asyncio.Event()
    release_errors: list[AttributeError] = []
    original_release = server.connections.release

    def release(connection: Any) -> None:
        try:
            original_release(connection)
        except AttributeError as exc:
            release_errors.append(exc)
        finally:
            released.set()

    # test-contract: seam observe connection release after the real transport closes
    monkeypatch.setattr(server.connections, "release", release)
    await client.close()
    await await_or_fail(released.wait(), what="connection replay release")
    assert not release_errors, f"connection release failed: {release_errors!r}"
    assert server.connections.replay.stream_id(closed_connection) is None
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

    # test-contract: seam retain real wire frames until the scope limit is crossed
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
        old_stream = joined["result"]["replay"]["stream_id"]
        frames = await _frames_before_status(client, 6)
        assert _wire(frames) == [
            ("turn_start", old_stream, 1),
            ("replay_stream_ended", old_stream, "evicted"),
        ], "scope eviction did not end the stream with exactly one event"
        assert _ended(frames[1], session_id, old_stream, "evicted")
        fresh = await _join(client, session_id, 4)
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(client, 7)) == [
            ("turn_start", fresh["result"]["replay"]["stream_id"], 1)
        ]
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
    assert replay.scope_for(server._client) is not None, "requesting scope was evicted"
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
    original = server.connections._fits
    rejected = False

    def reject_once(output: Any, size: int, frames: int = 1) -> bool:
        nonlocal rejected
        if not rejected:
            rejected = True
            return False
        return original(output, size, frames)

    # test-contract: seam make the real output admission overflow on one active frame
    monkeypatch.setattr(server.connections, "_fits", reject_once)
    released = _watch_release(server, monkeypatch)
    try:
        await _publish(server, session_id)
        assert rejected
        assert not server.connections.replay._subscriptions
        assert server.connections.replay._records == 0
        frames = await client.closed(4)
        assert [frame.get("error", {}).get("code") for frame in frames] == [-32011]
        await await_or_fail(released.wait(), what="overflowed connection released")
        replacement = await connect_replay(server, transport)
        try:
            await negotiate(replacement, observer=True)
            fresh = await _join(replacement, session_id, 5)
            assert fresh["result"]["replay"]["stream_id"] != joined["result"]["replay"]["stream_id"]
            await _publish(server, session_id)
            await _assert_events(replacement, fresh["result"]["replay"]["stream_id"], 1)
        finally:
            await replacement.close()
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
        ended = await _frames_before_status(active, 7)
        assert _wire(ended) == [("replay_stream_ended", old_stream, "session_replaced")], (
            "session replacement did not end the observer stream with exactly one event"
        )
        assert _ended(ended[0], current_session, old_stream, "session_replaced")
        await _publish(server, new_session)
        assert await _frames_before_status(active, 8) == [], "ended observer stream still got events"
        assert await _frames_before_status(pending, 9) == [], "pending join got an end event"
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
    assert [event.get("stream_id") for event in observed] == [stream_id] * len(observed)
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
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    plain = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    try:
        await negotiate(plain, replay=False)
        created = await plain.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        await negotiate(observer, observer=True)
        joined = await _join(observer, session_id)
        stream_id = joined["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        sequenced = await observer.event("turn_start")
        ordinary = await plain.event("turn_start")
        assert sequenced["stream_id"] == stream_id and sequenced["seq"] == 1
        assert "stream_id" not in ordinary and "seq" not in ordinary
        assert {key: value for key, value in sequenced.items() if key not in {"stream_id", "seq"}} == ordinary
    finally:
        await observer.close()
        await plain.close()
        await server.close()


async def _observer_without_join(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server, controller, session_id = await _started(tmp_path, transport)
    plain = await connect_replay(server, transport)
    unjoined = await connect_replay(server, transport)
    try:
        await negotiate(plain, replay=False, observer=True)
        await negotiate(unjoined, observer=True)
        stream_id = (await _join(controller, session_id))["result"]["replay"]["stream_id"]
        await _publish(server, session_id)
        assert server.connections is not None
        assert await server.connections.publish_controller(
            EventDescriptor.APPROVAL_REQUEST,
            session_id,
            data={"arguments": {"command": "secret"}},
        )
        events = [await controller.event("turn_start"), await controller.event("approval_request")]
        assert [(event["stream_id"], event["seq"]) for event in events] == [(stream_id, 1), (stream_id, 2)]
        assert await _frames_before_status(plain, 10) == [], "unnegotiated observer got controller events"
        assert await _frames_before_status(unjoined, 10) == [], "unjoined observer got controller events"
    finally:
        await unjoined.close()
        await plain.close()
        await controller.close()
        await server.close()


async def _join_failure(tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    assert server.connections is not None
    original = server.connections.run_read

    async def fail(connection: Any, callable: Callable[[], Any], *, session_id: str | None = None) -> Any:
        del connection, callable, session_id
        raise RuntimeError("history read failed")

    # test-contract: seam make the off-loop history read fail with a non-protocol error
    monkeypatch.setattr(server.connections, "run_read", fail)
    try:
        failed = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
        assert failed["error"]["message"] == "history read failed"
        monkeypatch.setattr(server.connections, "run_read", original)
        await _publish(server, session_id)
        frames = await _frames_before_status(client, 4)
        assert [frame["params"]["event"] for frame in frames] == ["turn_start"], "failed join kept the connection silent"
        assert "stream_id" not in frames[0]["params"]
        fresh = await _join(client, session_id, 5)
        await _publish(server, session_id)
        await _assert_events(client, fresh["result"]["replay"]["stream_id"], 1)
    finally:
        await client.close()
        await server.close()


async def _global_pressure_two_victims(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    try:
        stream_id = (await _join(client, session_id))["result"]["replay"]["stream_id"]
        assert server.connections is not None and server._client is not None
        replay = server.connections.replay
        frame = server._client.codec.notification(
            EventDescriptor.TURN_START, session_id, stream_id=stream_id, seq=1, data={"turn_id": "turn-0"}
        )
        assert frame is not None
        # incumbents a, a, a, b fit; adding the candidate fits only after two a-sized scopes go
        large, small = len(frame) // 2, len(frame) // 4
        cap = large + small + len(frame)
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_MAX_BYTES", cap)
        monkeypatch.setattr("zeta.server.session_replay.REPLAY_SCOPE_MAX_BYTES", cap)
        incumbents = [_scope(f"incumbent-{index}") for index in range(4)]
        for scope, size in zip(incumbents, (large, large, large, small), strict=True):
            assert replay.reserve(scope, size) is None
        await _publish(server, session_id)
        assert replay.scope_for(server._client) is not None, "requesting scope was evicted"
        assert [scope in replay.scopes() for scope in incumbents] == [False, False, True, True]
        await _assert_events(client, stream_id, 1)
        await _publish(server, session_id)
        await _assert_events(client, stream_id, 1, start=2)
    finally:
        await client.close()
        await server.close()


async def _join_during_swap(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, controller, old_session = await _started(tmp_path, transport)
    observer = await connect_replay(server, transport)
    await negotiate(observer, observer=True)
    assert server.connections is not None
    swapped, resume = asyncio.Event(), asyncio.Event()
    read_entered, read_release = asyncio.Event(), asyncio.Event()
    original_create = server.runtime.create_session
    original_read = server.connections.run_read

    async def gated_create(**kwargs: Any) -> Any:
        result = await original_create(**kwargs)
        swapped.set()
        await await_or_fail(resume.wait(), what="resume session swap")
        return result

    async def gated_read(connection: Any, callable: Callable[[], Any], *, session_id: str | None = None) -> Any:
        read_entered.set()
        await await_or_fail(read_release.wait(), what="release swap join read")
        return await original_read(connection, callable, session_id=session_id)

    # test-contract: seam hold the swap after the runtime switched, before it finishes
    monkeypatch.setattr(server.runtime, "create_session", gated_create)
    try:
        await controller.send(6, "new_session", {"provider": "codex"})
        await await_or_fail(swapped.wait(), what="runtime switched session")
        new_session = server.runtime.session_id
        assert new_session != old_session
        monkeypatch.setattr(server.connections, "run_read", gated_read)
        await observer.send(3, "session_history_and_subscribe", {"session_id": new_session})
        await await_or_fail(read_entered.wait(), what="swap join read entered")
        resume.set()
        created = await controller.response(6)
        assert created["result"]["session"]["session_id"] == new_session
        read_release.set()
        joined = await observer.response(3)
        assert "error" not in joined, "join against the new session was invalidated by the swap"
        await _publish(server, new_session)
        await _assert_events(observer, joined["result"]["replay"]["stream_id"], 1)
    finally:
        resume.set()
        read_release.set()
        await observer.close()
        await controller.close()
        await server.close()


async def _observer_write_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, controller, session_id = await _started(tmp_path, transport)
    observer = await connect_replay(server, transport)
    failing = asyncio.Event()
    writes: list[asyncio.Task[Any]] = []
    original = _Client._write_physical

    async def fail_observer(owner: _Client, payload: bytes) -> bool:
        if owner.role == "observer" and b'"seq":1' in payload:
            writes.append(asyncio.current_task())
            failing.set()
            return False
        return await original(owner, payload)

    try:
        await negotiate(observer, observer=True)
        stream_id = (await _join(controller, session_id))["result"]["replay"]["stream_id"]
        await _join(observer, session_id, 4)
        # test-contract: seam fail the observer's physical replay write only
        monkeypatch.setattr(_Client, "_write_physical", fail_observer)
        await _publish(server, session_id)
        await await_or_fail(failing.wait(), what="observer replay write entered")
        await await_or_fail(writes[0], what="observer replay write failure")
        assert server.connections is not None and server._client is not None
        assert server.connections.replay.stream_id(server._client) == stream_id, "observer failure ended the controller stream"
        await _publish(server, session_id)
        await _assert_events(controller, stream_id, 2)
    finally:
        await observer.close()
        await controller.close()
        await server.close()


async def _old_generation_failure(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    first_stream = (await _join(client, session_id))["result"]["replay"]["stream_id"]
    assert server.connections is not None
    entered, release, activated = asyncio.Event(), asyncio.Event(), asyncio.Event()
    writes: list[asyncio.Task[Any]] = []
    original = _Client._write_physical

    async def fail_old(owner: _Client, payload: bytes) -> bool:
        if first_stream.encode() in payload and b'"seq":1' in payload:
            writes.append(asyncio.current_task())
            entered.set()
            await await_or_fail(release.wait(), what="release old-generation frame")
            return False
        return await original(owner, payload)

    original_activate = server.connections.replay.activate_join

    def activate(connection: Any) -> None:
        original_activate(connection)
        activated.set()

    # test-contract: seam hold one old-generation frame in flight across a rejoin, then fail it
    monkeypatch.setattr(_Client, "_write_physical", fail_old)
    monkeypatch.setattr(server.connections.replay, "activate_join", activate)
    try:
        await _publish(server, session_id)
        await await_or_fail(entered.wait(), what="old-generation frame entered")
        await client.send(4, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(activated.wait(), what="new generation activated")
        assert server._client is not None
        second_stream = server.connections.replay.stream_id(server._client)
        assert second_stream is not None and second_stream != first_stream
        release.set()
        await await_or_fail(writes[0], what="old-generation write failure")
        assert server.connections.replay.stream_id(server._client) == second_stream, "old-generation failure ended the new stream"
        second = await client.response(4)
        assert second["result"]["replay"]["stream_id"] == second_stream
        await _publish(server, session_id)
        await _assert_events(client, second_stream, 1)
    finally:
        release.set()
        await client.close()
        await server.close()


async def _observer_settlement(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server, controller, session_id = await _started(tmp_path, transport)
    observer = await connect_replay(server, transport)
    try:
        await negotiate(observer, observer=True)
        stream_id = (await _join(observer, session_id))["result"]["replay"]["stream_id"]
        assert server.connections is not None
        admitted = server.connections.try_admit_controller(
            EventDescriptor.APPROVAL_END, session_id, data={"route": "settlement"}
        )
        assert admitted.kind == "admitted"
        frames = await _frames_before_status(observer, 4)
        assert [
            (frame["params"]["event"], frame["params"]["stream_id"], frame["params"]["seq"])
            for frame in frames
        ] == [("approval_end", stream_id, 1)], "joined observer missed approval settlement"
    finally:
        await observer.close()
        await controller.close()
        await server.close()


async def _observer_controller_absent(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, controller, session_id = await _started(tmp_path, transport)
    observer = await connect_replay(server, transport)
    assert server.connections is not None
    released = asyncio.Event()
    original_release = server.connections.release

    def release(connection: Any) -> None:
        original_release(connection)
        released.set()

    # test-contract: seam observe the real controller release after its transport closes
    monkeypatch.setattr(server.connections, "release", release)
    try:
        await negotiate(observer, observer=True)
        await controller.close()
        await await_or_fail(released.wait(), what="controller released")
        stream_id = (await _join(observer, session_id))["result"]["replay"]["stream_id"]
        await _publish(server, session_id, 2)
        frames = await _frames_before_status(observer, 4)
        assert [
            (frame["params"]["event"], frame["params"]["stream_id"], frame["params"]["seq"])
            for frame in frames
        ] == [("turn_start", stream_id, 1), ("turn_start", stream_id, 2)], "joined observer missed events without a controller"
    finally:
        await observer.close()
        await server.close()


async def _replay_only_backward_page(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    server, client, session_id = await _started(tmp_path, transport)
    try:
        for number in range(60):
            server.runtime.opened.store.append_message(
                with_message_origin(
                    Message(MessageRole.USER, [TextContent(f"message-{number}" + "x" * 100)]),
                    MessageOrigin.USER,
                )
            )
        newest = (await _join(client, session_id))["result"]
        assert newest["next_before"] is not None
        page = await client.request(
            4,
            "session_history",
            {
                "session_id": session_id,
                "before": newest["next_before"],
                "history_validator": newest["history_validator"],
            },
        )
        assert "error" not in page, f"session_replay_seq alone cannot page back: {page}"

        def numbers(rows: list[dict[str, Any]]) -> list[int]:
            return [
                int(row["content"][0]["text"].split("x", 1)[0].removeprefix("message-"))
                for row in rows
                if row["role"] == "user"
            ]

        older, recent = numbers(page["result"]["messages"]), numbers(newest["messages"])
        assert older and recent and max(older) < min(recent)
    finally:
        await client.close()
        await server.close()


async def _observer_with_stream(
    server: ZetaServer, transport: Transport, session_id: str
) -> tuple[ReplayClient, str]:
    observer = await connect_replay(server, transport)
    await negotiate(observer, observer=True)
    return observer, (await _join(observer, session_id, 30))["result"]["replay"]["stream_id"]


async def _controller_reconnect_keeps_observer(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, controller, session_id = await _started(tmp_path, transport)
    observer, stream_id = await _observer_with_stream(server, transport, session_id)
    assert server.connections is not None
    released = asyncio.Event()
    original_release = server.connections.release

    def release(connection: Any) -> None:
        original_release(connection)
        released.set()

    # test-contract: seam observe the real controller release after its transport closes
    monkeypatch.setattr(server.connections, "release", release)
    replacement: ReplayClient | None = None
    try:
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(observer, 31)) == [("turn_start", stream_id, 1)]
        await controller.close()
        await await_or_fail(released.wait(), what="controller released")
        replacement = await connect_replay(server, transport)
        await negotiate(replacement)
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(observer, 32)) == [("turn_start", stream_id, 2)], (
            "a controller disconnect ended the observer stream"
        )
        plain = await replacement.event("turn_start")
        assert "stream_id" not in plain and "seq" not in plain
    finally:
        if replacement is not None:
            await replacement.close()
        await observer.close()
        await server.close()


async def _approval_reconnect_rejoin(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    controller = await connect_replay(server, transport)
    await negotiate(controller, extra=("approval_reconnect",))
    created = await controller.request(2, "new_session", {"provider": "codex"})
    session_id = created["result"]["session"]["session_id"]
    observer, observer_stream = await _observer_with_stream(server, transport, session_id)
    controller_stream = (await _join(controller, session_id))["result"]["replay"]["stream_id"]
    assert server.connections is not None and server._client is not None
    old_controller = server._client
    entered, released = asyncio.Event(), asyncio.Event()
    original_write = _Client._write_physical
    original_release = server.connections.release

    async def hold_controller(owner: _Client, payload: bytes) -> bool:
        if owner is old_controller and controller_stream.encode() in payload:
            entered.set()
            await asyncio.Event().wait()
        return await original_write(owner, payload)

    def release(connection: Any) -> None:
        original_release(connection)
        if connection is old_controller:
            released.set()

    # test-contract: seam keep the old controller's replay frames in flight and queued at its close
    monkeypatch.setattr(_Client, "_write_physical", hold_controller)
    monkeypatch.setattr(server.connections, "release", release)
    replacement: ReplayClient | None = None
    try:
        await _publish(server, session_id, 2)
        await await_or_fail(entered.wait(), what="old controller frame in flight")
        await controller.close()
        await await_or_fail(released.wait(), what="old controller released")
        assert server.connections.replay.stream_id(old_controller) is None
        replacement = await connect_replay(server, transport)
        await negotiate(replacement, extra=("approval_reconnect",))
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(observer, 31)) == [
            ("turn_start", observer_stream, 1),
            ("turn_start", observer_stream, 2),
            ("turn_start", observer_stream, 3),
        ], "an approval_reconnect reconnect ended the observer stream"
        assert server.connections.replay._records == 0, "released controller frames leaked replay charges"
        plain = await replacement.event("turn_start")
        assert "stream_id" not in plain and "seq" not in plain
        fresh = (await _join(replacement, session_id, 4))["result"]["replay"]["stream_id"]
        assert fresh != controller_stream
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(replacement, 5)) == [("turn_start", fresh, 1)]
    finally:
        if replacement is not None:
            await replacement.close()
        await observer.close()
        await server.close()


async def _global_cap_self_eviction(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client, session_id = await _started(tmp_path, transport)
    stream_id = (await _join(client, session_id))["result"]["replay"]["stream_id"]
    entered, release = asyncio.Event(), asyncio.Event()
    original_write = _Client._write_physical

    async def gated(owner: _Client, payload: bytes) -> bool:
        if b'"seq":1' in payload and not entered.is_set():
            entered.set()
            await await_or_fail(release.wait(), what="release global-cap frame")
        return await original_write(owner, payload)

    # test-contract: seam retain real wire frames until the global limit is crossed
    monkeypatch.setattr(_Client, "_write_physical", gated)
    monkeypatch.setattr("zeta.server.session_replay.REPLAY_MAX_RECORDS", 2)
    try:
        await _publish(server, session_id)
        await await_or_fail(entered.wait(), what="first global-cap frame")
        await _publish(server, session_id, 2)
        release.set()
        frames = await _frames_before_status(client, 4)
        assert _wire(frames) == [
            ("turn_start", stream_id, 1),
            ("replay_stream_ended", stream_id, "evicted"),
        ], "global eviction did not end the stream with exactly one event"
        assert _ended(frames[1], session_id, stream_id, "evicted")
        await _publish(server, session_id)
        assert _wire(await _frames_before_status(client, 5)) == [("turn_start", None, None)]
    finally:
        release.set()
        await client.close()
        await server.close()


async def _stream_ended_unnegotiated(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    controller = await connect_replay(server, transport)
    plain = await connect_replay(server, transport)
    observer: ReplayClient | None = None
    try:
        await negotiate(controller, replay=False)
        await negotiate(plain, replay=False, observer=True)
        created = await controller.request(2, "new_session", {"provider": "codex"})
        old_session = created["result"]["session"]["session_id"]
        observer, stream_id = await _observer_with_stream(server, transport, old_session)
        await controller.request(3, "new_session", {"provider": "codex"})
        ended = await _frames_before_status(observer, 31)
        assert _wire(ended) == [("replay_stream_ended", stream_id, "session_replaced")]
        assert _ended(ended[0], old_session, stream_id, "session_replaced")
        assert all(
            frame.get("params", {}).get("event") != "replay_stream_ended"
            for frame in [*await _frames_before_status(controller, 4), *await _frames_before_status(plain, 5)]
        ), "a connection without session_replay_seq got replay_stream_ended"
    finally:
        if observer is not None:
            await observer.close()
        await plain.close()
        await controller.close()
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
    active_write_failure=_active_write_failure,
    active_admission_failure=_active_admission_failure,
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
    observer_without_join=_observer_without_join,
    join_failure=_join_failure,
    global_pressure_two_victims=_global_pressure_two_victims,
    join_during_swap=_join_during_swap,
    observer_write_failure=_observer_write_failure,
    old_generation_failure=_old_generation_failure,
    observer_settlement=_observer_settlement,
    observer_controller_absent=_observer_controller_absent,
    replay_only_backward_page=_replay_only_backward_page,
    controller_reconnect_keeps_observer=_controller_reconnect_keeps_observer,
    approval_reconnect_rejoin=_approval_reconnect_rejoin,
    global_cap_self_eviction=_global_cap_self_eviction,
    stream_ended_unnegotiated=_stream_ended_unnegotiated,
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
