"""real-socket routing of the two record resolvers while two sessions are resident:
a send goes to the record the connection owns, and a history read of a named
session goes to that session's record, even when another record is owned."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import frames_until_event, read_raw, send
from tests.test_server import _connect
from tests.test_server_in_flight_replay import HOLD, GatedBackend, _server
from tests.test_server_session_park import FEATURES, _open
from tests.test_server_turn_reconnect import _close
from zeta.protocol.types import TextContent
from zeta.server.protocol import ErrorDescriptor, ProtocolError
from zeta.server.session_controllers import SessionControllers


async def _park_a_then_attach_b(server, backend: GatedBackend, a_text: str):
    """Session A holds a busy turn and is parked; session B is attached and owned."""
    client = await _open(server, FEATURES)
    a = await client.new()
    await client.call("send", {"text": a_text})
    await await_or_fail(backend.held[0].wait(), what="A's turn is busy")
    b = await client.new()
    assert [row["session_id"] for row in (await client.call("status"))["parked_sessions"]] == [a]
    return client, a, b


@pytest.mark.asyncio
async def test_send_delivers_only_to_the_owned_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GatedBackend([
        ([HOLD], [TextContent("a")]),
        ([TextContent("b")], [TextContent("b")]),
    ])
    server = _server(tmp_path, backend)
    client = None
    try:
        client, a, b = await _park_a_then_attach_b(server, backend, "A-only")
        delivered: list[str] = []
        for record in server.controllers.records():
            deliveries = record.control.deliveries
            session_id = record.runtime.session_id

            async def send(use: Any, params: Any, features: Any, original=deliveries.send, session_id=session_id):
                delivered.append(session_id)
                return await original(use, params, features)

            monkeypatch.setattr(deliveries, "send", send)
        assert {record.runtime.session_id for record in server.controllers.records()} == {a, b}

        response = (await client.frames("send", {"text": "B-only", "session_id": b}))[-1]
        assert (response.get("error"), delivered) == (None, [b])
        await frames_until_event(client.reader, "agent_end")
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_session_history_reads_the_selected_record_not_the_owned_one(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([
        ([HOLD], [TextContent("a")]),
        ([TextContent("b")], [TextContent("b")]),
    ])
    server = _server(tmp_path, backend)
    client = None
    try:
        client, a, _b = await _park_a_then_attach_b(server, backend, "A-only")
        await client.call("send", {"text": "B-only"})
        await frames_until_event(client.reader, "agent_end")

        response = (await client.frames("session_history", {"session_id": a}))[-1]
        text = json.dumps(response)
        assert ("A-only" in text, "B-only" in text) == (True, False), response
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_idle_record_replaced_by_reattach_stays_resident_until_its_close_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GatedBackend([([HOLD], [TextContent("a")])])
    server = _server(tmp_path, backend)
    client = None
    release = asyncio.Event()
    entered = asyncio.Event()
    try:
        client, a, b = await _park_a_then_attach_b(server, backend, "A-only")
        record_b = server.controllers.attached()
        assert record_b.runtime.session_id == b
        original = record_b.control.close

        async def held_close() -> None:
            entered.set()
            await await_or_fail(release.wait(), what="close release")
            await original()

        monkeypatch.setattr(record_b.control, "close", held_close)
        await client.call("resume", {"session_id": a})
        await await_or_fail(entered.wait(), what="B's close starts")
        # closing: not selectable, but still resident (and counted) until closed
        assert record_b in server.controllers.records()
        assert server.controllers.is_closing(record_b.control)
        assert server.controllers.closing_session_ids() == [b]
        release.set()
        await await_or_fail(server.controllers.wait_closed(b), what="B's close")
        assert record_b not in server.controllers.records()
    finally:
        release.set()
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_failed_new_session_setup_answers_only_after_its_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GatedBackend([([HOLD], [TextContent("a")])])
    server = _server(tmp_path, backend)
    client = None
    release = asyncio.Event()
    entered = asyncio.Event()
    incoming: list[Any] = []
    try:
        client = await _open(server, FEATURES)
        await client.new()
        await client.call("send", {"text": "A-only"})
        await await_or_fail(backend.held[0].wait(), what="A's turn is busy")
        new_pair = server.new_pair

        def failing_pair():
            reservation = new_pair()
            record = reservation.record
            incoming.append(record)
            runtime = record.runtime
            original = runtime.close

            async def create_session(**kwargs: Any) -> None:
                raise RuntimeError("setup failed")

            async def held_close() -> None:
                entered.set()
                await await_or_fail(release.wait(), what="close release")
                await original()

            monkeypatch.setattr(runtime, "create_session", create_session)
            monkeypatch.setattr(runtime, "close", held_close)
            return reservation

        monkeypatch.setattr(server, "new_pair", failing_pair)
        request = asyncio.create_task(client.frames("new_session", {}))
        await await_or_fail(entered.wait(), what="the failed pair's close starts")
        # the failed generation stays reserved until its close ends; the error
        # answers only after that
        assert incoming and incoming[0] in server.controllers.records()
        assert incoming[0].closing and not request.done()
        release.set()
        assert "error" in (await await_or_fail(request, what="new_session"))[-1]
        assert incoming[0] not in server.controllers.records()
    finally:
        release.set()
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_each_request_kind_resolves_its_expected_record_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One send resolves one owned record, one named-session read one selected,
    and a sessionless read none, on the real socket path."""
    backend = GatedBackend([
        ([HOLD], [TextContent("a")]),
        ([TextContent("b")], [TextContent("b")]),
    ])
    server = _server(tmp_path, backend)
    client = None
    calls: list[str] = []
    try:
        client, a, _b = await _park_a_then_attach_b(server, backend, "A-only")
        for name in ("owned", "selected"):
            original = getattr(SessionControllers, name)

            def spy(self, *args: Any, _name=name, _original=original, **kwargs: Any):
                calls.append(_name)
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(SessionControllers, name, spy)
        counts = {}
        for method, params in (
            ("send", {"text": "B-only"}),
            ("session_history", {"session_id": a}),
            ("list_sessions", {}),
        ):
            calls.clear()
            response = (await client.frames(method, params))[-1]
            assert "error" not in response, response
            counts[method] = list(calls)
        assert counts == {
            "send": ["owned"],
            "session_history": ["selected"],
            "list_sessions": [],
        }
        await frames_until_event(client.reader, "agent_end")
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


def _legacy_error(code: int, message: str, data: str = "") -> bytes:
    return (
        f'{{"jsonrpc":"2.0","id":"r","error":{{"code":{code},"message":"{message}"{data}}}}}\n'
    ).encode()


_UNHEALTHY = _legacy_error(
    -32000, "approval persistence failed", ',"data":{"code":"approval_persistence_failed"}'
)


# the bytes b6f24f2df answers with no attached session: a session mutation's
# health check still runs before its version and feature checks
@pytest.mark.parametrize(("version", "method", "params", "unhealthy", "expected"), [
    ("1.0", "rename_session", {"session_id": "x"}, True, _UNHEALTHY),
    ("1.0", "rename_session", {"session_id": "x"}, False, _legacy_error(-32601, "request requires protocol 1.1")),
    ("1.2", "clear_steering", {}, True, _UNHEALTHY),
    ("1.0", "model_catalog", {}, False, _legacy_error(-32601, "request requires protocol 1.1")),
    ("1.2", "content_read", {}, False, _legacy_error(-32601, "content_read requires the negotiated content_read feature")),
    ("1.2", "ping", {}, False, _legacy_error(-32601, "ping requires the negotiated ping feature")),
    ("1.0", "session_history", {"session_id": "x"}, False, _legacy_error(-32601, "request requires protocol 1.1")),
    ("1.2", "run_work_list", {}, False, _legacy_error(-32601, "run_work_list requires the negotiated run_work feature")),
    ("1.0", "list_projects", {}, False, _legacy_error(-32601, "list_projects requires the negotiated projects feature")),
])
@pytest.mark.asyncio
async def test_legacy_version_and_feature_errors_keep_base_order_without_a_session(
    tmp_path: Path, version: str, method: str, params: dict[str, Any], unhealthy: bool, expected: bytes
) -> None:
    server = _server(tmp_path, GatedBackend([]))
    reader, writer = await _connect(server)
    try:
        await send(reader, writer, 1, "hello", {"protocol_version": version, "features": []})
        await read_raw(reader)
        if unhealthy:
            def reject() -> None:
                raise ProtocolError(
                    ErrorDescriptor.APPROVAL_PERSISTENCE_FAILED, "approval persistence failed"
                )

            server.controllers.attached().control._approvals.reject_if_unhealthy = reject
        await send(reader, writer, "r", method, params)
        assert await read_raw(reader) == expected
    finally:
        await _close(server, writer)
