"""real-socket routing of the two record resolvers while two sessions are resident:
a send goes to the record the connection owns, and a history read of a named
session goes to that session's record, even when another record is owned."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import frames_until_event
from tests.test_server_in_flight_replay import HOLD, GatedBackend, _server
from tests.test_server_session_park import FEATURES, _open
from tests.test_server_turn_reconnect import _close
from zeta.protocol.types import TextContent


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

            async def send(params: Any, features: Any, original=deliveries.send, session_id=session_id):
                delivered.append(session_id)
                return await original(params, features)

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
