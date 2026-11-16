from __future__ import annotations

import pytest

from tests.support.owned import await_or_fail
from tests.test_server_in_flight_replay import HOLD, GatedBackend, _server
from tests.test_server_session_park import FEATURES, _open


@pytest.mark.asyncio
async def test_exact_session_history_selects_a_parked_record_over_owned(tmp_path) -> None:
    backend = GatedBackend([([HOLD], [])])
    server = _server(tmp_path, backend)
    client = None
    try:
        client = await _open(server, FEATURES)
        session_a = await client.new()
        await client.call("send", {"text": "hold"})
        await await_or_fail(backend.held[0].wait(), what="held turn")
        session_b = await client.new()
        result = await client.call("session_history", {"session_id": session_a})
        assert result
        assert session_b != session_a
    finally:
        if client is not None:
            client.writer.close()
            await client.writer.wait_closed()
        await server.close()
