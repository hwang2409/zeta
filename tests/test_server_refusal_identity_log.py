"""Identity fields and rate limit on the controller -32001 refusal log.

Companion to test_server_stuck_holder_log.py: that file asserts the stuck-
holder dump; this file asserts the holder/refused identity fields the refusal
line names (peer pid on unix, remote address and User-Agent on websocket), that
no secret or other header leaks, and the per-holder rate limit with its
suppressed count. Every test crosses a real `zeta serve`; the rate limit uses
an injected clock.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")

_DIAG_LOGGER = "zeta.server.diagnostics"
_REFUSAL_PREFIX = "serve: controller hello refused -32001"


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


@asynccontextmanager
async def _serve(tmp_path: Path, transport: Transport) -> AsyncIterator[ZetaServer]:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    try:
        yield server
    finally:
        await server.close()


def _refusal_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == _DIAG_LOGGER and r.getMessage().startswith(_REFUSAL_PREFIX)
    ]


async def _refused_hello(client: ReplayClient, request_id: int) -> dict[str, object]:
    return await client.request(
        request_id,
        "hello",
        {
            "protocol_version": "1.1",
            "features": ["session_replay_seq"],
            "client_version": "1.2.3",
        },
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_refusal_names_both_identities(
    tmp_path: Path,
    transport: Transport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _serve(tmp_path, transport) as server:
        holder = await connect_replay(server, transport)
        try:
            await negotiate(holder)  # live holder
            with caplog.at_level(logging.WARNING, logger=_DIAG_LOGGER):
                refused_client = await connect_replay(server, transport)
                try:
                    refused = await _refused_hello(refused_client, 1)
                finally:
                    await refused_client.close()
                assert refused["error"]["code"] == -32001

                records = _refusal_records(caplog)
                assert len(records) == 1, [r.getMessage() for r in records]
                message = records[0].getMessage()
                assert "holder[" in message and "refused[" in message, message
                # The refused connection's hello client version is named.
                assert "client=1.2.3" in message, message
                if transport == "unix":
                    # Same-process peer: its pid is the test process pid.
                    assert f"pid={os.getpid()}" in message, message
                else:
                    assert 'ua="zeta-replay-test"' in message, message
                    # Only the User-Agent leaves the handshake: no token, no
                    # other header.
                    assert server._websocket is not None
                    assert server._websocket.token not in message, message
                    assert "Sec-WebSocket-Key" not in message, message
                    assert "Host:" not in message, message
        finally:
            await holder.close()


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_refusal_is_rate_limited_per_holder_with_suppressed_count(
    tmp_path: Path,
    transport: Transport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [1000.0]

    async with _serve(tmp_path, transport) as server:
        assert server.connections is not None
        server.connections._refusal_clock = lambda: now[0]
        holder = await connect_replay(server, transport)
        try:
            await negotiate(holder)
            with caplog.at_level(logging.WARNING, logger=_DIAG_LOGGER):
                async def refuse(request_id: int) -> None:
                    client = await connect_replay(server, transport)
                    try:
                        refused = await _refused_hello(client, request_id)
                    finally:
                        await client.close()
                    assert refused["error"]["code"] == -32001

                await refuse(1)  # first refusal logs
                assert len(_refusal_records(caplog)) == 1
                assert "suppressed=0" in _refusal_records(caplog)[0].getMessage()

                now[0] += 5.0  # inside the 30s window: suppressed
                await refuse(2)
                assert len(_refusal_records(caplog)) == 1

                now[0] += 30.0  # past the window: logs again, counting the suppressed one
                await refuse(3)
                records = _refusal_records(caplog)
                assert len(records) == 2, [r.getMessage() for r in records]
                assert "suppressed=1" in records[1].getMessage(), records[1].getMessage()
        finally:
            await holder.close()
