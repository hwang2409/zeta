"""Diagnostic log when a controller hello is refused for a held slot.

The slot_release lane names the holder on every -32001 controller refusal:
one bounded stderr line carrying the holder's and the refused connection's
transport, peer pid (unix) or remote address and User-Agent (websocket), hello
client version, connection age, time since the last received frame, running
request, and peer-closed state. When the holder's connection is already closed
while its request task still runs, the same line also carries the stuck request
and its await chain.

Every test crosses a real `zeta serve` over a unix socket and an authenticated
websocket. The rate limit uses an injected clock; transport-only states use a
bounded poll.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.server import ZetaServer
from zeta.server.handlers import registry
from zeta.server.protocol import MethodDescriptor

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")

_DIAG_LOGGER = "zeta.server.diagnostics"


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


async def _poll(predicate: Callable[[], bool], what: str) -> None:
    # Transport-level states (EOF seen, slot released) have no producer event;
    # yield to the loop until they hold, bounded by an explicit deadline.
    try:
        async with asyncio.timeout(10):
            while not predicate():
                await asyncio.sleep(0)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


async def _controller_hello(client: ReplayClient, request_id: int) -> dict[str, object]:
    return await client.request(
        request_id,
        "hello",
        {
            "protocol_version": "1.1",
            "features": ["session_replay_seq"],
            "client_version": "1.2.3",
        },
    )


def _diag_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _DIAG_LOGGER]


def _assert_only_user_agent(message: str, server: ZetaServer) -> None:
    # Only the User-Agent may leave the handshake; no token or other header.
    assert server._websocket is not None
    assert server._websocket.token not in message, message
    assert "Sec-WebSocket-Key" not in message, message
    assert "Host:" not in message, message


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_stuck_lost_holder_logs_identity_and_dump(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def gated_status(client, request_id, params):
        entered.set()
        await asyncio.wait_for(gate.wait(), 10)
        return {}

    monkeypatch.setitem(registry.HANDLERS, MethodDescriptor.STATUS, gated_status)

    async with _serve(tmp_path, transport) as server:
        holder = await connect_replay(server, transport)
        try:
            await negotiate(holder)
            # Start a request that blocks in its handler, then drop the socket.
            await holder.send(2, "status", {})
            registry_connections = server.connections
            assert registry_connections is not None
            await await_or_fail(entered.wait(), what="holder enters its request")
            await holder.close()
            await _poll(
                lambda: registry_connections._controller is not None
                and registry_connections._controller.connection.peer_closed(),
                what="serve sees the holder connection close",
            )

            with caplog.at_level(logging.WARNING, logger=_DIAG_LOGGER):
                first = await connect_replay(server, transport)
                try:
                    refused = await _controller_hello(first, 1)
                finally:
                    await first.close()
                assert refused["error"]["code"] == -32001

                records = _diag_records(caplog)
                assert len(records) == 1, [r.getMessage() for r in records]
                message = records[0].getMessage()
                # The holder identity names a stuck, peer-closed connection.
                assert "holder[" in message and "refused[" in message, message
                assert "peer_closed=yes" in message, message
                # Its stuck request and await chain are dumped.
                assert "method=status" in message, message
                assert "gated_status" in message, message
                if transport == "unix":
                    assert f"pid={os.getpid()}" in message, message
                else:
                    assert 'ua="zeta-replay-test"' in message, message
                    _assert_only_user_agent(message, server)

                # A second refused hello within the window adds no record.
                second = await connect_replay(server, transport)
                try:
                    again = await _controller_hello(second, 1)
                finally:
                    await second.close()
                assert again["error"]["code"] == -32001
                assert _diag_records(caplog) == records
        finally:
            gate.set()
            if server.connections is not None:
                await _poll(
                    lambda: server.connections is None
                    or server.connections._controller is None,
                    what="holder released after the gate opens",
                )


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_live_holder_refusal_logs_identity_without_dump(
    tmp_path: Path,
    transport: Transport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _serve(tmp_path, transport) as server:
        holder = await connect_replay(server, transport)
        try:
            await negotiate(holder)  # idle and open: the slot is held by a live holder
            with caplog.at_level(logging.WARNING, logger=_DIAG_LOGGER):
                refused_client = await connect_replay(server, transport)
                try:
                    refused = await _controller_hello(refused_client, 1)
                finally:
                    await refused_client.close()
                assert refused["error"]["code"] == -32001

                records = _diag_records(caplog)
                assert len(records) == 1, [r.getMessage() for r in records]
                message = records[0].getMessage()
                assert "holder[" in message and "refused[" in message, message
                assert "peer_closed=no" in message, message
                # A live holder refusal carries no stuck-request dump.
                assert "await chain" not in message, message
                assert "method=" not in message, message
                if transport == "websocket":
                    assert 'ua="zeta-replay-test"' in message, message
                    _assert_only_user_agent(message, server)
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
                        refused = await _controller_hello(client, request_id)
                    finally:
                        await client.close()
                    assert refused["error"]["code"] == -32001

                await refuse(1)  # first refusal logs
                assert len(_diag_records(caplog)) == 1
                assert "suppressed=0" in _diag_records(caplog)[0].getMessage()

                now[0] += 5.0  # inside the 30s window: suppressed
                await refuse(2)
                assert len(_diag_records(caplog)) == 1

                now[0] += 30.0  # past the window: logs again, counting the suppressed one
                await refuse(3)
                records = _diag_records(caplog)
                assert len(records) == 2, [r.getMessage() for r in records]
                assert "suppressed=1" in records[1].getMessage(), records[1].getMessage()
        finally:
            await holder.close()
