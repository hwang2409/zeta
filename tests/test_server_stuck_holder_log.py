"""Diagnostic log when a controller hello is refused while the holder is gone.

Slice 0 of the slot_release lane: on a -32001 refusal, when the holder's
connection is already closed (EOF / transport lost) but its request task still
runs, serve writes one bounded warning naming the stuck request and its await
chain. A live holder still gets the immediate -32001 and no dump.

Every test crosses a real `zeta serve` over a unix socket and an authenticated
websocket.
"""

from __future__ import annotations

import asyncio
import logging
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
    async def loop() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await await_or_fail(loop(), what=what)


async def _controller_hello(client: ReplayClient, request_id: int) -> dict[str, object]:
    return await client.request(
        request_id,
        "hello",
        {"protocol_version": "1.1", "features": ["session_replay_seq"]},
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.asyncio
async def test_stuck_lost_holder_is_logged_once(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = asyncio.Event()

    async def gated_status(client, request_id, params):
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
            await _poll(
                lambda: (
                    registry_connections._controller is not None
                    and getattr(
                        registry_connections._controller.connection, "_request", None
                    )
                    is not None
                ),
                what="holder enters its request",
            )
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

                records = [r for r in caplog.records if r.name == _DIAG_LOGGER]
                assert len(records) == 1, [r.getMessage() for r in records]
                message = records[0].getMessage()
                assert "method=status" in message, message
                assert "gated_status" in message, message  # an await-chain stack frame

                # A second refused hello against the same stuck holder adds no record.
                second = await connect_replay(server, transport)
                try:
                    again = await _controller_hello(second, 1)
                finally:
                    await second.close()
                assert again["error"]["code"] == -32001
                assert [r for r in caplog.records if r.name == _DIAG_LOGGER] == records
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
async def test_live_holder_refusal_has_no_dump(
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
                assert [r for r in caplog.records if r.name == _DIAG_LOGGER] == []
        finally:
            await holder.close()
