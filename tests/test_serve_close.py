from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tests.support.serve_close import close_serve


@dataclass
class _Server:
    fail: bool = False

    def __post_init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def close(self) -> None:
        self.started.set()
        await asyncio.sleep(0)
        self.closed = True
        if self.fail:
            raise RuntimeError("server close failed")


class _Writer:
    def __init__(self, server: _Server, *, fail: bool = False) -> None:
        self.server = server
        self.fail = fail
        self.closed = False
        self.wait_started_with_server_shutdown = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_started_with_server_shutdown = self.server.started.is_set()
        assert self.wait_started_with_server_shutdown, (
            "serve shutdown must start before waiting for a peer writer to close"
        )
        if self.fail:
            raise RuntimeError("writer close failed")


@pytest.mark.asyncio
async def test_close_serve_starts_server_shutdown_before_peer_wait() -> None:
    server = _Server()
    writer = _Writer(server)

    await close_serve(server, writer)

    assert server.closed
    assert writer.closed
    assert writer.wait_started_with_server_shutdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer_count", "writer_failure", "server_failure"),
    [
        pytest.param(1, False, False, id="success"),
        pytest.param(1, True, False, id="writer-wait-failure"),
        pytest.param(1, False, True, id="server-close-failure"),
        pytest.param(2, False, False, id="multiple-writers"),
    ],
)
async def test_close_serve_cleanup_outcomes(
    writer_count: int, writer_failure: bool, server_failure: bool
) -> None:
    server = _Server(fail=server_failure)
    writers = [
        _Writer(server, fail=writer_failure and index == 0)
        for index in range(writer_count)
    ]

    outcome: BaseException | None = None
    try:
        await close_serve(server, *writers)
    except RuntimeError as error:
        outcome = error

    assert server.closed
    assert all(writer.closed for writer in writers)
    assert all(writer.wait_started_with_server_shutdown for writer in writers)
    if writer_failure:
        assert isinstance(outcome, RuntimeError)
        assert str(outcome) == "writer close failed"
    elif server_failure:
        assert isinstance(outcome, RuntimeError)
        assert str(outcome) == "server close failed"
    else:
        assert outcome is None
