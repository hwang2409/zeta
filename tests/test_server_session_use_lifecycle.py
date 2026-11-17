"""A session's control and runtime close only after every worker read of it
has returned, even when the read's waiter is gone.

Each test drives a real serve server (unix and websocket) over real sockets. The
history worker is gated by a ``threading.Event`` after it has read the store,
so the test decides when the worker returns; there are no sleeps and no
wall-clock thresholds.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from tests.test_server_session_park import FEATURES as PARK_FEATURES
from zeta.protocol.types import TextContent
from zeta.server.handlers import history as history_module

pytest_plugins = ("tests.support.socket_fixtures",)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


async def _spin(cond: Callable[[], bool], *, what: str) -> None:
    try:
        async with asyncio.timeout(10):
            while not cond():
                await asyncio.sleep(0)
    except TimeoutError:
        raise AssertionError(f"{what} did not happen within 10 s") from None


async def _settle_until(cond: Callable[[], bool], *, turns: int = 2000) -> None:
    """Yield up to ``turns`` loop turns for ``cond``; bounded by turns, not time."""
    for _ in range(turns):
        if cond():
            return
        await asyncio.sleep(0)


class _GatedRead:
    """Gate every history worker after it has read the store; record each close
    of the watched record with whether the worker had returned by then."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.reading = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.closes: list[tuple[str, bool]] = []
        original = history_module.history

        def gated(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.reading.set()
            assert self.release.wait(10), "history worker was not released"
            self.returned.set()
            return result

        monkeypatch.setattr(history_module, "history", gated)

    def watch(self, record: Any) -> None:
        for label, owner in (("control", record.control), ("runtime", record.runtime)):
            original = owner.close

            async def close(*args: Any, _label: str = label, _original: Any = original, **kwargs: Any) -> Any:
                self.closes.append((_label, self.returned.is_set()))
                return await _original(*args, **kwargs)

            self.monkeypatch.setattr(owner, "close", close)

    async def worker_reading(self) -> None:
        assert await await_or_fail(
            asyncio.to_thread(self.reading.wait, 10), what="history worker reads"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_shutdown_waits_for_a_cancelled_history_worker(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server close cancels the controller's history request while its worker
    still reads; the session's control and runtime close only after that worker
    returns, once each."""
    gate = _GatedRead(monkeypatch)
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    closing: asyncio.Task[None] | None = None
    try:
        await negotiate(client)
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        record = server.controllers.attached()
        gate.watch(record)
        connection = server._client
        assert connection is not None
        await client.send(3, "session_history", {"session_id": session_id})
        await gate.worker_reading()
        handler = connection._run_task
        assert handler is not None

        closing = asyncio.create_task(server.close())
        await _spin(
            lambda: handler.done() and record.closing,
            what="server close cancels the history request and retires the record",
        )
        assert handler.cancelled() or handler.done()
        await _settle_until(lambda: bool(gate.closes))
        assert gate.closes == [], "the session closed under a running history worker"

        gate.release.set()
        await await_or_fail(closing, what="server close")
        assert gate.closes == [("control", True), ("runtime", True)]
    finally:
        gate.release.set()
        if closing is not None:
            await await_or_fail(asyncio.gather(closing, return_exceptions=True), what="close")
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_parked_session_closes_after_an_observer_history_worker(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session A is parked behind a busy turn; an observer reads A's history and
    drops its socket while the worker still reads. A's turn ends, so A is idle
    and closes, but only after that worker returns, once each."""
    gate = _GatedRead(monkeypatch)
    backend = GatedBackend([([HOLD], [TextContent("a")])])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    try:
        await negotiate(controller, extra=tuple(PARK_FEATURES), protocol_version="1.2")
        created = await controller.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        record = server.controllers.attached()
        gate.watch(record)
        await controller.request(3, "send", {"text": "hold"})
        await await_or_fail(backend.held[0].wait(), what="A's turn is busy")
        parked = await controller.request(4, "new_session", {"provider": "codex"})
        assert server.parking.get(session_id) is record, parked

        await negotiate(observer, observer=True)
        await observer.send(2, "session_history", {"session_id": session_id})
        await gate.worker_reading()
        observer.writer.write_eof()
        backend.open_all()
        await _spin(
            lambda: record.closing,
            what="the idle parked session starts its close",
        )
        await _settle_until(lambda: bool(gate.closes))
        assert gate.closes == [], "the parked session closed under a running history worker"

        gate.release.set()
        await _spin(lambda: len(gate.closes) == 2, what="the parked session closes")
        assert gate.closes == [("control", True), ("runtime", True)]
    finally:
        gate.release.set()
        backend.open_all()
        await observer.close()
        await controller.close()
        await server.close()
