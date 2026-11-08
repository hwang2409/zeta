from __future__ import annotations

import asyncio
import contextlib
import fcntl
import gc
import multiprocessing
import os
from pathlib import Path

import pytest

from tests.test_session_execution_owner import (
    _cleanup_process,
    _created_session,
    _join,
    _recv,
    _spawn,
    _try_open,
)
from zeta.core.session import SessionManager
from zeta.server import server as server_module
from zeta.server.server import ZetaServer
from zeta.storage_format import drain_releases, release_later

_DEADLINE_SECONDS = 10


def _exclusive_flock_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


class _DrainProbe:
    """Counts serve's idle drains so the test waits on them, not on a clock."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._count = 0
        self._waiters: list[tuple[int, asyncio.Future[None]]] = []

    def __call__(self) -> None:
        drain_releases()
        self._loop.call_soon_threadsafe(self._ran)

    def _ran(self) -> None:
        self._count += 1
        for target, waiter in self._waiters:
            if self._count >= target and not waiter.done():
                waiter.set_result(None)

    async def idle_ticks(self) -> None:
        """Wait for two drains: the first may have started before the caller."""

        waiter = self._loop.create_future()
        self._waiters.append((self._count + 2, waiter))
        # on the base no drain runs; the release assertion that follows fails
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(waiter, _DEADLINE_SECONDS)


def _failing_release() -> None:
    raise RuntimeError("release failed")


def test_idle_serve_releases_gc_dropped_session_to_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base fails at the first release assertion: serve keeps the flock."""

    # raising=False lets the unchanged base reach the behavior assertion
    monkeypatch.setattr(server_module, "IDLE_DRAIN_INTERVAL", 0.01, raising=False)
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    lease_path = home / "sessions" / session_id / "runtime.lease"
    manager = SessionManager(home, owner_surface="serve")

    async def exercise() -> None:
        probe = _DrainProbe(asyncio.get_running_loop())
        monkeypatch.setattr(server_module, "drain_releases", probe, raising=False)
        server = ZetaServer(home=home, port=0, provider="codex")
        await server.start()
        try:
            store = manager.open(session_id).store
            store._cycle = store  # only the collector can free it now
            del store
            # a failed release ahead of the store's must not stop the drain
            release_later(_failing_release)
            gc.collect()
            assert not _exclusive_flock_free(lease_path), (
                "the finalizer must defer the release, not run it"
            )

            await probe.idle_ticks()
            assert _exclusive_flock_free(lease_path), (
                "an idle serve kept a GC-released session lease"
            )

            result_recv, result_send = multiprocessing.Pipe(duplex=False)
            contender = _spawn(_try_open, str(home), session_id, result_send)
            try:
                result = _recv(result_recv)
                _join(contender)
            finally:
                _cleanup_process(contender)
                result_recv.close()
                result_send.close()
            assert result[0] == "opened", result

            # the tick survived the failed release and keeps draining
            again = manager.open(session_id).store
            again._cycle = again
            del again
            gc.collect()
            await probe.idle_ticks()
            assert _exclusive_flock_free(lease_path), (
                "the idle drain stopped after a failed release"
            )
        finally:
            drain = getattr(server, "_idle_drain", None)
            await asyncio.wait_for(server.close(), _DEADLINE_SECONDS)
        assert drain is not None and drain.cancelled()
        assert server._idle_drain is None

    try:
        asyncio.run(exercise())
    finally:
        manager._release_storage_lease()
