"""Single-worker daemon; clocks and waiting live outside tick."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import signal
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import DueOccurrence
from .runner import run_claimed
from .store import SQLiteStore
from .tick import tick

logger = logging.getLogger(__name__)


@contextmanager
def daemon_lock(home: Path):
    directory = home / "automations"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "daemon.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "an automation daemon is already running for this ZETA_HOME"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


async def serve(
    home: Path,
    *,
    stop: asyncio.Event | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    interval: float = 30,
    runner: Callable[..., Awaitable[None]] = run_claimed,
) -> None:
    if interval <= 0:
        raise ValueError("daemon interval must be positive")
    stopped = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = []
    if stop is None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
            installed.append(sig)
    worker: asyncio.Task[None] | None = None
    active_id: str | None = None
    try:
        with daemon_lock(home), SQLiteStore(home) as store:
            store.recover()
            try:
                while not stopped.is_set():
                    if worker is not None and worker.done():
                        error = None if worker.cancelled() else worker.exception()
                        if error is not None:
                            logger.error("automation worker failed: %s", error)
                            if active_id is not None:
                                store.fail_unfinished(active_id, str(error))
                        worker = None
                    due: tuple[DueOccurrence, ...] = tick(store, clock())
                    if worker is None:
                        for occurrence in due:
                            active_id = store.claim(occurrence)
                            if active_id is not None:
                                worker = asyncio.create_task(
                                    runner(store, occurrence, active_id, home=home)
                                )
                                break
                    try:
                        await asyncio.wait_for(stopped.wait(), timeout=interval)
                    except TimeoutError:
                        pass
            finally:
                if worker is not None:
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
