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

from .runner import run_claimed
from .store import SQLiteStore
from .tick import tick
from .webhook import DEFAULT_WEBHOOK_PORT, WebhookServer

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


async def _wait(stop: asyncio.Event, wake: asyncio.Event, interval: float) -> None:
    stop_task = asyncio.create_task(stop.wait())
    wake_task = asyncio.create_task(wake.wait())
    try:
        await asyncio.wait(
            {stop_task, wake_task},
            timeout=interval,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in (stop_task, wake_task):
            task.cancel()
        await asyncio.gather(stop_task, wake_task, return_exceptions=True)
    wake.clear()


async def serve(
    home: Path,
    *,
    stop: asyncio.Event | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    interval: float = 30,
    runner: Callable[..., Awaitable[None]] = run_claimed,
    webhook_host: str = "127.0.0.1",
    webhook_port: int = DEFAULT_WEBHOOK_PORT,
    allow_non_loopback: bool = False,
    on_ready: Callable[[str, int], None] | None = None,
) -> None:
    if interval <= 0:
        raise ValueError("daemon interval must be positive")
    stopped = stop or asyncio.Event()
    wake = asyncio.Event()
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
            webhook = WebhookServer(
                store,
                host=webhook_host,
                port=webhook_port,
                allow_non_loopback=allow_non_loopback,
                now=clock,
                wake=lambda: loop.call_soon_threadsafe(wake.set),
            )
            address = webhook.start()
            try:
                logger.info("webhook receiver listening on %s:%s", *address)
                if on_ready is not None:
                    on_ready(*address)
                while not stopped.is_set():
                    if worker is not None and worker.done():
                        error = None if worker.cancelled() else worker.exception()
                        if error is not None:
                            logger.error("automation worker failed: %s", error)
                            if active_id is not None:
                                store.fail_unfinished(active_id, str(error))
                        worker = None
                        active_id = None
                    if worker is None:
                        try:
                            claimed = store.claim_webhook(clock())
                        except Exception as exc:
                            logger.exception("failed to claim webhook delivery")
                            store.interrupt_oldest_webhook(str(exc))
                            claimed = None
                        if claimed is not None:
                            active_id = claimed.run_id
                            worker = asyncio.create_task(
                                runner(
                                    store,
                                    claimed.occurrence,
                                    claimed.run_id,
                                    home=home,
                                    webhook_body=claimed.delivery.body,
                                    webhook_headers=claimed.delivery.headers,
                                )
                            )
                        else:
                            for occurrence in tick(store, clock()):
                                active_id = store.claim(occurrence)
                                if active_id is not None:
                                    worker = asyncio.create_task(
                                        runner(
                                            store,
                                            occurrence,
                                            active_id,
                                            home=home,
                                        )
                                    )
                                    break
                    await _wait(stopped, wake, interval)
            finally:
                if worker is not None:
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                webhook.close()
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
