"""The serve read owner settles every read future on the event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tests.support.owned import await_or_fail
from zeta.server.connections import ConnectionRegistry

ROWS = (
    "success",
    "failure",
    "cancel_queued",
    "cancel_running",
    "close_queued",
    "close_running",
    "read_after_close",
    "duplicate_close",
)


def _fail() -> object:
    raise ValueError("read failed")


@pytest.mark.parametrize("row", ROWS)
async def test_read_owner_settles_every_read(
    row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    registry = ConnectionRegistry(asyncio.get_running_loop())
    release = threading.Event()
    workers: list[threading.Thread] = []
    queued_ran = threading.Event()
    tasks: list[asyncio.Task[object]] = []

    def hold(entered: threading.Event) -> object:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(30), "read worker was not released"
        return "late"

    async def hold_workers(count: int) -> None:
        gates = [threading.Event() for _ in range(count)]
        for gate in gates:
            tasks.append(
                asyncio.create_task(registry.run_blocking(lambda g=gate: hold(g)))
            )
        for gate in gates:
            assert await asyncio.to_thread(gate.wait, 10)

    async def queue_read() -> asyncio.Task[object]:
        task = asyncio.create_task(registry.run_blocking(queued_ran.set))
        tasks.append(task)
        await asyncio.sleep(0)  # let the task enqueue; not a timing proof
        return task

    async def settled(what: str) -> list[object]:
        return await await_or_fail(
            asyncio.gather(*tasks, return_exceptions=True), what=what
        )

    try:
        if row == "success":
            assert await await_or_fail(registry.run_blocking(lambda: 7), what="read") == 7
        elif row == "failure":
            with pytest.raises(ValueError, match="read failed"):
                await await_or_fail(registry.run_blocking(_fail), what="read")
        elif row == "cancel_queued":
            await hold_workers(2)
            queued = await queue_read()
            queued.cancel()
            await await_or_fail(
                asyncio.gather(queued, return_exceptions=True), what="queued cancel"
            )
            release.set()
            await settled("held reads after a queued cancel")
            assert queued.cancelled()
            assert [task.result() for task in tasks[:2]] == ["late", "late"]
        elif row == "cancel_running":
            await hold_workers(1)
            tasks[0].cancel()
            await settled("cancelled running read")
            assert tasks[0].cancelled()
            release.set()
            # The late hand-back is dropped and the owner keeps serving.
            assert await await_or_fail(registry.run_blocking(lambda: 8), what="read") == 8
        elif row in {"close_queued", "close_running", "duplicate_close"}:
            await hold_workers(2 if row == "close_queued" else 1)
            if row == "close_queued":
                await queue_read()
            await registry.aclose()
            if row == "duplicate_close":
                await registry.aclose()
            # The workers are still held: only close can settle these reads.
            await settled(f"{row} reads")
            assert all(task.cancelled() for task in tasks)
        else:
            await registry.aclose()
            with pytest.raises(RuntimeError, match="reads are closed"):
                await registry.run_blocking(lambda: 9)
    finally:
        release.set()
        await registry.aclose()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
    for worker in workers:
        await asyncio.to_thread(worker.join, 10)
        assert not worker.is_alive(), f"{worker.name} did not exit after close"
    assert not queued_ran.is_set(), "a cancelled or closed queued read ran"
    assert not errors
