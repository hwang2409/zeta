from __future__ import annotations

import asyncio
import threading

from zeta.session_retention.activation import RetentionActivationOwner


class _Planner:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release
        self.closed = False

    def run_pass(self, *, days: int):
        assert days == 30
        self.entered.set()
        self.release.wait(10)
        return type("Result", (), {"complete": True})()

    def close(self) -> None:
        self.closed = True


async def _wait(event: threading.Event) -> None:
    await asyncio.to_thread(event.wait, 10)


async def test_yield_to_open_closes_only_after_worker_pass() -> None:
    entered = threading.Event()
    release = threading.Event()
    planners: list[_Planner] = []

    def factory() -> _Planner:
        planner = _Planner(entered, release)
        planners.append(planner)
        return planner

    owner = RetentionActivationOwner(factory, lambda: 30, wake_seconds=0.01)
    owner.start()
    await _wait(entered)
    yielding = asyncio.create_task(owner.yield_to_open())
    await asyncio.sleep(0)
    assert not yielding.done()
    assert not planners[0].closed
    release.set()
    await asyncio.wait_for(yielding, 10)
    await owner.stop()
    assert planners[0].closed
    assert not owner.task


async def test_zero_days_skips_planner_pass() -> None:
    created = 0

    def factory():
        nonlocal created
        created += 1
        raise AssertionError("days=0 must not construct a planner")

    owner = RetentionActivationOwner(factory, lambda: 0, wake_seconds=0.01)
    owner.start()
    await asyncio.sleep(0)
    await owner.stop()
    assert created == 0
