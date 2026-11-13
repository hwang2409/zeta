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


async def test_yield_to_open_runs_operation_after_worker_pass(tmp_path) -> None:
    opened = threading.Event()
    owner = RetentionActivationOwner(
        lambda: _Planner(threading.Event(), threading.Event()),
        lambda: 0,
        home=tmp_path,
    )
    result = await owner.yield_to_open(lambda: (opened.set(), "opened")[1])
    assert result == "opened"
    assert opened.is_set()


async def test_yield_to_open_waits_for_planner_close() -> None:
    closed = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    class Planner(_Planner):
        def close(self) -> None:
            closed.set()
            super().close()

    owner = RetentionActivationOwner(
        lambda: Planner(entered, release), lambda: 30, wake_seconds=10
    )
    owner.start()
    await _wait(entered)
    opening = asyncio.create_task(owner.yield_to_open(lambda: closed.is_set()))
    await asyncio.to_thread(entered.wait, 10)
    release.set()
    assert await asyncio.wait_for(opening, 10)
    await owner.stop()
    assert not owner.task


async def test_yield_to_open_returns_when_owner_crashed() -> None:
    owner = RetentionActivationOwner(
        lambda: (_ for _ in ()).throw(RuntimeError("planner failed")), lambda: 30
    )
    owner.start()
    await asyncio.sleep(0)
    assert await asyncio.wait_for(owner.yield_to_open(), 10) is None
    await owner.stop()


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
