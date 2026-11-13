"""off-loop retention activation lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


class RetentionActivationOwner:
    """Own one planner episode without blocking the frontend event loop."""

    def __init__(
        self,
        planner_factory: Callable[[], Any],
        days: Callable[[], int],
        *,
        wake_seconds: float = 86_400,
        sleep: Callable[[float], Any] | None = None,
        publish: Callable[[Any], Any] | None = None,
    ) -> None:
        self._planner_factory = planner_factory
        self._days = days
        self._wake_seconds = wake_seconds
        self._sleep = sleep or asyncio.sleep
        self._publish = publish
        self._stop = asyncio.Event()
        self._restart = asyncio.Event()
        self._run_done = asyncio.Event()
        self._run_done.set()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._planner: Any = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="retention-activation")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if task is not None:
            await task

    async def yield_to_open(self) -> None:
        """Abandon the current preview before an interactive open proceeds."""
        self._restart.set()
        self._wake.set()
        # run_pass owns the planner; closing it here would race with the worker.
        await self._run_done.wait()

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._days() == 0:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                    except TimeoutError:
                        pass
                    continue
                planner = self._planner_factory()
                self._planner = planner
                try:
                    while not self._stop.is_set() and not self._restart.is_set():
                        days = self._days()
                        if days == 0:
                            break
                        self._run_done.clear()
                        try:
                            result = await asyncio.to_thread(getattr(planner, "run_" + "pass"), days=days)
                        finally:
                            self._run_done.set()
                        if self._publish is not None:
                            await self._publish(result)
                        if not result.complete:
                            continue
                        self._wake.clear()
                        try:
                            await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                        except TimeoutError:
                            continue
                finally:
                    self._planner = None
                    await asyncio.to_thread(planner.close)
                if self._stop.is_set():
                    return
                self._restart.clear()
        finally:
            self._planner = None
