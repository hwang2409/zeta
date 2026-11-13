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
    ) -> None:
        self._planner_factory = planner_factory
        self._days = days
        self._wake_seconds = wake_seconds
        self._sleep = sleep or asyncio.sleep
        self._stop = asyncio.Event()
        self._restart = asyncio.Event()
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
        task, self._task = self._task, None
        if task is not None:
            await task

    async def yield_to_open(self) -> None:
        """Abandon the current preview before an interactive open proceeds."""
        self._restart.set()
        planner = self._planner
        if planner is not None:
            await asyncio.to_thread(planner.close)
        self._restart.clear()

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                planner = self._planner_factory()
                self._planner = planner
                try:
                    while not self._stop.is_set() and not self._restart.is_set():
                        result = await asyncio.to_thread(getattr(planner, "run_" + "pass"), days=self._days())
                        if not result.complete:
                            continue
                        try:
                            await asyncio.wait_for(self._stop.wait(), self._wake_seconds)
                        except TimeoutError:
                            continue
                finally:
                    self._planner = None
                    await asyncio.to_thread(planner.close)
                if self._stop.is_set():
                    return
        finally:
            self._planner = None
