"""Explicit completion controls for tests that call providers directly."""

import asyncio

from zeta.providers.completion import (
    CompletionControl,
    CompletionRetryBudget,
    DeadlineClock,
    ResetPolicy,
)


class ManualDeadlineClock(DeadlineClock):
    def __init__(self) -> None:
        self.current = 0.0
        self.waiters: list[tuple[float, asyncio.Future[None]]] = []
        self.waiter_added = asyncio.Event()

    def now(self) -> float:
        return self.current

    async def wait_until(self, deadline: float) -> None:
        if deadline <= self.current:
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((deadline, future))
        self.waiter_added.set()
        try:
            await future
        finally:
            self.waiters = [item for item in self.waiters if item[1] is not future]

    def advance(self, value: float) -> None:
        self.current = value
        for deadline, future in tuple(self.waiters):
            if deadline <= value and not future.done():
                future.set_result(None)

    async def expire_next(self) -> None:
        while not self.waiters:
            self.waiter_added.clear()
            await self.waiter_added.wait()
        for _ in range(10):
            await asyncio.sleep(0)
        while not self.waiters:
            self.waiter_added.clear()
            await self.waiter_added.wait()
        deadline, future = min(self.waiters, key=lambda item: item[0])
        self.advance(deadline)
        while any(waiter is future for _, waiter in self.waiters):
            await asyncio.sleep(0)


def completion_control(
    *, reset: ResetPolicy = ResetPolicy.NO_RESET
) -> CompletionControl:
    return CompletionControl(
        abort=None,
        reset=reset,
        budget=CompletionRetryBudget(),
    )
