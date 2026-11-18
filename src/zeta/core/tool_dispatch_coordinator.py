"""FIFO admission for one tool dispatch scope."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ToolDispatchCoordinator:
    """Hold a slot through execution; exclusive calls fence both sides of the queue."""

    def __init__(self, max_parallel: int) -> None:
        self._max_parallel = max_parallel
        self._waiting: deque[object] = deque()
        self._active = 0
        self._exclusive = False
        self._changed = asyncio.Event()

    def _wake(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    @asynccontextmanager
    async def slot(self, *, parallel_safe: bool) -> AsyncIterator[None]:
        ticket = object()
        self._waiting.append(ticket)
        try:
            while (self._waiting[0] is not ticket or self._exclusive
                   or self._active >= self._max_parallel
                   or (not parallel_safe and self._active)):
                await self._changed.wait()
        except BaseException:
            self._waiting.remove(ticket)
            self._wake()
            raise
        self._waiting.popleft()
        self._active += 1
        self._exclusive = not parallel_safe
        self._wake()
        try:
            yield
        finally:
            self._active -= 1
            self._exclusive = False
            self._wake()
