"""Committed background-task lifecycle change signal."""

from __future__ import annotations

import asyncio


class TaskChangeSignal:
    """Wake supervisors after a committed task start, exit, or registry close."""

    def __init__(self) -> None:
        self.generation = 0
        self.closed = False
        self._changed = asyncio.Event()

    def advance(self, *, wake: bool = True) -> None:
        self.generation += 1
        if wake:
            self._changed.set()

    def close(self) -> None:
        self.closed = True
        self.advance()

    async def wait(self, generation: int) -> None:
        while self.generation == generation:
            self._changed.clear()
            if self.generation != generation:
                return
            await self._changed.wait()
