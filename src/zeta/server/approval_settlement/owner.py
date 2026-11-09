"""One owner for approval lifecycle background tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable

ApprovalKey = str | tuple[str, str]

logger = logging.getLogger(__name__)


class TaskOwner:
    """Create, observe, and bounded-drain approval lifecycle tasks."""

    def __init__(self) -> None:
        self._tasks: dict[asyncio.Task[object], ApprovalKey] = {}

    @property
    def tasks(self) -> dict[asyncio.Task[object], ApprovalKey]:
        return self._tasks

    def start(
        self, key: ApprovalKey, operation: Awaitable[object], kind: str
    ) -> asyncio.Task[object]:
        task = asyncio.create_task(operation)
        self._tasks[task] = key
        task.add_done_callback(lambda task: self._finished(task, kind))
        return task

    def _finished(self, task: asyncio.Task[object], kind: str) -> None:
        key = self._tasks.pop(task, None)
        if key is None or task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "approval settlement task failed",
                extra={"approval_key": key, "cause": repr(error), "kind": kind},
                exc_info=(type(error), error, error.__traceback__),
            )

    async def drain(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._tasks and (remaining := deadline - loop.time()) > 0:
            done, _ = await asyncio.wait(tuple(self._tasks), timeout=remaining)
            for task in done:
                self._finished(task, "settlement")
        if not self._tasks:
            return
        stuck = tuple(self._tasks)
        logger.error(
            "approval settlement tasks stuck during close",
            extra={"approval_keys": [self._tasks[task] for task in stuck]},
        )
        for task in stuck:
            task.cancel()
        if (remaining := deadline - loop.time()) > 0:
            await asyncio.wait(stuck, timeout=remaining)
