"""Bound worker-backed agent cleanup without abandoning its resources."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable

SHUTDOWN_BUDGET_SECONDS = 2.0
_LATE_CLEANUP_TASKS: set[asyncio.Task[None]] = set()
_LOGGER = logging.getLogger(__name__)


async def finish_cleanup_or_handoff(
    tasks: Iterable[asyncio.Task[object]],
    close: Callable[[], None] | None,
    *,
    deadline: float,
    label: str,
) -> None:
    """Wait through the shared deadline, then retain unfinished cleanup."""

    async def finish() -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
        if close is not None:
            await asyncio.to_thread(close)

    cleanup = asyncio.create_task(finish())
    timeout = max(0.0, deadline - asyncio.get_running_loop().time())
    done, _ = await asyncio.wait({cleanup}, timeout=timeout)
    if done:
        await cleanup
        return

    _LATE_CLEANUP_TASKS.add(cleanup)

    def cleanup_done(task: asyncio.Task[None]) -> None:
        _LATE_CLEANUP_TASKS.discard(task)
        try:
            task.result()
        except BaseException:
            _LOGGER.warning("late %s cleanup failed", label, exc_info=True)

    cleanup.add_done_callback(cleanup_done)


__all__ = ["SHUTDOWN_BUDGET_SECONDS", "finish_cleanup_or_handoff"]
