"""Cancellation-safe calls that transfer worker-thread results."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_owned_thread_call[T](
    function: Callable[[], T],
    adopt: Callable[[T], None] | None = None,
) -> T:
    """Run blocking work and transfer any result before cancellation propagates.

    ``adopt`` must be synchronous and non-blocking. If it raises, its exception
    takes precedence over cancellation and it must clean up the result itself.
    """

    worker = asyncio.get_running_loop().run_in_executor(None, function)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            cancelled = cancelled or (task is not None and task.cancelling() > 0)
            if worker.done():
                break
    result = worker.result()
    if adopt is not None:
        adopt(result)
    if cancelled:
        raise asyncio.CancelledError
    return result
