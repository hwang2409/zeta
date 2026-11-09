"""Cancellation-safe blocking calls that can return owned resources."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_owned_thread_call[T](
    function: Callable[[], T],
    close_on_cancel: Callable[[T], None] | None = None,
) -> T:
    """Run blocking work to completion before propagating cancellation."""

    worker = asyncio.get_running_loop().run_in_executor(None, function)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            task = asyncio.current_task()
            if not worker.done() or not (
                cancelled or (task is not None and task.cancelling() > 0)
            ):
                raise
            cancelled = True
            break
    task = asyncio.current_task()
    cancelled = cancelled or (task is not None and task.cancelling() > 0)
    try:
        result = worker.result()
    except BaseException:
        task = asyncio.current_task()
        if cancelled or (task is not None and task.cancelling() > 0):
            raise asyncio.CancelledError
        raise
    if cancelled:
        if close_on_cancel is not None:
            close_on_cancel(result)
        raise asyncio.CancelledError
    return result
