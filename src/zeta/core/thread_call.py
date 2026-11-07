"""Cancellation-safe calls that transfer worker-thread results."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_owned_thread_call[T](
    function: Callable[[], T],
    adopt: Callable[[T], None] | None = None,
) -> T:
    """Run blocking work and transfer any result before cancellation propagates."""

    worker = asyncio.create_task(asyncio.to_thread(function))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError:
            cancelled = True
    if adopt is not None:
        adopt(result)
    if cancelled:
        raise asyncio.CancelledError
    return result
