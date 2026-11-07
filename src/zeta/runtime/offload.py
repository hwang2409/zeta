"""Cancellation-safe adapters for synchronous runtime work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def await_joined_offload[**P, R](
    call: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run one thread call and never let cancellation detach its worker."""
    worker = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
    first_cancel: asyncio.CancelledError | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as exc:
            if first_cancel is None:
                first_cancel = exc
    try:
        result = worker.result()
    except BaseException:
        if first_cancel is not None:
            raise first_cancel
        raise
    if first_cancel is not None:
        raise first_cancel
    return result
