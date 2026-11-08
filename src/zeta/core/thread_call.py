"""Cancellation-safe blocking calls that can return owned resources."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


def _close_abandoned_result[T](
    worker: asyncio.Future[T], close_on_cancel: Callable[[T], None]
) -> None:
    try:
        result = worker.result()
    except BaseException:  # noqa: BLE001 - consume the abandoned worker outcome
        return
    close_on_cancel(result)


async def run_must_complete_thread_call[T](function: Callable[[], T]) -> T:
    """Run blocking work to completion before propagating caller cancellation."""

    worker = asyncio.get_running_loop().run_in_executor(None, function)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    result = worker.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def run_owned_thread_call[T](
    function: Callable[[], T],
    close_on_cancel: Callable[[T], None] | None = None,
) -> T:
    """Run blocking work and close an untransferred result after cancellation."""

    worker = asyncio.get_running_loop().run_in_executor(None, function)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if close_on_cancel is not None:
            worker.add_done_callback(
                lambda completed: _close_abandoned_result(completed, close_on_cancel)
            )
        else:
            worker.add_done_callback(
                lambda completed: _close_abandoned_result(
                    completed, lambda _result: None
                )
            )
        raise
