from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import IO


async def create_log_off_loop(open_log: Callable[[], IO[bytes]]) -> None:
    """Create a log off-loop without transferring its handle."""

    def create() -> None:
        with open_log():
            pass

    await asyncio.to_thread(create)


async def open_log_off_loop(open_log: Callable[[], IO[bytes]]) -> IO[bytes]:
    """Open a log off-loop and close it if the caller is canceled."""

    worker = asyncio.create_task(asyncio.to_thread(open_log))
    try:
        return await asyncio.shield(worker)
    except BaseException:
        try:
            handle = await asyncio.shield(worker)
        except BaseException as cleanup_error:  # noqa: BLE001 - drain worker outcome
            cleanup_error.add_note("log creation failed during cancellation")
        else:
            handle.close()
        raise
