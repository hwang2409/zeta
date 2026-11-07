from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import IO

from ...core.thread_call import run_owned_thread_call


async def create_log_off_loop(open_log: Callable[[], IO[bytes]]) -> None:
    """Create a log off-loop without transferring its handle."""

    def create() -> None:
        with open_log():
            pass

    await asyncio.to_thread(create)


async def open_log_off_loop(open_log: Callable[[], IO[bytes]]) -> IO[bytes]:
    """Open a log off-loop and close it if the caller is canceled."""

    return await run_owned_thread_call(open_log, lambda handle: handle.close())
