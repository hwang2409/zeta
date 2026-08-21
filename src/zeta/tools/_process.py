"""Process-group cleanup shared by shell tools."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Sequence


def _signal_group(process: asyncio.subprocess.Process, signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass
    except OSError:
        if process.returncode is None:
            process.kill()


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    process_tasks: Sequence[asyncio.Task[object]],
) -> None:
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.shield(asyncio.sleep(0.1))
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
    _signal_group(process, signal.SIGKILL)
    for task in process_tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if not task.cancelled():
            task.exception()
