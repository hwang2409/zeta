from __future__ import annotations

import asyncio
import contextlib
from typing import Any


async def await_submission_pipeline(app: Any, *, timeout: float = 30.0) -> None:
    """Wait behind all submissions already queued by a TUI app."""

    if timeout < 10.0:
        raise ValueError("submission timeout must be at least 10 seconds")
    barrier = asyncio.create_task(app._submissions.submit_text(""))
    try:
        await asyncio.wait_for(asyncio.shield(barrier), timeout=timeout)
    except BaseException:
        barrier.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(barrier, timeout=timeout)
        raise
