"""Bookkeeping for MCP client closes that ignored their bounded wait.

A client close() can ignore cancellation (anyio/httpx against a restarting
peer), so after `_drain_closes` cancels it the task may still be running. We
hold a strong reference so it is not garbage-collected while pending, keep at
most one such task per server so repeated re-binds cannot stack leaked tasks,
and count the live set process-wide for diagnostics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

_ABANDONED_CLOSES: set[asyncio.Task[object]] = set()


def live_abandoned_closes() -> int:
    """Number of abandoned client closes still running process-wide."""
    return len(_ABANDONED_CLOSES)


class AbandonedCloses:
    """Per-server owner of timed-out, cancelled-but-still-running closes."""

    def __init__(self) -> None:
        self._live: set[asyncio.Task[object]] = set()

    def __len__(self) -> int:
        return len(self._live)

    def __iter__(self) -> Iterator[asyncio.Task[object]]:
        return iter(self._live)

    def abandon(self, task: asyncio.Task[object]) -> None:
        """Track a timed-out close, keeping at most one live per server.

        An uncancellable close may never end, so a later one would only stack
        another leaked task. The event loop keeps the dropped task alive while
        it is pending; it is collected once it finally finishes.
        """
        if any(not pending.done() for pending in self._live):
            return
        self._live.add(task)
        _ABANDONED_CLOSES.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[object]) -> None:
        self._live.discard(task)
        _ABANDONED_CLOSES.discard(task)
