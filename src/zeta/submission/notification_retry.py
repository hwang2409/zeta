"""Bounded retry state for notification-driven turns."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal

RetryKind = Literal["wake", "history"]


class NotificationRetry:
    """Own one pending notification retry and its backoff timer."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._call_later = lambda delay, timer_callback: (
            asyncio.get_running_loop().call_later(delay, timer_callback)
        )
        self._kind: RetryKind | None = None
        self._ready = False
        self._handle: asyncio.TimerHandle | None = None
        self._delay = 1.0
        self._failure_reported = False

    @property
    def ready_kind(self) -> RetryKind | None:
        return self._kind if self._ready else None

    @property
    def pending(self) -> bool:
        return self._kind is not None

    @property
    def failure_reported(self) -> bool:
        return self._failure_reported

    def request_wake(self) -> None:
        if self._kind is None:
            self._kind = "wake"
            self._ready = True

    def started(self) -> None:
        self._ready = False

    def failed(self, kind: RetryKind) -> bool:
        """Schedule one retry and report whether this starts a failure episode."""

        self._kind = kind
        self._ready = False
        first_failure = not self._failure_reported
        self._failure_reported = True
        if self._handle is None:
            self._handle = self._call_later(self._delay, self._fire)
            self._delay = min(self._delay * 2, 30.0)
        return first_failure

    def recovered(self) -> None:
        self._kind = None
        self._ready = False
        self._delay = 1.0
        self._failure_reported = False
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def close(self) -> None:
        self.recovered()

    def _fire(self) -> None:
        self._handle = None
        self._ready = True
        self._callback()
