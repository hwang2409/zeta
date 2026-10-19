from __future__ import annotations

import asyncio
from collections.abc import Callable


class NotificationWakeRetry:
    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._handle: asyncio.TimerHandle | None = None
        self._delay = 1.0

    def schedule(self) -> None:
        if self._handle is None:
            self._handle = asyncio.get_running_loop().call_later(self._delay, self._run)
            self._delay = min(self._delay * 2, 30.0)

    def _run(self) -> None:
        self._handle = None
        self._callback()

    def attempt(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            self.schedule()
            raise
        self.success()

    def success(self) -> None:
        self._delay = 1.0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
