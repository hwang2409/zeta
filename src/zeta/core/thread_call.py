"""Off-loop calls whose result is closed when the awaiting side gives it up.

One owner for blocking work run on a worker thread: each call has one state
under a lock (pending, result_ready, taken, abandoned). Worker threads touch
only that state; only the loop touches the asyncio future. When the awaiting
side cancels, or the loop is closed before the hand-back, the result is closed
exactly once by whichever side comes second.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Closing these off the loop thread calls loop methods from the wrong thread.
_LOOP_AFFINE = (asyncio.Future, asyncio.BaseTransport, asyncio.StreamWriter)


class _OwnedCall[T]:
    __slots__ = (
        "_close", "_function", "_future", "_lock", "_loop", "_outcome", "_state"
    )

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        function: Callable[[], T],
        close: Callable[[T], None] | None,
    ) -> None:
        self._loop = loop
        self._function = function
        self._close = close
        self._lock = threading.Lock()
        self._state: Literal["pending", "result_ready", "taken", "abandoned"] = (
            "pending"
        )
        self._outcome: tuple[Any, BaseException | None] = (None, None)
        self._future: asyncio.Future[T] = loop.create_future()

    def run(self) -> None:
        """Run on a worker thread; never raises."""
        with self._lock:
            if self._state == "abandoned":
                return  # cancelled before start
        try:
            outcome: tuple[Any, BaseException | None] = (self._function(), None)
        except BaseException as exc:  # noqa: BLE001 - every outcome goes to the loop
            outcome = (None, exc)
        result = outcome[0]
        if self._close is not None and isinstance(result, _LOOP_AFFINE):
            # Refused: this result cannot be closed safely off the loop.
            refused = f"owned thread call cannot own loop-affine {type(result).__name__}"
            outcome = (None, TypeError(refused))
        with self._lock:
            if self._state != "abandoned":
                self._outcome = outcome
                try:
                    self._loop.call_soon_threadsafe(self._deliver)
                except RuntimeError:  # the loop is closed; nobody can take it
                    self._state = "abandoned"
                else:
                    self._state = "result_ready"
                    return
        if outcome[1] is None and self._close is not None:
            logger.debug("closing abandoned thread call result on the worker thread")
            self._close_result(outcome[0])

    def _deliver(self) -> None:
        with self._lock:
            if self._state != "result_ready" or self._future.done():
                return  # abandoned, or cancelled and the awaiter will abandon it
            result, error = self._outcome
        if error is None:
            self._future.set_result(result)
        else:
            self._future.set_exception(error)

    async def wait(self) -> T:
        try:
            result = await self._future
        except asyncio.CancelledError:
            self.abandon()
            raise
        with self._lock:
            if self._state == "abandoned":
                raise asyncio.CancelledError  # the owner closed after hand-back
            self._state = "taken"
        return result

    def abandon(self) -> None:
        """Give up the result on the loop thread; close it if it is already here."""
        with self._lock:
            previous, self._state = self._state, "abandoned"
        self._future.cancel()
        if previous == "result_ready" and self._outcome[1] is None:
            self._close_result(self._outcome[0])

    def _close_result(self, result: T) -> None:
        if self._close is None:
            return
        try:
            self._close(result)
        except Exception:  # a failed close must not mask the cancel
            logger.warning(
                "closing an abandoned thread call result failed", exc_info=True
            )


async def run_owned_thread_call[T](
    function: Callable[[], T],
    close_on_cancel: Callable[[T], None] | None = None,
) -> T:
    """Run blocking work in the default executor; close an untaken result.

    close_on_cancel may run on the worker thread, so it must be thread-safe
    (files, fds, plain objects). Loop-affine results are refused.
    """

    loop = asyncio.get_running_loop()
    call = _OwnedCall(loop, function, close_on_cancel)
    loop.run_in_executor(None, call.run)
    return await call.wait()


class ThreadCallWorkers:
    """A fixed pool of named worker threads running owned calls in order."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, *, workers: int, name: str
    ) -> None:
        self._loop = loop
        self._condition = threading.Condition()
        self._queue: deque[_OwnedCall[Any]] = deque()
        self._stopping = False
        self._calls: set[_OwnedCall[Any]] = set()  # loop thread only
        self.threads = [
            threading.Thread(target=self._work, daemon=True, name=f"{name}-{i}")
            for i in range(workers)
        ]
        for thread in self.threads:
            thread.start()

    async def run(self, callable: Callable[[], Any]) -> Any:
        call = _OwnedCall(self._loop, callable, None)
        with self._condition:
            if self._stopping:
                raise RuntimeError("reads are closed")
            self._queue.append(call)
            self._condition.notify()
        self._calls.add(call)
        try:
            return await call.wait()
        finally:
            self._calls.discard(call)
            with self._condition, contextlib.suppress(ValueError):
                self._queue.remove(call)  # cancelled before a worker took it

    def close(self) -> None:
        """Abandon every queued and running call; call on the loop thread."""
        with self._condition:
            self._stopping = True
            self._queue.clear()
            self._condition.notify_all()
        calls = tuple(self._calls)
        self._calls.clear()
        for call in calls:
            call.abandon()

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if not self._queue:
                    return
                call = self._queue.popleft()
            call.run()
