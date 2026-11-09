"""Off-loop calls whose result is closed when the awaiting side gives it up.

One owner for blocking work run on a worker thread: each call has one state
under a lock (pending, result_ready, taken, abandoned). Worker threads touch
only that state; only the loop touches the asyncio future. When the awaiting
side cancels, or the loop closes before the awaiter takes the result (even after
the hand-back was scheduled), the result is closed exactly once by whichever side
comes second. A close may run on a worker thread, so callers declare it
thread-safe when they register it.
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


class _Handoff:
    """The scheduled hand-back; a closed loop drops it without calling it."""

    __slots__ = ("call",)

    def __init__(self, call: _OwnedCall[Any]) -> None:
        self.call: _OwnedCall[Any] | None = call

    def __call__(self) -> None:
        call, self.call = self.call, None
        if call is not None:
            call._deliver()

    def __del__(self) -> None:
        call, self.call = self.call, None
        if call is not None:
            logger.debug("closing a thread call result whose hand-back was dropped")
            call._release_untaken()


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
        with self._lock:
            abandoned = self._state == "abandoned"
            if not abandoned:
                self._outcome, self._state = outcome, "result_ready"
        if abandoned:
            if outcome[1] is None:
                logger.debug("closing an abandoned thread call result")
                self._close_result(outcome[0])
            return
        handoff = _Handoff(self)
        try:
            self._loop.call_soon_threadsafe(handoff)
        except RuntimeError:  # the loop is closed; nobody can take it
            handoff.call = None
            self._release_untaken()

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
        except (asyncio.CancelledError, GeneratorExit):
            # GeneratorExit: a closed loop dropped the pending task and gc closed it.
            self.abandon()
            raise
        with self._lock:
            if self._state == "abandoned":
                raise asyncio.CancelledError  # the owner closed after hand-back
            self._state = "taken"
        return result

    def abandon(self) -> None:
        """Give up the result; close it if it is already here."""
        with self._lock:
            previous, self._state = self._state, "abandoned"
        if not self._loop.is_closed():
            self._future.cancel()
        if previous == "result_ready" and self._outcome[1] is None:
            self._close_result(self._outcome[0])

    def _release_untaken(self) -> None:
        """Close a result whose hand-back can no longer reach the awaiter."""
        with self._lock:
            if self._state != "result_ready":
                return  # taken, or abandon() already owns the close
            self._state = "abandoned"
        if self._outcome[1] is None:
            self._close_result(self._outcome[0])

    def _close_result(self, result: T) -> None:
        if self._close is None:
            return
        if not isinstance(result, _LOOP_AFFINE):
            self._close_now(result)
            return
        # A declared thread-safe close returned a loop-affine object: close it on
        # its loop, never on this thread.
        try:
            self._loop.call_soon_threadsafe(self._close_now, result)
        except RuntimeError:
            logger.error(
                "leaked loop-affine %s from a thread call: its loop is closed",
                type(result).__name__,
            )

    def _close_now(self, result: T) -> None:
        assert self._close is not None
        try:
            self._close(result)
        except Exception:  # a failed close must not mask the cancel
            logger.warning(
                "closing an abandoned thread call result failed", exc_info=True
            )


async def run_owned_thread_call[T](
    function: Callable[[], T],
    close_on_cancel: Callable[[T], None] | None = None,
    *,
    close_is_thread_safe: bool = False,
) -> T:
    """Run blocking work in the default executor; close an untaken result.

    close_on_cancel may run on a worker thread, so the caller must declare it
    thread-safe (files, fds, plain objects) with close_is_thread_safe=True.
    """

    if close_on_cancel is not None and not close_is_thread_safe:
        raise TypeError(
            "close_on_cancel may run on a worker thread; "
            "pass close_is_thread_safe=True for a thread-safe close"
        )
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
