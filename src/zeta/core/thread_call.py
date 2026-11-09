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
import functools
import logging
import threading
import weakref
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from ..storage_format import release_later

logger = logging.getLogger(__name__)

# Closing these off the loop thread calls loop methods from the wrong thread.
_LOOP_AFFINE = (asyncio.Future, asyncio.BaseTransport, asyncio.StreamWriter)


class _Handoff:
    """One callback handed to the loop; a closed loop drops it without calling it.

    Every callback this module schedules on the loop goes through schedule(), so
    a dropped one always runs its dropped action: at once when the loop refuses
    it, otherwise at the next release point (zeta.storage_format.drain_releases).
    A finalizer only queues the action; it must not run it.
    """

    __slots__ = ("__weakref__", "_dropped", "_on_loop")

    def __init__(
        self, on_loop: Callable[[], None], dropped: Callable[[], None]
    ) -> None:
        self._on_loop = on_loop
        self._dropped = weakref.finalize(self, release_later, dropped)

    @classmethod
    def schedule(
        cls,
        loop: asyncio.AbstractEventLoop,
        on_loop: Callable[[], None],
        dropped: Callable[[], None],
    ) -> None:
        handoff = cls(on_loop, dropped)
        try:
            loop.call_soon_threadsafe(handoff)
        except RuntimeError:  # the loop is closed; it will never run on_loop
            handoff._dropped.detach()
            dropped()

    def __call__(self) -> None:
        if self._dropped.detach() is not None:
            self._on_loop()


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
        _Handoff.schedule(self._loop, self._deliver, self._handoff_dropped)

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
        except GeneratorExit:
            # A closed loop dropped this task and gc closed it: defer the close.
            if self._mark_abandoned():
                release_later(self._close_result, self._outcome[0])
            raise
        with self._lock:
            if self._state == "abandoned":
                raise asyncio.CancelledError  # the owner closed after hand-back
            self._state = "taken"
        return result

    def abandon(self) -> None:
        """Give up the result on the loop thread; close it if it is already here."""
        ready = self._mark_abandoned()
        if not self._loop.is_closed():
            self._future.cancel()
        if ready:
            self._close_result(self._outcome[0])

    def _mark_abandoned(self) -> bool:
        """Mark abandoned; true when this side now owns closing the result."""
        with self._lock:
            previous, self._state = self._state, "abandoned"
        return previous == "result_ready" and self._outcome[1] is None

    def _handoff_dropped(self) -> None:
        logger.debug("closing a thread call result whose hand-back was dropped")
        self._release_untaken()

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
        # its loop, never on this thread. If the loop never runs the close, only
        # log the leak.
        _Handoff.schedule(
            self._loop,
            functools.partial(self._close_now, result),
            functools.partial(self._loop_affine_leaked, type(result).__name__),
        )

    def _loop_affine_leaked(self, kind: str) -> None:
        logger.error(
            "leaked loop-affine %s from thread call %s: its loop closed first",
            kind,
            getattr(self._function, "__qualname__", repr(self._function)),
        )

    def _close_now(self, result: T) -> None:
        assert self._close is not None
        try:
            self._close(result)
        except Exception:  # a failed close must not mask the cancel
            logger.warning(
                "closing an abandoned thread call result failed", exc_info=True
            )


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
