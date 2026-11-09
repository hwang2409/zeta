"""Own must-complete cleanup: one task, a caller abort, and a terminal bound."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol


class DeadlineClock(Protocol):
    def now(self) -> float: ...

    async def wait_until(self, deadline: float) -> None: ...


class MonotonicDeadlineClock:
    def now(self) -> float:
        return time.monotonic()

    async def wait_until(self, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        timer = loop.call_at(deadline, ready.set_result, None)
        try:
            # test-contract: seam this future is the injected deadline clock
            await ready
        finally:
            timer.cancel()


# the clock that runs must-complete close bounds (tests replace it)
CLOSE_CLOCK: DeadlineClock = MonotonicDeadlineClock()
# Must-complete cleanup tasks stay owned after their caller is cancelled.
_CLOSING_TASKS: set[asyncio.Task[None]] = set()


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class MustCompleteClose:
    """Own one bounded cleanup task that every caller waits for.

    A caller's cancellation never stops the cleanup. ``abort()`` cancels the
    cleanup once at its current await; it then finishes its remaining steps
    in a cancelling task, where transport closes do not wait. An abort is not
    a failure. The bound is separate and terminal: when it expires, the owner
    cancels the cleanup again and stops waiting for it, even if the cleanup
    swallowed earlier cancellations, and fails with ``<name> close did not
    finish within <bound> s`` (plus ``detail()`` when it names something).
    """

    def __init__(
        self, name: str, bound: float, detail: Callable[[], str] | None = None
    ) -> None:
        self.task: asyncio.Task[None] | None = None
        self._name = name
        self._bound = bound
        self._detail = detail
        self._abort: asyncio.Future[None] | None = None
        self._outcome: asyncio.CancelledError | None = None
        self._waiters = 0

    def start(self, close: Callable[[], Awaitable[None]]) -> None:
        if self.task is None:
            self._abort = asyncio.get_running_loop().create_future()
            self.task = asyncio.create_task(self._run(close))
            _CLOSING_TASKS.add(self.task)
            self.task.add_done_callback(self._release)

    def abort(self) -> None:
        """Stop waiting: the cleanup finishes its steps without waiting."""
        assert self._abort is not None
        if not self._abort.done():
            self._abort.set_result(None)

    async def _cleanup(self, close: Callable[[], Awaitable[None]]) -> None:
        try:
            await close()
        except asyncio.CancelledError as error:
            # only the owner cancels this task; a cancellation it did not send,
            # or one chained from a source error, is the source's outcome.
            if error.__cause__ is not None or not task_is_cancelling():
                self._outcome = error

    async def _run(self, close: Callable[[], Awaitable[None]]) -> None:
        # the cleanup task's first step runs before this owner can act on an
        # abort, so an aborted close still starts closing its source.
        cleanup = asyncio.create_task(self._cleanup(close), name=f"{self._name} close")
        _CLOSING_TASKS.add(cleanup)
        cleanup.add_done_callback(self._cleanup_done)
        clock = CLOSE_CLOCK
        bound = asyncio.create_task(clock.wait_until(clock.now() + self._bound))
        try:
            # test-contract: seam abort, the cleanup end, or the bound ends this wait
            await asyncio.wait(
                (cleanup, bound, self._abort), return_when=asyncio.FIRST_COMPLETED
            )
            if not (cleanup.done() or bound.done()):
                cleanup.cancel()  # abort: the cleanup stops waiting
                # test-contract: seam the cleanup end or the bound ends this wait
                await asyncio.wait((cleanup, bound), return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            cleanup.cancel()  # this owner itself is cancelled (loop shutdown)
            raise
        finally:
            bound.cancel()
        if not cleanup.done():
            cleanup.cancel()  # terminal: stop waiting, whatever the cleanup does
            raise RuntimeError(self._expired_message())
        if (error := cleanup.exception()) is not None:
            raise error

    def _expired_message(self) -> str:
        message = f"{self._name} close did not finish within {self._bound:g} s"
        detail = self._detail() if self._detail is not None else ""
        return f"{message}: {detail}" if detail else message

    def _cleanup_done(self, cleanup: asyncio.Task[None]) -> None:
        _CLOSING_TASKS.discard(cleanup)
        assert self.task is not None
        if self.task.done() and not cleanup.cancelled():
            # the owner stopped waiting at its bound: report a late failure
            self._report(cleanup.exception())

    def _release(self, task: asyncio.Task[None]) -> None:
        _CLOSING_TASKS.discard(task)
        if not task.cancelled() and self._waiters == 0:
            # nobody waits for this outcome any more: report it
            self._report(task.exception())

    def _report(self, error: BaseException | None) -> None:
        if error is not None:
            asyncio.get_running_loop().call_exception_handler(
                {"message": f"{self._name} close failed", "exception": error}
            )

    async def wait(self) -> None:
        assert self.task is not None
        self._waiters += 1
        try:
            # test-contract: seam the owner bound ends this task
            await asyncio.shield(self.task)
        finally:
            self._waiters -= 1
        if self._outcome is not None:
            raise self._outcome
