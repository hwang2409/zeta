"""One turn budget shared by the phases before provider dispatch."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from ...providers.must_complete_close import DeadlineClock

PRE_DISPATCH_TIMEOUT_SECONDS = 15 * 60


class PreDispatchTimeout(TimeoutError):
    code = "pre_dispatch_timeout"
    provider_error = False


class PreDispatchWatchdog:
    def __init__(
        self, clock: DeadlineClock, stalled: Callable[[str, str], None]
    ) -> None:
        self._clock = clock
        self._stalled = stalled
        self._deadline = clock.now() + PRE_DISPATCH_TIMEOUT_SECONDS
        self._timer: asyncio.Task[None] | None = None

    async def wait[T](
        self,
        stage: str,
        description: str,
        operation: Callable[[], Coroutine[Any, Any, T]],
    ) -> T:
        if self._clock.now() >= self._deadline:
            raise self._timeout(stage, description)
        if self._timer is None:
            self._timer = asyncio.create_task(self._clock.wait_until(self._deadline))
        work = asyncio.create_task(operation())
        try:
            # test-contract: seam the injected deadline bounds the owned stage
            done, _ = await asyncio.wait(
                (self._timer, work), return_when=asyncio.FIRST_COMPLETED
            )
            if self._timer in done or self._clock.now() >= self._deadline:
                raise self._timeout(stage, description)
            return work.result()
        finally:
            if not work.done():
                work.cancel()
            # test-contract: durability a submitted write must settle before the next turn
            await asyncio.gather(work, return_exceptions=True)

    def _timeout(self, stage: str, description: str) -> PreDispatchTimeout:
        minutes = PRE_DISPATCH_TIMEOUT_SECONDS // 60
        message = f"turn stalled for {minutes} min while {description} (stage: {stage})"
        self._stalled(f"stalled {minutes}m in {stage}, cancelling", message)
        return PreDispatchTimeout(message)

    async def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            await asyncio.gather(self._timer, return_exceptions=True)
            self._timer = None
