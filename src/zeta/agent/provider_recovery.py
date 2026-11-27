"""One recovery schedule per provider account in a live child tree."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..providers.must_complete_close import DeadlineClock, MonotonicDeadlineClock
from ..providers.recovery_probe import Readiness, probe_readiness

_LOG = logging.getLogger(__name__)
_BACKOFF = (5, 10, 20, 40, 80, 160, 300)


@dataclass(slots=True)
class RecoverySchedule:
    probe_at: str
    step: int


@dataclass(slots=True)
class _Subscriber:
    backend: object
    deadline: float
    activate: Callable[[RecoverySchedule], Awaitable[None]]
    update: Callable[[RecoverySchedule, bool], Awaitable[None]]
    waiting: bool = True


@dataclass(slots=True)
class _Account:
    next_at: float
    step: int = 0
    runs: dict[str, _Subscriber] = field(default_factory=dict)
    task: asyncio.Task[None] | None = None


class ProviderRecoveryCoordinator:
    def __init__(self, *, clock: DeadlineClock | None = None, wall=None, probe=probe_readiness) -> None:
        self.clock = clock or MonotonicDeadlineClock()
        self.wall = wall or (lambda: datetime.now(UTC))
        self._probe = probe
        self._accounts: dict[object, _Account] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def schedule(self, account, previous: RecoverySchedule | None) -> RecoverySchedule:
        if (state := self._accounts.get(account)) is not None:
            return self._schedule(state)
        if previous is not None:
            return previous
        return RecoverySchedule((self.wall() + timedelta(seconds=_BACKOFF[0])).isoformat(), 0)

    def _schedule(self, state: _Account) -> RecoverySchedule:
        remaining = max(0, state.next_at - self.clock.now())
        return RecoverySchedule((self.wall() + timedelta(seconds=remaining)).isoformat(), state.step)

    def subscribe(self, account, run_id: str, backend, *, deadline: str,
                  schedule: RecoverySchedule, activate, update) -> None:
        if self._closed:
            return
        wall, now = self.wall(), self.clock.now()

        def to_mono(stamp: str) -> float:
            return now + (datetime.fromisoformat(stamp) - wall).total_seconds()

        state = self._accounts.get(account)
        if state is None:
            state = self._accounts[account] = _Account(to_mono(schedule.probe_at), schedule.step)
        state.runs[run_id] = _Subscriber(backend, to_mono(deadline), activate, update)
        if state.task is None or state.task.done():
            state.task = asyncio.create_task(self._run(state))
            self._tasks.add(state.task)
            state.task.add_done_callback(self._done)

    def replaying(self, account, run_id: str) -> None:
        if (state := self._accounts.get(account)) is not None and (run := state.runs.get(run_id)) is not None:
            run.waiting = False

    def unsubscribe(self, account, run_id: str) -> None:
        if (state := self._accounts.get(account)) is None:
            return
        state.runs.pop(run_id, None)
        if not state.runs:
            del self._accounts[account]
            if state.task is not None and state.task is not asyncio.current_task():
                state.task.cancel()

    replay_progress = unsubscribe

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            _LOG.error("provider recovery task failed", exc_info=error)

    async def _run(self, state: _Account) -> None:
        while not self._closed:
            waiting = tuple(run for run in state.runs.values() if run.waiting)
            if not waiting:
                return
            edge = min(state.next_at, *(run.deadline for run in waiting))
            if edge > self.clock.now():
                await self.clock.wait_until(edge)
            now = self.clock.now()
            await self._expire(state, now)
            ready = tuple(run for run in state.runs.values() if run.waiting)
            if not ready or now < state.next_at:
                continue
            try:
                # A readiness request is never allowed to pin a suspended tree indefinitely.
                async with asyncio.timeout(min(30, max(run.deadline for run in ready) - now)):
                    result = await self._probe(ready[0].backend)
            except TimeoutError:
                result = Readiness.RETRY
            now = self.clock.now()
            state.step = min(state.step + 1, len(_BACKOFF) - 1)
            state.next_at = now + _BACKOFF[state.step]
            await self._fan_out(state, result, now)

    async def _expire(self, state: _Account, now: float) -> None:
        for run in tuple(state.runs.values()):
            if run.waiting and run.deadline <= now:
                run.waiting = False
                await run.update(self._schedule(state), True)

    async def _fan_out(self, state: _Account, result: Readiness, now: float) -> None:
        schedule = self._schedule(state)
        for run in tuple(state.runs.values()):
            if not run.waiting:
                continue
            if result is Readiness.STOP or run.deadline <= now:
                run.waiting = False
                await run.update(schedule, True)
            elif result is Readiness.READY:
                run.waiting = False
                await run.activate(schedule)
            else:
                await run.update(schedule, False)

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._accounts.clear()
