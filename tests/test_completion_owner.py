from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from zeta.protocol.types import (
    ErrorInfo,
    Message,
    StreamEvent,
    StreamEventType,
    ToolSchema,
)
from zeta.providers.completion import (
    AttemptSignal,
    CompletionControl,
    CompletionRetryBudget,
    CompletionRun,
    DeadlineClock,
    ProviderAttempt,
    ResetPolicy,
    TransportActivity,
)


async def _within[T](awaitable: Awaitable[T]) -> T:
    return await asyncio.wait_for(awaitable, 10)


async def _collect(completion: Any) -> list[StreamEvent]:
    return [event async for event in completion]


class _ManualClock(DeadlineClock):
    def __init__(self) -> None:
        self.current = 0.0
        self.waiters: list[tuple[float, asyncio.Future[None]]] = []
        self.waiter_added = asyncio.Event()

    def now(self) -> float:
        return self.current

    async def wait_until(self, deadline: float) -> None:
        if deadline <= self.current:
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((deadline, future))
        self.waiter_added.set()
        try:
            await await_or_fail(future, what="future")
        finally:
            self.waiters = [item for item in self.waiters if item[1] is not future]

    def advance(self, value: float) -> None:
        self.current = value
        for deadline, future in tuple(self.waiters):
            if deadline <= value and not future.done():
                future.set_result(None)


class _SequenceAttempt(ProviderAttempt):
    def __init__(self, *items: AttemptSignal | BaseException) -> None:
        self.items = list(items)
        self.closed = 0

    def __aiter__(self) -> _SequenceAttempt:
        return self

    async def __anext__(self) -> AttemptSignal:
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed += 1


class _AttemptBackend:
    provider = "test"

    def __init__(
        self,
        attempts: Sequence[ProviderAttempt],
        clock: DeadlineClock,
        wait: Any = None,
    ) -> None:
        self.attempts = list(attempts)
        self.created = 0
        self.clock = clock
        self.client_close_started = False
        self.wait = wait or _no_wait

    def create_attempt(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        token: str | None = None,
    ) -> ProviderAttempt:
        del messages, tool_schemas, token
        attempt = self.attempts[self.created]
        self.created += 1
        return attempt

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        *,
        control: CompletionControl,
    ) -> CompletionRun:
        return CompletionRun(
            self,
            messages,
            tool_schemas,
            control=control,
            clock=self.clock,
            wait=self.wait,
        )


async def _no_wait(_delay: float) -> None:
    return None


class _RetryableError(RuntimeError):
    retryable = True
    retry_after = 0.0




def _control(
    *,
    reset: ResetPolicy = ResetPolicy.NO_RESET,
    budget: CompletionRetryBudget | None = None,
) -> CompletionControl:
    return CompletionControl(
        abort=None,
        reset=reset,
        budget=budget or CompletionRetryBudget(),
    )


@pytest.mark.asyncio
async def test_commit_rejects_retry_after_visible_output() -> None:
    clock = _ManualClock()
    first = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="visible"),
        ),
        _RetryableError("late disconnect"),
    )
    backend = _AttemptBackend([first, _SequenceAttempt()], clock)
    completion = backend.complete(
        [], [], control=_control(reset=ResetPolicy.ALLOW_ASSISTANT_RESET)
    )
    try:
        event = await _within(completion.__anext__())
        assert event.delta == "visible"
        completion.commit()
        completion.commit()
        with pytest.raises(_RetryableError, match="late disconnect"):
            await _within(completion.__anext__())
        assert backend.created == 1
    finally:
        await _within(completion.aclose())
    assert first.closed == 1


@pytest.mark.asyncio
async def test_allow_assistant_reset_retries_partial_memory_output() -> None:
    clock = _ManualClock()
    first = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="partial"),
        ),
        _RetryableError("repairable"),
    )
    second = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="complete"),
        ),
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_END),
        ),
    )
    backend = _AttemptBackend([first, second], clock)
    completion = backend.complete(
        [], [], control=_control(reset=ResetPolicy.ALLOW_ASSISTANT_RESET)
    )
    try:
        events = [event async for event in completion]
    finally:
        await _within(completion.aclose())
    assert [event.type for event in events] == [
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.RETRY,
        StreamEventType.ASSISTANT_RESET,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
    ]
    assert events[-2].delta == "complete"
    assert first.closed == second.closed == 1






@pytest.mark.asyncio
async def test_provider_error_event_remains_public_error_event() -> None:
    clock = _ManualClock()
    error = ErrorInfo("invalid_request", "bad request")
    attempt = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.ERROR, error=error),
        )
    )
    backend = _AttemptBackend([attempt], clock)
    completion = backend.complete([], [], control=_control())
    try:
        event = await _within(completion.__anext__())
        assert event.type is StreamEventType.ERROR
        assert event.error == error
    finally:
        await _within(completion.aclose())


class _StallError(_RetryableError):
    is_stall = True
    stall_seconds = 90.0


@pytest.mark.asyncio
async def test_retry_after_and_shared_attempt_budget_are_owner_policy() -> None:
    clock = _ManualClock()
    delays: list[float] = []

    async def wait(delay: float) -> None:
        delays.append(delay)
        clock.current += delay

    errors = [_SequenceAttempt(_RetryableError("busy")) for _ in range(5)]
    backend = _AttemptBackend(errors, clock, wait=wait)
    budget = CompletionRetryBudget()
    completion = backend.complete([], [], control=_control(budget=budget))
    try:
        with pytest.raises(_RetryableError, match="busy"):
            async for _event in completion:
                pass
    finally:
        await _within(completion.aclose())
    assert backend.created == 5
    assert len(delays) == 4
    assert all(attempt.closed == 1 for attempt in errors)


@pytest.mark.asyncio
async def test_stall_retry_count_is_bounded_by_owner() -> None:
    clock = _ManualClock()
    attempts = [_SequenceAttempt(_StallError("stalled")) for _ in range(3)]
    backend = _AttemptBackend(attempts, clock)
    completion = backend.complete([], [], control=_control())
    try:
        with pytest.raises(_StallError, match="stalled"):
            async for _event in completion:
                pass
    finally:
        await _within(completion.aclose())
    assert backend.created == 3
    assert all(attempt.closed == 1 for attempt in attempts)


@pytest.mark.asyncio
async def test_retry_window_starts_at_first_failure_not_request_start() -> None:
    clock = _ManualClock()

    class _LateFailure(ProviderAttempt):
        def __aiter__(self) -> _LateFailure:
            return self

        async def __anext__(self) -> AttemptSignal:
            clock.current = 753.0
            raise _RetryableError("late server error")

        async def aclose(self) -> None:
            return None

    second = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_END),
        )
    )
    backend = _AttemptBackend([_LateFailure(), second], clock)
    budget = CompletionRetryBudget()
    completion = backend.complete([], [], control=_control(budget=budget))
    try:
        events = [event async for event in completion]
    finally:
        await _within(completion.aclose())
    assert events[-1].type is StreamEventType.MESSAGE_END
    assert backend.created == 2
    assert budget.started_at == 753.0


class _DrivenAttempt(ProviderAttempt):
    def __init__(self) -> None:
        self.signals: asyncio.Queue[AttemptSignal] = asyncio.Queue()
        self.reads: asyncio.Queue[int] = asyncio.Queue()
        self.read_count = 0
        self.read_cancelled = False
        self.closed = 0

    def __aiter__(self) -> _DrivenAttempt:
        return self

    async def __anext__(self) -> AttemptSignal:
        self.read_count += 1
        self.reads.put_nowait(self.read_count)
        try:
            return await self.signals.get()
        except asyncio.CancelledError:
            self.read_cancelled = True
            raise

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_transport_deadline_retries_attempt_through_public_completion() -> None:
    clock = _ManualClock()
    stalled = _DrivenAttempt()
    success = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_END),
        )
    )
    backend = _AttemptBackend([stalled, success], clock)
    completion = backend.complete([], [], control=_control())
    drain = asyncio.create_task(_collect(completion))
    try:
        assert await _within(stalled.reads.get()) == 1
        await _within(clock.waiter_added.wait())
        clock.advance(90.0)
        events = await _within(drain)
    finally:
        if not drain.done():
            drain.cancel()
            await _within(asyncio.gather(drain, return_exceptions=True))
        await _within(completion.aclose())
    assert [event.type for event in events] == [
        StreamEventType.RETRY,
        StreamEventType.MESSAGE_END,
    ]
    assert events[0].data["is_stall"] is True
    assert stalled.read_cancelled is True
    assert stalled.closed == 1


@pytest.mark.asyncio
async def test_transport_only_activity_does_not_reset_model_progress_deadline() -> None:
    clock = _ManualClock()
    stalled = _DrivenAttempt()
    backend = _AttemptBackend([stalled, _DrivenAttempt(), _DrivenAttempt()], clock)
    completion = backend.complete([], [], control=_control())
    next_event = asyncio.create_task(completion.__anext__())
    try:
        for index in range(10):
            assert await _within(stalled.reads.get()) == index + 1
            await _within(clock.waiter_added.wait())
            clock.waiter_added.clear()
            clock.advance(float(89 * (index + 1)))
            stalled.signals.put_nowait(
                AttemptSignal(TransportActivity.TRANSPORT_ONLY)
            )
        assert await _within(stalled.reads.get()) == 11
        await _within(clock.waiter_added.wait())
        clock.advance(900.0)
        retry = await _within(next_event)
    finally:
        if not next_event.done():
            next_event.cancel()
        await _within(asyncio.gather(next_event, return_exceptions=True))
        await _within(completion.aclose())
    assert retry.type is StreamEventType.RETRY
    assert retry.data["is_stall"] is True
    assert stalled.read_cancelled is True
