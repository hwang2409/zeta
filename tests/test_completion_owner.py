from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Sequence
from typing import Any

import httpx
import pytest

from tests.support.owned import await_or_fail
from zeta.accounts import AccountCredentialLockTimeoutError
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
    TaskDeadlineExceeded,
    TransportActivity,
)
from zeta.providers.ollama import OllamaBackend


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

    def add_waiter(self, deadline: float) -> asyncio.Future[None]:
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((deadline, future))
        self.waiter_added.set()
        return future

    async def wait_until(self, deadline: float) -> None:
        if deadline <= self.current:
            return
        future = self.add_waiter(deadline)
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




class _DeadlineBlockedStep:
    def __init__(self, clock: _ManualClock, deadline: float) -> None:
        self.clock = clock
        self.deadline = deadline
        self.started = asyncio.Event()
        self.deadline_observed = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def __call__(self, _delay: float | None = None) -> str:
        self.started.set()
        try:
            await self.clock.wait_until(self.deadline)
            self.deadline_observed.set()
            await await_or_fail(self.release.wait(), what="release blocked policy step")
            return "refreshed-token"
        except asyncio.CancelledError:
            self.cancelled = True
            self.deadline_observed.set()
            raise


class _BlockedRefreshAttempt(_SequenceAttempt):
    def __init__(self, refresh: _DeadlineBlockedStep) -> None:
        unauthorized = RuntimeError("expired token")
        unauthorized.status_code = 401  # type: ignore[attr-defined]
        super().__init__(unauthorized)
        self.refresh = refresh

    async def refresh_auth(self) -> str:
        return await self.refresh()

def _control(
    *,
    reset: ResetPolicy = ResetPolicy.NO_RESET,
    budget: CompletionRetryBudget | None = None,
) -> CompletionControl:
    return CompletionControl(
        abort=None,
        reset=reset,
        budget=budget or CompletionRetryBudget(),
        task_deadline=None,
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
async def test_account_lock_timeout_waits_then_retries_same_completion() -> None:
    clock = _ManualClock()
    delays: list[float] = []

    async def wait(delay: float) -> None:
        delays.append(delay)
        clock.current += delay

    first = _SequenceAttempt(AccountCredentialLockTimeoutError())
    second = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="done"),
        ),
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_END),
        ),
    )
    backend = _AttemptBackend([first, second], clock, wait=wait)
    completion = backend.complete([], [], control=_control())
    try:
        events = await _collect(completion)
    finally:
        await _within(completion.aclose())

    retry = next(event for event in events if event.type is StreamEventType.RETRY)
    assert retry.data["kind"] == "account_lock_wait"
    assert retry.data["text"].startswith("waiting for account lock; retrying in ")
    assert not any(event.type is StreamEventType.ERROR for event in events)
    assert events[-2].delta == "done"
    assert backend.created == 2
    assert len(delays) == 1


def test_account_lock_before_generic_error_does_not_extend_generic_window() -> None:
    budget = CompletionRetryBudget()
    account_error = AccountCredentialLockTimeoutError()
    generic_error = _RetryableError("busy")

    assert budget.start_attempt("test", 0.0)
    assert budget.plan(account_error, 90.0) is not None
    assert budget.start_attempt("test", 100.0)
    assert budget.plan(generic_error, 100.0) is not None
    assert budget.plan(generic_error, 160.0) is not None
    assert budget.plan(generic_error, 160.001) is None


@pytest.mark.asyncio
async def test_account_lock_retry_does_not_consume_transient_attempt_budget() -> None:
    clock = _ManualClock()

    async def wait(delay: float) -> None:
        clock.current += delay

    attempts = [
        _SequenceAttempt(AccountCredentialLockTimeoutError()),
        *[_SequenceAttempt(_RetryableError("busy")) for _ in range(5)],
    ]
    backend = _AttemptBackend(attempts, clock, wait=wait)
    completion = backend.complete([], [], control=_control())
    try:
        with pytest.raises(_RetryableError, match="busy"):
            await _collect(completion)
    finally:
        await _within(completion.aclose())

    assert backend.created == 6


@pytest.mark.asyncio
async def test_lock_waits_do_not_grant_a_sixth_provider_attempt_to_a_later_request() -> None:
    clock = _ManualClock()

    async def wait(delay: float) -> None:
        clock.current += delay

    done = _SequenceAttempt(
        AttemptSignal(
            TransportActivity.MODEL_PROGRESS,
            StreamEvent(StreamEventType.MESSAGE_END),
        )
    )
    attempts = [
        _SequenceAttempt(AccountCredentialLockTimeoutError()),
        *[_SequenceAttempt(_RetryableError("busy")) for _ in range(4)],
        done,
        _SequenceAttempt(_RetryableError("must not start")),
    ]
    backend = _AttemptBackend(attempts, clock, wait=wait)
    budget = CompletionRetryBudget()
    first = backend.complete([], [], control=_control(budget=budget))
    try:
        await _collect(first)
    finally:
        await _within(first.aclose())
    assert backend.created == 6

    later = backend.complete([], [], control=_control(budget=budget))
    try:
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await _collect(later)
    finally:
        await _within(later.aclose())
    assert backend.created == 6


@pytest.mark.asyncio
async def test_account_lock_exhaustion_stops_after_five_attempts() -> None:
    clock = _ManualClock()

    async def wait(delay: float) -> None:
        clock.current += delay

    attempts = [
        _SequenceAttempt(AccountCredentialLockTimeoutError()) for _ in range(5)
    ]
    backend = _AttemptBackend(attempts, clock, wait=wait)
    completion = backend.complete([], [], control=_control())
    try:
        with pytest.raises(AccountCredentialLockTimeoutError):
            await _collect(completion)
    finally:
        await _within(completion.aclose())

    assert backend.created == 5


@pytest.mark.asyncio
async def test_task_deadline_bounds_blocked_retry_waiter() -> None:
    clock = _ManualClock()
    retry_wait = _DeadlineBlockedStep(clock, 30.0)
    backend = _AttemptBackend(
        [
            _SequenceAttempt(_RetryableError("busy")),
            _SequenceAttempt(
                AttemptSignal(
                    TransportActivity.MODEL_PROGRESS,
                    StreamEvent(StreamEventType.MESSAGE_END),
                )
            ),
        ],
        clock,
        wait=retry_wait,
    )
    completion = backend.complete([], [], control=_deadline_control(30.0))
    next_event: asyncio.Task[StreamEvent] | None = None
    try:
        retry = await _within(completion.__anext__())
        assert retry.type is StreamEventType.RETRY
        next_event = asyncio.create_task(completion.__anext__())
        await _within(retry_wait.started.wait())
        clock.advance(30.0)
        await _within(retry_wait.deadline_observed.wait())
        retry_wait.release.set()
        outcome = (await _within(asyncio.gather(next_event, return_exceptions=True)))[0]
        assert retry_wait.cancelled
        assert isinstance(outcome, TaskDeadlineExceeded)
    finally:
        retry_wait.release.set()
        if next_event is not None and not next_event.done():
            next_event.cancel()
            await _within(asyncio.gather(next_event, return_exceptions=True))
        await _within(completion.aclose())


@pytest.mark.asyncio
async def test_task_deadline_bounds_blocked_auth_refresh() -> None:
    clock = _ManualClock()
    refresh = _DeadlineBlockedStep(clock, 30.0)
    backend = _AttemptBackend(
        [
            _BlockedRefreshAttempt(refresh),
            _SequenceAttempt(
                AttemptSignal(
                    TransportActivity.MODEL_PROGRESS,
                    StreamEvent(StreamEventType.MESSAGE_END),
                )
            ),
        ],
        clock,
    )
    completion = backend.complete([], [], control=_deadline_control(30.0))
    next_event = asyncio.create_task(completion.__anext__())
    try:
        await _within(refresh.started.wait())
        clock.advance(30.0)
        await _within(refresh.deadline_observed.wait())
        refresh.release.set()
        outcome = (await _within(asyncio.gather(next_event, return_exceptions=True)))[0]
        assert refresh.cancelled
        assert isinstance(outcome, TaskDeadlineExceeded)
    finally:
        refresh.release.set()
        if not next_event.done():
            next_event.cancel()
            await _within(asyncio.gather(next_event, return_exceptions=True))
        await _within(completion.aclose())


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


class _BlockingCloseAttempt(_SequenceAttempt):
    """One stream item, then a response-context close that blocks."""

    def __init__(
        self,
        *,
        slow_cancel: asyncio.Event | None = None,
        clock: _ManualClock | None = None,
        deadline: float = 0.0,
    ) -> None:
        super().__init__(
            AttemptSignal(
                TransportActivity.MODEL_PROGRESS,
                StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="partial"),
            )
        )
        self.slow_cancel = slow_cancel
        self.clock = clock
        self.deadline = deadline
        self.close_started = asyncio.Event()
        self.deadline_observed = asyncio.Event()
        self.release = asyncio.Event()
        self.close_cancelled = asyncio.Event()
        self.close_finished = asyncio.Event()

    async def aclose(self) -> None:
        waiter = self.clock.add_waiter(self.deadline) if self.clock is not None else None
        self.close_started.set()
        try:
            if waiter is None:
                await await_or_fail(
                    asyncio.Event().wait(), what="blocking response close"
                )
            else:
                await await_or_fail(waiter, what="response close deadline")
                self.deadline_observed.set()
                await await_or_fail(self.release.wait(), what="release response close")
        except asyncio.CancelledError:
            self.close_cancelled.set()
            self.deadline_observed.set()
            if self.slow_cancel is not None:
                await await_or_fail(self.slow_cancel.wait(), what="slow close cancellation")
            raise
        finally:
            if waiter is not None and self.clock is not None:
                self.clock.waiters = [
                    item for item in self.clock.waiters if item[1] is not waiter
                ]
            self.close_finished.set()


def _deadline_control(task_deadline: float) -> CompletionControl:
    return CompletionControl(
        abort=None,
        reset=ResetPolicy.ALLOW_ASSISTANT_RESET,
        budget=CompletionRetryBudget(),
        task_deadline=task_deadline,
    )


@pytest.mark.asyncio
async def test_task_deadline_bounds_the_stream_close() -> None:
    clock = _ManualClock()
    attempt = _BlockingCloseAttempt(clock=clock, deadline=30.0)
    backend = _AttemptBackend([attempt], clock)
    completion = backend.complete([], [], control=_deadline_control(30.0))
    event = await _within(completion.__anext__())
    assert event.delta == "partial"
    closing = asyncio.create_task(completion.aclose())
    try:
        await _within(attempt.close_started.wait())
        clock.advance(30.0)
        await _within(attempt.deadline_observed.wait())
        attempt.release.set()
        outcome = (await _within(asyncio.gather(closing, return_exceptions=True)))[0]
        assert attempt.close_cancelled.is_set()
        assert isinstance(outcome, TaskDeadlineExceeded)
        assert attempt.close_finished.is_set()
        assert not backend.client_close_started
    finally:
        if not closing.done():
            closing.cancel()
        await _within(asyncio.gather(closing, return_exceptions=True))


@pytest.mark.asyncio
async def test_outer_cancel_during_stream_close_drains_attempt() -> None:
    clock = _ManualClock()
    slow_cancel = asyncio.Event()
    attempt = _BlockingCloseAttempt(slow_cancel=slow_cancel)
    backend = _AttemptBackend([attempt], clock)
    completion = backend.complete([], [], control=_deadline_control(30.0))
    await _within(completion.__anext__())
    closing = asyncio.create_task(completion.aclose())
    try:
        await _within(attempt.close_started.wait())
        closing.cancel()
        await _within(attempt.close_cancelled.wait())
        assert not closing.done()
        slow_cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await _within(closing)
        assert attempt.close_finished.is_set()
        assert not backend.client_close_started
    finally:
        slow_cancel.set()
        if not closing.done():
            closing.cancel()
        await _within(asyncio.gather(closing, return_exceptions=True))


@pytest.mark.asyncio
async def test_outer_cancel_during_task_deadline_drain_waits_for_close() -> None:
    clock = _ManualClock()
    slow_cancel = asyncio.Event()
    attempt = _BlockingCloseAttempt(slow_cancel=slow_cancel)
    backend = _AttemptBackend([attempt], clock)
    completion = backend.complete([], [], control=_deadline_control(30.0))
    await _within(completion.__anext__())
    closing = asyncio.create_task(completion.aclose())
    try:
        await _within(attempt.close_started.wait())
        await _within(clock.waiter_added.wait())
        clock.advance(30.0)
        await _within(attempt.close_cancelled.wait())
        closing.cancel()
        assert not closing.done()
        slow_cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await _within(closing)
        assert attempt.close_finished.is_set()
        assert not backend.client_close_started
    finally:
        slow_cancel.set()
        if not closing.done():
            closing.cancel()
        await _within(asyncio.gather(closing, return_exceptions=True))


@pytest.mark.asyncio
async def test_cancelled_attempt_close_still_lets_the_lease_close() -> None:
    # The shape the bench serve probe hung in: cancellation must not skip source cleanup.
    lines = [
        {"message": {"role": "assistant", "content": "."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"Content-Type": "application/x-ndjson"}, content=body
            )
        )
    )
    backend = OllamaBackend(model="stub", base_url="http://ollama.test", client=client)
    attempt = backend.create_attempt([], [])
    try:
        while True:
            signal = await await_or_fail(attempt.__anext__(), what="next signal")
            if signal.event is not None and signal.event.type is StreamEventType.MESSAGE_END:
                break
        closing = asyncio.create_task(attempt.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        await await_or_fail(asyncio.gather(closing, return_exceptions=True), what="cancelled close")
        await await_or_fail(attempt.aclose(), what="retried close")
        assert backend._lease._active_borrows == 0, "the closed attempt still holds a borrow"
        await await_or_fail(backend.aclose(), what="backend close")
    finally:
        await await_or_fail(client.aclose(), what="client close")
