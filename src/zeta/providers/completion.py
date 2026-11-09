"""Own one provider completion from attempt creation through cleanup."""

from __future__ import annotations

import asyncio
import errno
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Protocol, runtime_checkable

import httpx

from ..core.abort import AbortSignal
from ..protocol.types import (
    Message,
    StreamEvent,
    StreamEventType,
    ToolSchema,
    ToolUseContent,
)
from ..runtime._async_close import is_control_exception, merge_exception
from .must_complete_close import (
    DeadlineClock,
    MonotonicDeadlineClock,
    MustCompleteClose,
    task_is_cancelling,
)
from .rate_limit_hold import (
    RATE_LIMIT_CREDENTIAL_POLL_SECONDS,
    RATE_LIMIT_PROBE_SECONDS,
    credential_fingerprint,
    is_rate_limit_hold,
    rate_limit_hold_text,
    retry_after_seconds,
    subscription_limit_wait,
)

MAX_PROVIDER_ATTEMPTS = 5
MAX_PROVIDER_RETRY_SECONDS = 60.0
MAX_PROVIDER_RETRY_DELAY_SECONDS = 30.0
MAX_ACCOUNT_LOCK_ATTEMPTS = 5
MAX_ACCOUNT_LOCK_RETRY_SECONDS = 120.0
TRANSPORT_IDLE_SECONDS = 90.0
MAX_STALL_RETRIES = 2
MODEL_PROGRESS_IDLE_SECONDS = 15 * 60.0
_RETRYABLE_STATUS_CODES = {
    408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529
}
_TERMINAL_ERROR_CODES = {
    "auth_error",
    "authentication_error",
    "context_length_exceeded",
    "permission_denied",
    "permission_error",
}


class ResetPolicy(Enum):
    ALLOW_ASSISTANT_RESET = auto()
    NO_RESET = auto()


class TransportActivity(Enum):
    TRANSPORT_ONLY = auto()
    MODEL_PROGRESS = auto()


@dataclass(frozen=True, slots=True)
class AttemptSignal:
    activity: TransportActivity
    event: StreamEvent | None = None



def _first_provider_error(*errors: BaseException | None) -> BaseException | None:
    for error in errors:
        current = error
        while isinstance(current, BaseException):
            if not is_control_exception(current):
                return current
            current = current.__cause__
    return None


def select_completion_outcome(
    primary: BaseException | None,
    producer: BaseException | None,
    cleanup: BaseException,
) -> BaseException:
    """Select the visible outcome while preserving provider failure causes."""
    selected = merge_exception(primary, cleanup)
    if is_control_exception(selected):
        provider_error = _first_provider_error(producer, cleanup, primary)
        if provider_error is not None:
            selected.__cause__ = provider_error
            selected.__context__ = provider_error
    return selected


@dataclass(frozen=True, slots=True)
class RetryPlan:
    attempt: int
    reason: str
    delay: float
    is_stall: bool = False
    is_account_lock: bool = False
    is_rate_limit_hold: bool = False
    retry_after: float | None = None
    subscription_limit: Any = None


@dataclass(slots=True)
class CompletionRetryBudget:
    """Bound all attempts for related completion requests."""

    attempts: int = 0
    max_attempts: int = MAX_PROVIDER_ATTEMPTS
    retry_wait_seconds: float = 0.0
    original_error: Any | None = None
    exhausted: bool = False
    max_stall_retries: int = MAX_STALL_RETRIES
    stall_retries: int = 0
    excluded_stall_seconds: float = 0.0
    records: list[dict[str, object]] = field(default_factory=list)
    attempt_records: list[dict[str, object]] = field(default_factory=list)
    started_at: float | None = None
    account_lock_attempts: int = 0
    account_lock_started_at: float | None = None
    account_lock_exclusion_started_at: float | None = None
    excluded_account_lock_seconds: float = 0.0
    attempt_started_at: float | None = None
    committed: bool = False

    def start_attempt(
        self,
        owner: str,
        now: float,
        *,
        is_stall: bool = False,
        is_rate_limit_hold: bool = False,
    ) -> bool:
        del is_stall
        if is_rate_limit_hold:
            self.attempt_started_at = now
            self.attempt_records.append(
                {"attempt": self.attempts, "owner": owner, "probe": True}
            )
            return True
        # lock waits have their own bound in _plan_account_lock; they do not use provider attempts
        if self.attempts - self.account_lock_attempts >= self.max_attempts:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return False
        self.attempts += 1
        self.attempt_started_at = now
        self.attempt_records.append({"attempt": self.attempts, "owner": owner})
        return True

    def plan(
        self,
        source: BaseException | object,
        now: float,
        *,
        event_data: Mapping[str, Any] | None = None,
    ) -> RetryPlan | None:
        first_failure = self.original_error is None
        if first_failure:
            self.original_error = source
        limit = getattr(source, "subscription_limit", None)
        if limit is not None and not self.exhausted:
            wait = subscription_limit_wait(limit.resets_at.timestamp(), time.time())
            return RetryPlan(
                attempt=self.attempts,
                reason="subscription_limit",
                delay=wait,
                is_rate_limit_hold=True,
                retry_after=wait,
                subscription_limit=limit,
            )
        if self.exhausted or not retryable_provider_error(source, event_data):
            return None
        is_account_lock = getattr(source, "code", None) == "account_lock_timeout"
        if is_account_lock:
            return self._plan_account_lock(source, now, event_data)
        if self.started_at is None:
            self.started_at = now
            self.excluded_stall_seconds = 0.0
            self.excluded_account_lock_seconds = 0.0
            self.account_lock_exclusion_started_at = None
        elif self.account_lock_exclusion_started_at is not None:
            self.excluded_account_lock_seconds += (
                now - self.account_lock_exclusion_started_at
            )
            self.account_lock_exclusion_started_at = None
        generic_attempts = self.attempts - self.account_lock_attempts
        explicit_retry_after = retry_after_seconds(source, event_data)
        if generic_attempts >= self.max_attempts:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return None
        delay = retry_wait_seconds(source, generic_attempts, event_data)
        is_stall = bool(getattr(source, "is_stall", False))
        stall_seconds = getattr(source, "stall_seconds", 0.0)
        has_stall_duration = isinstance(stall_seconds, (int, float)) and stall_seconds > 0
        if is_stall:
            if has_stall_duration and not first_failure:
                self.excluded_stall_seconds += float(stall_seconds)
            if self.stall_retries >= self.max_stall_retries:
                self.exhausted = True
                self.records.append({"decision": "budget-exhausted"})
                return None
            self.stall_retries += 1
        assert self.started_at is not None
        elapsed = (
            now
            - self.started_at
            - self.excluded_stall_seconds
            - self.excluded_account_lock_seconds
        )
        if (
            is_rate_limit_hold(source, explicit_retry_after)
            and elapsed + delay > MAX_PROVIDER_RETRY_SECONDS
        ):
            return RetryPlan(
                attempt=self.attempts,
                reason=retry_reason(source),
                delay=min(delay, RATE_LIMIT_PROBE_SECONDS),
                is_rate_limit_hold=True,
                retry_after=explicit_retry_after,
            )
        if (not is_stall or has_stall_duration) and elapsed + delay > MAX_PROVIDER_RETRY_SECONDS:
            self.exhausted = True
            record: dict[str, object] = {"decision": "budget-exhausted"}
            if explicit_retry_after is not None:
                record["retry_after"] = explicit_retry_after
            self.records.append(record)
            return None
        return RetryPlan(
            attempt=self.attempts + 1,
            reason=retry_reason(source),
            delay=delay,
            is_stall=is_stall,
        )

    def _plan_account_lock(
        self,
        source: BaseException | object,
        now: float,
        event_data: Mapping[str, Any] | None,
    ) -> RetryPlan | None:
        if self.account_lock_exclusion_started_at is None:
            self.account_lock_exclusion_started_at = self.attempt_started_at or now
        if self.account_lock_started_at is None:
            self.account_lock_started_at = self.account_lock_exclusion_started_at
        self.account_lock_attempts += 1
        delay = retry_wait_seconds(source, self.account_lock_attempts, event_data)
        elapsed = now - self.account_lock_started_at
        if (
            self.account_lock_attempts >= MAX_ACCOUNT_LOCK_ATTEMPTS
            or elapsed + delay > MAX_ACCOUNT_LOCK_RETRY_SECONDS
        ):
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return None
        return RetryPlan(
            attempt=self.attempts + 1,
            reason=retry_reason(source),
            delay=delay,
            is_account_lock=True,
        )

    def record_retry(self, plan: RetryPlan) -> None:
        self.retry_wait_seconds += plan.delay
        self.records.append(
            {
                "attempt": plan.attempt,
                "reason": plan.reason,
                "delay": plan.delay,
                "decision": "retried",
            }
        )


@dataclass(frozen=True, slots=True)
class CompletionControl:
    abort: AbortSignal | None
    reset: ResetPolicy
    budget: CompletionRetryBudget
    task_deadline: float | None
    # the session's account identity; a change ends a rate-limit hold at once
    account_generation: Callable[[], object] | None = None
    # records a subscription limit and returns the user's recovery suggestion, if any
    account_limit: Callable[[object], Awaitable[dict[str, object] | None]] | None = None


class ProviderAttempt(Protocol):
    def __aiter__(self) -> ProviderAttempt: ...

    async def __anext__(self) -> AttemptSignal: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class AuthCapableProviderAttempt(ProviderAttempt, Protocol):
    async def refresh_auth(self) -> str: ...


_ACTIVITY_REPORTER: ContextVar[Callable[[TransportActivity], None] | None] = (
    ContextVar("provider_activity_reporter", default=None)
)


def report_transport_activity(
    activity: TransportActivity = TransportActivity.TRANSPORT_ONLY,
) -> None:
    reporter = _ACTIVITY_REPORTER.get()
    if reporter is not None:
        reporter(activity)


def report_provider_record(event_type: str) -> None:
    activity = (
        TransportActivity.TRANSPORT_ONLY
        if event_type
        in {"keepalive", "ping", "response.in_progress", "response.metadata"}
        else TransportActivity.MODEL_PROGRESS
    )
    report_transport_activity(activity)


def is_completion_sentinel(event: StreamEvent) -> bool:
    """Return whether ``event`` ends a completion; a truncated end is a salvage."""
    return (
        event.type is StreamEventType.MESSAGE_END
        and event.data.get("truncated") is not True
    )


_ATTEMPT_END = object()


# how long an attempt close may run before its owner stops it; below the
# lease drain bound, so a stuck close releases its borrow before the lease
# close names it.
ATTEMPT_CLOSE_BOUND = 20.0


class IteratorProviderAttempt:
    """Expose decoder activity and output through one owned attempt handle."""

    def __init__(self, source: AsyncIterator[StreamEvent]) -> None:
        self._source = source
        self._close = MustCompleteClose("provider attempt", ATTEMPT_CLOSE_BOUND)
        self._producer: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[AttemptSignal | BaseException | object] = (
            asyncio.Queue()
        )
        self._resume = asyncio.Event()
        self._resume.set()
        self._resume_after_event = False
        self._producer_error: BaseException | None = None
        self._error_delivered = False

    def __aiter__(self) -> IteratorProviderAttempt:
        return self

    def _report(self, activity: TransportActivity) -> None:
        self._queue.put_nowait(AttemptSignal(activity))

    async def _produce(self) -> None:
        token = _ACTIVITY_REPORTER.set(self._report)
        try:
            while True:
                # test-contract: seam the owner cancels the attempt producer
                await self._resume.wait()
                self._resume.clear()
                event = await self._source.__anext__()
                await self._queue.put(
                    AttemptSignal(TransportActivity.MODEL_PROGRESS, event)
                )
        except BaseException as error:  # noqa: BLE001 - forward control exceptions
            self._producer_error = error
            await self._queue.put(error)
        finally:
            _ACTIVITY_REPORTER.reset(token)
            await self._queue.put(_ATTEMPT_END)

    async def __anext__(self) -> AttemptSignal:
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce())
        if self._resume_after_event:
            self._resume_after_event = False
            self._resume.set()
        item = await self._queue.get()
        if item is _ATTEMPT_END:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._error_delivered = True
            raise item
        assert isinstance(item, AttemptSignal)
        if item.event is not None:
            self._resume_after_event = True
        return item

    async def aclose(self) -> None:
        # the close runs as one owned task: a cancelled caller (task deadline,
        # client disconnect, server close) cannot stop it before the source
        # closes and releases what it holds (an HTTP lease borrow); it only
        # tells the close not to wait on slow transport closes.
        self._close.start(self._close_source)
        if task_is_cancelling():
            self._close.abort()
        try:
            await self._close.wait()
        except asyncio.CancelledError:
            self._close.abort()
            raise

    async def _close_source(self) -> None:
        producer = self._producer
        if producer is not None:
            if producer.cancel():
                await _drain_cancelled(producer)
                # this close cancelled the producer; a bare cancellation is not
                # a source outcome (a chained one still carries the source error).
                if (
                    isinstance(self._producer_error, asyncio.CancelledError)
                    and self._producer_error.__cause__ is None
                ):
                    self._producer_error = None
            else:
                # test-contract: seam producer completion must drain fully
                await asyncio.gather(producer, return_exceptions=True)
        close = getattr(self._source, "aclose", None)
        if close is not None:
            try:
                # test-contract: seam the owner abort or bound ends this source close
                await close()
            except asyncio.CancelledError as error:
                if self._producer_error is not None:
                    raise error from self._producer_error
                raise
        if self._undelivered_error is not None:
            raise self._undelivered_error

    @property
    def _undelivered_error(self) -> BaseException | None:
        error = self._producer_error
        if error is None or self._error_delivered:
            return None
        if isinstance(error, asyncio.CancelledError) and error.__cause__ is None:
            return None
        return error


async def _drain_cancelled(producer: asyncio.Task[None]) -> None:
    """Wait for a cancelled producer to unwind; an owner abort does not skip it."""
    try:
        # test-contract: seam producer cancellation must drain fully
        await asyncio.gather(producer, return_exceptions=True)
    except asyncio.CancelledError:
        # abort: the gather passed the cancellation on to the producer, and the
        # close keeps waiting for it to release what it holds. this wait is not
        # bounded here: the owner's terminal bound cancels it and stops waiting.
        # test-contract: seam the owner bound ends this drain
        await asyncio.gather(producer, return_exceptions=True)


class AuthIteratorProviderAttempt(IteratorProviderAttempt):
    def __init__(
        self,
        source: AsyncIterator[StreamEvent],
        refresh: Callable[[], Awaitable[str]],
        map_error: Callable[[BaseException], BaseException] | None = None,
    ) -> None:
        super().__init__(source)
        self._refresh = refresh
        self._map_error = map_error

    async def __anext__(self) -> AttemptSignal:
        try:
            return await super().__anext__()
        except BaseException as error:
            if self._map_error is None:
                raise
            mapped = self._map_error(error)
            if mapped is error:
                raise
            raise mapped from error

    async def refresh_auth(self) -> str:
        return await self._refresh()


class _MethodAdapter:
    def __init__(self, backend: object, method: Callable[..., AsyncIterator[StreamEvent]]) -> None:
        self._backend = backend
        self._method = method
        self.provider = str(getattr(backend, "provider", "scripted"))

    def create_attempt(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        token: str | None = None,
    ) -> ProviderAttempt:
        del token
        return IteratorProviderAttempt(self._method(self._backend, messages, tool_schemas))


def owned_completion(method: Callable[..., AsyncIterator[StreamEvent]]) -> Callable[..., CompletionRun]:
    """Adapt a simple scripted decoder without giving it completion policy."""

    @wraps(method)
    def wrapper(
        backend: object,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        *,
        control: CompletionControl,
    ) -> CompletionRun:
        return complete(
            _MethodAdapter(backend, method),
            messages,
            tool_schemas,
            control=control,
            clock=getattr(backend, "_completion_clock", None),
            wait=getattr(backend, "_completion_wait", None),
        )

    return wrapper


class ProviderAdapter(Protocol):
    provider: str

    def create_attempt(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        token: str | None = None,
    ) -> ProviderAttempt: ...


class PrematureCompletionError(RuntimeError):
    provider_error = False
    retryable = True
    retry_reason = "stream_error"
    code = "stream_error"


class CompletionCleanupError(RuntimeError):
    """Closing a successful provider attempt failed."""


class ProviderAccountSwitched(Exception):
    """The session switched accounts during a hold; redispatch through the new backend."""


class TaskDeadlineExceeded(TimeoutError):
    """The enclosing finish-gate task deadline expired."""


class ProviderStallError(RuntimeError):
    retryable = True
    retry_reason = "stream_stall"
    is_stall = True

    def __init__(self, kind: str, seconds: float) -> None:
        super().__init__(f"provider {kind} stalled for {seconds:.0f}s")
        self.code = "timeout"
        self.stall_seconds = seconds


_ABANDONED = object()


class CompletionRun(AsyncIterator[StreamEvent]):
    """Public handle for one owner-managed completion."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        control: CompletionControl,
        *,
        clock: DeadlineClock | None = None,
        wait: Callable[[float], Awaitable[None]] | None = None,
        transport_idle: float = TRANSPORT_IDLE_SECONDS,
        model_idle: float = MODEL_PROGRESS_IDLE_SECONDS,
    ) -> None:
        self._adapter = adapter
        self._messages = messages
        self._tool_schemas = tool_schemas
        self._control = control
        self._clock = clock or MonotonicDeadlineClock()
        self._wait = wait or asyncio.sleep
        self._transport_idle = transport_idle
        self._model_idle = model_idle
        self._attempt: ProviderAttempt | None = None
        self._token: str | None = None
        self._refreshed = False
        self._committed = False
        self._emitted = False
        self._closed = False
        self._attempt_salvaged = False
        self._pending: deque[StreamEvent] = deque()
        self._pending_retry: RetryPlan | None = None
        self._transport_deadline = 0.0
        self._model_deadline = 0.0
        self._last_model_progress: float | None = None
        self._maximum_progress_gap = 0.0
        self._retry_usage: dict[str, Any] | None = None
        self._rate_limit_hold = False
        self._rate_limit_state: object | None = None

    def __aiter__(self) -> CompletionRun:
        return self

    def commit(self) -> None:
        self._committed = True
        self._control.budget.committed = True

    async def __anext__(self) -> StreamEvent:
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._closed:
                raise StopAsyncIteration
            if self._pending_retry is not None:
                plan = self._pending_retry
                self._pending_retry = None
                await self._wait_for_retry(plan.delay)
                self._control.budget.record_retry(plan)
            if self._attempt is None:
                await self._open_attempt()
            assert self._attempt is not None
            try:
                signal = await self._read_signal(self._attempt)
            except StopAsyncIteration:
                if self._attempt_salvaged:
                    await self._finish_attempt()
                    self._closed = True
                    raise
                error = PrematureCompletionError(
                    "provider stream ended before completion"
                )
                retry = await self._retry_and_finish(error)
                if retry:
                    continue
                self._closed = True
                raise error
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    deadline = self._control.task_deadline
                    if deadline is not None and self._clock.now() >= deadline:
                        current = asyncio.current_task()
                        if current is not None:
                            current.uncancel()
                        error = TaskDeadlineExceeded(
                            "finish-gate task safety bound expired"
                        )
                    await self._finish_attempt(error)
                    self._closed = True
                    raise
                context = getattr(error, "diagnostic_context", None)
                if isinstance(context, dict):
                    context["max_progress_gap_s"] = self._maximum_progress_gap
                retry = await self._retry_and_finish(error)
                if retry:
                    continue
                self._closed = True
                raise
            now = self._clock.now()
            self._transport_deadline = now + self._transport_idle
            if self._model_deadline == float("inf"):
                self._model_deadline = now + self._model_idle
            if signal.activity is TransportActivity.MODEL_PROGRESS:
                if self._last_model_progress is not None:
                    self._maximum_progress_gap = max(
                        self._maximum_progress_gap,
                        now - self._last_model_progress,
                    )
                self._last_model_progress = now
                self._model_deadline = now + self._model_idle
            event = signal.event
            if event is None:
                continue
            # an untruncated MESSAGE_END is the provider completion sentinel:
            # publish it and close the attempt without another read. a
            # truncated one is a salvage: it stands only if the attempt then
            # ends cleanly; a later failure is planned like any other.
            terminal_event = is_completion_sentinel(event)
            if event.type is StreamEventType.MESSAGE_END:
                self._attempt_salvaged = not terminal_event
                usage = event.data.get("usage")
                if isinstance(usage, Mapping):
                    self._retry_usage = dict(usage)
            if event.type is StreamEventType.ERROR and event.error is not None:
                retry = await self._retry_and_finish(event.error, event.data)
                if retry:
                    continue
                self._closed = True
            self._observe(event)
            if terminal_event:
                await self._finish_attempt()
                self._closed = True
            return event

    async def _open_attempt(self) -> None:
        now = self._clock.now()
        if not self._control.budget.start_attempt(
            self._adapter.provider,
            now,
            is_rate_limit_hold=self._rate_limit_hold,
        ):
            self._closed = True
            raise RuntimeError("provider retry budget exhausted")
        self._rate_limit_hold = False
        self._attempt_salvaged = False
        self._attempt = self._adapter.create_attempt(
            self._messages, self._tool_schemas, self._token
        )
        self._transport_deadline = now + self._transport_idle
        self._model_deadline = now + self._model_idle

    async def _read_signal(self, attempt: ProviderAttempt) -> AttemptSignal:
        task_deadline = self._control.task_deadline
        deadline = min(
            self._transport_deadline,
            self._model_deadline,
            float("inf") if task_deadline is None else task_deadline,
        )
        if task_deadline is not None and task_deadline <= min(
            self._transport_deadline, self._model_deadline
        ):
            deadline_error: BaseException = TaskDeadlineExceeded(
                "finish-gate task safety bound expired"
            )
        else:
            kind = (
                "model progress"
                if self._model_deadline < self._transport_deadline
                else "transport"
            )
            seconds = (
                self._model_idle
                if kind == "model progress"
                else self._transport_idle
            )
            deadline_error = ProviderStallError(kind, seconds)
        return await self._await_step(
            attempt.__anext__(),
            deadline=deadline,
            deadline_error=deadline_error,
            deadline_wins_tie=(
                task_deadline is not None and task_deadline == deadline
            ),
        )

    def _observe(self, event: StreamEvent) -> None:
        public_output = (
            event.message is not None
            or event.content is not None
            or event.delta is not None
            or event.tool_call is not None
        )
        completed_tool = event.data.get("tool_call_completed") is True
        if event.data.get("truncated") is not True:
            completed_tool = completed_tool or (
                event.tool_call is not None and "tool_call_delta" not in event.data
            )
            completed_tool = completed_tool or isinstance(event.content, ToolUseContent)
            completed_tool = completed_tool or (
                event.message is not None
                and any(
                    isinstance(block, ToolUseContent) for block in event.message.content
                )
            )
        self._emitted = self._emitted or public_output
        if completed_tool or (
            public_output and self._control.reset is ResetPolicy.NO_RESET
        ):
            self.commit()

    async def _retry_and_finish(
        self,
        error: BaseException | object,
        event_data: Mapping[str, Any] | None = None,
    ) -> bool:
        try:
            return await self._retry(error, event_data)
        finally:
            await self._finish_attempt(error)

    async def _retry(
        self,
        error: BaseException | object,
        event_data: Mapping[str, Any] | None = None,
    ) -> bool:
        if self._committed or self._control.budget.committed:
            self._control.budget.records.append(
                {
                    "decision": "skipped-after-output",
                    "reason": "tool_call_completed",
                }
            )
            self._record_failure_decision(error, will_retry=False)
            return False
        if self._control.abort and self._control.abort.is_set():
            self._record_failure_decision(error, will_retry=False)
            return False
        if self._is_unauthorized(error):
            if self._refreshed or not isinstance(
                self._attempt, AuthCapableProviderAttempt
            ):
                return False
            self._token = await self._await_step(
                self._attempt.refresh_auth(),
                deadline=self._control.task_deadline,
                deadline_error=TaskDeadlineExceeded(
                    "finish-gate task safety bound expired"
                ),
            )
            self._refreshed = True
            return True
        retryable = retryable_provider_error(error, event_data)
        plan = self._control.budget.plan(
            error, self._clock.now(), event_data=event_data
        )
        self._record_failure_decision(error, will_retry=plan is not None)
        if plan is None:
            if retryable:
                record_exhausted = getattr(
                    self._adapter, "_record_retry_exhausted", None
                )
                if record_exhausted is not None:
                    record_exhausted(
                        error,
                        max(0, self._control.budget.attempts - 1),
                    )
            return False
        if self._emitted and self._control.reset is ResetPolicy.NO_RESET:
            self._record_failure_decision(error, will_retry=False)
            return False
        notice = self._retry_notice(plan, error)
        if self._retry_usage is not None:
            notice.data["usage"] = self._retry_usage
            self._retry_usage = None
        self._pending.append(notice)
        if self._emitted:
            self._pending.append(StreamEvent(StreamEventType.ASSISTANT_RESET))
            self._emitted = False
        if plan.subscription_limit is not None and self._control.account_limit is not None:
            # a suggestion only: the runtime never switches accounts by itself
            suggestion = await self._control.account_limit(error)
            if suggestion is not None:
                self._pending.append(StreamEvent(StreamEventType.ACCOUNT_LIMIT, data=suggestion))
        self._pending_retry = plan
        self._rate_limit_hold = plan.is_rate_limit_hold
        if plan.is_rate_limit_hold:
            self._rate_limit_state = self._hold_generation()
        return True

    def _hold_generation(self) -> tuple[object, object]:
        account = self._control.account_generation
        return (
            credential_fingerprint(getattr(self._adapter, "token_store", None)),
            account() if account is not None else None,
        )

    async def _wait_for_retry(self, delay: float) -> None:
        if self._rate_limit_hold:
            await self._wait_for_rate_limit_hold(delay)
            return
        await self._wait_step(delay)

    async def _wait_for_rate_limit_hold(self, delay: float) -> None:
        """Wait out a hold; a re-login wakes it, an account switch ends this run."""

        initial = self._rate_limit_state
        assert isinstance(initial, tuple)
        remaining = delay
        while remaining > 0:
            step = min(remaining, RATE_LIMIT_CREDENTIAL_POLL_SECONDS)
            await self._wait_step(step)
            credential, account = self._hold_generation()
            if account != initial[1]:
                # this adapter belongs to the old account; the caller redispatches
                self._closed = True
                raise ProviderAccountSwitched("session account changed during rate-limit hold")
            if credential != initial[0]:
                return
            remaining -= step

    async def _wait_step(self, delay: float) -> None:
        await self._await_step(
            self._wait(delay),
            deadline=self._control.task_deadline,
            deadline_error=TaskDeadlineExceeded(
                "finish-gate task safety bound expired"
            ),
            abort=self._control.abort,
        )

    async def _await_step(
        self,
        step: Awaitable[Any],
        *,
        deadline: float | None,
        deadline_error: BaseException,
        abort: AbortSignal | None = None,
        deadline_wins_tie: bool = True,
    ) -> Any:
        if deadline is None and abort is None:
            # test-contract: seam caller cancellation directly owns this step
            return await step
        timer_task = (
            asyncio.create_task(self._clock.wait_until(deadline))
            if deadline is not None
            else None
        )
        step_task = asyncio.ensure_future(step)
        # test-contract: seam the abort signal bounds this owner wait
        abort_task = asyncio.create_task(abort.wait()) if abort is not None else None
        tasks = tuple(
            task for task in (step_task, timer_task, abort_task) if task is not None
        )
        try:
            # test-contract: seam timer or abort bounds the owned step tasks
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if timer_task is not None and timer_task in done and (
                deadline_wins_tie or step_task not in done
            ):
                raise deadline_error
            if abort_task is not None and abort_task in done:
                raise asyncio.CancelledError
            return step_task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # test-contract: seam every losing step task must drain fully
            await self._drain_preserving_cancellation(*tasks)

    def _record_failure_decision(self, error: object, *, will_retry: bool) -> None:
        if getattr(error, "stream_diagnostic", None) is None:
            return
        diagnostic = error.stream_diagnostic
        if isinstance(diagnostic, tuple) and len(diagnostic) == 4:
            error.stream_diagnostic = (*diagnostic[:3], self._control.budget.attempts)
        from .stream_diagnostics import StreamDiagnostics

        StreamDiagnostics.record_failure_decision(
            error,
            provider=self._adapter.provider,
            will_retry=will_retry,
        )

    def _retry_notice(self, plan: RetryPlan, error: object) -> StreamEvent:
        if plan.is_rate_limit_hold:
            assert plan.retry_after is not None
            limit = plan.subscription_limit
            data: dict[str, Any] = {
                "kind": "provider_rate_limit_hold",
                "limit": "rate" if limit is None else "subscription",
                "delay": plan.delay,
                "retry_after": plan.retry_after,
                "until": time.time() + plan.retry_after,
                "provider": self._adapter.provider,
                "attempt": plan.attempt,
                "reason": plan.reason,
            }
            if limit is not None:
                data["until"] = limit.resets_at.timestamp()
                data["account"] = str(limit.account)
                data["text"] = f"{limit.account} {rate_limit_hold_text(data)}"
                return StreamEvent(StreamEventType.RETRY, data=data)
            data["text"] = (
                f"{self._adapter.provider} {rate_limit_hold_text(data)} "
                "(log in again or switch account to resume sooner)"
            )
            return StreamEvent(StreamEventType.RETRY, data=data)
        if plan.is_account_lock:
            return StreamEvent(
                StreamEventType.RETRY,
                data={
                    "text": (
                        "waiting for account lock; retrying in "
                        f"{format_retry_delay(plan.delay)}s"
                    ),
                    "kind": "account_lock_wait",
                    "attempt": plan.attempt,
                    "retry": self._control.budget.account_lock_attempts,
                    "delay": plan.delay,
                    "reason": plan.reason,
                },
            )
        if plan.is_stall:
            stall_attempt = self._control.budget.stall_retries
            return StreamEvent(
                StreamEventType.RETRY,
                data={
                    "text": (
                        "provider stalled, retrying "
                        f"({stall_attempt}/{self._control.budget.max_stall_retries})"
                        f" in {format_retry_delay(plan.delay)}s"
                    ),
                    "kind": "provider_stall",
                    "attempt": plan.attempt,
                    "reason": plan.reason,
                    "retry": stall_attempt,
                    "delay": plan.delay,
                    "is_stall": True,
                },
            )
        retry_text = (
            f"{retry_error_label(error)}, retry scheduled"
            if getattr(error, "retry_reason", None) == "stream_error"
            else (
                f"{retry_error_label(error)}, retrying in {plan.delay:g}s "
                f"(attempt {plan.attempt}/{MAX_PROVIDER_ATTEMPTS})"
            )
        )
        return StreamEvent(
            StreamEventType.RETRY,
            data={
                "text": retry_text,
                "kind": "provider_retry",
                "attempt": plan.attempt,
                "retry": plan.attempt - 1,
                "delay": plan.delay,
                "reason": plan.reason,
            },
        )

    @staticmethod
    async def _drain_preserving_cancellation(*tasks: asyncio.Future[Any]) -> None:
        drain = asyncio.gather(*tasks, return_exceptions=True)
        cancelled = False
        while not drain.done():
            try:
                # test-contract: seam cancellation cannot interrupt owner cleanup
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _finish_attempt(self, primary: BaseException | object | None = None) -> None:
        attempt, self._attempt = self._attempt, None
        if attempt is None:
            return
        try:
            await self._close_attempt(attempt)
        except BaseException as cleanup_error:
            primary_error = primary if isinstance(primary, BaseException) else None
            if isinstance(cleanup_error, TaskDeadlineExceeded) and not (
                primary_error is not None and is_control_exception(primary_error)
            ):
                # the task deadline is terminal and wins over any outcome
                # except the caller's own cancellation.
                raise cleanup_error from primary_error
            producer_error = getattr(attempt, "_producer_error", None)
            if (
                primary_error is None
                and isinstance(producer_error, BaseException)
                and is_control_exception(producer_error)
                and not is_control_exception(cleanup_error)
            ):
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "provider completion cleanup failed",
                        "exception": cleanup_error,
                        "provider": self._adapter.provider,
                    }
                )
                return
            if primary_error is None and primary is not None:
                if not is_control_exception(cleanup_error):
                    asyncio.get_running_loop().call_exception_handler(
                        {
                            "message": "provider completion cleanup failed",
                            "exception": cleanup_error,
                            "provider": self._adapter.provider,
                        }
                    )
                    return
                raise
            selected = select_completion_outcome(
                primary_error,
                producer_error if isinstance(producer_error, BaseException) else None,
                cleanup_error,
            )
            if primary_error is None and selected is cleanup_error and not is_control_exception(cleanup_error):
                raise CompletionCleanupError(str(cleanup_error)) from cleanup_error
            if selected is not cleanup_error and not is_control_exception(cleanup_error):
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "provider completion cleanup failed",
                        "exception": cleanup_error,
                        "provider": self._adapter.provider,
                    }
                )
            raise selected

    async def _close_attempt(self, attempt: ProviderAttempt) -> None:
        await self._await_step(
            attempt.aclose(),
            deadline=self._control.task_deadline,
            deadline_error=TaskDeadlineExceeded(
                "finish-gate task safety bound expired"
            ),
        )

    async def aclose(self) -> None:
        if self._closed and self._attempt is None:
            return
        self._closed = True
        # the caller abandoned an unfinished attempt: an ordinary cleanup
        # failure is reported, a control exception still propagates.
        await self._finish_attempt(_ABANDONED)

    @staticmethod
    def _is_unauthorized(error: object) -> bool:
        return getattr(error, "status_code", None) == 401


def complete(
    adapter: ProviderAdapter,
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    control: CompletionControl,
    clock: DeadlineClock | None = None,
    wait: Callable[[float], Awaitable[None]] | None = None,
    transport_idle: float = TRANSPORT_IDLE_SECONDS,
    max_stall_retries: int = MAX_STALL_RETRIES,
) -> CompletionRun:
    """Run one owned completion with the backend's configured stall limits."""
    control.budget.max_stall_retries = max_stall_retries
    return CompletionRun(
        adapter,
        messages,
        tool_schemas,
        control,
        clock=clock or getattr(adapter, "_completion_clock", None),
        wait=wait or getattr(adapter, "_completion_wait", None),
        transport_idle=transport_idle,
        # a silent stream must reach the user's transport limit, not stall earlier as "model progress"
        model_idle=max(MODEL_PROGRESS_IDLE_SECONDS, transport_idle),
    )


def transient_network_error(error: BaseException | object) -> bool:
    current = error if isinstance(error, BaseException) else None
    transient = False
    while current is not None:
        if isinstance(current, PermissionError) or (
            isinstance(current, OSError) and current.errno in {errno.EACCES, errno.EPERM}
        ):
            # a denied permission does not heal on retry, whatever wraps it.
            return False
        transient = transient or isinstance(
            current, (ConnectionError, TimeoutError, httpx.TransportError)
        )
        current = current.__cause__
    return transient


def retryable_provider_error(
    error: BaseException | object,
    event_data: Mapping[str, Any] | None = None,
) -> bool:
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    if getattr(error, "subscription_limit", None) is not None:
        return False
    if code in _TERMINAL_ERROR_CODES:
        return False
    if type(status_code) is int and 400 <= status_code < 500 and status_code not in {408, 409, 425, 429}:
        return False
    return bool(
        code == "server_error"
        or status_code in _RETRYABLE_STATUS_CODES
        or getattr(error, "retryable", False)
        or getattr(error, "is_stall", False)
        or transient_network_error(error)
        or code in {"timeout", "transport_error"}
        or (
            code == "http_error"
            and isinstance(getattr(error, "message", None), str)
            and error.message.strip().lower() == "request failed"
        )
        or event_data
        and event_data.get("retryable") is True
    )

def retry_wait_seconds(
    error: BaseException | object,
    retry_number: int,
    event_data: Mapping[str, Any] | None = None,
) -> float:
    explicit_retry_after = retry_after_seconds(error, event_data)
    if explicit_retry_after is not None:
        return explicit_retry_after
    base = min(MAX_PROVIDER_RETRY_DELAY_SECONDS, float(2 ** (retry_number - 1)))
    return min(MAX_PROVIDER_RETRY_DELAY_SECONDS, random.uniform(base * 0.5, base * 1.5))


def retry_reason(error: BaseException | object) -> str:
    value = getattr(error, "retry_reason", None)
    if isinstance(value, str) and value:
        return value
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    if getattr(error, "status_code", None) is not None:
        return "http_error"
    return "transport_error" if transient_network_error(error) else "backend_error"


def format_retry_delay(delay: float) -> str:
    return f"{delay:.1f}".rstrip("0").rstrip(".")


def retry_error_label(error: BaseException | object) -> str:
    status_code = getattr(error, "status_code", None)
    reason = getattr(error, "retry_reason", None)
    if status_code == 429 or reason == "rate_limit_error":
        return "429 rate limited"
    if status_code == 529 or reason == "overloaded_error":
        return "529 overloaded"
    if getattr(error, "retry_reason", None) == "stream_error":
        return "provider error (stream_error)"
    if status_code is not None:
        return f"{status_code} server error"
    if getattr(error, "retryable", False):
        return "provider overloaded"
    if transient_network_error(error):
        return "network error"
    return "provider error"
