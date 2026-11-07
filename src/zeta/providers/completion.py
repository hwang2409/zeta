"""Own one provider completion from attempt creation through cleanup."""

from __future__ import annotations

import asyncio
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

MAX_PROVIDER_ATTEMPTS = 5
MAX_PROVIDER_RETRY_SECONDS = 60.0
MAX_PROVIDER_RETRY_DELAY_SECONDS = 30.0
TRANSPORT_IDLE_SECONDS = 90.0
MODEL_PROGRESS_IDLE_SECONDS = 15 * 60.0

_RETRYABLE_STATUS_CODES = {
    408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529
}
_TERMINAL_ERROR_CODES = {
    "account_lock_timeout",
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
            await ready
        finally:
            timer.cancel()


def is_control_exception(value: BaseException) -> bool:
    return isinstance(value, (asyncio.CancelledError, GeneratorExit))


def _control_priority(value: BaseException) -> int:
    if isinstance(value, asyncio.CancelledError):
        return 2
    if isinstance(value, GeneratorExit):
        return 1
    return 0


def merge_exception(
    primary: BaseException | None,
    cleanup: BaseException,
) -> BaseException:
    if primary is None:
        return cleanup
    primary_is_control = is_control_exception(primary)
    cleanup_is_control = is_control_exception(cleanup)
    if cleanup_is_control and not primary_is_control:
        cleanup.__cause__ = primary
        cleanup.__context__ = primary
        return cleanup
    if (
        cleanup_is_control
        and primary_is_control
        and _control_priority(cleanup) > _control_priority(primary)
    ):
        return cleanup
    if primary_is_control and not cleanup_is_control:
        primary.__context__ = cleanup
    return primary


@dataclass(frozen=True, slots=True)
class RetryPlan:
    attempt: int
    reason: str
    delay: float
    is_stall: bool = False


@dataclass(slots=True)
class CompletionRetryBudget:
    """Bound all attempts for related completion requests."""

    attempts: int = 0
    max_attempts: int = MAX_PROVIDER_ATTEMPTS
    retry_wait_seconds: float = 0.0
    original_error: Any | None = None
    exhausted: bool = False
    max_stall_retries: int = 2
    stall_retries: int = 0
    excluded_stall_seconds: float = 0.0
    records: list[dict[str, object]] = field(default_factory=list)
    attempt_records: list[dict[str, object]] = field(default_factory=list)
    started_at: float | None = None
    committed: bool = False

    def start_attempt(self, owner: str, now: float, *, is_stall: bool = False) -> bool:
        if self.attempts >= self.max_attempts:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return False
        if (
            not is_stall
            and self.started_at is not None
            and now - self.started_at - self.excluded_stall_seconds
            >= MAX_PROVIDER_RETRY_SECONDS
        ):
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return False
        self.attempts += 1
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
            self.started_at = now
            self.excluded_stall_seconds = 0.0
        if self.exhausted or not retryable_provider_error(source, event_data):
            return None
        if self.attempts >= self.max_attempts:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return None
        explicit_retry_after = retry_after_seconds(source, event_data)
        delay = retry_wait_seconds(source, self.attempts, event_data)
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
        elapsed = now - self.started_at - self.excluded_stall_seconds
        if (not is_stall or has_stall_duration) and elapsed + delay > MAX_PROVIDER_RETRY_SECONDS:
            self.exhausted = True
            record: dict[str, object] = {"decision": "budget-exhausted"}
            if explicit_retry_after is not None:
                record["retry_after"] = explicit_retry_after
                explain_unhonored_retry_after(source, explicit_retry_after)
            self.records.append(record)
            return None
        return RetryPlan(
            attempt=self.attempts + 1,
            reason=retry_reason(source),
            delay=delay,
            is_stall=is_stall,
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


_ATTEMPT_END = object()


class IteratorProviderAttempt:
    """Expose decoder activity and output through one owned attempt handle."""

    def __init__(self, source: AsyncIterator[StreamEvent]) -> None:
        self._source = source
        self._closed = False
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
        if self._closed:
            return
        self._closed = True
        if self._producer is not None and not self._producer.done():
            self._producer.cancel()
        if self._producer is not None:
            await asyncio.gather(self._producer, return_exceptions=True)
        close = getattr(self._source, "aclose", None)
        if close is not None:
            await close()
        if (
            self._producer_error is not None
            and not self._error_delivered
            and (
                not isinstance(self._producer_error, asyncio.CancelledError)
                or self._producer_error.__cause__ is not None
            )
        ):
            raise self._producer_error


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


class ProviderStallError(RuntimeError):
    retryable = True
    retry_reason = "stream_stall"
    is_stall = True

    def __init__(self, kind: str, seconds: float) -> None:
        super().__init__(f"provider {kind} stalled for {seconds:.0f}s")
        self.code = "timeout"
        self.stall_seconds = seconds


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
        self._attempt_completed = False
        self._pending: deque[StreamEvent] = deque()
        self._pending_retry: RetryPlan | None = None
        self._transport_deadline = 0.0
        self._model_deadline = 0.0
        self._last_model_progress: float | None = None
        self._maximum_progress_gap = 0.0
        self._retry_usage: dict[str, Any] | None = None

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
                if self._attempt_completed:
                    await self._finish_attempt()
                    self._closed = True
                    raise
                error = PrematureCompletionError(
                    "provider stream ended before completion"
                )
                retry = await self._retry(error)
                await self._finish_attempt(error)
                if retry:
                    continue
                self._closed = True
                raise error
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    await self._finish_attempt(error)
                    self._closed = True
                    raise
                context = getattr(error, "diagnostic_context", None)
                if isinstance(context, dict):
                    context["max_progress_gap_s"] = self._maximum_progress_gap
                retry = await self._retry(error)
                await self._finish_attempt(error)
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
            if event.type is StreamEventType.MESSAGE_END:
                self._attempt_completed = event.data.get("truncated") is not True
                usage = event.data.get("usage")
                if isinstance(usage, Mapping):
                    self._retry_usage = dict(usage)
            if event.type is StreamEventType.ERROR and event.error is not None:
                retry = await self._retry(event.error, event.data)
                await self._finish_attempt(event.error)
                if retry:
                    continue
            self._observe(event)
            if self._attempt_completed:
                await self._finish_attempt()
                self._closed = True
            return event

    async def _open_attempt(self) -> None:
        now = self._clock.now()
        if not self._control.budget.start_attempt(self._adapter.provider, now):
            self._closed = True
            raise RuntimeError("provider retry budget exhausted")
        self._attempt_completed = False
        self._attempt = self._adapter.create_attempt(
            self._messages, self._tool_schemas, self._token
        )
        self._transport_deadline = now + self._transport_idle
        self._model_deadline = now + self._model_idle

    async def _read_signal(self, attempt: ProviderAttempt) -> AttemptSignal:
        deadline = min(self._transport_deadline, self._model_deadline)
        read_task = asyncio.create_task(attempt.__anext__())
        timer_task = asyncio.create_task(self._clock.wait_until(deadline))
        try:
            done, _ = await asyncio.wait(
                (read_task, timer_task), return_when=asyncio.FIRST_COMPLETED
            )
            if read_task in done:
                return read_task.result()
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
            raise ProviderStallError(kind, seconds)
        finally:
            for task in (read_task, timer_task):
                if not task.done():
                    task.cancel()
            await self._drain_preserving_cancellation(read_task, timer_task)

    def _observe(self, event: StreamEvent) -> None:
        public_output = (
            event.message is not None
            or event.content is not None
            or event.delta is not None
            or event.tool_call is not None
        )
        completed_tool = event.data.get("tool_call_completed") is True
        completed_tool = completed_tool or (
            event.tool_call is not None and "tool_call_delta" not in event.data
        )
        completed_tool = completed_tool or isinstance(event.content, ToolUseContent)
        completed_tool = completed_tool or (
            event.message is not None
            and any(isinstance(block, ToolUseContent) for block in event.message.content)
        )
        self._emitted = self._emitted or public_output
        if completed_tool or (
            public_output and self._control.reset is ResetPolicy.NO_RESET
        ):
            self.commit()

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
            self._token = await self._attempt.refresh_auth()
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
        notice = self._retry_notice(plan, error)
        if self._retry_usage is not None:
            notice.data["usage"] = self._retry_usage
            self._retry_usage = None
        self._pending.append(notice)
        if self._emitted:
            if self._control.reset is ResetPolicy.NO_RESET:
                return False
            self._pending.append(StreamEvent(StreamEventType.ASSISTANT_RESET))
            self._emitted = False
        self._pending_retry = plan
        return True

    async def _wait_for_retry(self, delay: float) -> None:
        wait_task = asyncio.create_task(self._wait(delay))
        abort = self._control.abort
        abort_task = asyncio.create_task(abort.wait()) if abort is not None else None
        tasks = (wait_task,) if abort_task is None else (wait_task, abort_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if abort_task is not None and abort_task in done:
                raise asyncio.CancelledError
            await wait_task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
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
        return StreamEvent(
            StreamEventType.RETRY,
            data={
                "text": (
                    f"{retry_error_label(error)}, retrying in {plan.delay:g}s "
                    f"(attempt {plan.attempt}/{MAX_PROVIDER_ATTEMPTS})"
                ),
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
            await attempt.aclose()
        except BaseException as cleanup_error:
            primary_error = primary if isinstance(primary, BaseException) else None
            if primary is not None and primary_error is None:
                if is_control_exception(cleanup_error):
                    raise
            else:
                selected = merge_exception(primary_error, cleanup_error)
                if selected is cleanup_error:
                    if primary is None and not is_control_exception(cleanup_error):
                        raise CompletionCleanupError(str(cleanup_error)) from cleanup_error
                    raise selected
            if not is_control_exception(cleanup_error):
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "provider completion cleanup failed",
                        "exception": cleanup_error,
                        "provider": self._adapter.provider,
                    }
                )

    async def aclose(self) -> None:
        if self._closed and self._attempt is None:
            return
        self._closed = True
        await self._finish_attempt()

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
) -> CompletionRun:
    return CompletionRun(
        adapter,
        messages,
        tool_schemas,
        control,
        clock=clock or getattr(adapter, "_completion_clock", None),
        wait=wait or getattr(adapter, "_completion_wait", None),
    )


def transient_network_error(error: BaseException | object) -> bool:
    current = error if isinstance(error, BaseException) else None
    while current is not None:
        if isinstance(current, (ConnectionError, TimeoutError, httpx.TransportError)):
            return True
        current = current.__cause__
    return False


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


def retry_after_seconds(
    error: BaseException | object,
    event_data: Mapping[str, Any] | None = None,
) -> float | None:
    value = getattr(error, "retry_after", None)
    if value is None and event_data is not None:
        value = event_data.get("retry_after")
    return max(0.0, float(value)) if type(value) in {int, float} else None


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


def explain_unhonored_retry_after(source: BaseException | object, delay: float) -> None:
    if isinstance(source, BaseException):
        source.args = (
            (
                f"{source}; provider requested Retry-After of {delay:g}s, longer than "
                f"the remaining turn retry budget (maximum "
                f"{MAX_PROVIDER_RETRY_SECONDS:g}s); not retrying early"
            ),
        )


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
    if status_code is not None:
        return f"{status_code} server error"
    if getattr(error, "retryable", False):
        return "provider overloaded"
    if transient_network_error(error):
        return "network error"
    return "provider error"
