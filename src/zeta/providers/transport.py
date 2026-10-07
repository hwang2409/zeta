"""Shared async transport cleanup and exception precedence."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

from ..protocol.types import StreamEvent, StreamEventType
from .retry_policy import (
    MAX_PROVIDER_ATTEMPTS,
    ProviderRetryBudget,
    current_retry_budget,
    retry_error_label,
)

ErrorT = TypeVar("ErrorT", bound=RuntimeError)
DEFAULT_STREAM_STALL_SECONDS = 90.0
DEFAULT_STREAM_STALL_RETRIES = 2


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return parsed if parsed >= 0 else None


def format_retry_delay(delay: float) -> str:
    return f"{delay:.1f}".rstrip("0").rstrip(".")


def stall_retry_notice(
    retry_number: int, delay: float, max_retries: int
) -> StreamEvent:
    """Build a stall RETRY event; ``is_stall`` also signals the loop to
    clear its accumulated partial before the retry resumes streaming."""

    return StreamEvent(
        StreamEventType.RETRY,
        data={
            "text": (
                f"provider stalled, retrying ({retry_number}/{max_retries})"
                f" in {format_retry_delay(delay)}s"
            ),
            "retry": retry_number,
            "delay": delay,
            "is_stall": True,
        },
    )


def stall_retry_kwargs(stall_retries: int) -> dict[str, object]:
    """Wire ``retry_provider_completion`` with the shared stall predicate."""

    return {
        "is_stall": lambda error: getattr(error, "is_stall", False),
        "stall_notice": lambda n, d, _e: stall_retry_notice(n, d, stall_retries),
        "max_stall_retries": stall_retries,
    }


class StreamFinished:
    """Mutable flag the provider decoder sets when the completion sentinel
    arrives, so ``sse_lines``' stall watchdog treats any post-completion
    silence as a clean EOF instead of a spurious stall retry."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False


async def wait_for_response_headers[E: RuntimeError](
    enter: Awaitable[httpx.Response],
    seconds: float,
    provider: str,
    error_class: type[E],
) -> httpx.Response:
    """Apply the stream stall limit before SSE iteration can start."""

    try:
        return await asyncio.wait_for(enter, timeout=seconds if seconds > 0 else None)
    except TimeoutError as exc:
        raise error_class(
            f"{provider} response headers stalled for {seconds:.0f}s", is_stall=True
        ) from exc


def sse_lines[E: RuntimeError](
    response: httpx.Response,
    seconds: float,
    provider: str,
    error_class: type[E],
    finished: StreamFinished | None = None,
) -> AsyncIterator[str]:
    """Yield SSE lines from ``response`` and raise ``error_class`` on stall.

    A ``finished`` flag lets the caller signal that the provider has already
    emitted its completion sentinel; post-completion silence is then treated
    as a clean EOF instead of a stall retry."""

    def on_stall(elapsed: float) -> E:
        return error_class(
            f"{provider} stream stalled for {elapsed:.0f}s", is_stall=True
        )

    is_finished = (lambda: finished.value) if finished is not None else None
    return stall_watchdog(
        response.aiter_lines(),
        seconds=seconds,
        on_stall=on_stall,
        is_finished=is_finished,
    )


def provider_retry_notice(
    retry_number: int, delay: float, error: RuntimeError
) -> StreamEvent:
    """Build the shared visible notice for a transient provider retry."""

    return StreamEvent(
        StreamEventType.RETRY,
        data={
            "text": (
                f"{retry_error_label(error)}, retrying in "
                f"{format_retry_delay(delay)}s "
                f"(attempt {retry_number + 1}/{MAX_PROVIDER_ATTEMPTS})"
            ),
            "retry": retry_number,
            "delay": delay,
        },
    )


async def retry_provider_completion(
    first: Callable[[], AsyncIterator[StreamEvent]],
    retry: Callable[[str], AsyncIterator[StreamEvent]],
    refresh: Callable[[], Awaitable[str]],
    is_unauthorized: Callable[[RuntimeError], bool],
    auth_exhausted: Callable[[RuntimeError], RuntimeError],
    notice: Callable[[int, float, RuntimeError], StreamEvent],
    on_exhausted: Callable[[RuntimeError, int], None],
    is_stall: Callable[[RuntimeError], bool] = lambda _error: False,
    stall_notice: Callable[[int, float, RuntimeError], StreamEvent] | None = None,
    max_stall_retries: int = DEFAULT_STREAM_STALL_RETRIES,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    defer_truncated_message_end: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Retry failures before the provider emits its first stream event.

    Once stream output starts, the agent loop owns retry safety and reset
    ordering. Both layers consume the same turn-level retry budget.
    """

    sleep = asyncio.sleep if sleep is None else sleep
    budget = current_retry_budget()
    if budget is None:
        budget = ProviderRetryBudget()
        if not budget.start_attempt("transport"):
            raise RuntimeError("provider retry budget exhausted")
    attempt_factory: Callable[[], AsyncIterator[StreamEvent]] = first
    refreshed = False
    stall_retries = 0
    while True:
        attempt = attempt_factory()
        started = False
        committed = False
        error: RuntimeError | None = None
        pending_truncated_end: StreamEvent | None = None
        try:
            async for value in attempt:
                started = True
                committed = committed or (
                    value.message is not None
                    or value.content is not None
                    or value.delta is not None
                    or value.tool_call is not None
                )
                if (
                    defer_truncated_message_end
                    and value.type is StreamEventType.MESSAGE_END
                    and value.data.get("truncated")
                ):
                    pending_truncated_end = value
                    continue
                yield value
        except RuntimeError as exc:
            error = exc
        finally:
            await attempt.aclose()
        if error is None:
            if pending_truncated_end is not None:
                yield pending_truncated_end
            return
        stalled = is_stall(error)
        # MESSAGE_START carries response.created metadata, not assistant content.
        # A stall after that event can still be retried safely.
        if started and (not stalled or committed):
            if pending_truncated_end is not None:
                yield pending_truncated_end
            raise error
        if is_unauthorized(error):
            if refreshed:
                raise auth_exhausted(error) from error
            if not budget.start_attempt("transport"):
                raise error
            token = await refresh()
            refreshed = True
            attempt_factory = lambda token=token: retry(token)
            continue
        if stalled and stall_retries >= max_stall_retries:
            on_exhausted(error, budget.attempts - 1)
            raise error
        plan = budget.plan(
            error,
            owner="transport",
            event_data={"retryable": True} if stalled else None,
        )
        if plan is None:
            if budget.exhausted:
                on_exhausted(error, budget.attempts - 1)
            raise error
        if stalled:
            stall_retries += 1
            emit = stall_notice if stall_notice is not None else notice
            retry_event = emit(stall_retries, plan.delay, error)
        else:
            retry_event = notice(budget.attempts, plan.delay, error)
        yield retry_event
        await sleep(plan.delay)
        budget.record_retry(plan)
        if not budget.start_attempt("transport", is_stall=stalled):
            on_exhausted(error, budget.attempts - 1)
            raise error


async def stall_watchdog[T](
    source: AsyncIterable[T],
    *,
    seconds: float,
    on_stall: Callable[[float], BaseException],
    is_finished: Callable[[], bool] | None = None,
) -> AsyncIterator[T]:
    """Yield from ``source`` and raise ``on_stall(elapsed)`` when no item
    arrives within ``seconds``. A non-positive ``seconds`` disables the
    watchdog. Every item resets the timer, so keepalives count as activity.

    ``is_finished`` lets the caller mark the stream complete; when set, a
    timeout returns cleanly (EOF) instead of raising a stall — protecting
    already-completed responses from a spurious post-sentinel retry."""

    if seconds <= 0:
        async for item in source:
            yield item
        return
    iterator = source.__aiter__()
    try:
        while True:
            try:
                item = await asyncio.wait_for(iterator.__anext__(), timeout=seconds)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                if is_finished is not None and is_finished():
                    return
                raise on_stall(seconds) from exc
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def is_control_exception(value: BaseException) -> bool:
    return isinstance(value, (asyncio.CancelledError, GeneratorExit))


def control_priority(value: BaseException) -> int:
    if isinstance(value, asyncio.CancelledError):
        return 2
    if isinstance(value, GeneratorExit):
        return 1
    return 0


def request_error[ErrorT: RuntimeError](
    cause: httpx.HTTPError, error_type: type[ErrorT]
) -> ErrorT:
    error = error_type("request failed")
    error.__cause__ = cause
    return error


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
        and control_priority(cleanup) > control_priority(primary)
    ):
        return cleanup
    if primary_is_control and not cleanup_is_control:
        primary.__context__ = cleanup
    return primary


async def cleanup_transport[ErrorT: RuntimeError](
    *,
    stream_context: object | None,
    entered: bool,
    client: object,
    owns_client: bool,
    primary_exception: BaseException | None,
    http_error_type: type[ErrorT],
) -> BaseException | None:
    if entered and stream_context is not None:
        try:
            await stream_context.__aexit__(  # type: ignore[attr-defined]
                type(primary_exception) if primary_exception else None,
                primary_exception,
                primary_exception.__traceback__ if primary_exception else None,
            )
        except BaseException as exc:
            cleanup = (
                request_error(exc, http_error_type)
                if isinstance(exc, httpx.HTTPError)
                else exc
            )
            primary_exception = merge_exception(primary_exception, cleanup)
    if owns_client:
        try:
            await client.aclose()  # type: ignore[attr-defined]
        except BaseException as exc:
            cleanup = (
                request_error(exc, http_error_type)
                if isinstance(exc, httpx.HTTPError)
                else exc
            )
            primary_exception = merge_exception(primary_exception, cleanup)
    if task_is_cancelling():
        primary_exception = merge_exception(
            primary_exception, asyncio.CancelledError()
        )
    return primary_exception
