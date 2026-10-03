"""Shared async transport cleanup and exception precedence."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import replace
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

from ..protocol.types import StreamEvent, StreamEventType, ToolUseContent

ErrorT = TypeVar("ErrorT", bound=RuntimeError)
MAX_PROVIDER_ATTEMPTS = 5
MAX_PROVIDER_RETRY_WINDOW_SECONDS = 180.0
INITIAL_PROVIDER_RETRY_WAIT_SECONDS = 5.0
MAX_RETRY_WAIT_SECONDS = MAX_PROVIDER_RETRY_WINDOW_SECONDS
DEFAULT_STREAM_STALL_SECONDS = 90.0
DEFAULT_STREAM_STALL_RETRIES = 2


def transient_network_error(error: RuntimeError) -> bool:
    """Return whether a transport or request timeout is in the cause chain."""

    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, (httpx.TransportError, TimeoutError)):
            return True
        cause = cause.__cause__
    return False


def retryable_provider_error(error: RuntimeError) -> bool:
    """Return whether a provider error is safe to retry before streaming."""

    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}:
        return True
    if getattr(error, "retryable", False):
        return True
    return transient_network_error(error)


def retry_wait_seconds(error: RuntimeError, retry_number: int) -> float:
    """Return a bounded exponential wait, honoring a provider retry hint."""

    retry_after = getattr(error, "retry_after", None)
    if type(retry_after) is int or type(retry_after) is float:
        return max(0.0, min(MAX_RETRY_WAIT_SECONDS, float(retry_after)))
    base = min(
        MAX_RETRY_WAIT_SECONDS,
        INITIAL_PROVIDER_RETRY_WAIT_SECONDS * 2 ** (retry_number - 1),
    )
    return min(MAX_RETRY_WAIT_SECONDS, random.uniform(base * 0.5, base * 1.5))


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


def stream_event_retry_safe(event: StreamEvent) -> bool:
    """Return false once an attempt has exposed a tool call to a consumer."""

    if event.tool_call is not None or isinstance(event.content, ToolUseContent):
        return False
    return event.message is None or not any(
        isinstance(block, ToolUseContent) for block in event.message.content
    )


def _discard_partial_notice(event: StreamEvent) -> StreamEvent:
    return replace(event, data={**event.data, "discard_partial": True})


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


def retry_error_label(error: RuntimeError) -> str:
    status_code = getattr(error, "status_code", None)
    retry_reason = getattr(error, "retry_reason", None)
    if status_code == 429 or retry_reason == "rate_limit_error":
        return "429 rate limited"
    if status_code == 529 or retry_reason == "overloaded_error":
        return "529 overloaded"
    if status_code is not None:
        return f"{status_code} server error"
    if getattr(error, "retryable", False):
        return "provider overloaded"
    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, (httpx.TransportError, TimeoutError)):
            return "network error"
        cause = cause.__cause__
    return "provider error"


async def retry_provider_completion(
    first: Callable[[], AsyncIterator[StreamEvent]],
    retry: Callable[[str], AsyncIterator[StreamEvent]],
    refresh: Callable[[], Awaitable[str]],
    is_unauthorized: Callable[[RuntimeError], bool],
    auth_exhausted: Callable[[RuntimeError], RuntimeError],
    is_started: Callable[[StreamEvent], bool],
    is_retryable: Callable[[RuntimeError], bool],
    notice: Callable[[int, float, RuntimeError], StreamEvent],
    on_exhausted: Callable[[RuntimeError, int], None],
    is_stall: Callable[[RuntimeError], bool] = lambda _error: False,
    stall_notice: Callable[[int, float, RuntimeError], StreamEvent] | None = None,
    max_stall_retries: int = DEFAULT_STREAM_STALL_RETRIES,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
    retry_window_seconds: float = MAX_PROVIDER_RETRY_WINDOW_SECONDS,
    defer_truncated_message_end: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Retry transient failures within one bounded provider-attempt window.

    A started attempt is retried only while it has emitted no tool call. The
    retry event tells consumers to discard that attempt's partial assistant
    state before the next attempt starts.
    """

    sleep = asyncio.sleep if sleep is None else sleep
    clock = time.monotonic if clock is None else clock
    attempt_factory: Callable[[], AsyncIterator[StreamEvent]] = first
    started_at = clock()
    refreshed = False
    retries = 0
    stall_retries = 0
    while True:
        attempt = attempt_factory()
        started = False
        retry_safe = True
        error: RuntimeError | None = None
        pending_truncated_end: StreamEvent | None = None
        try:
            async for value in attempt:
                started = started or is_started(value)
                retry_safe = retry_safe and stream_event_retry_safe(value)
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
        if stalled and not retry_safe:
            if pending_truncated_end is not None:
                yield pending_truncated_end
            raise error
        if is_unauthorized(error):
            if started:
                if pending_truncated_end is not None:
                    yield pending_truncated_end
                raise error
            if refreshed:
                raise auth_exhausted(error) from error
            token = await refresh()
            refreshed = True
            attempt_factory = lambda token=token: retry(token)
            continue
        retryable = stalled or (
            is_retryable(error)
            and not (
                started and (not retry_safe or not transient_network_error(error))
            )
        )
        if not retryable:
            if pending_truncated_end is not None:
                yield pending_truncated_end
            raise error
        delay = retry_wait_seconds(error, retries + 1)
        window_exhausted = clock() - started_at + delay > retry_window_seconds
        stall_exhausted = stalled and stall_retries >= max_stall_retries
        if (
            retries >= MAX_PROVIDER_ATTEMPTS - 1
            or window_exhausted
            or stall_exhausted
        ):
            on_exhausted(error, retries)
            if pending_truncated_end is not None:
                yield pending_truncated_end
            raise error
        retries += 1
        if stalled:
            stall_retries += 1
            emit = stall_notice if stall_notice is not None else notice
            retry_event = emit(stall_retries, delay, error)
        else:
            retry_event = notice(retries, delay, error)
        if pending_truncated_end is not None:
            usage = pending_truncated_end.data.get("usage")
            if usage is not None:
                retry_event = replace(
                    retry_event, data={**retry_event.data, "usage": usage}
                )
        yield _discard_partial_notice(retry_event) if started else retry_event
        await sleep(delay)


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
