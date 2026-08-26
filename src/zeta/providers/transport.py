"""Shared async transport cleanup and exception precedence."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

ErrorT = TypeVar("ErrorT", bound=RuntimeError)
MAX_PROVIDER_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 10.0


def retryable_provider_error(error: RuntimeError) -> bool:
    """Return whether a provider error is safe to retry before streaming."""

    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504, 529}:
        return True
    if getattr(error, "retryable", False):
        return True
    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, httpx.TransportError):
            return True
        cause = cause.__cause__
    return False


def retry_wait_seconds(error: RuntimeError, retry_number: int) -> float:
    """Return a bounded exponential wait, honoring a provider retry hint."""

    retry_after = getattr(error, "retry_after", None)
    if type(retry_after) is int or type(retry_after) is float:
        return max(0.0, min(MAX_RETRY_WAIT_SECONDS, float(retry_after)))
    base = min(MAX_RETRY_WAIT_SECONDS, float(2 ** (retry_number - 1)))
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


def retry_error_label(error: RuntimeError) -> str:
    status_code = getattr(error, "status_code", None)
    retry_reason = getattr(error, "retry_reason", None)
    if status_code == 429 or retry_reason == "rate_limit_error":
        return "429 rate limited"
    if (
        status_code == 529
        or retry_reason == "overloaded_error"
        or getattr(error, "retryable", False)
    ):
        return "529 overloaded"
    if status_code is not None:
        return f"{status_code} server error"
    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, httpx.TransportError):
            return "connection error"
        cause = cause.__cause__
    return "provider error"


async def retry_provider_completion[T](
    first: Callable[[], AsyncIterator[T]],
    retry: Callable[[str], AsyncIterator[T]],
    refresh: Callable[[], Awaitable[str]],
    is_unauthorized: Callable[[RuntimeError], bool],
    auth_exhausted: Callable[[RuntimeError], RuntimeError],
    is_started: Callable[[T], bool],
    is_retryable: Callable[[RuntimeError], bool],
    notice: Callable[[int, float, RuntimeError], T],
    on_exhausted: Callable[[RuntimeError, int], None],
) -> AsyncIterator[T]:
    """Retry pre-stream provider failures with one optional auth refresh."""

    attempt_factory: Callable[[], AsyncIterator[T]] = first
    refreshed = False
    retries = 0
    while True:
        attempt = attempt_factory()
        started = False
        error: RuntimeError | None = None
        try:
            async for value in attempt:
                started = started or is_started(value)
                yield value
        except RuntimeError as exc:
            error = exc
        finally:
            await attempt.aclose()
        if error is None:
            return
        if started:
            raise error
        if is_unauthorized(error):
            if refreshed:
                raise auth_exhausted(error) from error
            token = await refresh()
            refreshed = True
            attempt_factory = lambda token=token: retry(token)
            continue
        if not is_retryable(error):
            raise error
        if retries >= MAX_PROVIDER_RETRIES:
            on_exhausted(error, retries)
            raise error
        retries += 1
        delay = retry_wait_seconds(error, retries)
        yield notice(retries, delay, error)
        await asyncio.sleep(delay)


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
