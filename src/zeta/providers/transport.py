"""Shared async transport cleanup and exception precedence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import httpx

from .completion import (
    MustCompleteClose,
    is_control_exception,
    merge_exception,
    report_transport_activity,
    task_is_cancelling,
)
from .stream_diagnostics import StreamDiagnostics

ErrorT = TypeVar("ErrorT", bound=RuntimeError)
DEFAULT_STREAM_STALL_SECONDS = 90.0  # Transport idle, including missing keepalives.
# Long hidden reasoning is normal; diagnostics will let us safely tighten this later.
MODEL_PROGRESS_STALL_SECONDS = 15 * 60.0
DEFAULT_STREAM_STALL_RETRIES = 2
_BACKGROUND_CLOSE_TASKS: set[asyncio.Task[object]] = set()
_BACKGROUND_CLEANUP_REPORT_TASKS: set[asyncio.Task[object]] = set()
_MAX_BACKGROUND_CLEANUP_REPORTS = 16
_dropped_cleanup_reports = 0
# how long a lease close waits for open borrows before it closes the client anyway
LEASE_DRAIN_BOUND = 30.0
CloseFailureReporter = Callable[[int, BaseException], None]


@dataclass(frozen=True, slots=True)
class CloseFailure:
    index: int
    error: BaseException


@dataclass(frozen=True, slots=True)
class CloseOutcome:
    """Failures from one aggregate close; an empty tuple means success."""

    failures: tuple[CloseFailure, ...]


def _exception_type_name(error: BaseException) -> str:
    name = f"{type(error).__module__}.{type(error).__qualname__}"
    return name.encode()[:128].decode(errors="ignore")


def _primary_outcome(primary_exception: BaseException | None) -> str:
    if isinstance(primary_exception, asyncio.CancelledError):
        return "cancelled"
    return "error" if primary_exception is not None else "success"


def _cleanup_failure_context(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None = None,
) -> dict[str, object]:
    return {
        "exception_type": _exception_type_name(error),
        "stage": stage,
        "index": index,
        "primary_outcome": _primary_outcome(primary_exception),
    }


def _report_loop_cleanup_failure(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None = None,
) -> None:
    asyncio.get_running_loop().call_exception_handler(
        _cleanup_failure_context(
            error,
            stage=stage,
            index=index,
            primary_exception=primary_exception,
        )
    )


async def _report_cleanup_failure(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None = None,
    diagnostics_path: Path | None = None,
    provider: str = "provider",
) -> None:
    if diagnostics_path is None:
        _report_loop_cleanup_failure(
            error,
            stage=stage,
            index=index,
            primary_exception=primary_exception,
        )
        return
    context = _cleanup_failure_context(
        error,
        stage=stage,
        index=index,
        primary_exception=primary_exception,
    )
    try:
        await StreamDiagnostics.record_cleanup_failure(
            diagnostics_path,
            provider=provider,
            stage=stage,
            index=index,
            exception_type=str(context["exception_type"]),
            primary_outcome=str(context["primary_outcome"]),
        )
    except BaseException as diagnostic_error:
        if is_control_exception(diagnostic_error):
            raise merge_exception(primary_exception, diagnostic_error)
        _report_loop_cleanup_failure(
            diagnostic_error,
            stage=f"{stage}_diagnostic",
            index=index,
            primary_exception=primary_exception,
        )


def _release_background_close(
    task: asyncio.Task[object],
    index: int,
    report_failure: CloseFailureReporter | None,
) -> None:
    _BACKGROUND_CLOSE_TASKS.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is None or is_control_exception(error):
        return
    if report_failure is None:
        _report_loop_cleanup_failure(error, stage="background_close", index=index)
    else:
        with contextlib.suppress(BaseException):
            report_failure(index, error)


def _retain_background_closes(
    tasks: tuple[asyncio.Task[object], ...],
    report_failure: CloseFailureReporter | None,
) -> None:
    for index, task in enumerate(tasks):
        _BACKGROUND_CLOSE_TASKS.add(task)
        task.add_done_callback(
            lambda done, index=index: _release_background_close(
                done, index, report_failure
            )
        )


def _schedule_cleanup_failure(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None,
    diagnostics_path: Path | None,
    provider: str,
) -> None:
    global _dropped_cleanup_reports
    if len(_BACKGROUND_CLEANUP_REPORT_TASKS) >= _MAX_BACKGROUND_CLEANUP_REPORTS:
        _dropped_cleanup_reports += 1
        _report_loop_cleanup_failure(
            RuntimeError(),
            stage=f"{stage}_diagnostic_dropped",
            index=_dropped_cleanup_reports,
            primary_exception=primary_exception,
        )
        return
    task = asyncio.create_task(
        _report_cleanup_failure(
            error,
            stage=stage,
            index=index,
            primary_exception=primary_exception,
            diagnostics_path=diagnostics_path,
            provider=provider,
        )
    )
    _BACKGROUND_CLEANUP_REPORT_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_CLEANUP_REPORT_TASKS.discard)
    _retain_background_closes((task,), None)


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






class SSEEventAccumulator[ErrorT: RuntimeError]:
    """Accumulate one JSON SSE event while preserving provider error types."""

    __slots__ = ("data", "error_type", "event", "provider")

    def __init__(self, provider: str, error_type: type[ErrorT]) -> None:
        self.provider = provider
        self.error_type = error_type
        self.event = "message"
        self.data: list[str] = []

    def feed(self, line: str) -> tuple[str, dict[str, Any]] | None:
        if not line:
            return self.finish()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            self.event = value
        elif field == "data":
            self.data.append(value)
        return None

    def finish(self) -> tuple[str, dict[str, Any]] | None:
        if not self.data:
            self.event = "message"
            return None
        event = self.event
        value = "\n".join(self.data)
        self.event = "message"
        self.data = []
        if value == "[DONE]":
            return event, {"type": "done"}
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise self.error_type(f"{self.provider} returned invalid SSE JSON") from exc
        if not isinstance(payload, dict):
            raise self.error_type(f"{self.provider} SSE payload is not an object")
        return event, payload


class HTTPClientLease:
    """Own one completion client and drain active borrows before closing it."""

    __slots__ = (
        "_background_reporter_registered",
        "_borrowers",
        "_close",
        "_drained",
        "_retired",
        "client",
        "connection_setup_count",
        "owned",
    )

    def __init__(
        self, configured: httpx.AsyncClient | None, *, timeout: object = None
    ) -> None:
        self.owned = configured is None
        self.client = configured or httpx.AsyncClient(timeout=timeout)
        self.connection_setup_count = int(self.owned)
        self._borrowers: Counter[str] = Counter()
        self._background_reporter_registered = False
        self._close = MustCompleteClose("http client", LEASE_DRAIN_BOUND)
        self._drained = asyncio.Event()
        self._drained.set()
        self._retired = False

    @property
    def _active_borrows(self) -> int:
        return self._borrowers.total()

    @contextlib.asynccontextmanager
    async def borrow(self, borrower: str) -> AsyncIterator[httpx.AsyncClient]:
        if self._retired:
            raise RuntimeError("completion backend is retired")
        self._borrowers[borrower] += 1
        self._drained.clear()
        try:
            yield self.client
        finally:
            self._borrowers[borrower] -= 1
            if not +self._borrowers:
                self._borrowers.clear()
                self._drained.set()

    def retire(self) -> None:
        self._retired = True

    async def _close_owned(self) -> None:
        try:
            # test-contract: seam the close owner bounds this drain
            await self._drained.wait()
        finally:
            # the drain bound expired or the drain ended: close the client
            # either way, and name any borrow that never ended.
            if self.owned:
                await self.client.aclose()
            if self._borrowers:
                raise RuntimeError(
                    "http client closed with borrows still open: "
                    + ", ".join(sorted(+self._borrowers))
                )

    def start_close(self) -> None:
        self.retire()
        self._close.start(self._close_owned)

    async def aclose(
        self, report_failure: CloseFailureReporter | None = None
    ) -> None:
        self.start_close()
        assert self._close.task is not None
        try:
            await self._close.wait()
        except asyncio.CancelledError:
            if report_failure is not None and not self._background_reporter_registered:
                self._background_reporter_registered = True
                _retain_background_closes((self._close.task,), report_failure)
            raise

    async def refresh(
        self, refresh_token: Callable[[httpx.AsyncClient], Awaitable[str]]
    ) -> str:
        async with self.borrow("token refresh") as client:
            return await refresh_token(client)


async def build_response_error[ErrorT: RuntimeError](
    response: httpx.Response,
    build: Callable[[int, bytes, Mapping[str, str]], ErrorT],
    limit: int = 8192,
    *,
    stall_seconds: float,
    on_stall: Callable[[float], BaseException],
) -> ErrorT:
    """Build a provider error from a bounded streaming response body."""

    body = bytearray()
    del stall_seconds, on_stall
    async for chunk in response.aiter_bytes():
        report_transport_activity()
        body.extend(chunk[: limit - len(body)])
        if len(body) >= limit:
            break
    return build(response.status_code, bytes(body), response.headers)


class StreamProgressDiagnostics:
    """Measure decoded model gaps without owning any deadline."""

    __slots__ = ("last_at", "max_gap_s")

    _NON_PROGRESS_EVENTS: ClassVar[frozenset[str]] = frozenset(
        {"keepalive", "ping", "response.in_progress", "response.metadata"}
    )

    def __init__(self) -> None:
        self.last_at = time.monotonic()
        self.max_gap_s = 0.0

    def observe(self, record: tuple[str, dict[str, Any]]) -> None:
        event, payload = record
        if payload.get("type", event) not in self._NON_PROGRESS_EVENTS:
            now = time.monotonic()
            self.max_gap_s = max(self.max_gap_s, now - self.last_at)
            self.last_at = now

    def maximum_gap_s(self) -> float:
        return max(self.max_gap_s, time.monotonic() - self.last_at)


class StreamFinished:
    """Mutable flag the provider decoder sets when the completion sentinel
    arrives, so ``sse_lines``' stall watchdog treats any post-completion
    silence as a clean EOF instead of a spurious stall retry."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False








async def sse_lines[E: RuntimeError](
    response: httpx.Response,
    seconds: float,
    provider: str,
    error_class: type[E],
    finished: StreamFinished | None = None,
) -> AsyncIterator[str]:
    """Yield low-level SSE lines while the completion owner tracks activity."""
    del seconds, provider, error_class, finished
    async for line in response.aiter_lines():
        report_transport_activity()
        yield line








def request_error[ErrorT: RuntimeError](
    cause: httpx.HTTPError, error_type: type[ErrorT]
) -> ErrorT:
    error = error_type("request failed")
    error.__cause__ = cause
    return error


async def close_transport_resources(
    *closes: Awaitable[object],
    report_failure: CloseFailureReporter | None = None,
) -> CloseOutcome:
    """Close every owned resource and return each ordinary failure."""

    created: list[asyncio.Task[object]] = []
    try:
        for close in closes:
            created.append(asyncio.create_task(close))
        tasks = tuple(created)
        if task_is_cancelling():
            await asyncio.sleep(0)
            for task in tasks:
                task.cancel()
        results = tuple(await asyncio.gather(*tasks, return_exceptions=True))
        cancellation = next(
            (result for result in results if isinstance(result, asyncio.CancelledError)),
            None,
        )
        if cancellation is not None:
            raise cancellation
        return CloseOutcome(
            tuple(
                CloseFailure(index, result)
                for index, result in enumerate(results)
                if isinstance(result, BaseException)
            )
        )
    finally:
        tasks = tuple(created)
        pending = tuple(task for task in tasks if not task.done())
        for task in pending:
            task.cancel()
        if task_is_cancelling():
            _retain_background_closes(tasks, report_failure)
        elif pending:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                _retain_background_closes(tasks, report_failure)
                raise


async def close_resource(resource: object) -> None:
    """Close one async resource through the shared cancellation-safe owner."""

    outcome = await close_transport_resources(
        resource.aclose()  # type: ignore[attr-defined]
    )
    if outcome.failures:
        raise outcome.failures[0].error


async def cleanup_transport[ErrorT: RuntimeError](
    *,
    stream_context: object | None,
    entered: bool,
    client: object,
    owns_client: bool,
    primary_exception: BaseException | None,
    http_error_type: type[ErrorT],
    diagnostics_path: Path | None = None,
    provider: str = "provider",
) -> BaseException | None:
    """Close one request while preserving and diagnosing its primary outcome."""

    closes: list[Awaitable[object]] = []
    resources: list[str] = []
    if entered and stream_context is not None:
        closes.append(
            stream_context.__aexit__(  # type: ignore[attr-defined]
                type(primary_exception) if primary_exception else None,
                primary_exception,
                primary_exception.__traceback__ if primary_exception else None,
            )
        )
        resources.append("response")
    if owns_client:
        closes.append(client.aclose())  # type: ignore[attr-defined]
        resources.append("client")

    def schedule_failure(index: int, error: BaseException) -> None:
        diagnostic_error = (
            request_error(error, http_error_type)
            if isinstance(error, httpx.HTTPError)
            else error
        )
        _schedule_cleanup_failure(
            diagnostic_error,
            stage=resources[index],
            index=index,
            primary_exception=primary_exception,
            diagnostics_path=diagnostics_path,
            provider=provider,
        )

    try:
        outcome = await close_transport_resources(
            *closes, report_failure=schedule_failure
        )
    except asyncio.CancelledError as exc:
        raise merge_exception(primary_exception, exc)
    for failure in outcome.failures:
        if is_control_exception(failure.error):
            primary_exception = merge_exception(primary_exception, failure.error)
        else:
            diagnostic_error = (
                request_error(failure.error, http_error_type)
                if isinstance(failure.error, httpx.HTTPError)
                else failure.error
            )
            await _report_cleanup_failure(
                diagnostic_error,
                stage=resources[failure.index],
                index=failure.index,
                primary_exception=primary_exception,
                diagnostics_path=diagnostics_path,
                provider=provider,
            )
    if task_is_cancelling():
        primary_exception = merge_exception(primary_exception, asyncio.CancelledError())
    if isinstance(primary_exception, asyncio.CancelledError):
        raise primary_exception
    return primary_exception
