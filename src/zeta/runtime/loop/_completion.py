"""Turn-completion lifecycle helpers shared by the agent loop.

This module owns completion cleanup and both safe retry policies. Provider
retries are allowed only before the current user turn has committed assistant
state or dispatched a tool.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...core.abort import AbortSignal
from ...protocol.types import (
    ContentBlock,
    ErrorInfo,
    Message,
    StreamEvent,
    StreamEventType,
    ToolUseContent,
)

MAX_TURN_PROVIDER_RETRIES = 3
MAX_TURN_PROVIDER_RETRY_SECONDS = 60.0
MAX_TURN_PROVIDER_RETRY_DELAY_SECONDS = 30.0
MAX_ERROR_MESSAGE = 400


def _error_info(error: BaseException, *, provider_error: bool = False) -> ErrorInfo:
    """Normalize provider and transport failures for the transcript."""

    code = getattr(error, "code", None)
    if type(code) is not str or not code:
        if isinstance(error, TimeoutError):
            code = "timeout"
        elif isinstance(error, (ConnectionError, httpx.TransportError)):
            code = "transport_error"
        else:
            cause = error.__cause__
            while cause is not None:
                if isinstance(cause, (ConnectionError, TimeoutError, httpx.TransportError)):
                    code = "transport_error"
                    break
                cause = cause.__cause__
            else:
                code = "backend_error"
    try:
        message = str(error).strip()
    except Exception:  # noqa: BLE001 - malformed exception text is recoverable
        message = ""
    if not message:
        message = type(error).__name__
    if len(message) > MAX_ERROR_MESSAGE:
        message = f"{message[: MAX_ERROR_MESSAGE - 3]}..."
    return ErrorInfo(
        code,
        message,
        status_code=getattr(error, "status_code", None),
        provider_error=provider_error,
    )


async def close_completion(
    completion: AsyncIterator[StreamEvent] | None,
) -> BaseException | None:
    if completion is None:
        return None
    close = getattr(completion, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except BaseException as exc:  # noqa: BLE001 - preserve close errors
        return exc
    return None


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def can_retry_context(
    error: ErrorInfo, retrying: bool, partial: list[ContentBlock], message: Message | None
) -> bool:
    return (
        error.code == "context_length_exceeded"
        and not retrying
        and not partial
        and message is None
    )


@dataclass(frozen=True, slots=True)
class ProviderRetryPlan:
    attempt: int
    reason: str
    delay: float

    def metadata(self) -> dict[str, object]:
        return {"attempt": self.attempt, "reason": self.reason, "delay": self.delay}


@dataclass(slots=True)
class ProviderRetryState:
    """Bound retry count, elapsed budget, and transcript-safe metadata."""

    started_at: float = field(default_factory=time.monotonic)
    retries: int = 0
    original_error: ErrorInfo | None = None
    exhausted: bool = False
    records: list[dict[str, object]] = field(default_factory=list)


def provider_attempt_uncommitted(
    turn_committed: bool,
    partial: list[ContentBlock],
    message: Message | None,
) -> bool:
    """Return whether retry can discard this attempt without repeating effects."""

    if turn_committed or message is not None:
        return False
    return not any(isinstance(block, ToolUseContent) for block in partial)


def plan_provider_retry(
    error: ErrorInfo,
    source: BaseException | ErrorInfo,
    *,
    event_data: Mapping[str, Any] | None,
    state: ProviderRetryState,
    safe: bool,
    aborted: bool,
    clock: Any = time.monotonic,
) -> ProviderRetryPlan | None:
    """Plan one safe retry, or return ``None`` and preserve terminal behavior."""

    if state.original_error is None:
        state.original_error = error
    if aborted or not safe or not _is_retryable_provider_failure(source, event_data):
        return None
    if state.retries >= MAX_TURN_PROVIDER_RETRIES:
        state.exhausted = True
        return None

    delay = _provider_retry_delay(source, event_data, state.retries + 1)
    if clock() - state.started_at + delay > MAX_TURN_PROVIDER_RETRY_SECONDS:
        state.exhausted = True
        return None

    state.retries += 1
    plan = ProviderRetryPlan(state.retries + 1, error.code, delay)
    state.records.append(plan.metadata())
    return plan


def provider_retry_notice(plan: ProviderRetryPlan, *, discard_partial: bool) -> StreamEvent:
    return StreamEvent(
        type=StreamEventType.RETRY,
        data={
            "kind": "provider_retry",
            "text": (
                f"provider error ({plan.reason}), retrying in {plan.delay:g}s "
                f"(attempt {plan.attempt}/{MAX_TURN_PROVIDER_RETRIES + 1})"
            ),
            "retry": plan.attempt - 1,
            "attempt": plan.attempt,
            "reason": plan.reason,
            "delay": plan.delay,
            "discard_partial": discard_partial,
        },
    )


async def wait_for_provider_retry(
    delay: float, abort_signal: AbortSignal | None
) -> bool:
    """Wait for backoff, returning false immediately when the user aborts."""

    if abort_signal is None:
        await asyncio.sleep(delay)
        return True
    if abort_signal.is_set():
        return False
    sleep_task = asyncio.create_task(asyncio.sleep(delay))
    abort_task = asyncio.create_task(abort_signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        return sleep_task in done and not abort_signal.is_set()
    finally:
        for task in (sleep_task, abort_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleep_task, abort_task, return_exceptions=True)


def _is_retryable_provider_failure(
    source: BaseException | ErrorInfo,
    event_data: Mapping[str, Any] | None,
) -> bool:
    status_code = getattr(source, "status_code", None)
    if status_code == 429 or status_code in {500, 502, 503, 504, 520, 521, 522, 523, 524, 529}:
        return True
    if getattr(source, "retryable", False):
        return True
    if isinstance(source, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    cause = source.__cause__ if isinstance(source, BaseException) else None
    while cause is not None:
        if isinstance(cause, (ConnectionError, TimeoutError, httpx.TransportError)):
            return True
        cause = cause.__cause__
    code = source.code if isinstance(source, ErrorInfo) else getattr(source, "code", None)
    message = source.message if isinstance(source, ErrorInfo) else str(source)
    if code in {"timeout", "transport_error"}:
        return True
    if code == "http_error" and message.strip().lower() == "request failed":
        return True
    return bool(event_data and event_data.get("retryable") is True)


def _provider_retry_delay(
    source: BaseException | ErrorInfo,
    event_data: Mapping[str, Any] | None,
    retry_number: int,
) -> float:
    retry_after = getattr(source, "retry_after", None)
    if retry_after is None and event_data is not None:
        retry_after = event_data.get("retry_after")
    if type(retry_after) in {int, float}:
        return max(0.0, min(MAX_TURN_PROVIDER_RETRY_DELAY_SECONDS, float(retry_after)))
    base = min(MAX_TURN_PROVIDER_RETRY_DELAY_SECONDS, float(2 ** (retry_number - 1)))
    return min(
        MAX_TURN_PROVIDER_RETRY_DELAY_SECONDS,
        random.uniform(base * 0.5, base * 1.5),
    )
