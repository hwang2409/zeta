"""Turn-completion lifecycle helpers shared by the agent loop.

This module owns completion cleanup and both safe retry policies. Provider
retries are allowed only before the current user turn has committed assistant
state or dispatched a tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
from ...providers.retry_policy import (
    MAX_PROVIDER_ATTEMPTS,
    RetryPlan,
)

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


@dataclass(slots=True)
class ProviderAttemptState:
    """Exposure and persistence state for one streamed provider attempt."""

    started: bool = False
    tool_call_exposed: bool = False
    persisted: bool = False

    def observe(self, event: StreamEvent) -> None:
        if (
            event.type
            in {
                StreamEventType.MESSAGE_START,
                StreamEventType.MESSAGE_UPDATE,
                StreamEventType.MESSAGE_END,
            }
            or event.message is not None
            or event.content is not None
            or event.delta is not None
            or event.tool_call is not None
        ):
            self.started = True
        if event.tool_call is not None or isinstance(event.content, ToolUseContent):
            self.tool_call_exposed = True
        if event.message is not None and any(
            isinstance(block, ToolUseContent) for block in event.message.content
        ):
            self.tool_call_exposed = True

    @property
    def can_retry(self) -> bool:
        return self.started and not self.persisted and not self.tool_call_exposed


def provider_retry_notice(plan: RetryPlan) -> StreamEvent:
    """Tell consumers that a retry is scheduled, without clearing output yet."""

    label = "provider stalled" if plan.is_stall else f"provider error ({plan.reason})"
    data: dict[str, object] = {
        "kind": "provider_retry",
        "text": (
            f"{label}, retry scheduled in {plan.delay:g}s "
            f"(attempt {plan.attempt}/{MAX_PROVIDER_ATTEMPTS})"
        ),
        "retry": plan.attempt - 1,
        "attempt": plan.attempt,
        "reason": plan.reason,
        "delay": plan.delay,
    }
    if plan.is_stall:
        data["is_stall"] = True
    return StreamEvent(type=StreamEventType.RETRY, data=data)


def assistant_reset_event() -> StreamEvent:
    """Tell output adapters to drop the unfinished assistant response."""

    return StreamEvent(StreamEventType.ASSISTANT_RESET)


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
