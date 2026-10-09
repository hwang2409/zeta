"""Turn-completion lifecycle helpers shared by the agent loop.

This module owns completion cleanup and both safe retry policies. Provider
retries are allowed only before the current user turn has committed assistant
state or dispatched a tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
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
from ...providers.retry_policy import (
    MAX_PROVIDER_ATTEMPTS,
    ProviderRetryBudget,
    RetryPlan,
)

MAX_ERROR_MESSAGE = 400


async def provider_events(
    completion: AsyncIterator[StreamEvent],
    on_started: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Preserve stream ordering while exposing the provider-start seam."""

    finished = object()
    events: asyncio.Queue[StreamEvent | object] = asyncio.Queue(maxsize=1)
    consumed = asyncio.Event()
    started = asyncio.Event()

    async def produce() -> None:
        started.set()
        try:
            async for event in completion:
                await events.put(event)
                await consumed.wait()
                consumed.clear()
        finally:
            task = asyncio.current_task()
            if task is None or not task.cancelling():
                await events.put(finished)

    producer = asyncio.create_task(produce())
    try:
        await started.wait()
        if producer.done() and not producer.cancelled():
            error = producer.exception()
            if error is not None:
                raise error
        if on_started is not None:
            await on_started()
        while (event := await events.get()) is not finished:
            assert isinstance(event, StreamEvent)
            try:
                yield event
            finally:
                consumed.set()
        await producer
    finally:
        if not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)


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


def ensure_context_reduced(current: int | None, previous: int | None) -> None:
    if current is not None and previous is not None and current >= previous:
        raise RuntimeError(
            "provider context overflow recovery could not reduce the prompt below "
            f"{previous} estimated tokens; raise --token-budget only if the provider "
            "limit also increased, or start a new session"
        )


def provider_prompt_tokens(error: BaseException | ErrorInfo | None) -> int | None:
    value = getattr(error, "provider_prompt_tokens", None)
    return value if type(value) is int and value >= 0 else None


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
    tool_call_completed: bool = False
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
        if event.data.get("tool_call_completed") is True:
            self.tool_call_completed = True
        elif event.tool_call is not None and "tool_call_delta" not in event.data:
            # Providers such as Ollama emit only complete tool calls.
            self.tool_call_completed = True
        if isinstance(event.content, ToolUseContent):
            self.tool_call_completed = True
        if event.message is not None and any(
            isinstance(block, ToolUseContent) for block in event.message.content
        ):
            self.tool_call_completed = True

    @property
    def can_retry(self) -> bool:
        return self.started and not self.persisted and not self.tool_call_completed

    def _retry_block_reason(
        self, *, reset_supported: bool, turn_aborted: bool
    ) -> str | None:
        if self.persisted:
            return "assistant_persisted"
        if self.tool_call_completed:
            return "tool_call_completed"
        if not reset_supported:
            return "assistant_reset_not_supported"
        if turn_aborted:
            return "turn_aborted"
        return None

    def retry_plan(
        self,
        budget: ProviderRetryBudget | None,
        source: BaseException | object,
        *,
        reset_supported: bool,
        turn_aborted: bool,
        event_data: Mapping[str, Any],
    ) -> RetryPlan | None:
        if budget is None:
            return None
        reason = self._retry_block_reason(
            reset_supported=reset_supported,
            turn_aborted=turn_aborted,
        )
        if self.started and reason is not None:
            budget.records.append(
                {"decision": "skipped-after-output", "reason": reason}
            )
            return None
        if not self.can_retry:
            return None
        return budget.plan(source, owner="loop", event_data=event_data)


def start_provider_attempt(
    budget: ProviderRetryBudget | None,
) -> ProviderRetryBudget:
    budget = budget or ProviderRetryBudget()
    if not budget.start_attempt("loop"):
        raise RuntimeError("provider retry budget exhausted")
    return budget


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
