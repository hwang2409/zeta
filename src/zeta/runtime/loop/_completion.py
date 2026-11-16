"""Turn-completion lifecycle helpers shared by the agent loop.

This module owns completion cleanup and both safe retry policies. Provider
retries are allowed only before the current user turn has committed assistant
state or dispatched a tool.
"""

from __future__ import annotations

import asyncio
import errno
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Self

import httpx

from ...protocol.types import (
    ContentBlock,
    ErrorInfo,
    Message,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    ThinkingContent,
)

if TYPE_CHECKING:
    from ...core.store import ConversationStore
    from ...core.store._client_delivery import ClientSteeringBatch
    from ...telemetry import SpanHandle

MAX_ERROR_MESSAGE = 400


async def provider_events(
    completion: AsyncIterator[StreamEvent],
    on_started: Callable[[], Awaitable[None]],
) -> AsyncIterator[StreamEvent]:
    """Run ``on_started`` when provider consumption starts."""

    await on_started()
    async for event in completion:
        yield event


def observed_completion(
    completion: AsyncIterator[StreamEvent],
    store: ConversationStore,
    steering: ClientSteeringBatch | None,
) -> AsyncIterator[StreamEvent]:
    """Publish steering when provider consumption starts, keeping event order."""

    if steering is None:
        return completion
    return provider_events(
        completion, partial(store.deliver_client_steering, steering)
    )


def _is_permission_error(error: BaseException) -> bool:
    return isinstance(error, PermissionError) or (
        isinstance(error, OSError) and error.errno in {errno.EACCES, errno.EPERM}
    )


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _classify_error(error: BaseException) -> str:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_permission_error(current):
            return "permission_error"
        current = current.__cause__

    code = getattr(error, "code", None)
    if type(code) is str and code:
        return code

    for current in _exception_chain(error):
        if _is_permission_error(current):
            return "permission_error"
        if isinstance(current, TimeoutError):
            return "timeout" if current is error else "transport_error"
        if isinstance(current, (ConnectionError, httpx.TransportError)):
            return "transport_error"
        if isinstance(current, OSError):
            return "os_error"
    return "backend_error"


def _error_info(error: BaseException, *, provider_error: bool = False) -> ErrorInfo:
    """Normalize provider and transport failures for the transcript."""

    code = _classify_error(error)
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


def _sanitize_account_message(message: Message) -> Message:
    content = [
        ThinkingContent(block.text) if isinstance(block, ThinkingContent) else block
        for block in message.content
        if not isinstance(block, RedactedThinkingContent)
    ]
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=message.metadata,
    )


async def subscription_limit_failure(
    account_context: Any,
    provider_error: ErrorInfo,
    source: object,
) -> tuple[ErrorInfo, StreamEvent | None]:
    if account_context is None:
        return provider_error, None
    limit_data = await account_context.subscription_limit_event(source)
    if limit_data is None:
        return provider_error, None
    error = replace(
        provider_error,
        code="subscription_limit",
        message=(
            f"{limit_data['account']} reached its subscription limit; "
            f"resets {limit_data['reset_local']}"
        ),
    )
    return error, StreamEvent(StreamEventType.ACCOUNT_LIMIT, data=limit_data)




def account_generation(loop: Any) -> Callable[[], object]:
    """Read the session's backend and account identity; a change ends a rate-limit hold."""

    context = getattr(loop, "account_context", None)
    return lambda: (
        loop.backend,
        None if context is None else (context.active_provider, context.switch_sequence),
    )


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


def _usage_tokens(usage: object) -> tuple[int, int] | None:
    """Normalized (input, output) tokens; input includes cache reads and writes, as GenAI counts them."""

    if not isinstance(usage, Mapping):
        return None
    counts = [
        value if type(value := usage.get(key)) is int and value >= 0 else 0
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
    ]
    return sum(counts[:3]), counts[3]


class InferenceObservation:
    """The one owner of a logical model call's span.

    ``start`` opens the span; leaving the ``with`` block ends it exactly once on
    every exit (return, continue, any ``BaseException``). An exception in flight
    is the span's error; otherwise ``error`` (an exception or an ERROR event's
    ``ErrorInfo``) is. Usage sums every attempt, because retries are one call.
    """

    def __init__(self) -> None:
        self._span: SpanHandle | None = None
        self.error: BaseException | ErrorInfo | None = None
        self._finished_usage: tuple[int, int] | None = None
        self._attempt_usage: tuple[int, int] | None = None
        self._model: str | None = None
        self._finish_reason: str | None = None

    def __enter__(self) -> Self:
        return self

    def start(self, span: SpanHandle) -> None:
        span.__enter__()
        self._span = span

    def observe(self, event: StreamEvent) -> None:
        if self._span is None:
            return
        if event.type is StreamEventType.MESSAGE_START:
            model = event.data.get("model")
            if type(model) is str:
                self._model = model
        elif event.type is StreamEventType.MESSAGE_END:
            self._attempt_usage = _usage_tokens(event.data.get("usage")) or self._attempt_usage
            reason = event.data.get("stop_reason")
            if type(reason) is str:
                self._finish_reason = reason
        elif event.type is StreamEventType.RETRY and "reason" in event.data:
            delay = event.data.get("delay")
            self._span.retry(
                attempt=event.data.get("attempt", 0),
                reason=event.data["reason"],
                delay=round(delay * 1000) if isinstance(delay, (int, float)) else 0,
                after="retry_after" in event.data,
            )
            # the retry notice repeats the failed attempt's usage, so it replaces, never adds to, that attempt's MESSAGE_END
            failed = _usage_tokens(event.data.get("usage")) or self._attempt_usage
            if failed is not None:
                done = self._finished_usage or (0, 0)
                self._finished_usage = (done[0] + failed[0], done[1] + failed[1])
            self._attempt_usage = None
            self._finish_reason = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        span, self._span = self._span, None
        if span is None:
            return False
        try:
            usage = [part for part in (self._finished_usage, self._attempt_usage) if part is not None]
            if usage:
                input_tokens = sum(part[0] for part in usage)
                output_tokens = sum(part[1] for part in usage)
                span.set_usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens)
            span.set_response(finish_reason=self._finish_reason, model=self._model)
            if exc is None and isinstance(self.error, ErrorInfo):
                span.fail(None)
        finally:
            if exc is None and isinstance(self.error, BaseException):
                span.__exit__(type(self.error), self.error, None)
            else:
                span.__exit__(exc_type, exc, traceback)
        return False
