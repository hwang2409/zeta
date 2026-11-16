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
from typing import TYPE_CHECKING, Any

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


def observe_inference(span: SpanHandle, event: StreamEvent) -> None:
    """Copy retry decisions, normalized usage, and the stop reason onto the call's span."""

    if event.type is StreamEventType.RETRY and "reason" in event.data:
        delay = event.data.get("delay")
        span.retry(
            attempt=event.data.get("attempt", 0),
            reason=event.data["reason"],
            delay=round(delay * 1000) if isinstance(delay, (int, float)) else 0,
            after="retry_after" in event.data,
        )
    elif event.type is StreamEventType.MESSAGE_END:
        usage = event.data.get("usage")
        if isinstance(usage, Mapping):
            # normalized input_tokens exclude cache reads and writes; the GenAI count includes them
            input_tokens = sum(
                value
                for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                if type(value := usage.get(key)) is int and value >= 0
            )
            output_tokens = usage.get("output_tokens")
            output_tokens = output_tokens if type(output_tokens) is int and output_tokens >= 0 else 0
            span.set_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        reason = event.data.get("stop_reason")
        if type(reason) is str:
            span.set_response(finish_reason=reason)


def end_inference(span: SpanHandle | None, error: BaseException | object | None) -> None:
    """End the logical call's span; a non-exception error (an ERROR event) is ``_OTHER``."""

    if span is None:
        return
    if isinstance(error, BaseException):
        span.__exit__(type(error), error, None)
        return
    if error is not None:
        span.fail(None)
    span.__exit__(None, None, None)
