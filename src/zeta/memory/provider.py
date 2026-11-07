"""Completion backend for project-memory work, not live session requests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolSchema,
)
from ..providers.completion import (
    CompletionControl,
    CompletionRetryBudget,
    ResetPolicy,
)
from .reconciler import ReconciliationResponse

_RESPONSE_BYTE_LIMIT: ContextVar[int | None] = ContextVar(
    "memory_response_byte_limit", default=None
)


@contextmanager
def use_response_byte_limit(limit: int) -> Iterator[None]:
    """Stop a reconciliation stream after this many response bytes."""
    token = _RESPONSE_BYTE_LIMIT.set(limit)
    try:
        yield
    finally:
        _RESPONSE_BYTE_LIMIT.reset(token)


def _bounded_response(parts: list[str], limit: int) -> str:
    return "".join(parts).encode()[:limit].decode("utf-8", errors="ignore")


async def complete_reconciliation(
    backend: CompletionBackend,
    prompt: str,
    budget: CompletionRetryBudget,
) -> ReconciliationResponse:
    """Return one accepted response under an explicit shared retry budget."""
    request = [Message(MessageRole.USER, [TextContent(prompt)])]
    final: Message | None = None
    usage: dict[str, int] = {}
    response_parts: list[str] = []
    response_bytes = 0
    response_limit = _RESPONSE_BYTE_LIMIT.get()
    completion = backend.complete(
        request,
        _NO_TOOLS,
        control=CompletionControl(
            abort=None,
            reset=ResetPolicy.ALLOW_ASSISTANT_RESET,
            budget=budget,
        ),
    )
    try:
        async for event in completion:
            if event.type is StreamEventType.ASSISTANT_RESET:
                response_parts.clear()
                response_bytes = 0
                final = None
                usage = {}
                continue
            if event.type is StreamEventType.ERROR and event.error is not None:
                raise RuntimeError(event.error.message)
            if event.type is StreamEventType.MESSAGE_UPDATE and event.delta:
                response_parts.append(event.delta)
                response_bytes += len(event.delta.encode())
                if response_limit is not None and response_bytes > response_limit:
                    return ReconciliationResponse(
                        _bounded_response(response_parts, response_limit),
                        usage,
                        truncated=True,
                    )
            if event.type is StreamEventType.MESSAGE_END and event.message is not None:
                final = event.message
                raw_usage = event.data.get("usage")
                if isinstance(raw_usage, dict):
                    usage = {
                        key: value
                        for key, value in raw_usage.items()
                        if isinstance(key, str)
                        and type(value) is int
                        and value >= 0
                    }
    finally:
        await completion.aclose()
    if final is None:
        raise RuntimeError("memory reconciler returned no final message")
    text = "".join(
        block.text for block in final.content if isinstance(block, TextContent)
    )
    if response_limit is not None and len(text.encode()) > response_limit:
        return ReconciliationResponse(
            _bounded_response([text], response_limit), usage, truncated=True
        )
    return ReconciliationResponse(text, usage)


_NO_TOOLS: Sequence[ToolSchema] = ()
