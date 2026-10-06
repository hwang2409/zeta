"""Completion-backend adapter for one memory reconciliation request."""

from __future__ import annotations

from collections.abc import Sequence

from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolSchema,
)
from .reconciler import ReconciliationResponse


async def complete_reconciliation(
    backend: CompletionBackend, prompt: str
) -> ReconciliationResponse:
    """Return the final text from a tool-free reconciliation completion."""
    request = [Message(MessageRole.USER, [TextContent(prompt)])]
    final: Message | None = None
    usage: dict[str, int] = {}
    async for event in backend.complete(request, _NO_TOOLS):
        if event.type is StreamEventType.MESSAGE_END and event.message is not None:
            final = event.message
            raw_usage = event.data.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    key: value
                    for key, value in raw_usage.items()
                    if isinstance(key, str) and type(value) is int and value >= 0
                }
    if final is None:
        raise RuntimeError("memory reconciler returned no final message")
    return ReconciliationResponse(
        "".join(
            block.text for block in final.content if isinstance(block, TextContent)
        ),
        usage,
    )


_NO_TOOLS: Sequence[ToolSchema] = ()
