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


async def complete_reconciliation(backend: CompletionBackend, prompt: str) -> str:
    """Return the final text from a tool-free reconciliation completion."""
    request = [Message(MessageRole.USER, [TextContent(prompt)])]
    final: Message | None = None
    async for event in backend.complete(request, _NO_TOOLS):
        if event.type is StreamEventType.MESSAGE_END and event.message is not None:
            final = event.message
    if final is None:
        raise RuntimeError("memory reconciler returned no final message")
    return "".join(
        block.text for block in final.content if isinstance(block, TextContent)
    )


_NO_TOOLS: Sequence[ToolSchema] = ()
