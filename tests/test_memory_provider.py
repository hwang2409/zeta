from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from zeta.memory.provider import complete_reconciliation
from zeta.protocol.types import (
    CompletionBackend,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolSchema,
)


class _PostStreamRetryBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("timeout", "unsafe provider detail"),
                data={"retry_after": 0},
            )
            return
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent('{"changes":[]}')]),
            data={"usage": {"input_tokens": 3, "output_tokens": 2}},
        )


@pytest.mark.asyncio
async def test_reconciliation_retries_after_partial_stream_before_commit() -> None:
    backend = _PostStreamRetryBackend()

    response = await complete_reconciliation(backend, "safe prompt")

    assert backend.calls == 2
    assert response.text == '{"changes":[]}'
    assert response.usage == {"input_tokens": 3, "output_tokens": 2}
