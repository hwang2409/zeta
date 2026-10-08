from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from zeta.memory.provider import complete_reconciliation, use_response_byte_limit
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
from zeta.providers.retry_policy import ProviderRetryBudget, use_retry_budget


class _BudgetExhaustingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools
        self.calls += 1
        if self.calls < 5:
            yield StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("timeout", "safe timeout"),
                data={"retry_after": 0},
            )
            return
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("not json")]),
            data={"usage": {"input_tokens": 1}},
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


class _OversizedStreamingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.requested_third = False

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="12345")
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="67890")
        self.requested_third = True
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="unsafe tail")


@pytest.mark.asyncio
async def test_bounded_reconciliation_stops_reading_oversized_stream() -> None:
    backend = _OversizedStreamingBackend()

    with use_response_byte_limit(8):
        response = await complete_reconciliation(backend, "safe prompt")

    assert response.text == "12345678"
    assert response.truncated is True
    assert backend.requested_third is False


@pytest.mark.asyncio
async def test_original_and_repair_share_one_provider_attempt_budget() -> None:
    backend = _BudgetExhaustingBackend()
    budget = ProviderRetryBudget()

    with use_retry_budget(budget):
        response = await complete_reconciliation(backend, "original")
        assert response.text == "not json"
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await complete_reconciliation(backend, "repair")

    assert backend.calls == 5
    assert budget.attempts == 5


@pytest.mark.asyncio
async def test_reconciliation_retries_after_partial_stream_before_commit() -> None:
    backend = _PostStreamRetryBackend()

    response = await complete_reconciliation(backend, "safe prompt")

    assert backend.calls == 2
    assert response.text == '{"changes":[]}'
    assert response.usage == {"input_tokens": 3, "output_tokens": 2}
