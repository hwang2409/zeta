"""Offline streaming backend used by the interactive CLI smoke mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)


class FakeInteractiveBackend(CompletionBackend):
    """Small streaming backend for offline CLI smoke tests."""

    def __init__(self, *, delay: float = 0.03, model: str = "offline") -> None:
        self.delay = delay
        self.model = model
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        self.calls.append(list(messages))
        prompt = ""
        for message in reversed(messages):
            if message.role is MessageRole.USER:
                prompt = "".join(
                    block.text
                    for block in message.content
                    if isinstance(block, TextContent)
                )
                break
        response = (
            f"you said: {prompt}\n\n"
            "the fake provider is streaming this response offline.\n"
            "try queueing another message while this turn runs."
        )
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for chunk in _chunks(response, 9):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=chunk)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
            data={"usage": {"input_tokens": len(prompt), "output_tokens": len(response)}},
        )


def _chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)]
