"""Small deterministic backend used by ``zeta serve --provider fake``."""

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


class ServerFakeBackend(CompletionBackend):
    def __init__(self, *, delay: float = 0.01, model: str = "offline") -> None:
        self.delay = delay
        self.model = model

    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[dict[str, Any]]
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        prompt = ""
        for message in reversed(messages):
            if message.role is MessageRole.USER:
                prompt = "".join(
                    block.text for block in message.content if isinstance(block, TextContent)
                )
                break
        text = f"you said: {prompt}"
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for index in range(0, len(text), 8):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=text[index : index + 8])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(text)]),
            data={"usage": {"input_tokens": len(prompt), "output_tokens": len(text)}},
        )


__all__ = ["ServerFakeBackend"]
