"""Deterministic completion backend for core-loop tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from .types import (
    CompletionBackend,
    ContentBlock,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    content: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    delay: float = 0.0


class FakeBackend(CompletionBackend):
    def __init__(self, turns: Sequence[ScriptedTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[tuple[list[Message], list[ToolSchema]]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append((list(messages), list(tool_schemas)))
        turn = self.turns[index]
        blocks = [*turn.content, *(ToolUseContent(call) for call in turn.tool_calls)]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for index, block in enumerate(blocks):
            if index and turn.delay:
                await asyncio.sleep(turn.delay)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        if turn.delay:
            await asyncio.sleep(turn.delay)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(role=MessageRole.ASSISTANT, content=blocks),
        )
