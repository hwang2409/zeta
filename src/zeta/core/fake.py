"""Deterministic completion backend for core-loop tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from ..types import (
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
    usage: dict[str, int] = field(default_factory=dict)


class FakeBackend(CompletionBackend):
    def __init__(
        self,
        turns: Sequence[ScriptedTurn],
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.turns = list(turns)
        self.calls: list[tuple[list[Message], list[ToolSchema]]] = []
        self.completion_close_count = 0
        self.close_error = close_error
        self._previous_request: tuple[bytes, ...] | None = None

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append((list(messages), list(tool_schemas)))
        turn = self.turns[index]
        request = tuple(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            for value in [
                *(message.to_dict() for message in messages),
                {"tools": list(tool_schemas)},
            ]
        )
        usage = dict(turn.usage)
        if usage:
            cache_read = _common_prefix_length(self._previous_request, request)
            self._previous_request = request
            usage["cache_read_input_tokens"] = cache_read
            usage["cache_creation_input_tokens"] = sum(map(len, request)) - cache_read
        blocks = [*turn.content, *(ToolUseContent(call) for call in turn.tool_calls)]
        try:
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
                data={"usage": usage} if usage else {},
            )
        finally:
            self.completion_close_count += 1
            if self.close_error is not None:
                raise self.close_error


def _common_prefix_length(
    previous: tuple[bytes, ...] | None, current: tuple[bytes, ...]
) -> int:
    if previous is None:
        return 0
    length = 0
    for previous_part, current_part in zip(previous, current):
        if previous_part != current_part:
            return length
        length += len(previous_part)
    return length
