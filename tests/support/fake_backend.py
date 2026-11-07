"""Deterministic completion backend for core-loop tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

from zeta.protocol.types import (
    CompletionBackend,
    ContentBlock,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
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
    stop_reason: str | None = None


class FakeBackend(CompletionBackend):
    def __init__(
        self,
        turns: Sequence[ScriptedTurn],
        *,
        close_error: Exception | None = None,
        request_serializer: Callable[[Sequence[Message], Sequence[ToolSchema]], bytes]
        | None = None,
    ) -> None:
        self.turns = list(turns)
        self.calls: list[tuple[list[Message], list[ToolSchema]]] = []
        self.completion_close_count = 0
        self.close_error = close_error
        self.request_serializer = request_serializer
        self.request_bytes: list[bytes] = []
        self._previous_request: bytes | None = None
        self._actor_prompt: str | None = None
        self._multi_actor_script = any(
            call.name == "agent" and call.arguments.get("background") is True
            for turn in turns
            for call in turn.tool_calls
        )

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        if self._multi_actor_script:
            actor_prompt = _first_public_input(messages)
            if self._actor_prompt is None:
                self._actor_prompt = actor_prompt
            elif actor_prompt != self._actor_prompt:
                raise AssertionError(
                    "FakeBackend cannot serve multiple actors; use PerActorFakeBackend "
                    f"for {self._actor_prompt!r} and {actor_prompt!r}"
                )
        index = len(self.calls)
        self.calls.append((list(messages), list(tool_schemas)))
        turn = self.turns[index]
        if self.request_serializer is None:
            request = json.dumps(
                [
                    *(message.to_dict() for message in messages),
                    {"tools": list(tool_schemas)},
                ],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        else:
            request = self.request_serializer(messages, tool_schemas)
        self.request_bytes.append(request)
        usage = dict(turn.usage)
        cache_read = _common_prefix_length(self._previous_request, request)
        self._previous_request = request
        if usage:
            usage["cache_read_input_tokens"] = cache_read
            usage["cache_creation_input_tokens"] = len(request) - cache_read
        blocks = [*turn.content, *(ToolUseContent(call) for call in turn.tool_calls)]
        try:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            for index, block in enumerate(blocks):
                if index and turn.delay:
                    await asyncio.sleep(turn.delay)
                yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
            if turn.delay:
                await asyncio.sleep(turn.delay)
            data: dict[str, object] = {}
            if usage:
                data["usage"] = usage
            if turn.stop_reason is not None:
                data["stop_reason"] = turn.stop_reason
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(role=MessageRole.ASSISTANT, content=blocks),
                data=data,
            )
        finally:
            self.completion_close_count += 1
            if self.close_error is not None:
                raise self.close_error


def _first_public_input(messages: Sequence[Message]) -> str:
    for message in messages:
        if message.role is MessageRole.USER:
            for block in message.content:
                if isinstance(block, TextContent):
                    return block.text
    raise AssertionError("scripted multi-actor backend call has no public input")


def _common_prefix_length(previous: bytes | None, current: bytes) -> int:
    if previous is None:
        return 0
    length = min(len(previous), len(current))
    for index in range(length):
        if previous[index] != current[index]:
            return index
    return length
