"""Deterministic completion backend for core-loop tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field

import pytest

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
from zeta.providers.completion import owned_completion


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    content: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    delay: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


ACTOR_WAIT_TIMEOUT = 30
_CREATED_FAKE_BACKENDS: list[FakeBackend] = []


@pytest.fixture(autouse=True)
def reject_shared_fake_backend() -> Iterator[None]:
    _CREATED_FAKE_BACKENDS.clear()
    yield
    violations = [
        backend._actor_violation
        for backend in _CREATED_FAKE_BACKENDS
        if backend._actor_violation is not None
    ]
    _CREATED_FAKE_BACKENDS.clear()
    if violations:
        first, second = violations[0]
        pytest.fail(
            f"one FakeBackend served actors {first!r} and {second!r}; "
            "use PerActorFakeBackend from tests/support/fake_backend.py"
        )


def consume_fake_backend_actor_violations() -> list[tuple[str, str]]:
    violations = [
        backend._actor_violation
        for backend in _CREATED_FAKE_BACKENDS
        if backend._actor_violation is not None
    ]
    for backend in _CREATED_FAKE_BACKENDS:
        backend._actor_violation = None
    return violations


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
        self._actor_violation: tuple[str, str] | None = None
        self._scripted_child_prompts = {
            prompt
            for turn in turns
            for call in turn.tool_calls
            if call.name == "agent" and call.arguments.get("background") is True
            if isinstance(prompt := call.arguments.get("prompt"), str)
        }
        _CREATED_FAKE_BACKENDS.append(self)

    @owned_completion
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        if self._scripted_child_prompts:
            actor_prompt = _script_actor_prompt(messages, self._scripted_child_prompts)
            if self._actor_prompt is None:
                self._actor_prompt = actor_prompt
            elif actor_prompt != self._actor_prompt and self._actor_violation is None:
                self._actor_violation = (self._actor_prompt, actor_prompt)
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


def _public_input_prompt(messages: Sequence[Message], prompts: Collection[str]) -> str:
    public_inputs = [
        block.text
        for message in messages
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    prompt = next((text for text in public_inputs if text in prompts), None)
    if prompt is None:
        raise AssertionError(f"no scripted backend for actor prompt {public_inputs!r}")
    return prompt


class PerActorFakeBackend:
    def __init__(
        self,
        parent_turns: list[ScriptedTurn],
        child_turns: dict[str, list[ScriptedTurn]],
        *,
        parent_prompt: str | Collection[str],
        gated_child_call: int = 1,
        gated_child_prompts: Collection[str] | None = None,
        parent_waits_for_children: bool = True,
    ) -> None:
        self.parent_prompts = (
            {parent_prompt} if isinstance(parent_prompt, str) else set(parent_prompt)
        )
        self.parent = FakeBackend(parent_turns)
        self.children = {
            prompt: FakeBackend(turns) for prompt, turns in child_turns.items()
        }
        self.child_started = {prompt: asyncio.Event() for prompt in self.children}
        self.child_calls = dict.fromkeys(self.children, 0)
        self.release_children = asyncio.Event()
        self.gated_child_call = gated_child_call
        self.parent_waits_for_children = parent_waits_for_children
        self.gated_child_prompts = (
            set(self.children)
            if gated_child_prompts is None
            else set(gated_child_prompts)
        )

    @owned_completion
    async def complete(self, messages, tool_schemas):
        actor_prompt = _public_input_prompt(
            messages, {*self.parent_prompts, *self.children}
        )
        child_prompt = actor_prompt if actor_prompt in self.children else None
        backend = (
            self.children[child_prompt] if child_prompt is not None else self.parent
        )
        if child_prompt is not None:
            self.child_calls[child_prompt] += 1
        if (
            child_prompt in self.gated_child_prompts
            and self.child_calls[child_prompt] == self.gated_child_call
        ):
            self.child_started[child_prompt].set()
            try:
                await asyncio.wait_for(self.release_children.wait(), ACTOR_WAIT_TIMEOUT)
            except TimeoutError as error:
                raise AssertionError("children were not released") from error
        stream = FakeBackend.complete.__wrapped__(backend, messages, tool_schemas)
        if (
            child_prompt is None
            and self.parent.calls
            and self.parent_waits_for_children
        ):
            try:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *(started.wait() for started in self.child_started.values())
                        ),
                        ACTOR_WAIT_TIMEOUT,
                    )
                except TimeoutError as error:
                    raise AssertionError("child did not start") from error
                yield await anext(stream)
            finally:
                self.release_children.set()
        async for event in stream:
            yield event


def _script_actor_prompt(messages: Sequence[Message], child_prompts: set[str]) -> str:
    public_inputs = [
        block.text
        for message in messages
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    if not public_inputs:
        raise AssertionError("scripted multi-actor backend call has no public input")
    return next(
        (text for text in public_inputs if text in child_prompts), public_inputs[0]
    )


def _common_prefix_length(previous: bytes | None, current: bytes) -> int:
    if previous is None:
        return 0
    length = min(len(previous), len(current))
    for index in range(length):
        if previous[index] != current[index]:
            return index
    return length
