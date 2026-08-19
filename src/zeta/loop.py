"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from .store import ConversationStore
from .types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)


class ToolExecutor(Protocol):
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute one call and return its durable result."""


ToolHandler = Callable[
    [dict[str, Any]], str | ToolResult | Awaitable[str | ToolResult]
]


class DictToolExecutor:
    def __init__(self, handlers: Mapping[str, ToolHandler]) -> None:
        self.handlers = dict(handlers)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        handler = self.handlers[tool_call.name]
        result = handler(tool_call.arguments)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        return ToolResult(tool_call.id, str(result))


async def _close_completion(
    completion: AsyncIterator[StreamEvent] | None,
) -> BaseException | None:
    if completion is None:
        return None
    close = getattr(completion, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except BaseException as exc:
        return exc
    return None


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _validated_tool_result(result: object, expected_id: str) -> ToolResult:
    if not isinstance(result, ToolResult):
        return ToolResult(expected_id, "invalid tool result: expected ToolResult", True)
    if type(result.tool_call_id) is not str or not result.tool_call_id:
        return ToolResult(expected_id, "invalid tool result: call id", True)
    if type(result.content) is not str:
        return ToolResult(expected_id, "invalid tool result: content", True)
    if type(result.is_error) is not bool:
        return ToolResult(expected_id, "invalid tool result: is_error", True)
    if result.tool_call_id != expected_id:
        return ToolResult(
            expected_id,
            f"tool result id mismatch: expected {expected_id}, got {result.tool_call_id}",
            is_error=True,
        )
    return result


class AgentLoop:
    def __init__(
        self,
        backend: CompletionBackend,
        store: ConversationStore,
        *,
        tools: Mapping[str, ToolHandler] | ToolExecutor | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int = 10,
    ) -> None:
        self.backend = backend
        self.store = store
        self.tool_executor = (
            DictToolExecutor(tools) if isinstance(tools, Mapping) else tools
        )
        self.tool_schemas = list(tool_schemas or [])
        if not self.tool_schemas and isinstance(tools, Mapping):
            self.tool_schemas = [{"name": name} for name in tools]
        self.max_turns = max_turns

    def run_turn(self, user_text: str) -> AsyncIterator[StreamEvent]:
        return self._run_turn(user_text)

    async def _run_turn(self, user_text: str) -> AsyncIterator[StreamEvent]:
        self.store.append_message(
            Message(MessageRole.USER, [TextContent(user_text)])
        )
        yield StreamEvent(StreamEventType.AGENT_START)

        for turn_number in range(1, self.max_turns + 1):
            yield StreamEvent(
                StreamEventType.TURN_START,
                data={"turn": turn_number},
            )
            partial_blocks: list[ContentBlock] = []
            assistant_message: Message | None = None
            completion: AsyncIterator[StreamEvent] | None = None
            try:
                completion = self.backend.complete(
                    self.store.messages(), self.tool_schemas
                )
                async for event in completion:
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial_blocks.append(event.content)
                        if event.delta is not None:
                            partial_blocks.append(TextContent(event.delta))
                    if event.message is not None and event.type is StreamEventType.MESSAGE_END:
                        assistant_message = event.message
                    yield event
            except asyncio.CancelledError:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except GeneratorExit:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except Exception as exc:
                await _close_completion(completion)
                if _task_is_cancelling():
                    self._persist_partial_for_control(partial_blocks, assistant_message)
                    raise asyncio.CancelledError() from exc
                self._persist_partial(partial_blocks, assistant_message)
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("backend_error", str(exc)),
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            cleanup_error = await _close_completion(completion)
            if _task_is_cancelling():
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise asyncio.CancelledError()
            if cleanup_error is not None:
                self._persist_partial(partial_blocks, assistant_message)
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("backend_error", str(cleanup_error)),
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return

            if assistant_message is None and partial_blocks:
                assistant_message = Message(MessageRole.ASSISTANT, partial_blocks)
            if assistant_message is None:
                assistant_message = Message(MessageRole.ASSISTANT)
            self.store.append_message(assistant_message)
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            if not calls:
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return

            for tool_call in calls:
                yield StreamEvent(
                    StreamEventType.TOOL_EXECUTION_START,
                    tool_call=tool_call,
                )
                try:
                    if self.tool_executor is None:
                        raise KeyError(f"no executor for tool {tool_call.name}")
                    result = await self.tool_executor.execute(tool_call)
                except Exception as exc:
                    result = ToolResult(tool_call.id, str(exc), is_error=True)
                result = _validated_tool_result(result, tool_call.id)
                self.store.append_message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        [TextContent(result.content)],
                        tool_result=result,
                    )
                )
                yield StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=tool_call,
                    tool_result=result,
                )
            yield StreamEvent(
                StreamEventType.TURN_END,
                message=assistant_message,
                data={"turn": turn_number, "tool_calls": len(calls)},
            )

        yield StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("max_turns", f"maximum turns reached: {self.max_turns}"),
        )
        yield StreamEvent(StreamEventType.AGENT_END)

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        if assistant_message is not None:
            self.store.append_message(assistant_message)
        elif partial_blocks:
            self.store.append_message(Message(MessageRole.ASSISTANT, partial_blocks))

    def _persist_partial_for_control(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        try:
            self._persist_partial(partial_blocks, assistant_message)
        except Exception as exc:
            warnings.warn(
                f"failed to persist partial state: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
