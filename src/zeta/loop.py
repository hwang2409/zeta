"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Mapping, Sequence

from .store import ConversationStore
from .tools import ToolHandler, ToolRegistry
from .types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)


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
        tools: Mapping[str, ToolHandler] | ToolRegistry | None = None,
        registry: ToolRegistry | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int = 10,
    ) -> None:
        self.backend = backend
        self.store = store
        if registry is not None and tools is not None:
            raise ValueError("pass only one tool registry")
        if registry is not None:
            self.tool_registry = registry
        elif isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, Mapping):
            self.tool_registry = ToolRegistry(store.cwd, register_builtin=False)
            schemas_by_name = {
                schema.get("name"): schema
                for schema in (tool_schemas or [])
                if isinstance(schema.get("name"), str)
            }
            for name, handler in tools.items():
                schema = schemas_by_name.get(name, {})
                parameters = schema.get("parameters", schema.get("input_schema"))
                if parameters is None:
                    parameters = {
                        key: value
                        for key, value in schema.items()
                        if key not in {"name", "description", "cache_control"}
                    }
                self.tool_registry.register(
                    name,
                    handler,
                    description=(
                        schema.get("description", "")
                        if isinstance(schema.get("description", ""), str)
                        else ""
                    ),
                    parameters=parameters,
                )
        elif tools is None:
            self.tool_registry = ToolRegistry(store.cwd)
        else:
            raise TypeError("tools must be a mapping or ToolRegistry")
        self.tool_schemas = list(
            tool_schemas
            if tool_schemas is not None else self.tool_registry.schemas
        )
        self.max_turns = max_turns

    def abort(self) -> None:
        """Signal the active tool batch before the caller cancels the turn."""

        self.tool_registry.abort()

    def run_turn(self, user_text: str) -> AsyncIterator[StreamEvent]:
        return self._run_turn(user_text)

    async def _run_turn(self, user_text: str) -> AsyncIterator[StreamEvent]:
        self.store.append_message(
            Message(MessageRole.USER, [TextContent(user_text)])
        )
        yield StreamEvent(StreamEventType.AGENT_START)

        for turn_number in range(1, self.max_turns + 1):
            self.tool_registry.start_batch()
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

            completed_tool_ids: set[str] = set()
            try:
                call_index = 0
                while call_index < len(calls):
                    parallel_calls: list[ToolCall] = []
                    if self.tool_registry is not None:
                        definition = self.tool_registry.definitions_by_name.get(
                            calls[call_index].name
                        )
                        if definition is not None and definition.parallel_safe:
                            parallel_calls.append(calls[call_index])
                            while call_index + len(parallel_calls) < len(calls):
                                next_call = calls[call_index + len(parallel_calls)]
                                next_definition = self.tool_registry.definitions_by_name.get(
                                    next_call.name
                                )
                                if next_definition is None or not next_definition.parallel_safe:
                                    break
                                parallel_calls.append(next_call)
                    if len(parallel_calls) > 1:
                        for tool_call in parallel_calls:
                            yield StreamEvent(
                                StreamEventType.TOOL_EXECUTION_START,
                                tool_call=tool_call,
                            )
                        results = await self.tool_registry.execute_many(parallel_calls)
                        for tool_call, result in zip(parallel_calls, results, strict=True):
                            result = _validated_tool_result(result, tool_call.id)
                            self._append_tool_result(result)
                            completed_tool_ids.add(tool_call.id)
                            yield StreamEvent(
                                StreamEventType.TOOL_EXECUTION_END,
                                tool_call=tool_call,
                                tool_result=result,
                            )
                        call_index += len(parallel_calls)
                        continue

                    tool_call = calls[call_index]
                    yield StreamEvent(
                        StreamEventType.TOOL_EXECUTION_START,
                        tool_call=tool_call,
                    )
                    try:
                        result = await self.tool_registry.execute(tool_call)
                    except Exception as exc:
                        result = ToolResult(tool_call.id, str(exc), is_error=True)
                    result = _validated_tool_result(result, tool_call.id)
                    self._append_tool_result(result)
                    completed_tool_ids.add(tool_call.id)
                    yield StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                    call_index += 1
            except (asyncio.CancelledError, GeneratorExit):
                self._append_cancelled_tool_results(calls, completed_tool_ids)
                raise
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

    def _append_tool_result(self, result: ToolResult) -> None:
        self.store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [TextContent(result.content)],
                tool_result=result,
            )
        )

    def _append_cancelled_tool_results(
        self,
        calls: Sequence[ToolCall],
        completed_tool_ids: set[str],
    ) -> None:
        for call in calls:
            if call.id in completed_tool_ids:
                continue
            self._append_tool_result(
                ToolResult(call.id, "tool execution canceled", is_error=True)
            )

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        if assistant_message is not None:
            self.store.append_message(assistant_message)
        else:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent) or block.signature
            ]
            if durable_blocks:
                self.store.append_message(Message(MessageRole.ASSISTANT, durable_blocks))

    def _persist_partial_for_control(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        try:
            self._persist_partial(partial_blocks, assistant_message)
        except Exception as exc:
            try:
                warnings.warn(
                    f"failed to persist partial state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except:
                pass
