"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Mapping, Sequence

from .approval import ApprovalPolicy
from .context import ContextAssembler
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
        approval_policy: ApprovalPolicy | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int = 10,
        context_assembler: ContextAssembler | None = None,
        system_prompt: str | Message = "",
        token_budget: int = 100_000,
        retained_tail: int = 8,
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
        if (
            approval_policy is not None
            and self.tool_registry.approval_policy is not None
            and self.tool_registry.approval_policy is not approval_policy
        ):
            raise ValueError("pass only one approval policy")
        if approval_policy is not None:
            self.tool_registry.set_approval_policy(approval_policy)
        if self.tool_registry.approval_policy is not None:
            self.tool_registry.bind_approval_store(store)
        self.tool_schemas = list(
            tool_schemas
            if tool_schemas is not None else self.tool_registry.schemas
        )
        self.max_turns = max_turns
        self.context_assembler = context_assembler or ContextAssembler(
            store,
            token_budget=token_budget,
            retained_tail=retained_tail,
            system_prompt=system_prompt,
            backend=backend,
        )

    def abort(self) -> None:
        """Signal the active tool batch before the caller cancels the turn."""

        self.tool_registry.abort()

    def run_turn(self, user_text: str) -> AsyncIterator[StreamEvent]:
        return self._run_turn(user_text)

    async def resume_pending_tool(self, request_id: str) -> ToolResult | None:
        """Finish a durable approval request before starting another turn."""

        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return None
        tool_call = state[0]
        self.tool_registry.start_batch()
        try:
            result = await self.tool_registry.execute(tool_call)
        except Exception as exc:
            result = ToolResult(tool_call.id, str(exc), is_error=True)
        return self._finalize_tool_results(
            [tool_call], [_validated_tool_result(result, tool_call.id)]
        )[0]

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
            context = await self.context_assembler.assemble(backend=self.backend)
            try:
                completion = self.backend.complete(
                    context, self.tool_schemas
                )
                async for event in completion:
                    self.context_assembler.observe_event(event)
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
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message
                )
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
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message
                )
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
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            approval_requests: list[tuple[str, ToolCall]] = []
            for tool_call in calls:
                request = self.tool_registry.prepare_approval(tool_call)
                if request is not None:
                    approval_requests.append((request.request_id, request.tool_call))
            self.store.append_message_with_approval_requests(
                assistant_message,
                approval_requests,
            )
            if not calls:
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return

            completed_tool_indexes: set[int] = set()
            parallel_tasks: dict[asyncio.Task[ToolResult], tuple[int, ToolCall]] = {}
            parallel_results: list[ToolResult | None] = [None] * len(calls)
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
                        parallel_tasks = {
                            asyncio.create_task(self.tool_registry.execute(tool_call)): (
                                call_index + offset,
                                tool_call,
                            )
                            for offset, tool_call in enumerate(parallel_calls)
                        }
                        while parallel_tasks:
                            done, _ = await asyncio.wait(
                                parallel_tasks,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for task in done:
                                index, tool_call = parallel_tasks.pop(task)
                                parallel_results[index] = _validated_tool_result(
                                    task.result(),
                                    tool_call.id,
                                )
                        batch_end = call_index + len(parallel_calls)
                        results = self._finalize_tool_results(
                            parallel_calls,
                            parallel_results[call_index:batch_end],
                        )
                        parallel_results[call_index:batch_end] = [None] * len(parallel_calls)
                        completed_tool_indexes.update(range(call_index, batch_end))
                        for tool_call, result in zip(parallel_calls, results, strict=True):
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
                    result = self._finalize_tool_results([tool_call], [result])[0]
                    completed_tool_indexes.add(call_index)
                    yield StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                    call_index += 1
            except (asyncio.CancelledError, GeneratorExit):
                pending_tasks: list[asyncio.Task[ToolResult]] = []
                for task, (index, tool_call) in list(parallel_tasks.items()):
                    parallel_tasks.pop(task)
                    if task.done():
                        if task.cancelled():
                            continue
                        try:
                            result = _validated_tool_result(task.result(), tool_call.id)
                        except BaseException:
                            continue
                        parallel_results[index] = result
                    else:
                        task.cancel()
                        pending_tasks.append(task)
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                pending_indexes = [
                    index for index in range(len(calls)) if index not in completed_tool_indexes
                ]
                self._finalize_tool_results(
                    [calls[index] for index in pending_indexes],
                    [parallel_results[index] for index in pending_indexes],
                )
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

    def _finalize_tool_results(
        self,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]:
        results = [
            result or ToolResult(call.id, "tool execution canceled", is_error=True)
            for call, result in zip(calls, slots, strict=True)
        ]
        for result in results:
            self.store.append_message(
                Message(
                    MessageRole.TOOL_RESULT,
                    [TextContent(result.content)],
                    tool_result=result,
                )
            )
        return results

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
            self._persist_partial_with_cancelled_tools(
                partial_blocks, assistant_message
            )
        except Exception as exc:
            try:
                warnings.warn(
                    f"failed to persist partial state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except:
                pass

    def _persist_partial_with_cancelled_tools(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message)
        blocks = (
            assistant_message.content
            if assistant_message is not None
            else partial_blocks
        )
        calls = [
            block.tool_call
            for block in blocks
            if isinstance(block, ToolUseContent)
        ]
        self._finalize_tool_results(calls, [None] * len(calls))
