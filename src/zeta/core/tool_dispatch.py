"""Dispatch tool calls while preserving streamed lifecycle events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from ..types import (
    StreamEvent,
    StreamEventType,
    StructuredToolResult,
    ToolCall,
    ToolResult,
)

TaskResult = TypeVar("TaskResult")


class _ToolDefinition(Protocol):
    parallel_safe: bool


class _ToolRegistry(Protocol):
    definitions_by_name: Mapping[str, _ToolDefinition]

    def execute(
        self,
        tool_call: ToolCall,
        *,
        _stream_sink: Callable[[StreamEvent], None],
        _lifecycle_sink: Callable[..., None],
    ) -> Coroutine[Any, Any, StructuredToolResult]: ...


class _AgentLoop(Protocol):
    tool_registry: _ToolRegistry | None

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, TaskResult],
    ) -> asyncio.Task[TaskResult]: ...

    def _finalize_tool_results(
        self,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]: ...


async def dispatch_tool_calls(
    loop: _AgentLoop,
    calls: Sequence[ToolCall],
    validate_result: Callable[[object, str], ToolResult],
) -> AsyncIterator[StreamEvent]:
    """Execute one tool batch and yield its lifecycle and result events."""

    completed_tool_indexes: set[int] = set()
    # Cap advisory output at 128 events; final results stay complete.
    stream_updates: asyncio.Queue[StreamEvent] = asyncio.Queue(
        maxsize=128 + len(calls) * 3
    )

    def enqueue_tool_update(event: StreamEvent) -> None:
        retained: list[StreamEvent] = []
        while not stream_updates.empty():
            retained.append(stream_updates.get_nowait())
        update_count = sum(
            item.type is StreamEventType.TOOL_EXECUTION_UPDATE for item in retained
        )
        if update_count >= 128:
            first_update = next(
                index
                for index, item in enumerate(retained)
                if item.type is StreamEventType.TOOL_EXECUTION_UPDATE
            )
            del retained[first_update]
        for queued in retained:
            stream_updates.put_nowait(queued)
        stream_updates.put_nowait(event)

    def enqueue_tool_lifecycle(
        kind: str,
        tool_call: ToolCall,
        data: Mapping[str, object] | None = None,
        tool_result: ToolResult | None = None,
    ) -> None:
        event_type = {
            "approval_start": StreamEventType.TOOL_APPROVAL_START,
            "approval_end": StreamEventType.TOOL_APPROVAL_END,
            "execution_start": StreamEventType.TOOL_EXECUTION_START,
            "execution_end": StreamEventType.TOOL_EXECUTION_END,
        }.get(kind)
        if event_type is not None:
            stream_updates.put_nowait(
                StreamEvent(
                    event_type,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    data={} if data is None else dict(data),
                )
            )

    active_task: asyncio.Task[StructuredToolResult] | None = None
    parallel_tasks: dict[asyncio.Task[StructuredToolResult], tuple[int, ToolCall]] = {}
    parallel_results: list[ToolResult | None] = [None] * len(calls)
    try:
        call_index = 0
        while call_index < len(calls):
            parallel_calls: list[ToolCall] = []
            if loop.tool_registry is not None:
                definition = loop.tool_registry.definitions_by_name.get(
                    calls[call_index].name
                )
                if definition is not None and definition.parallel_safe:
                    parallel_calls.append(calls[call_index])
                    while call_index + len(parallel_calls) < len(calls):
                        next_call = calls[call_index + len(parallel_calls)]
                        next_definition = loop.tool_registry.definitions_by_name.get(
                            next_call.name
                        )
                        if next_definition is None or not next_definition.parallel_safe:
                            break
                        parallel_calls.append(next_call)
            if len(parallel_calls) > 1:
                parallel_tasks = {
                    loop._create_task(
                        loop.tool_registry.execute(
                            tool_call,
                            _stream_sink=enqueue_tool_update,
                            _lifecycle_sink=(
                                lambda kind, call=tool_call, data=None, tool_result=None: enqueue_tool_lifecycle(
                                    kind, call, data, tool_result
                                )
                            ),
                        )
                    ): (call_index + offset, tool_call)
                    for offset, tool_call in enumerate(parallel_calls)
                }
                while parallel_tasks:
                    update_task = loop._create_task(stream_updates.get())
                    done, _ = await asyncio.wait(
                        (*parallel_tasks, update_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if update_task in done:
                        yield update_task.result()
                    else:
                        update_task.cancel()
                        await asyncio.gather(
                            update_task,
                            return_exceptions=True,
                        )
                    for task in done:
                        if task is update_task:
                            continue
                        index, tool_call = parallel_tasks.pop(task)
                        try:
                            task_result = task.result()
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            task_result = ToolResult(
                                tool_call.id,
                                str(exc),
                                is_error=True,
                            )
                        parallel_results[index] = validate_result(
                            task_result,
                            tool_call.id,
                        )
                while not stream_updates.empty():
                    yield stream_updates.get_nowait()
                batch_end = call_index + len(parallel_calls)
                results = loop._finalize_tool_results(
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
            active_task = loop._create_task(
                loop.tool_registry.execute(
                    tool_call,
                    _stream_sink=enqueue_tool_update,
                    _lifecycle_sink=(
                        lambda kind, call=tool_call, data=None, tool_result=None: enqueue_tool_lifecycle(
                            kind, call, data, tool_result
                        )
                    ),
                )
            )
            try:
                while True:
                    update_task = loop._create_task(stream_updates.get())
                    done, _ = await asyncio.wait(
                        (active_task, update_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if update_task in done:
                        yield update_task.result()
                    else:
                        update_task.cancel()
                        await asyncio.gather(
                            update_task,
                            return_exceptions=True,
                        )
                    if active_task in done:
                        while not stream_updates.empty():
                            yield stream_updates.get_nowait()
                        break
                result = active_task.result()
                active_task = None
            except Exception as exc:
                result = ToolResult(tool_call.id, str(exc), is_error=True)
            result = validate_result(result, tool_call.id)
            result = loop._finalize_tool_results([tool_call], [result])[0]
            completed_tool_indexes.add(call_index)
            yield StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=tool_call,
                tool_result=result,
            )
            call_index += 1
    except (asyncio.CancelledError, GeneratorExit):
        pending_tasks: list[asyncio.Task[StructuredToolResult]] = []
        await asyncio.sleep(0)
        if active_task is not None and not active_task.done():
            active_task.cancel()
            pending_tasks.append(active_task)
        for task, (index, tool_call) in list(parallel_tasks.items()):
            parallel_tasks.pop(task)
            if task.done():
                if task.cancelled():
                    continue
                try:
                    result = validate_result(task.result(), tool_call.id)
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
        loop._finalize_tool_results(
            [calls[index] for index in pending_indexes],
            [parallel_results[index] for index in pending_indexes],
        )
        raise
