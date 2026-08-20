import asyncio
import warnings
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEventType,
    StreamEvent,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
)


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


@pytest.mark.asyncio
async def test_single_turn_without_tools(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    store = ConversationStore(tmp_path)

    events = await collect(AgentLoop(backend, store).run_turn("hi"))

    assert events[-1].type is StreamEventType.AGENT_END
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_tool_call_then_next_completion(tmp_path: Path) -> None:
    first_call = ToolCall("call-1", "echo", {"value": "one"})
    second_call = ToolCall("call-2", "echo", {"value": "two"})
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("using tools")], [first_call, second_call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    async def echo(arguments: dict[str, str]) -> str:
        return arguments["value"]

    events = await collect(AgentLoop(backend, store, tools={"echo": echo}).run_turn("start"))

    assert [event.type for event in events] == [
        StreamEventType.AGENT_START,
        StreamEventType.TURN_START,
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_END,
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_END,
        StreamEventType.TURN_END,
        StreamEventType.TURN_START,
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
        StreamEventType.TURN_END,
        StreamEventType.AGENT_END,
    ]

    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
        MessageRole.TOOL_RESULT,
        MessageRole.ASSISTANT,
    ]
    assert [message.tool_result.content for message in backend.calls[1][0][-2:] if message.tool_result] == [
        "one",
        "two",
    ]


@pytest.mark.asyncio
async def test_parallel_cancellation_persists_resolved_results(tmp_path: Path) -> None:
    first_call = ToolCall("call-1", "first", {})
    second_call = ToolCall("call-2", "second", {})
    backend = FakeBackend([ScriptedTurn([], [first_call, second_call])])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    second_release = asyncio.Event()

    async def first(arguments: dict[str, object]) -> str:
        return "one"

    async def second(arguments: dict[str, object]) -> str:
        await second_release.wait()
        return "two"

    registry.register("first", first, parallel_safe=True)
    registry.register("second", second, parallel_safe=True)
    loop = AgentLoop(backend, store, registry=registry)

    async def consume() -> None:
        async for event in loop.run_turn("start"):
            if (
                event.type is StreamEventType.TOOL_EXECUTION_END
                and event.tool_call is not None
                and event.tool_call.id == first_call.id
            ):
                second_release.set()
                await asyncio.sleep(0)
                asyncio.current_task().cancel()

    with pytest.raises(asyncio.CancelledError):
        await consume()

    results = {
        message.tool_result.tool_call_id: message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    }
    assert results[first_call.id].content == "one"
    assert results[second_call.id].content == "two"
    assert not results[first_call.id].is_error
    assert not results[second_call.id].is_error


@pytest.mark.asyncio
async def test_tool_error_is_a_result_and_loop_continues(tmp_path: Path) -> None:
    call = ToolCall("call-1", "fail", {})
    backend = FakeBackend(
        [ScriptedTurn([], [call]), ScriptedTurn([TextContent("recovered")])]
    )
    store = ConversationStore(tmp_path)

    async def fail(arguments: dict[str, str]) -> str:
        raise RuntimeError("tool broke")

    events = await collect(AgentLoop(backend, store, tools={"fail": fail}).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None and result.is_error
    assert "tool broke" in result.content
    assert events[-1].type is StreamEventType.AGENT_END


@pytest.mark.asyncio
async def test_wrong_tool_result_id_becomes_expected_error_result(tmp_path: Path) -> None:
    call = ToolCall("expected", "echo", {})
    backend = FakeBackend(
        [ScriptedTurn([], [call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    def echo(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("wrong", "bad result")

    await collect(AgentLoop(backend, store, tools={"echo": echo}).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.tool_call_id == "expected"
    assert result.is_error
    assert "mismatch" in result.content


@pytest.mark.asyncio
async def test_wrong_typed_tool_result_becomes_valid_error_result(tmp_path: Path) -> None:
    call = ToolCall("expected", "echo", {})
    backend = FakeBackend(
        [ScriptedTurn([], [call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    def echo(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(call.id, 123)  # type: ignore[arg-type]

    await collect(AgentLoop(backend, store, tools={"echo": echo}).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.tool_call_id == call.id
    assert result.is_error
    assert "invalid tool result" in result.content
    assert ConversationStore(tmp_path, session_id=store.session_id).messages()


@pytest.mark.parametrize("output", [None, 123, {"value": "bad"}])
@pytest.mark.asyncio
async def test_invalid_tool_handler_output_becomes_error_result(
    tmp_path: Path,
    output: object,
) -> None:
    call = ToolCall("expected", "echo", {})
    backend = FakeBackend(
        [ScriptedTurn([], [call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    def echo(arguments: dict[str, object]) -> object:
        return output

    await collect(AgentLoop(backend, store, tools={"echo": echo}).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.tool_call_id == call.id
    assert result.is_error
    assert "invalid tool handler result" in result.content


async def close_after(
    events: AsyncIterator[StreamEvent],
    event_type: StreamEventType,
) -> list[StreamEvent]:
    seen: list[StreamEvent] = []
    async for event in events:
        seen.append(event)
        if event.type is event_type:
            await events.aclose()
            break
    return seen


@pytest.mark.asyncio
async def test_aclose_after_message_update_persists_partial_state(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("partial")])],
        close_error=RuntimeError("close failed"),
    )
    store = ConversationStore(tmp_path)

    stream = AgentLoop(backend, store).run_turn("start")
    await close_after(
        stream,
        StreamEventType.MESSAGE_UPDATE,
    )

    assert backend.completion_close_count == 1
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert store.messages()[-1].content[0].text == "partial"


@pytest.mark.asyncio
async def test_aclose_after_message_end_persists_complete_state(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("complete")])],
        close_error=RuntimeError("close failed"),
    )
    store = ConversationStore(tmp_path)

    await close_after(
        AgentLoop(backend, store).run_turn("start"),
        StreamEventType.MESSAGE_END,
    )

    assert store.messages()[-1].content[0].text == "complete"


@pytest.mark.asyncio
async def test_plain_async_iterator_completes_without_aclose(tmp_path: Path) -> None:
    class PlainCompletion:
        def __init__(self) -> None:
            self.events = [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent("plain"),
                ),
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [TextContent("plain")],
                    ),
                ),
            ]
            self.index = 0

        def __aiter__(self) -> "PlainCompletion":
            return self

        async def __anext__(self) -> StreamEvent:
            if self.index == len(self.events):
                raise StopAsyncIteration
            event = self.events[self.index]
            self.index += 1
            return event

    class PlainBackend(CompletionBackend):
        def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            return PlainCompletion()

    events = await collect(
        AgentLoop(PlainBackend(), ConversationStore(tmp_path)).run_turn("start")
    )

    assert all(event.type is not StreamEventType.ERROR for event in events)
    assert events[-1].type is StreamEventType.AGENT_END


@pytest.mark.asyncio
async def test_cancellation_persists_partial_state(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("partial"), TextContent("more")], delay=0.1)],
        close_error=RuntimeError("close failed"),
    )
    store = ConversationStore(tmp_path)
    task = asyncio.create_task(collect(AgentLoop(backend, store).run_turn("start")))
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.completion_close_count == 1
    assert len(store.messages()) == 2
    assert store.messages()[-1].role is MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_cancellation_keeps_control_error_when_partial_persist_fails(
    tmp_path: Path,
) -> None:
    class WaitingBackend(CompletionBackend):
        def __init__(self) -> None:
            self.update_seen = asyncio.Event()

        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("partial"),
            )
            self.update_seen.set()
            await asyncio.Event().wait()

    backend = WaitingBackend()
    store = ConversationStore(tmp_path)
    task = asyncio.create_task(collect(AgentLoop(backend, store).run_turn("start")))
    await backend.update_seen.wait()

    with patch.object(
        store,
        "append_message",
        side_effect=OSError("disk full"),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_aclose_keeps_control_error_when_partial_persist_fails(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    stream = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("partial")])]),
        store,
    ).run_turn("start")

    async for event in stream:
        if event.type is StreamEventType.MESSAGE_UPDATE:
            with patch.object(
                store,
                "append_message",
                side_effect=OSError("disk full"),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    await stream.aclose()
            break


@pytest.mark.asyncio
async def test_max_turns_stops(tmp_path: Path) -> None:
    call = ToolCall("call-1", "echo", {})
    backend = FakeBackend([ScriptedTurn([], [call]), ScriptedTurn([], [call])])
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, tools={"echo": lambda arguments: "ok"}, max_turns=1).run_turn("start")
    )

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.code == "max_turns"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_backend_error_is_typed_and_user_state_is_persisted(tmp_path: Path) -> None:
    class BrokenBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            raise RuntimeError("backend broke")
            yield

    store = ConversationStore(tmp_path)
    events = await collect(AgentLoop(BrokenBackend(), store).run_turn("start"))

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.code == "backend_error"
    assert len(store.messages()) == 1
