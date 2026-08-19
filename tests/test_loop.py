import asyncio
from pathlib import Path

import pytest

from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.types import (
    MessageRole,
    StreamEventType,
    TextContent,
    ToolCall,
)


async def collect(events):
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
async def test_cancellation_persists_partial_state(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("partial"), TextContent("more")], delay=0.1)]
    )
    store = ConversationStore(tmp_path)
    task = asyncio.create_task(collect(AgentLoop(backend, store).run_turn("start")))
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(store.messages()) == 2
    assert store.messages()[-1].role is MessageRole.ASSISTANT


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
    class BrokenBackend:
        async def complete(self, messages, tool_schemas):
            raise RuntimeError("backend broke")
            yield

    store = ConversationStore(tmp_path)
    events = await collect(AgentLoop(BrokenBackend(), store).run_turn("start"))

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.code == "backend_error"
    assert len(store.messages()) == 1
