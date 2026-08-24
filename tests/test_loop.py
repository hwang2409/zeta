import asyncio
import json
import warnings
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.context import ContextAssembler
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.loop import _validated_tool_result
from zeta.tools import ToolRegistry, ToolStreamPublisher
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
async def test_empty_system_prompt_is_not_sent_to_backend(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    store = ConversationStore(tmp_path)

    await collect(AgentLoop(backend, store, system_prompt="").run_turn("hi"))

    assert all(
        message.role is not MessageRole.SYSTEM for message in backend.calls[0][0]
    )


def test_validated_tool_result_joins_multiple_text_blocks() -> None:
    result = _validated_tool_result(
        {
            "content": [
                {"type": "text", "text": "one", "truncated": False, "full_size": 3},
                {"type": "text", "text": "two", "truncated": False, "full_size": 3},
            ],
            "isError": False,
            "structuredContent": None,
        },
        "call-1",
    )

    assert result.content == "one\ntwo"


def test_validated_tool_result_uses_utf8_bytes_in_truncation_marker() -> None:
    result = _validated_tool_result(
        {
            "content": [
                {"type": "text", "text": "é", "truncated": True, "full_size": 10}
            ],
            "isError": False,
            "structuredContent": None,
        },
        "call-1",
    )

    assert result.content == "é\n[truncated: 2 of 10 bytes]"


def test_validated_tool_result_has_no_marker_when_not_truncated() -> None:
    result = _validated_tool_result(
        {
            "content": [
                {"type": "text", "text": "é", "truncated": False, "full_size": 2}
            ],
            "isError": False,
            "structuredContent": None,
        },
        "call-1",
    )

    assert result.content == "é"


@pytest.mark.asyncio
async def test_agent_loop_preserves_mixed_tool_blocks(tmp_path: Path) -> None:
    call = ToolCall("mixed-1", "mixed", {})
    blocks = [
        {"type": "text", "text": "answer", "truncated": False, "full_size": 6},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {
            "type": "resource",
            "resource": {"uri": "file:///tmp/note.txt", "text": "note"},
        },
    ]

    async def mixed(arguments: dict[str, object]) -> dict[str, object]:
        return {
            "content": blocks,
            "isError": False,
            "structuredContent": None,
        }

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("mixed", mixed)

    await collect(AgentLoop(backend, store, registry=registry).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.content == "answer\n[image block]\n[resource: file:///tmp/note.txt]"
    assert result.content_blocks == blocks
    replayed_result = backend.calls[1][0][-1].tool_result
    assert replayed_result is not None
    assert replayed_result.content_blocks == blocks


@pytest.mark.asyncio
async def test_agent_loop_rejects_invalid_block_metadata_before_persistence(
    tmp_path: Path,
) -> None:
    call = ToolCall("invalid-metadata-1", "invalid", {})

    async def invalid(arguments: dict[str, object]) -> dict[str, object]:
        return {
            "content": [
                {
                    "type": "image",
                    "data": "aGVsbG8=",
                    "mimeType": "image/png",
                    "annotations": {"extra": object()},
                }
            ],
            "isError": False,
            "structuredContent": None,
        }

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("invalid", invalid)

    await collect(AgentLoop(backend, store, registry=registry).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.is_error is True
    assert "unsupported fields" in result.content
    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    assert reopened.messages()[2].tool_result is not None


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
async def test_bash_streams_in_order_and_persists_one_result(tmp_path: Path) -> None:
    call = ToolCall(
        "stream-1",
        "bash",
        {
            "cmd": (
                "printf 'one\\n'; sleep 0.05; "
                "printf 'two\\n'; sleep 0.05; printf 'three\\n'"
            )
        },
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    events = await collect(AgentLoop(backend, store).run_turn("start"))

    tool_events = [
        event
        for event in events
        if event.tool_call is not None and event.tool_call.id == call.id
    ]
    assert [event.type for event in tool_events] == [
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_UPDATE,
        StreamEventType.TOOL_EXECUTION_UPDATE,
        StreamEventType.TOOL_EXECUTION_UPDATE,
        StreamEventType.TOOL_EXECUTION_END,
    ]
    assert [event.delta for event in tool_events[1:-1]] == [
        "one\n",
        "two\n",
        "three\n",
    ]

    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert len(results) == 1
    assert results[0].content == tool_events[-1].tool_result.content
    persisted_context = backend.calls[1][0]
    assert sum(message.tool_result is not None for message in persisted_context) == 1
    assert all(entry.type == "message" for entry in store.entries)
    assert "tool_execution_update" not in json.dumps(
        [entry.to_dict() for entry in store.entries]
    )
    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    replayed_context = await ContextAssembler(reopened).assemble()
    assert sum(message.tool_result is not None for message in replayed_context) == 1


@pytest.mark.asyncio
async def test_stream_publisher_closes_before_delayed_output(tmp_path: Path) -> None:
    calls = [
        ToolCall("late-1", "stream", {}),
        ToolCall("late-2", "stream", {}),
    ]
    late_tasks: list[asyncio.Task[None]] = []
    backend = FakeBackend([ScriptedTurn(tool_calls=calls)])

    async def stream(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: ToolStreamPublisher,
    ) -> str:
        del arguments, abort_signal
        publisher.publish("now", "stdout")

        async def publish_late() -> None:
            await asyncio.sleep(0)
            publisher.publish("late", "stdout")

        late_tasks.append(asyncio.create_task(publish_late()))
        return "final"

    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        tools={"stream": stream},
        max_turns=1,
    )
    events = await collect(loop.run_turn("start"))
    await asyncio.gather(*late_tasks)

    for call in calls:
        tool_events = [event for event in events if event.tool_call == call]
        assert [event.type for event in tool_events] == [
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_UPDATE,
            StreamEventType.TOOL_EXECUTION_END,
        ]
        assert tool_events[1].delta == "now"


@pytest.mark.asyncio
async def test_bash_streams_stdout_and_stderr_labels(tmp_path: Path) -> None:
    call = ToolCall(
        "split-1",
        "bash",
        {"cmd": "printf out; printf err >&2"},
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    events = await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            max_turns=1,
        ).run_turn("start")
    )

    updates = [
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE
    ]
    assert {event.data["stream"] for event in updates} == {"stdout", "stderr"}
    assert {event.delta for event in updates} == {"out", "err"}


@pytest.mark.asyncio
async def test_bash_stream_preserves_split_utf8_code_points(tmp_path: Path) -> None:
    call = ToolCall(
        "utf8-1",
        "bash",
        {"cmd": "printf '\\303'; sleep 0.03; printf '\\251'"},
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    events = await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            max_turns=1,
        ).run_turn("start")
    )

    updates = [
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE
    ]
    assert "".join(event.delta or "" for event in updates) == "é"
    assert all("�" not in (event.delta or "") for event in updates)


@pytest.mark.asyncio
async def test_bash_cancel_stops_updates_before_terminal_event(tmp_path: Path) -> None:
    call = ToolCall(
        "cancel-1",
        "bash",
        {"cmd": "printf first; sleep 5; printf second"},
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    events: list[StreamEvent] = []

    async for event in loop.run_turn("start"):
        events.append(event)
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            loop.abort()

    tool_events = [
        event
        for event in events
        if event.tool_call is not None and event.tool_call.id == call.id
    ]
    end_index = next(
        index
        for index, event in enumerate(tool_events)
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert all(
        event.type is not StreamEventType.TOOL_EXECUTION_UPDATE
        for event in tool_events[end_index + 1 :]
    )
    assert tool_events[end_index].tool_result.is_error is True
    assert tool_events[end_index].tool_result.content == "tool execution canceled"
    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert len(results) == 1
    assert results[0].content == "tool execution canceled"
    assert "tool_execution_update" not in json.dumps(
        [entry.to_dict() for entry in store.entries]
    )


@pytest.mark.asyncio
async def test_streamed_message_log_is_cadence_stable(tmp_path: Path) -> None:
    async def run(cadence: str, root: Path) -> str:
        call = ToolCall("stable-1", "stream", {})

        async def stream(
            arguments: dict[str, object],
            abort_signal: object,
            publisher: ToolStreamPublisher,
        ) -> str:
            del arguments, abort_signal
            if cadence == "small":
                publisher.publish("a", "stdout")
                await asyncio.sleep(0.01)
                publisher.publish("b", "stdout")
            else:
                publisher.publish("ab", "stdout")
            return "ab"

        backend = FakeBackend(
            [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
        )
        store = ConversationStore(root)
        await collect(
            AgentLoop(backend, store, tools={"stream": stream}).run_turn("start")
        )
        messages = [
            entry.data["message"]
            for entry in store.entries
            if entry.type == "message"
        ]
        return json.dumps(messages, sort_keys=True, separators=(",", ":"))

    small_chunks = await run("small", tmp_path / "small")
    large_chunks = await run("large", tmp_path / "large")
    assert small_chunks == large_chunks


@pytest.mark.asyncio
async def test_stream_update_queue_drops_oldest_without_truncating_result(
    tmp_path: Path,
) -> None:
    call = ToolCall("burst-1", "stream", {})
    chunks = [f"chunk-{index}\n" for index in range(200)]
    final_text = "".join(chunks)
    publish_duration = 0.0
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])

    async def stream(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: ToolStreamPublisher,
    ) -> str:
        nonlocal publish_duration
        del arguments, abort_signal
        started = asyncio.get_running_loop().time()
        for chunk in chunks:
            publisher.publish(chunk, "stdout")
        publish_duration = asyncio.get_running_loop().time() - started
        return final_text

    events: list[StreamEvent] = []
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        tools={"stream": stream},
        max_turns=1,
    )
    async for event in loop.run_turn("start"):
        events.append(event)
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            await asyncio.sleep(0.001)

    updates = [
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE
    ]
    end = next(
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert publish_duration < 0.1
    assert len(updates) == 128
    assert updates[0].delta == "chunk-72\n"
    assert updates[-1].delta == "chunk-199\n"
    assert end.tool_result.content == final_text


@pytest.mark.asyncio
async def test_parallel_cancellation_persists_resolved_results(tmp_path: Path) -> None:
    first_call = ToolCall("call-1", "first", {})
    second_call = ToolCall("call-2", "second", {})
    backend = FakeBackend([ScriptedTurn([], [first_call, second_call])])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)

    async def first(arguments: dict[str, object]) -> str:
        return "one"

    async def second(arguments: dict[str, object]) -> str:
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
                await asyncio.sleep(0)
                asyncio.current_task().cancel()

    task = asyncio.create_task(consume())
    with pytest.raises(asyncio.CancelledError):
        await task

    results = {
        message.tool_result.tool_call_id: message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    }
    assert results[first_call.id].content == "one"
    assert results[second_call.id].content == "two"
    assert not results[first_call.id].is_error
    assert not results[second_call.id].is_error


@pytest.mark.parametrize(
    "aborted_index",
    [0, 1],
    ids=["first-canceled", "second-canceled"],
)
@pytest.mark.asyncio
async def test_parallel_cancellation_keeps_call_order(
    tmp_path: Path,
    aborted_index: int,
) -> None:
    calls = [
        ToolCall("call-1", "first", {}),
        ToolCall("call-2", "second", {}),
    ]
    backend = FakeBackend([ScriptedTurn([], calls)])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    started = [asyncio.Event(), asyncio.Event()]
    completed = asyncio.Event()

    async def first(arguments: dict[str, object]) -> str:
        started[0].set()
        if aborted_index == 0:
            await asyncio.Event().wait()
        completed.set()
        return "one"

    async def second(arguments: dict[str, object]) -> str:
        started[1].set()
        if aborted_index == 1:
            await asyncio.Event().wait()
        completed.set()
        return "two"

    registry.register("first", first, parallel_safe=True)
    registry.register("second", second, parallel_safe=True)
    loop = AgentLoop(backend, store, registry=registry)
    task = asyncio.create_task(collect(loop.run_turn("start")))
    await started[0].wait()
    await started[1].wait()
    await completed.wait()
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert [result.tool_call_id for result in results] == [call.id for call in calls]
    assert [result.is_error for result in results] == [
        aborted_index == 0,
        aborted_index == 1,
    ]


@pytest.mark.asyncio
async def test_parallel_duplicate_ids_use_indexed_results(tmp_path: Path) -> None:
    calls = [
        ToolCall("same-id", "first", {}),
        ToolCall("same-id", "second", {}),
    ]
    backend = FakeBackend([ScriptedTurn([], calls)])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("first", lambda arguments: "one", parallel_safe=True)
    registry.register("second", lambda arguments: "two", parallel_safe=True)

    await collect(AgentLoop(backend, store, registry=registry).run_turn("start"))

    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert [result.tool_call_id for result in results] == ["same-id", "same-id"]
    assert [result.content for result in results] == ["one", "two"]


@pytest.fixture(scope="module")
def finalize_store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("finalize-tool-results")


def reverse_completion_slots() -> list[ToolResult | None]:
    slots: list[ToolResult | None] = [None, None]
    slots[1] = ToolResult("call-2", "two")
    slots[0] = ToolResult("call-1", "one")
    return slots


@pytest.mark.parametrize(
    ("calls", "slots", "expected_contents", "expected_errors"),
    [
        pytest.param(
            [ToolCall("call-1", "first", {}), ToolCall("call-2", "second", {})],
            [ToolResult("call-1", "one"), ToolResult("call-2", "two")],
            ["one", "two"],
            [False, False],
            id="all-resolved-in-order",
        ),
        pytest.param(
            [ToolCall("call-1", "first", {}), ToolCall("call-2", "second", {})],
            reverse_completion_slots(),
            ["one", "two"],
            [False, False],
            id="all-resolved-reverse-completion",
        ),
        pytest.param(
            [
                ToolCall("call-1", "first", {}),
                ToolCall("call-2", "second", {}),
                ToolCall("call-3", "third", {}),
            ],
            [ToolResult("call-1", "one"), None, ToolResult("call-3", "three")],
            ["one", "tool execution canceled", "three"],
            [False, True, False],
            id="mixed-real-and-canceled",
        ),
        pytest.param(
            [ToolCall("same-id", "first", {}), ToolCall("same-id", "second", {})],
            [ToolResult("same-id", "one"), ToolResult("same-id", "two")],
            ["one", "two"],
            [False, False],
            id="duplicate-ids",
        ),
        pytest.param([], [], [], [], id="zero-calls"),
        pytest.param(
            [ToolCall("call-1", "first", {})],
            [None],
            ["tool execution canceled"],
            [True],
            id="one-canceled-call",
        ),
    ],
)
def test_finalize_tool_results(
    finalize_store_path: Path,
    calls: list[ToolCall],
    slots: list[ToolResult | None],
    expected_contents: list[str],
    expected_errors: list[bool],
) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(finalize_store_path))

    results = loop._finalize_tool_results(calls, slots)

    assert [result.content for result in results] == expected_contents
    assert [result.is_error for result in results] == expected_errors
    persisted = [
        message.tool_result
        for message in loop.store.messages()
        if message.tool_result is not None
    ]
    assert [result.content for result in persisted] == expected_contents


@pytest.mark.asyncio
async def test_parallel_results_persist_in_call_order(tmp_path: Path) -> None:
    first_call = ToolCall("call-1", "first", {})
    second_call = ToolCall("call-2", "second", {})
    backend = FakeBackend([ScriptedTurn([], [first_call, second_call])])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_release = asyncio.Event()
    second_release = asyncio.Event()

    async def first(arguments: dict[str, object]) -> str:
        first_started.set()
        await first_release.wait()
        return "one"

    async def second(arguments: dict[str, object]) -> str:
        second_started.set()
        await second_release.wait()
        return "two"

    registry.register("first", first, parallel_safe=True)
    registry.register("second", second, parallel_safe=True)
    loop = AgentLoop(backend, store, registry=registry)

    task = asyncio.create_task(collect(loop.run_turn("start")))
    await first_started.wait()
    await second_started.wait()
    second_release.set()
    first_release.set()
    await task

    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert [result.tool_call_id for result in results] == [
        first_call.id,
        second_call.id,
    ]
    assert [result.content for result in results] == ["one", "two"]


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
