import asyncio
import json
import stat
import warnings
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import zeta.providers.anthropic as anthropic_module
import zeta.providers.codex as codex_module
import zeta.providers.retry_policy as retry_policy_module
from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.prompts import load_identity
from zeta.protocol.types import (
    CompletionBackend,
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
from zeta.providers.payload_common import HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER
from zeta.providers.retry_policy import current_retry_budget
from zeta.providers.transport import retry_provider_completion
from zeta.runtime.loop.agent import _validated_tool_result
from zeta.runtime.loop.cache_trace import CacheTrace
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry, ToolStreamPublisher


def anthropic_request_bytes(
    messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
) -> bytes:
    payload = anthropic_module.build_request_payload(
        messages,
        tool_schemas,
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )
    return anthropic_module.serialize_request_payload(payload)


def codex_request_bytes(
    messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
) -> bytes:
    payload = codex_module.build_responses_payload(
        messages,
        tool_schemas,
        model="codex-test",
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class RetryableProviderFailure(RuntimeError):
    code = "http_error"

    def __init__(
        self,
        message: str = "request failed",
        *,
        status_code: int | None = None,
        retry_after: float | None = 0.0,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


class AttemptBackend(CompletionBackend):
    def __init__(self, attempts: Sequence[Sequence[StreamEvent] | BaseException]) -> None:
        self.attempts = list(attempts)
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        attempt = self.attempts[self.calls]
        self.calls += 1
        if isinstance(attempt, BaseException):
            raise attempt
        for event in attempt:
            if event.type is StreamEventType.ERROR:
                yield event
                return
            yield event
        if attempt and attempt[-1].type is StreamEventType.MESSAGE_END:
            return
        raise RetryableProviderFailure()


class ParallelChildFailureBackend(CompletionBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        prompt = next(
            (
                block.text
                for block in reversed(messages[-1].content)
                if isinstance(block, TextContent)
            ),
            "",
        )
        if prompt == "die":
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("child partial"),
            )
            raise ConnectionError("child connection dropped")
        if prompt == "live":
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(
                    MessageRole.ASSISTANT,
                    [TextContent("sibling complete")],
                ),
            )
            return
        if any(message.tool_result is not None for message in messages):
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(
                    MessageRole.ASSISTANT,
                    [TextContent("parent survived")],
                ),
            )
            return
        calls = [
            ToolCall(
                "child-dies",
                "agent",
                {"prompt": "die", "description": "failing child"},
            ),
            ToolCall(
                "child-lives",
                "agent",
                {"prompt": "live", "description": "healthy sibling"},
            ),
        ]
        message = Message(
            MessageRole.ASSISTANT,
            [TextContent("delegating"), *(ToolUseContent(call) for call in calls)],
        )
        yield StreamEvent(StreamEventType.MESSAGE_END, message=message)


@pytest.mark.asyncio
async def test_single_turn_without_tools(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    store = ConversationStore(tmp_path)

    events = await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi"))

    assert events[-1].type is StreamEventType.AGENT_END
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert store.messages()[-1].metadata["response_state"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "request_serializer"),
    [
        pytest.param("anthropic", anthropic_request_bytes, id="anthropic"),
        pytest.param("codex", codex_request_bytes, id="codex"),
    ],
)
async def test_notification_turn_serializes_notification_as_actionable_input(
    tmp_path: Path,
    provider: str,
    request_serializer,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("acknowledged")])],
        request_serializer=request_serializer,
    )
    store = ConversationStore(tmp_path)
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("waiting")]))
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await collect(loop.run_notification_turn())

    payload = json.loads(backend.request_bytes[0])
    expected_prefix = (
        f"{HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER}\n"
        "durable notifications (kind is agent_completion when omitted):\n"
    )
    if provider == "anthropic":
        notification_text = next(
            block["text"]
            for message in payload["messages"]
            for block in message["content"]
            if block.get("text", "").startswith(
                HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER
            )
        )
        assert payload["messages"][-1]["role"] == "user"
        assert all(
            not block.get("text", "").startswith(
                HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER
            )
            for block in payload["system"]
        )
    else:
        notification_text = next(
            part["text"]
            for item in payload["input"]
            for part in item.get("content", [])
            if part.get("text", "").startswith(
                HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER
            )
        )
        assert payload["input"][-1]["role"] == "user"
        assert HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER not in payload["instructions"]
    assert notification_text.startswith(expected_prefix)
    assert "child-1" in notification_text
    assert "done" in notification_text
    await loop.close()


@pytest.mark.asyncio
async def test_fake_usage_reports_cache_reads_on_consecutive_turns(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first")], usage={"output_tokens": 5}),
            ScriptedTurn([TextContent("second")], usage={"output_tokens": 5}),
            ScriptedTurn([TextContent("third")], usage={"output_tokens": 5}),
        ],
        request_serializer=anthropic_request_bytes,
    )
    loop = AgentLoop(backend, ConversationStore(tmp_path), tool_schemas=[], skill_catalog=SkillCatalog.empty())

    await collect(loop.run_turn("first prompt"))
    assert loop.context_assembler.cache_read_input_tokens_this_session == 0
    await collect(loop.run_turn("second prompt"))
    second_read = loop.context_assembler.cache_read_input_tokens_this_session
    assert second_read > 0
    await collect(loop.run_turn("third prompt"))

    assert loop.context_assembler.cache_read_input_tokens_this_session > second_read
    assert loop.context_assembler.cache_creation_input_tokens_this_session > 0
    first_payload = json.loads(backend.request_bytes[0])
    assert first_payload["system"][0]["text"] == (
        "You are Claude Code, Anthropic's official CLI for Claude."
    )

    baseline_backend = FakeBackend(
        [
            ScriptedTurn([TextContent("same")], usage={"output_tokens": 1}),
            ScriptedTurn([TextContent("same")], usage={"output_tokens": 1}),
        ],
        request_serializer=anthropic_request_bytes,
    )
    same_request = [Message(MessageRole.USER, [TextContent("same")])]
    await collect(baseline_backend.complete(same_request, []))
    baseline_events = await collect(baseline_backend.complete(same_request, []))
    baseline_read = baseline_events[-1].data["usage"]["cache_read_input_tokens"]

    mutated_backend = FakeBackend(
        [
            ScriptedTurn([TextContent("same")], usage={"output_tokens": 1}),
            ScriptedTurn([TextContent("same")], usage={"output_tokens": 1}),
        ],
    )

    def mutate_first_request(
        messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> bytes:
        payload = anthropic_module.build_request_payload(
            messages,
            tool_schemas,
            model="claude-test",
            max_tokens=4096,
            thinking_budget=2048,
        )
        if not mutated_backend.request_bytes:
            payload["system"][0]["text"] += "x"
        return anthropic_module.serialize_request_payload(payload)

    mutated_backend.request_serializer = mutate_first_request
    await collect(mutated_backend.complete(same_request, []))
    changed_events = await collect(mutated_backend.complete(same_request, []))

    assert (
        changed_events[-1].data["usage"]["cache_read_input_tokens"]
        < baseline_read
    )


@pytest.mark.asyncio
async def test_opt_in_cache_trace_records_reuse_without_prompt_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ZETA_CACHE_TRACE", "1")
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first")], usage={"input_tokens": 10}),
            ScriptedTurn([TextContent("second")], usage={"input_tokens": 10}),
        ]
    )
    backend.model = "gpt-5.6-luna"
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path / "session"),
        tool_schemas=[
            {
                "name": "sensitive_tool",
                "description": "private tool description",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        system_prompt="private system instructions",
        skill_catalog=SkillCatalog.empty(),
    )

    await collect(loop.run_turn("private first question"))
    await collect(loop.run_turn("private second question"))

    trace = tmp_path / "home" / "logs" / "cache-trace.jsonl"
    raw = trace.read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 2
    assert rows[0]["shared_prefix_messages"] is None
    assert rows[1]["shared_prefix_messages"] == 2
    assert rows[1]["same_tools"] is True
    assert rows[1]["provider"] == "FakeBackend"
    assert rows[1]["model"] == "gpt-5.6-luna"
    assert rows[1]["tool_count"] == 1
    assert rows[1]["cache_read_tokens"] > 0
    assert rows[1]["cache_hit_rate"] > 0
    assert rows[1]["duration_seconds"] >= 0
    assert "private" not in raw
    assert "sensitive_tool" not in raw
    assert stat.S_IMODE(trace.stat().st_mode) == 0o600
    assert loop._cache_trace is not None
    assert all(
        isinstance(value, bytes)
        for value in loop._cache_trace.previous_messages or ()
    )
    await loop.close()


def test_cache_trace_compares_against_last_completed_request(tmp_path: Path) -> None:
    trace = CacheTrace(tmp_path / "trace.jsonl", "session", 0)
    backend = FakeBackend([])
    system = Message(MessageRole.SYSTEM, [TextContent("stable")])
    original = Message(MessageRole.USER, [TextContent("original")])
    changed = Message(MessageRole.USER, [TextContent("changed")])
    first = trace.start([system, original], [], backend, 1, False, False)
    trace.observe(
        first,
        StreamEvent(StreamEventType.MESSAGE_END, data={"usage": {"input_tokens": 2}}),
    )
    trace.start([system, changed], [], backend, 2, False, False)

    resumed = trace.start([system, original, changed], [], backend, 3, False, False)

    assert resumed["shared_prefix_messages"] == 2


@pytest.mark.asyncio
async def test_unsigned_thinking_is_not_persisted_with_assistant_message(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([ThinkingContent("partial"), TextContent("answer")])]
    )
    store = ConversationStore(tmp_path)

    await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi"))

    assert store.messages()[-1].content == [TextContent("answer")]


@pytest.mark.asyncio
async def test_header_only_thinking_is_persisted_with_assistant_message(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([ThinkingContent("")]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi"))

    # Header-only thinking stays durable even though the empty turn is nudged.
    assert any(
        message.content == [ThinkingContent("")] for message in store.messages()
    )


@pytest.mark.asyncio
async def test_tool_lifecycle_events_separate_approval_from_execution(
    tmp_path: Path,
) -> None:
    call = ToolCall("approval-live", "echo", {"value": "ok"})
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    loop = AgentLoop(
        backend,
        store,
        tools={"echo": lambda arguments: arguments["value"]},
        approval_policy=policy,
        max_turns=1,
skill_catalog=SkillCatalog.empty(),
    )

    events: list[StreamEvent] = []
    async for event in loop.run_turn("start"):
        events.append(event)
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            assert policy.approve(call.id)

    lifecycle = [
        event.type
        for event in events
        if event.tool_call is not None and event.tool_call.id == call.id
    ]
    assert lifecycle == [
        StreamEventType.TOOL_APPROVAL_START,
        StreamEventType.TOOL_APPROVAL_END,
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_END,
    ]


@pytest.mark.asyncio
async def test_default_system_prompt_is_zeta_identity(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    store = ConversationStore(tmp_path)

    await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi"))

    assert backend.calls[0][0][0].content[0].text == load_identity(catalog=SkillCatalog.empty())


@pytest.mark.asyncio
async def test_empty_system_prompt_is_not_sent_to_backend(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    store = ConversationStore(tmp_path)

    await collect(AgentLoop(backend, store, system_prompt="", skill_catalog=SkillCatalog.empty()).run_turn("hi"))

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
        {
            "type": "image",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
            "mimeType": "image/png",
        },
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
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register("mixed", mixed)

    await collect(AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.content == (
        "answer\n[image block] media_type=image/png dimensions=1x1 bytes=70\n"
        "[resource: file:///tmp/note.txt]"
    )
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
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
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
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register("invalid", invalid)

    await collect(AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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

    events = await collect(AgentLoop(backend, store, tools={"echo": echo}, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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
    assert store.messages()[1].metadata["response_state"] == "completed"
    assert store.messages()[-1].metadata["response_state"] == "completed"
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

    events = await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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
skill_catalog=SkillCatalog.empty(),
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
skill_catalog=SkillCatalog.empty(),
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
skill_catalog=SkillCatalog.empty(),
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
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
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
            AgentLoop(backend, store, tools={"stream": stream}, skill_catalog=SkillCatalog.empty()).run_turn("start")
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
skill_catalog=SkillCatalog.empty(),
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
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())

    async def first(arguments: dict[str, object]) -> str:
        return "one"

    async def second(arguments: dict[str, object]) -> str:
        return "two"

    registry.register("first", first, parallel_safe=True)
    registry.register("second", second, parallel_safe=True)
    loop = AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty())

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
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
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
    loop = AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty())
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
async def test_parallel_duplicate_ids_fail_before_dispatch(tmp_path: Path) -> None:
    calls = [
        ToolCall("same-id", "first", {}),
        ToolCall("same-id", "second", {}),
    ]
    backend = FakeBackend([ScriptedTurn([], calls)])
    store = ConversationStore(tmp_path)

    with pytest.raises(ValueError, match="duplicate tool call id"):
        await collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert not (store.session_dir / "agents").exists()


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
    loop = AgentLoop(FakeBackend([]), ConversationStore(finalize_store_path), skill_catalog=SkillCatalog.empty())

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
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
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
    loop = AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty())

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

    events = await collect(AgentLoop(backend, store, tools={"fail": fail}, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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

    await collect(AgentLoop(backend, store, tools={"echo": echo}, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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

    await collect(AgentLoop(backend, store, tools={"echo": echo}, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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

    await collect(AgentLoop(backend, store, tools={"echo": echo}, skill_catalog=SkillCatalog.empty()).run_turn("start"))

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

    stream = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
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
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start"),
        StreamEventType.MESSAGE_END,
    )

    assert store.messages()[-1].content[0].text == "complete"


@pytest.mark.asyncio
async def test_aclose_during_active_tool_persists_canceled_result(tmp_path: Path) -> None:
    call = ToolCall("active-close", "blocked", {})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)
    started = asyncio.Event()

    async def blocked(arguments: dict[str, object]) -> str:
        del arguments
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    stream = AgentLoop(
        backend,
        store,
        tools={"blocked": blocked},
        max_turns=1,
skill_catalog=SkillCatalog.empty(),
    ).run_turn("start")
    async for event in stream:
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            await started.wait()
            await stream.aclose()
            break

    result = store.messages()[-1].tool_result
    assert result is not None
    assert result.content == "tool execution canceled"
    assert result.is_error is True


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
        AgentLoop(PlainBackend(), ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty()).run_turn("start")
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
    update_seen = asyncio.Event()

    async def collect_after_update() -> None:
        async for event in AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start"):
            if event.type is StreamEventType.MESSAGE_UPDATE:
                update_seen.set()

    task = asyncio.create_task(collect_after_update())
    await asyncio.wait_for(update_seen.wait(), timeout=10)
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
    task = asyncio.create_task(collect(AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")))
    await backend.update_seen.wait()

    with patch.object(
        store,
        "append_message",
        side_effect=OSError("disk full"),
    ), warnings.catch_warnings():
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
skill_catalog=SkillCatalog.empty(),
    ).run_turn("start")

    async for event in stream:
        if event.type is StreamEventType.MESSAGE_UPDATE:
            with patch.object(
                store,
                "append_message",
                side_effect=OSError("disk full"),
            ), warnings.catch_warnings():
                warnings.simplefilter("error")
                await stream.aclose()
            break


@pytest.mark.asyncio
async def test_max_turns_stops(tmp_path: Path) -> None:
    call = ToolCall("call-1", "echo", {})
    backend = FakeBackend([ScriptedTurn([], [call]), ScriptedTurn([], [call])])
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, tools={"echo": lambda arguments: "ok"}, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.code == "max_turns"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_truncated_stream_persists_stop_reason_metadata(tmp_path: Path) -> None:
    class TruncatedBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("partial")]),
                data={
                    "truncated": True,
                    "stop_reason": "max_tokens",
                    "usage": {"output_tokens": 37},
                },
            )
            raise ConnectionError("stream disconnected")

    store = ConversationStore(tmp_path)

    await collect(
        AgentLoop(
            TruncatedBackend(), store, skill_catalog=SkillCatalog.empty()
        ).run_turn("start")
    )

    partial = store.messages()[-1]
    assert partial.content == [TextContent("partial")]
    assert partial.metadata["stop_reason"] == "max_tokens"
    assert partial.metadata["output_tokens"] == 37
    assert partial.metadata["turn_failed"] is True


@pytest.mark.asyncio
async def test_cancel_during_retry_backoff_does_not_persist_discarded_partial(
    tmp_path: Path,
) -> None:
    retry_seen = asyncio.Event()

    class WaitingRetryBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="discard me")
            yield StreamEvent(
                StreamEventType.RETRY,
                data={"text": "retrying"},
            )
            yield StreamEvent(StreamEventType.ASSISTANT_RESET)
            await asyncio.Event().wait()

    store = ConversationStore(tmp_path)
    events: list[StreamEvent] = []

    async def consume() -> None:
        async for event in AgentLoop(
            WaitingRetryBackend(), store, skill_catalog=SkillCatalog.empty()
        ).run_turn("start"):
            events.append(event)
            if event.type is StreamEventType.RETRY:
                retry_seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(retry_seen.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any(
        event.type is StreamEventType.MESSAGE_END
        and event.message is not None
        and event.message.content == [TextContent("discard me")]
        for event in events
    )
    assert all(
        message.content != [TextContent("discard me")] for message in store.messages()
    )


@pytest.mark.asyncio
async def test_discarded_partial_is_not_finalized_when_stream_then_ends(
    tmp_path: Path,
) -> None:
    class PartialOnlyRetryBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="discard me")
            yield StreamEvent(
                StreamEventType.RETRY,
                data={"text": "retrying"},
            )
            yield StreamEvent(StreamEventType.ASSISTANT_RESET)

    store = ConversationStore(tmp_path)
    events = await collect(
        AgentLoop(
            PartialOnlyRetryBackend(), store, skill_catalog=SkillCatalog.empty()
        ).run_turn("start")
    )

    assert not any(
        event.type is StreamEventType.MESSAGE_END
        and event.message is not None
        and event.message.content == [TextContent("discard me")]
        for event in events
    )
    assert all(
        message.content != [TextContent("discard me")] for message in store.messages()
    )


@pytest.mark.asyncio
async def test_canceled_turn_persists_stop_reason_metadata(tmp_path: Path) -> None:
    class WaitingAfterMetadataBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("partial")]),
                data={
                    "truncated": True,
                    "stop_reason": "max_tokens",
                    "usage": {"output_tokens": 23},
                },
            )
            await asyncio.Event().wait()

    store = ConversationStore(tmp_path)
    metadata_seen = asyncio.Event()

    async def consume() -> None:
        async for event in AgentLoop(
            WaitingAfterMetadataBackend(), store, skill_catalog=SkillCatalog.empty()
        ).run_turn("start"):
            if event.type is StreamEventType.MESSAGE_END:
                metadata_seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(metadata_seen.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    partial = store.messages()[-1]
    assert partial.content == [TextContent("partial")]
    assert partial.metadata["stop_reason"] == "max_tokens"
    assert partial.metadata["output_tokens"] == 23
    assert partial.metadata["response_state"] == "aborted"


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
    events = await collect(AgentLoop(BrokenBackend(), store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.code == "backend_error"
    messages = store.messages()
    assert len(messages) == 2
    assert messages[-1].metadata["turn_failed"] is True
    assert messages[-1].metadata["turn_error"]["code"] == "backend_error"
    assert messages[-1].metadata["response_state"] == "failed"


@pytest.mark.asyncio
async def test_slow_context_assembly_does_not_exhaust_retry_budget(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    store = ConversationStore(tmp_path)
    assembler = ContextAssembler(store, backend=backend)
    original_assemble = assembler.assemble
    now = [0.0]

    async def slow_assemble(*args, **kwargs):
        messages = await original_assemble(*args, **kwargs)
        now[0] += 61.0
        return messages

    assembler.assemble = slow_assemble  # type: ignore[method-assign]
    with patch(
        "zeta.providers.retry_policy.time.monotonic", side_effect=lambda: now[0]
    ):
        events = await collect(
            AgentLoop(
                backend,
                store,
                context_assembler=assembler,
                skill_catalog=SkillCatalog.empty(),
            ).run_turn("start")
        )

    assert len(backend.calls) == 1
    assert not any(event.type is StreamEventType.ERROR for event in events)


@pytest.mark.asyncio
async def test_loop_owned_stall_retry_starts_after_real_90s_silence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    budget = None
    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)

    async def advance(delay: float, _abort_signal=None) -> bool:
        nonlocal now
        now += delay
        return True

    monkeypatch.setattr("zeta.runtime.loop.agent.wait_for_provider_retry", advance)

    class StallError(RetryableProviderFailure):
        is_stall = True
        stall_seconds = 90.0

    class AlternatingStallBackend(CompletionBackend):
        def __init__(self) -> None:
            self.stream_calls = 0

        async def complete(self, messages, tool_schemas):
            nonlocal budget
            del messages, tool_schemas
            budget = current_retry_budget()

            async def stream() -> AsyncIterator[StreamEvent]:
                nonlocal now
                self.stream_calls += 1
                if self.stream_calls == 1:
                    yield StreamEvent(StreamEventType.MESSAGE_START)
                    yield StreamEvent(
                        StreamEventType.MESSAGE_UPDATE,
                        content=TextContent("discarded"),
                    )
                now += 90.0
                raise StallError("terminal stall", retry_after=30.0)

            async def retry(_token: str) -> AsyncIterator[StreamEvent]:
                async for event in stream():
                    yield event

            async def refresh() -> str:
                raise AssertionError("auth refresh is unreachable")

            async for event in retry_provider_completion(
                stream,
                retry,
                refresh,
                lambda _error: False,
                lambda error: error,
                lambda number, delay, _error: StreamEvent(
                    StreamEventType.RETRY,
                    data={"retry": number, "delay": delay},
                ),
                lambda _error, _retries: None,
                is_stall=lambda error: getattr(error, "is_stall", False),
                stall_notice=lambda number, delay, _error: StreamEvent(
                    StreamEventType.RETRY,
                    data={
                        "retry": number,
                        "delay": delay,
                        "is_stall": True,
                    },
                ),
                max_stall_retries=2,
                sleep=advance,
            ):
                yield event

    backend = AlternatingStallBackend()
    store = ConversationStore(tmp_path)
    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    retries = [event for event in events if event.type is StreamEventType.RETRY]
    assert backend.stream_calls == 3
    assert len(retries) == 2
    assert budget is not None
    assert budget.stall_retries == 2
    assert budget.excluded_stall_seconds == 270.0
    assert now - budget.started_at - budget.excluded_stall_seconds == 60.0
    assert now == 3 * 90.0 + 2 * 30.0
    assert now <= 330.0
    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error is not None
    assert events[-2].error.message == "terminal stall"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata_event",
    [
        pytest.param(
            StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                data={"index": (0, "raw", 0)},
            ),
            id="codex-raw-reasoning",
        ),
        pytest.param(
            StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                data={"thinking_signature_delta": "signature", "index": 0},
            ),
            id="anthropic-signature",
        ),
        pytest.param(
            StreamEvent(StreamEventType.MESSAGE_START),
            id="generic-message-start",
        ),
    ],
)
async def test_metadata_only_provider_events_are_retry_safe(
    tmp_path: Path,
    metadata_event: StreamEvent,
) -> None:
    class StallError(RetryableProviderFailure):
        is_stall = True
        stall_seconds = 90.0

    class MetadataThenStallBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            del messages, tool_schemas

            async def stream() -> AsyncIterator[StreamEvent]:
                self.calls += 1
                if self.calls == 1:
                    yield metadata_event
                    raise StallError("stalled", retry_after=0.0)
                yield StreamEvent(StreamEventType.MESSAGE_START)
                yield StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [TextContent("final")],
                    ),
                )

            async def retry(_token: str) -> AsyncIterator[StreamEvent]:
                async for event in stream():
                    yield event

            async def refresh() -> str:
                raise AssertionError("auth refresh is unreachable")

            async for event in retry_provider_completion(
                stream,
                retry,
                refresh,
                lambda _error: False,
                lambda error: error,
                lambda number, delay, _error: StreamEvent(
                    StreamEventType.RETRY,
                    data={"retry": number, "delay": delay},
                ),
                lambda _error, _retries: None,
                is_stall=lambda error: getattr(error, "is_stall", False),
                stall_notice=lambda number, delay, _error: StreamEvent(
                    StreamEventType.RETRY,
                    data={
                        "retry": number,
                        "delay": delay,
                        "is_stall": True,
                    },
                ),
                max_stall_retries=1,
                sleep=lambda _delay: asyncio.sleep(0),
            ):
                yield event

    backend = MetadataThenStallBackend()
    store = ConversationStore(tmp_path)
    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    assert backend.calls == 2
    assert sum(event.type is StreamEventType.RETRY for event in events) == 1
    assistant_messages = [
        message for message in store.messages() if message.role is MessageRole.ASSISTANT
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == [TextContent("final")]
    assert (
        sum(
            event.type is StreamEventType.MESSAGE_END and event.message is not None
            for event in events
        )
        == 1
    )
    assert not any(
        event.type is StreamEventType.MESSAGE_UPDATE
        and (event.content is not None or event.delta is not None)
        for event in events
    )
    assert not any(
        isinstance(block, ThinkingContent)
        for message in assistant_messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_retry_after_midstream_request_failed_succeeds(tmp_path: Path) -> None:
    backend = AttemptBackend(
        [
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent("discard me"),
                ),
            ],
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent("final"),
                ),
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, [TextContent("final")]),
                ),
            ],
        ]
    )
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    assert backend.calls == 2
    assert [message.content for message in store.messages()] == [
        [TextContent("start")],
        [TextContent("final")],
    ]
    retry = next(event for event in events if event.type is StreamEventType.RETRY)
    assert retry.data["kind"] == "provider_retry"
    assert retry.data["attempt"] == 2
    assert retry.data["reason"] == "http_error"
    assert "discard_partial" not in retry.data
    reset_index = next(
        index
        for index, event in enumerate(events)
        if event.type is StreamEventType.ASSISTANT_RESET
    )
    final_index = next(
        index
        for index, event in enumerate(events)
        if event.type is StreamEventType.MESSAGE_UPDATE
        and event.content == TextContent("final")
    )
    assert reset_index < final_index
    assert store.messages()[-1].metadata["provider_retries"] == [
        {
            "attempt": 2,
            "reason": "http_error",
            "delay": 0.0,
            "decision": "retried",
        }
    ]


@pytest.mark.asyncio
async def test_retry_after_codex_stream_error_retryable(tmp_path: Path) -> None:
    failure = codex_module.CodexStreamError(
        "Codex output item completed with open blocks",
        retryable=True,
    )
    class RetryingCodexBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            del messages, tool_schemas
            self.calls += 1
            yield StreamEvent(StreamEventType.MESSAGE_START)
            if self.calls == 1:
                raise failure
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
            )

    backend = RetryingCodexBackend()

    await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    assert backend.calls == 2


@pytest.mark.asyncio
async def test_no_retry_after_streamed_tool_call_exposed(tmp_path: Path) -> None:
    call = ToolCall("echo-1", "echo", {"value": "once"})
    backend = AttemptBackend(
        [
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(StreamEventType.MESSAGE_UPDATE, tool_call=call),
            ],
            [
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, [TextContent("wrong")]),
                )
            ],
        ]
    )

    events = await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    assert backend.calls == 1
    assert not any(event.type is StreamEventType.RETRY for event in events)
    assert events[-2].type is StreamEventType.ERROR


@pytest.mark.asyncio
async def test_no_retry_after_tool_call_executed(tmp_path: Path) -> None:
    call = ToolCall("echo-1", "echo", {"value": "once"})
    backend = AttemptBackend(
        [
            [
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [ToolUseContent(call)],
                    ),
                )
            ],
            RetryableProviderFailure(),
        ]
    )
    executions: list[str] = []

    events = await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            tools={"echo": lambda arguments: executions.append(arguments["value"]) or "ok"},
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    assert executions == ["once"]
    assert backend.calls == 2
    assert not any(event.type is StreamEventType.RETRY for event in events)
    assert events[-2].error is not None
    assert events[-2].error.message == "request failed"


@pytest.mark.asyncio
async def test_no_retry_after_user_abort(tmp_path: Path) -> None:
    signal = AbortSignal()

    class AbortingBackend(CompletionBackend):
        calls = 0

        async def complete(self, messages, tool_schemas):
            del messages, tool_schemas
            self.calls += 1
            signal.abort()
            raise RetryableProviderFailure()
            yield

    backend = AbortingBackend()
    events = await collect(
        AgentLoop(
            backend,
            ConversationStore(tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start", abort_signal=signal)
    )

    assert backend.calls == 1
    assert not any(event.type is StreamEventType.RETRY for event in events)


@pytest.mark.asyncio
async def test_abort_during_backoff_keeps_ui_and_store_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def blocked_sleep(_delay: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr("zeta.runtime.loop._completion.asyncio.sleep", blocked_sleep)
    signal = AbortSignal()
    retry_seen = asyncio.Event()
    backend = AttemptBackend(
        [
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="keep me"),
            ],
            [
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, [TextContent("wrong")]),
                )
            ],
        ]
    )
    store = ConversationStore(tmp_path)
    events: list[StreamEvent] = []

    async def consume() -> None:
        async for event in AgentLoop(
            backend, store, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", abort_signal=signal):
            events.append(event)
            if event.type is StreamEventType.RETRY:
                retry_seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(retry_seen.wait(), timeout=10)
    signal.abort()
    await asyncio.wait_for(task, timeout=10)

    retry = next(event for event in events if event.type is StreamEventType.RETRY)
    assert retry.data.get("discard_partial") is not True
    assert not any(event.type.value == "assistant_reset" for event in events)
    assert backend.calls == 1
    assert store.messages()[-1].content == [TextContent("keep me")]
    assert store.messages()[-1].metadata["turn_failed"] is True


@pytest.mark.asyncio
async def test_retry_budget_exhausted_fails_with_original_error(tmp_path: Path) -> None:
    backend = AttemptBackend(
        [
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("http_error", "first failure"),
                    data={"retryable": True, "retry_after": 0.0},
                ),
            ],
            *[
                [
                    StreamEvent(StreamEventType.MESSAGE_START),
                    StreamEvent(
                        StreamEventType.ERROR,
                        error=ErrorInfo("http_error", f"failure {number}"),
                        data={"retryable": True, "retry_after": 0.0},
                    ),
                ]
                for number in range(2, 6)
            ],
        ]
    )
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    assert backend.calls == 5
    assert sum(event.type is StreamEventType.RETRY for event in events) == 4
    assert events[-2].error is not None
    assert events[-2].error.message == "first failure"
    assert store.messages()[-1].metadata["turn_error"]["message"] == "first failure"


@pytest.mark.asyncio
async def test_429_retry_after_honored(tmp_path: Path) -> None:
    backend = AttemptBackend(
        [
            [
                StreamEvent(StreamEventType.MESSAGE_START),
                StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("http_error", "rate limited", status_code=429),
                    data={"retryable": True, "retry_after": 1.25},
                ),
            ],
            [
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
                )
            ],
        ]
    )

    with patch("zeta.runtime.loop._completion.asyncio.sleep") as sleep:
        await collect(
            AgentLoop(
                backend,
                ConversationStore(tmp_path),
                skill_catalog=SkillCatalog.empty(),
            ).run_turn("start")
        )

    sleep.assert_awaited_once_with(1.25)


@pytest.mark.asyncio
async def test_child_agent_survives_transient_provider_error(tmp_path: Path) -> None:
    call = ToolCall(
        "child-1",
        "agent",
        {"prompt": "child work", "description": "retrying child"},
    )

    class RetryingChildBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            del messages, tool_schemas
            self.calls += 1
            if self.calls == 1:
                yield StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [ToolUseContent(call)],
                    ),
                )
                return
            if self.calls == 2:
                yield StreamEvent(StreamEventType.MESSAGE_START)
                raise RetryableProviderFailure()
            text = "child survived" if self.calls == 3 else "parent done"
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent(text)]),
            )

    backend = RetryingChildBackend()
    store = ConversationStore(tmp_path)

    await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    assert backend.calls == 4
    result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    )
    assert result.is_error is False
    assert result.content.startswith("child survived")
    child_path = Path(result.structured_content["child_session_path"])
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    assert child_store.messages()[-1].content == [TextContent("child survived")]
    assert child_store.messages()[-1].metadata["provider_retries"][0]["attempt"] == 2


@pytest.mark.asyncio
async def test_child_retries_post_stream_even_when_root_serve_client_not_negotiated(
    tmp_path: Path,
) -> None:
    call = ToolCall(
        "child-1",
        "agent",
        {"prompt": "child work", "description": "retrying child"},
    )

    class PostStreamRetryChildBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            self.calls += 1
            is_child = any(
                message.role is MessageRole.USER
                and any(
                    block.text == "child work"
                    for block in message.content
                    if isinstance(block, TextContent)
                )
                for message in messages
            )
            del tool_schemas
            if self.calls == 1:
                yield StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [ToolUseContent(call)],
                    ),
                )
                return
            if self.calls == 2:
                yield StreamEvent(StreamEventType.MESSAGE_START)
                yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="child partial")
                raise RetryableProviderFailure()
            if self.calls == 3 and is_child:
                yield StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(
                        MessageRole.ASSISTANT,
                        [TextContent("child survived")],
                    ),
                )
                return
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("parent done")]),
            )

    backend = PostStreamRetryChildBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    loop.post_stream_provider_retry = False

    events = await collect(loop.run_turn("start"))

    assert backend.calls == 4
    result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    )
    assert result is not None and result.is_error is False
    assert result.content.startswith("child survived")
    assert not any(event.type is StreamEventType.ASSISTANT_RESET for event in events)
    assert not any(event.delta == "child partial" for event in events)
    await loop.close()


@pytest.mark.asyncio
async def test_provider_error_event_persists_partial_state_and_ends_turn(
    tmp_path: Path,
) -> None:
    class ErrorEventBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("partial"),
            )
            yield StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("stream_error", "provider disconnected"),
            )

    store = ConversationStore(tmp_path)
    events = await collect(AgentLoop(ErrorEventBackend(), store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert [event.type for event in events][-2:] == [
        StreamEventType.ERROR,
        StreamEventType.AGENT_END,
    ]
    assert StreamEventType.TURN_END not in [event.type for event in events]
    assert [message.content[0].text for message in store.messages()[1:]] == [
        "partial"
    ]
    assert store.messages()[-1].metadata["turn_failed"] is True
    assert store.messages()[-1].metadata["turn_error"]["code"] == "stream_error"


@pytest.mark.asyncio
async def test_clean_stream_exit_becomes_provider_failure(tmp_path: Path) -> None:
    class IncompleteBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("partial"),
            )

    store = ConversationStore(tmp_path)
    events = await collect(AgentLoop(IncompleteBackend(), store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert events[-2].type is StreamEventType.ERROR
    assert events[-2].error == ErrorInfo(
        "stream_error", "provider stream ended before completion"
    )
    assert StreamEventType.TURN_END not in [event.type for event in events]
    assert store.turn_in_flight() is False
    assert store.messages()[-1].metadata["turn_failed"] is True


@pytest.mark.asyncio
async def test_no_output_failure_closes_turn_and_persists_marker(tmp_path: Path) -> None:
    class BrokenBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            raise RuntimeError("provider broke")
            yield

    store = ConversationStore(tmp_path)
    await collect(AgentLoop(BrokenBackend(), store, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    messages = store.messages()
    assert messages[-1].content == []
    assert messages[-1].metadata["turn_failed"] is True
    assert store.turn_in_flight() is False


@pytest.mark.asyncio
async def test_failed_child_returns_error_and_sibling_survives(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    events = await collect(
        AgentLoop(ParallelChildFailureBackend(), store, skill_catalog=SkillCatalog.empty()).run_turn("delegate")
    )

    assert events[-1].type is StreamEventType.AGENT_END
    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert [result.tool_call_id for result in results] == [
        "child-dies",
        "child-lives",
    ]
    assert results[0].is_error
    assert "child connection dropped" in results[0].content
    assert results[1].content.startswith("sibling complete")
    assert store.messages()[-1].content[0].text == "parent survived"
    child_path = Path(results[0].structured_content["child_session_path"])
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    assert child_store.messages()[-1].metadata["turn_failed"] is True


@pytest.mark.asyncio
async def test_child_setup_failure_does_not_cancel_parallel_sibling(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(ParallelChildFailureBackend(), store, skill_catalog=SkillCatalog.empty())
    original_ensure = AgentLoop._ensure_mcp_servers
    child_ids: list[str] = []

    async def fail_first_child(self: AgentLoop) -> None:
        child_ids.append(self.store.session_id)
        if self.store.session_id == "1":
            raise httpx.ConnectError(
                "child setup disconnected",
                request=httpx.Request("GET", "https://test.invalid"),
            )
        await original_ensure(self)

    with patch.object(AgentLoop, "_ensure_mcp_servers", fail_first_child):
        await collect(loop.run_turn("delegate"))

    assert "1" in child_ids
    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    ]
    assert results[0] is not None and results[0].is_error
    assert "child setup disconnected" in results[0].content
    assert results[1] is not None and results[1].content.startswith("sibling complete")
    child_path = Path(results[0].structured_content["child_session_path"])
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    assert child_store.turn_in_flight() is False


@pytest.mark.asyncio
async def test_thinking_only_nudge_runs_even_at_max_turns(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([ThinkingContent("planning", "sig-1")], stop_reason="end_turn"),
            ScriptedTurn([TextContent("visible answer")], stop_reason="end_turn"),
        ]
    )
    store = ConversationStore(tmp_path / "recovers")

    events = await collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("hi")
    )

    assert len(backend.calls) == 2
    assert events[-1].type is StreamEventType.AGENT_END
    assert store.turn_in_flight() is False
    messages = store.messages()
    nudge_index = next(
        index
        for index, message in enumerate(messages)
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    )
    assert messages[nudge_index + 1].role is MessageRole.ASSISTANT
    assert messages[-1].content == [TextContent("visible answer")]

    still_empty = FakeBackend(
        [
            ScriptedTurn([ThinkingContent("planning", "sig-1")], stop_reason="end_turn"),
            ScriptedTurn(
                [ThinkingContent("still planning", "sig-2")],
                stop_reason="end_turn",
            ),
            ScriptedTurn([TextContent("must not run")], stop_reason="end_turn"),
        ]
    )
    empty_store = ConversationStore(tmp_path / "still-empty")

    empty_events = await collect(
        AgentLoop(
            still_empty,
            empty_store,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("hi")
    )

    assert len(still_empty.calls) == 2
    assert empty_events[-1].type is StreamEventType.AGENT_END
    assert empty_store.turn_in_flight() is False


@pytest.mark.asyncio
async def test_thinking_only_reply_is_nudged_once_and_recovers(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([ThinkingContent("planning", "sig-1")], stop_reason="end_turn"),
            ScriptedTurn([TextContent("here is the answer")], stop_reason="end_turn"),
        ]
    )
    store = ConversationStore(tmp_path)

    await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi")
    )

    # The thinking-only reply triggered exactly one extra completion.
    assert len(backend.calls) == 2
    nudges = [
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    ]
    assert len(nudges) == 1
    assert nudges[0].role is MessageRole.USER
    # The nudged completion delivered the visible reply as the final message.
    assert store.messages()[-1].content == [TextContent("here is the answer")]


@pytest.mark.asyncio
async def test_thinking_only_reply_nudged_at_most_once_per_turn(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([ThinkingContent("planning", "sig-1")], stop_reason="end_turn"),
            ScriptedTurn([ThinkingContent("still thinking", "sig-2")], stop_reason="end_turn"),
        ]
    )
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi")
    )

    # A second thinking-only reply is not nudged again: no unbounded loop.
    assert len(backend.calls) == 2
    nudges = [
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    ]
    assert len(nudges) == 1
    assert events[-1].type is StreamEventType.AGENT_END


@pytest.mark.asyncio
async def test_thinking_only_max_tokens_is_not_nudged(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([ThinkingContent("planning", "sig-1")], stop_reason="max_tokens")]
    )
    store = ConversationStore(tmp_path)

    events = await collect(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()).run_turn("hi")
    )

    # Truncated at the output-token limit: surfaced via metadata, not nudged.
    assert len(backend.calls) == 1
    assert not [
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    ]
    assistant = store.messages()[-1]
    assert assistant.role is MessageRole.ASSISTANT
    assert assistant.metadata["stop_reason"] == "max_tokens"
    assert events[-1].type is StreamEventType.AGENT_END
