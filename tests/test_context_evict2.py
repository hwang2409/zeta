from pathlib import Path

import pytest

from zeta.context_strategies import context_strategies
from zeta.context_strategies.evict2 import evict2_messages
from zeta.core.context import CompactionPolicy, ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def tool_pair(
    name: str,
    call_id: str,
    output: str,
    *,
    arguments: dict[str, object] | None = None,
    error: bool = False,
) -> tuple[Message, Message]:
    call = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(
                ToolCall(call_id, name, arguments or {"path": "MAINTAINERS.md"})
            )
        ],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult(call_id, output, is_error=error),
    )
    return call, result


def rendered_text(messages: list[Message]) -> str:
    values: list[str] = []
    for message in messages:
        values.extend(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        if message.tool_result is not None:
            values.append(message.tool_result.content)
    return "\n".join(values)


def assert_payload_pairing(messages: list[Message]) -> None:
    anthropic = build_messages_payload(
        messages, [], model="claude-test", max_tokens=2048, thinking_budget=1024
    )["messages"]
    assert {
        block["id"]
        for message in anthropic
        for block in message["content"]
        if block["type"] == "tool_use"
    } == {
        block["tool_use_id"]
        for message in anthropic
        for block in message["content"]
        if block["type"] == "tool_result"
    }

    codex = build_responses_payload(messages, [], model="gpt-test")["input"]
    assert {
        item["call_id"] for item in codex if item.get("type") == "function_call"
    } == {
        item["call_id"]
        for item in codex
        if item.get("type") == "function_call_output"
    }

    ollama = build_ollama_messages(messages)
    assert {
        call["function"]["name"]
        for message in ollama
        for call in message.get("tool_calls", [])
    } == {
        message["tool_name"] for message in ollama if message["role"] == "tool"
    }


def test_evict2_flags_are_recognized() -> None:
    assert context_strategies("evict2,evict2sum,recall") == {
        "evict2",
        "evict2sum",
        "recall",
    }


def test_evict2_digest_is_deterministic_and_retains_load_bearing_lines() -> None:
    output = "\n".join(
        [
            "# Maintainers",
            "first useful line",
            *(f"unimportant filler {index}" for index in range(80)),
            "MUST run the complete validation suite",
            "Do not force-push this branch",
            "last useful line",
        ]
    )
    call, result = tool_pair("read", "read-1", output)
    records = [(10, call), (11, result)]

    first = evict2_messages(
        records, fixed_tokens=0, target_tokens=1, recall_enabled=True
    )
    second = evict2_messages(
        records, fixed_tokens=0, target_tokens=1, recall_enabled=True
    )

    assert [message.to_dict() for message in first.messages] == [
        message.to_dict() for message in second.messages
    ]
    digest = first.messages[1].tool_result
    assert digest is not None
    assert "MAINTAINERS.md" in digest.content
    assert "85 lines" in digest.content
    assert "# Maintainers" in digest.content
    assert "MUST run the complete validation suite" in digest.content
    assert "Do not force-push this branch" in digest.content
    assert "last useful line" in digest.content
    assert "recall_history seq_start=11" in digest.content
    assert "re-read only if the file may have changed" in digest.content
    assert len(digest.content) <= 500


def test_evict2_deduplicates_repeated_unchanged_reads() -> None:
    first_call, first_result = tool_pair("read", "read-1", "same output\n" * 300)
    second_call, second_result = tool_pair("read", "read-2", "same output\n" * 300)
    records = [
        (1, first_call),
        (2, first_result),
        (3, second_call),
        (4, second_result),
    ]

    result = evict2_messages(
        records, fixed_tokens=0, target_tokens=250, recall_enabled=False
    )
    output = rendered_text(result.messages)

    assert output.count("MAINTAINERS.md") == 1
    assert "read 2 times" in output
    assert "re-run" not in output
    assert_payload_pairing(result.messages)


def test_evict2_keeps_parallel_pairing_for_every_provider() -> None:
    calls = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(ToolCall("read-a", "read", {"path": "a.md"})),
            ToolUseContent(ToolCall("read-b", "read", {"path": "b.md"})),
        ],
    )
    records = [
        (1, calls),
        (2, Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("read-a", "a" * 4000))),
        (3, Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("read-b", "b" * 4000))),
    ]

    result = evict2_messages(
        records, fixed_tokens=0, target_tokens=100, recall_enabled=False
    )

    assert_payload_pairing(result.messages)


class DecisionPolicy(CompactionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[list[Message]] = []

    async def summarize_chunked(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.inputs.append(list(messages))
        return "Decision: use SQLite for durable state."


@pytest.mark.asyncio
async def test_evict2_has_hysteresis_no_churn_and_replays_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict2,recall")
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="evict2")
    store.append_message(text(MessageRole.USER, "early requirement"))
    call, result = tool_pair("read", "read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    store.append_message(text(MessageRole.ASSISTANT, "old reasoning " * 500))
    store.append_message(text(MessageRole.USER, "latest request verbatim"))
    assembler = ContextAssembler(store, token_budget=700, retained_tail=1)

    assembled = [await assembler.assemble_context() for _ in range(4)]

    markers = [entry for entry in store.replay() if entry.type == "compaction"]
    assert len(markers) == 1
    assert markers[0].data["kind"] == "eviction2"
    assert {context.digest for context in assembled} == {assembled[0].digest}
    assert "latest request verbatim" in rendered_text(assembled[0].messages)
    assert_payload_pairing(assembled[0].messages)

    reopened = ConversationStore(sessions, session_id="evict2")
    replayed = await ContextAssembler(
        reopened, token_budget=700, retained_tail=1
    ).assemble_context()
    assert replayed.digest == assembled[0].digest
    assert [message.to_dict() for message in replayed.messages] == [
        message.to_dict() for message in assembled[0].messages
    ]
    assert len([entry for entry in reopened.replay() if entry.type == "compaction"]) == 1


@pytest.mark.asyncio
async def test_evict2sum_summarizes_only_assistant_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict2sum,recall")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "choose storage"))
    call, result = tool_pair("read", "read-1", "tool output " * 1200)
    store.append_message(call)
    store.append_message(result)
    store.append_message(
        text(MessageRole.ASSISTANT, "I decided to use SQLite for durable state. " * 300)
    )
    store.append_message(text(MessageRole.USER, "continue"))
    policy = DecisionPolicy()

    assembled = await ContextAssembler(
        store,
        token_budget=700,
        retained_tail=1,
        compaction_policy=policy,
    ).assemble()

    assert len(policy.inputs) == 1
    assert policy.inputs[0]
    assert all(message.role is MessageRole.ASSISTANT for message in policy.inputs[0])
    assert all(message.tool_result is None for message in policy.inputs[0])
    assert "Decision: use SQLite for durable state." in rendered_text(assembled)
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["summary"] == "Decision: use SQLite for durable state."


@pytest.mark.asyncio
async def test_evict2sum_fallback_survives_empty_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "evict2sum,recall")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "choose storage"))
    call, result = tool_pair("read", "read-1", "tool output " * 1200)
    store.append_message(call)
    store.append_message(result)
    store.append_message(
        text(MessageRole.ASSISTANT, "I decided to use SQLite. " * 300)
    )
    store.append_message(text(MessageRole.USER, "continue"))
    telemetry: list[dict[str, object]] = []
    backend = FakeBackend([ScriptedTurn([TextContent(" ")])] * 30)

    assembled = await ContextAssembler(
        store,
        token_budget=700,
        retained_tail=1,
        backend=backend,
        telemetry_sink=telemetry.append,
    ).assemble()

    assert "[automatic fallback summary:" in rendered_text(assembled)
    assert len(backend.calls) >= 3
    assert telemetry[-1]["fallback_count"] >= 1
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["kind"] == "eviction2sum"
    assert marker.data["summary"].startswith("[automatic fallback summary:")
