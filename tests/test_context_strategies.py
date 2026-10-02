import json
from pathlib import Path

import pytest

from zeta.core.context import ContextAssembler
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
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def result_text(result: dict[str, object]) -> str:
    content = result["content"]
    assert isinstance(content, list)
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


@pytest.mark.asyncio
async def test_default_strategy_matches_pre_strategy_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZETA_CONTEXT_STRATEGY", raising=False)
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "hello"))
    store.append_message(text(MessageRole.ASSISTANT, "hi"))

    assembled = await ContextAssembler(store, system_prompt="system").assemble()

    assert [message.to_dict() for message in assembled] == [
        text(MessageRole.SYSTEM, "system").to_dict(),
        text(MessageRole.USER, "hello").to_dict(),
        text(MessageRole.ASSISTANT, "hi").to_dict(),
    ]


def test_recall_tool_is_not_registered_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZETA_CONTEXT_STRATEGY", raising=False)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    assert "recall_history" not in registry.registered_names


@pytest.mark.asyncio
async def test_recall_range_returns_exact_hidden_structured_messages_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "recall")
    store = ConversationStore(tmp_path)
    user = store.append_message(text(MessageRole.USER, "needle alpha"))
    call = ToolCall("call-1", "read", {"path": "README.md"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    tool_result = store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("ignored duplicate")],
            tool_result=ToolResult(call.id, "exact tool output"),
        )
    )
    store.append_compaction_marker("summary", user.seq, tool_result.seq)
    before = store.path.read_bytes()
    registry = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )

    output = await registry.execute(
        ToolCall(
            "recall-1",
            "recall_history",
            {"seq_start": user.seq, "seq_end": tool_result.seq},
        ),
        _skip_approval=True,
    )

    rendered = result_text(output)
    assert f"seq {user.seq}" in rendered
    assert '"role":"user"' in rendered
    assert '"id":"call-1","name":"read"' in rendered
    assert '"args":{"path":"README.md"}' in rendered
    assert '"tool_result":{"tool_call_id":"call-1"' in rendered
    assert '"content":"exact tool output"' in rendered
    assert store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_recall_range_truncation_has_usable_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "recall")
    store = ConversationStore(tmp_path)
    first = store.append_message(text(MessageRole.USER, "a" * 30))
    second = store.append_message(text(MessageRole.ASSISTANT, "second " * 20))
    store.append_compaction_marker("summary", first.seq, second.seq)
    registry = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )

    output = await registry.execute(
        ToolCall(
            "recall-truncated",
            "recall_history",
            {"seq_start": first.seq, "seq_end": second.seq, "max_chars": 150},
        ),
        _skip_approval=True,
    )

    rendered = result_text(output)
    assert len(rendered) <= 150
    assert "[truncated;" in rendered
    assert f"seq_start={second.seq}, seq_end={second.seq}" in rendered


@pytest.mark.asyncio
async def test_recall_search_only_finds_hidden_active_branch_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "recall")
    store = ConversationStore(tmp_path)
    root = store.append_message(text(MessageRole.USER, "root"))
    store.append_message(text(MessageRole.ASSISTANT, "inactive forbidden secret"))
    store.append_message_fork(root.id)
    active = store.append_message(text(MessageRole.USER, "active searchable needle"))
    store.append_compaction_marker("active summary", active.seq, active.seq)
    registry = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty()
    )

    found = await registry.execute(
        ToolCall("recall-2", "recall_history", {"query": "searchable needle"}),
        _skip_approval=True,
    )
    absent = await registry.execute(
        ToolCall("recall-3", "recall_history", {"query": "forbidden secret"}),
        _skip_approval=True,
    )

    assert f"seq {active.seq}" in result_text(found)
    assert "active searchable needle" in result_text(found)
    assert "inactive forbidden secret" not in result_text(found)
    assert "inactive forbidden secret" not in result_text(absent)


@pytest.mark.asyncio
async def test_recall_header_is_only_enabled_with_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(text(MessageRole.USER, "old"))
    store.append_compaction_marker("summary", first.seq, first.seq)

    monkeypatch.delenv("ZETA_CONTEXT_STRATEGY", raising=False)
    default_messages = await ContextAssembler(store).assemble()
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "recall")
    recall_messages = await ContextAssembler(store).assemble()

    default_summary = default_messages[1].content[0]
    recall_summary = recall_messages[1].content[0]
    assert isinstance(default_summary, TextContent)
    assert isinstance(recall_summary, TextContent)
    assert default_summary.text == "summary"
    assert recall_summary.text.startswith(
        f"[compacted history seq {first.seq}–{first.seq}; "
        "use recall_history to retrieve exact messages]\n"
    )


@pytest.mark.asyncio
async def test_budget_readout_is_tail_only_not_persisted_and_payloads_pair_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "budget")
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "inspect"))
    call = ToolCall("call-1", "read", {"path": "README.md"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(call.id, "done"),
        )
    )
    before = store.path.read_bytes()

    assembled = await ContextAssembler(
        store, system_prompt="large " * 200, token_budget=10_000
    ).assemble()

    assert store.path.read_bytes() == before
    assert assembled[-1].role is MessageRole.USER
    tail = assembled[-1].content[0]
    assert isinstance(tail, TextContent)
    assert tail.text.startswith("[context budget]")
    assert "? system" in tail.text
    assert all(
        "[context budget]" not in block.text
        for message in assembled[:-1]
        for block in message.content
        if isinstance(block, TextContent)
    )

    base_messages = assembled[:-1]
    anthropic_base = build_messages_payload(
        base_messages, [], model="claude-test", max_tokens=10_000, thinking_budget=1024
    )
    anthropic = build_messages_payload(
        assembled, [], model="claude-test", max_tokens=10_000, thinking_budget=1024
    )
    anthropic_messages = anthropic["messages"]
    tool_use_index = next(
        index
        for index, message in enumerate(anthropic_messages)
        if any(block.get("type") == "tool_use" for block in message["content"])
    )
    following = anthropic_messages[tool_use_index + 1]
    assert following["role"] == "user"
    assert [block["type"] for block in following["content"]] == [
        "tool_result",
        "text",
    ]
    assert anthropic_messages[:-1] == anthropic_base["messages"][:-1]
    assert following["content"][:-1] == anthropic_base["messages"][-1]["content"]

    codex_base = build_responses_payload(base_messages, [], model="gpt-test")
    codex = build_responses_payload(assembled, [], model="gpt-test")
    kinds = [item.get("type", item.get("role")) for item in codex["input"]]
    assert kinds[-2:] == ["function_call_output", "user"]
    assert codex["input"][:-1] == codex_base["input"]
    assert codex["input"][-1]["content"][0]["text"].startswith("[context budget]")

    ollama = build_ollama_messages(assembled)
    assert [message["role"] for message in ollama[-2:]] == ["tool", "user"]
    assert ollama[-1]["content"].startswith("[context budget]")


@pytest.mark.asyncio
async def test_telemetry_written_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disabled = tmp_path / "disabled.jsonl"
    monkeypatch.delenv("ZETA_CONTEXT_TELEMETRY", raising=False)
    store = ConversationStore(tmp_path / "off")
    store.append_message(text(MessageRole.USER, "hello"))
    await ContextAssembler(store).assemble()
    assert not disabled.exists()

    telemetry = tmp_path / "events.jsonl"
    monkeypatch.setenv("ZETA_CONTEXT_TELEMETRY", str(telemetry))
    monkeypatch.setenv("ZETA_CONTEXT_STRATEGY", "recall,budget")
    enabled = ConversationStore(tmp_path / "on")
    enabled.append_message(text(MessageRole.USER, "old"))
    enabled.append_message(text(MessageRole.USER, "tail"))
    assembler = ContextAssembler(
        enabled,
        token_budget=40,
        retained_tail=1,
        token_counter=lambda message: (
            30 if message.role is MessageRole.USER else 1
        ),
        backend=FakeBackend([ScriptedTurn([TextContent("summary")])]),
    )
    await assembler.assemble()

    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    request = next(event for event in events if event["event"] == "request")
    compaction = next(event for event in events if event["event"] == "compaction")
    assert request["strategy"] == "budget,recall"
    assert request["compaction"] is True
    assert request["budget_readout"] is True
    assert request["budget"] == 40
    assert request["est_tokens"] > 0
    assert compaction["source_seq_start"] == 1
    assert compaction["source_seq_end"] == 1
    assert compaction["summary_chars"] == len("summary")
    assert compaction["duration_s"] >= 0
    assert compaction["map_calls"] == 0

    registry = ToolRegistry(
        tmp_path, session_store=enabled, skill_catalog=SkillCatalog.empty()
    )
    await registry.execute(
        ToolCall("recall-telemetry", "recall_history", {"seq_start": 1, "seq_end": 1}),
        _skip_approval=True,
    )
    recall = [json.loads(line) for line in telemetry.read_text().splitlines()][-1]
    assert recall["event"] == "recall_history"
    assert recall["mode"] == "range"
    assert recall["chars"] > 0
