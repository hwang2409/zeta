import json
from pathlib import Path

import pytest

from zeta.agent.tool_results import validated_tool_result
from zeta.context_accounting import message_token_count
from zeta.context_eviction import evict_messages, eviction_view
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ImageContent,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def _assistant_receipt(seq: int, kind: str = "assistant text") -> Message:
    return Message(
        MessageRole.ASSISTANT,
        [TextContent(f"[{kind} evicted · seq {seq}]")],
        metadata={
            "context_evicted": True,
            "eviction_receipt": "assistant",
            "source_seq": seq,
        },
    )


def _notification_receipt(seq: int) -> Message:
    return Message(
        MessageRole.SYSTEM,
        [TextContent(f"[notification receipt] seq {seq}")],
        metadata={
            "context_evicted": True,
            "eviction_receipt": "notification",
            "source_seq": seq,
            "zeta_event": "agent_notifications",
            "notifications": [],
        },
    )


def _tool_receipt_pair(seq: int, name: str, call_id: str) -> list[tuple[int, Message]]:
    call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall(call_id, name, {"path": f"{call_id}.txt"}))],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult(
            call_id,
            f"[semantic {name} digest · seq {seq + 1}] exact receipt",
        ),
        metadata={
            "context_evicted": True,
            "eviction_receipt": "tool_result",
            "source_seq": seq + 1,
        },
    )
    return [(seq, call), (seq + 1, result)]


def _range_text(message: Message) -> str:
    assert message.role is MessageRole.ASSISTANT
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


def test_range_receipt_coalesces_runs_of_old_receipts() -> None:
    records = [
        (10, _notification_receipt(10)),
        (11, _notification_receipt(11)),
        (12, _assistant_receipt(12, "assistant reasoning")),
        *_tool_receipt_pair(13, "read", "read-1"),
        (15, _notification_receipt(15)),
        (16, _notification_receipt(16)),
        (17, _notification_receipt(17)),
        *_tool_receipt_pair(18, "bash", "bash-1"),
    ]

    result = evict_messages(records, fixed_tokens=0, target_tokens=1)

    ranges = [message for message in result.messages if message.metadata.get("eviction_range")]
    assert len(ranges) == 1
    receipt = _range_text(ranges[0])
    assert "[evicted range] seq 10-14:" in receipt
    assert "1 tool result (read 1)" in receipt
    assert "2 notifications" in receipt
    assert "1 assistant note" in receipt
    assert receipt.endswith(
        "recall_history seq_start=10, seq_end=14 for exact content"
    )
    assert "bash" not in receipt  # Protected bash-call tail remains outside the range.


def test_reasoning_evicted_messages_with_raw_text_or_images_break_ranges() -> None:
    records = [
        (
            1,
            Message(
                MessageRole.ASSISTANT,
                [ThinkingContent("private reasoning"), TextContent("keep this answer")],
            ),
        ),
        (
            2,
            Message(
                MessageRole.ASSISTANT,
                [ThinkingContent("more reasoning"), ImageContent("aW1hZ2U=", "image/png")],
            ),
        ),
    ]

    first = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=2,
        token_counter=lambda message: (
            100
            if any(isinstance(block, ThinkingContent) for block in message.content)
            else 1
        ),
    )
    replay = [
        (int(message.metadata["source_seq"]), message) for message in first.messages
    ]
    second = evict_messages(replay, fixed_tokens=0, target_tokens=1)

    assert not any(message.metadata.get("eviction_range") for message in second.messages)
    assert [message.to_dict() for message in second.messages] == [
        message.to_dict() for message in first.messages
    ]
    assert any(
        isinstance(block, TextContent) and block.text == "keep this answer"
        for block in second.messages[0].content
    )
    assert any(
        isinstance(block, ImageContent) for block in second.messages[1].content
    )


def test_range_receipt_never_coalesces_user_messages() -> None:
    user = with_message_origin(
        Message(MessageRole.USER, [TextContent("keep my exact words")]),
        MessageOrigin.USER,
    )
    records = [
        (1, _assistant_receipt(1)),
        (2, _assistant_receipt(2)),
        (3, user),
        (4, _assistant_receipt(4)),
        (5, _assistant_receipt(5)),
    ]

    result = evict_messages(records, fixed_tokens=0, target_tokens=1)

    assert sum(bool(message.metadata.get("eviction_range")) for message in result.messages) == 2
    kept_user = next(message for message in result.messages if message.role is MessageRole.USER)
    assert kept_user.to_dict() == user.to_dict()


def test_range_receipt_keeps_provider_payload_valid() -> None:
    records = [
        *_tool_receipt_pair(1, "read", "read-1"),
        *_tool_receipt_pair(3, "read", "read-2"),
        *_tool_receipt_pair(5, "search", "search-1"),
        *_tool_receipt_pair(7, "search", "protected-1"),
    ]

    messages = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        unconsumed_source_seqs={8},
    ).messages

    assert len(messages) == 3
    call_ids = {
        block.tool_call.id
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
    result_ids = {
        message.tool_result.tool_call_id
        for message in messages
        if message.tool_result is not None
    }
    assert call_ids == result_ids == {"protected-1"}
    assert build_messages_payload(
        messages, [], model="claude-test", max_tokens=2048, thinking_budget=1024
    )["messages"]
    assert build_responses_payload(messages, [], model="gpt-test")["input"]
    assert build_ollama_messages(messages)


@pytest.mark.asyncio
async def test_range_receipt_recall_returns_exact_originals(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    originals = [_assistant_receipt(seq) for seq in range(1, 5)]
    entries = [store.append_message(message) for message in originals]
    records = [(entry.seq, message) for entry, message in zip(entries, originals, strict=True)]
    result = evict_messages(records, fixed_tokens=0, target_tokens=1)
    assert sum(bool(message.metadata.get("eviction_range")) for message in result.messages) == 1
    store.append_compaction_marker(
        "evicted",
        entries[0].seq,
        entries[-1].seq,
        kind="evict",
        view=eviction_view(records, result),
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        call = ToolCall(
            "recall-range",
            "recall_history",
            {"seq_start": entries[0].seq, "seq_end": entries[-1].seq},
        )
        recalled = validated_tool_result(await registry.execute(call), call.id).content
    finally:
        await registry.close()
        store.close()

    for entry, original in zip(entries, originals, strict=True):
        encoded = json.dumps(original.to_dict(), sort_keys=True, separators=(",", ":"))
        assert f"seq {entry.seq}: {encoded}" in recalled


def test_range_receipts_are_stable_as_history_grows() -> None:
    initial = [(seq, _assistant_receipt(seq)) for seq in range(1, 7)]
    first = evict_messages(initial, fixed_tokens=0, target_tokens=1)
    first_range = next(
        message.to_dict()
        for message in first.messages
        if message.metadata.get("eviction_range")
    )
    replay = [
        (int(message.metadata["source_seq"]), message) for message in first.messages
    ]
    replay.extend((seq, _assistant_receipt(seq)) for seq in range(7, 11))

    grown = evict_messages(replay, fixed_tokens=0, target_tokens=1)

    ranges = [
        message.to_dict()
        for message in grown.messages
        if message.metadata.get("eviction_range")
    ]
    assert ranges[0] == first_range
    assert len(ranges) == 2


def test_long_session_eviction_frees_target_with_range_receipts() -> None:
    records = [(seq, _assistant_receipt(seq)) for seq in range(1, 4_001)]
    before = sum(message_token_count(message) for _, message in records)

    result = evict_messages(records, fixed_tokens=0, target_tokens=500)

    assert before > 20_000
    assert result.reached_target is True
    assert result.tokens_after <= 500
    assert result.tokens_after == sum(message_token_count(message) for message in result.messages)
    assert sum(bool(message.metadata.get("eviction_range")) for message in result.messages) == 1


def test_all_eviction_candidates_use_supplied_counter() -> None:
    call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("read-1", "read", {"path": "tiny.txt"}))],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("read-1", "tiny"),
    )
    records = [(1, call), (2, result)]

    def adversarial_counter(message: Message) -> int:
        return 100_001 if message.metadata.get("context_evicted") else 2

    evicted = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=adversarial_counter,
    )

    assert [message.to_dict() for message in evicted.messages] == [
        message.to_dict() for _, message in records
    ]
    assert evicted.tokens_before == evicted.tokens_after == 4
    assert evicted.items_evicted == 0

    user = Message(MessageRole.USER, [TextContent("break the receipt runs")])
    receipted_records = [
        (1, _assistant_receipt(1)),
        (2, _assistant_receipt(2)),
        (3, user),
        (4, _assistant_receipt(4)),
        (5, _assistant_receipt(5)),
    ]

    def range_adversarial_counter(message: Message) -> int:
        if message.metadata.get("eviction_range"):
            return 1 if message.metadata["source_seq"] == 1 else 100
        return 100 if message.metadata.get("source_seq", 99) <= 2 else 2

    range_evicted = evict_messages(
        receipted_records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=range_adversarial_counter,
    )

    assert len(range_evicted.messages) == 4
    assert range_evicted.messages[0].metadata.get("eviction_range") is True
    assert [message.to_dict() for message in range_evicted.messages[-2:]] == [
        message.to_dict() for _, message in receipted_records[-2:]
    ]
    assert range_evicted.tokens_before == 206
    assert range_evicted.tokens_after == 7
    assert range_evicted.items_evicted == 2
