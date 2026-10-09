import json
from pathlib import Path
from typing import Any

import pytest

from zeta.agent.tool_results import validated_tool_result
from zeta.context_accounting import message_token_count
from zeta.context_eviction import evict_messages as _real_evict_messages
from zeta.context_eviction import eviction_view
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ContentBlock,
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


def _source_messages(records: list[tuple[int, Message]]) -> dict[int, Message]:
    sources: dict[int, Message] = {}
    for seq, message in records:
        if any(isinstance(block, ToolUseContent) for block in message.content):
            sources[seq] = message
        elif message.tool_result is not None:
            sources[seq] = Message(
                message.role,
                tool_result=ToolResult(
                    message.tool_result.tool_call_id, f"original raw result {seq}"
                ),
            )
        else:
            sources[seq] = Message(
                message.role, [TextContent(f"original raw message {seq}")]
            )
    return sources


def _evict_messages(
    records: list[tuple[int, Message]],
    **kwargs: Any,
):
    kwargs.setdefault("source_messages", _source_messages(records))
    return _real_evict_messages(records, **kwargs)


def _sources_with_tool_result(
    records: list[tuple[int, Message]], seq: int, content: str
) -> dict[int, Message]:
    sources = _source_messages(records)
    result = dict(records)[seq].tool_result
    assert result is not None
    sources[seq] = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult(result.tool_call_id, content),
    )
    return sources


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

    result = _evict_messages(records, fixed_tokens=0, target_tokens=1)

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


def _legacy_tool_receipt_pair(
    seq: int, content: str, *, name: str = "read"
) -> list[tuple[int, Message]]:
    call_id = f"legacy-{seq}"
    return [
        (
            seq,
            Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(ToolCall(call_id, name, {"path": "secret.txt"}))],
            ),
        ),
        (
            seq + 1,
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult(call_id, content),
                metadata={
                    "context_evicted": True,
                    "source_seq": seq + 1,
                    "eviction_content_digest": "0" * 64,
                },
            ),
        ),
    ]


def test_legacy_semantic_digest_with_appended_raw_text_breaks_range() -> None:
    records = [
        *_legacy_tool_receipt_pair(
            1, "[semantic read digest · seq 2] RAW SECRET"
        ),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages=_sources_with_tool_result(records, 2, "RAW SECRET"),
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)
    assert result.messages[1].tool_result is not None
    assert result.messages[1].tool_result.content.endswith("RAW SECRET")


def test_legacy_workflow_receipt_with_appended_raw_text_breaks_range() -> None:
    records = [
        *_legacy_tool_receipt_pair(
            1,
            "[workflow result receipt] recall_history seq_start=2, seq_end=2 "
            "for exact result\nRAW SECRET",
            name="recall_history",
        ),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages=_sources_with_tool_result(records, 2, "RAW SECRET"),
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)
    assert result.messages[1].tool_result is not None
    assert result.messages[1].tool_result.content.endswith("RAW SECRET")


def test_legacy_orchestration_receipt_with_appended_raw_text_breaks_range() -> None:
    records = [
        *_legacy_tool_receipt_pair(
            1,
            "[orchestration result receipt] {}"
            "; recall_history seq_start=2, seq_end=2 for exact result\nRAW SECRET",
            name="agent",
        ),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages=_sources_with_tool_result(records, 2, "RAW SECRET"),
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)
    assert result.messages[1].tool_result is not None
    assert result.messages[1].tool_result.content.endswith("RAW SECRET")


def test_legacy_notification_receipt_with_appended_raw_text_breaks_range() -> None:
    malicious = Message(
        MessageRole.SYSTEM,
        [
            TextContent(
                "[notification receipt] recall_history seq_start=1, seq_end=1 "
                "for exact notification\nRAW SECRET"
            )
        ],
        metadata={
            "context_evicted": True,
            "source_seq": 1,
            "eviction_content_digest": "0" * 64,
        },
    )

    result = _evict_messages(
        [(1, malicious), (2, _assistant_receipt(2))],
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages={
            1: Message(MessageRole.SYSTEM, [TextContent("RAW SECRET")]),
            2: Message(MessageRole.ASSISTANT, [TextContent("original answer")]),
        },
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)
    block = result.messages[0].content[0]
    assert isinstance(block, TextContent)
    assert block.text.endswith("RAW SECRET")


def test_provenance_rule_coalesces_legacy_receipts_without_shared_content() -> None:
    records = [
        *_legacy_tool_receipt_pair(
            1,
            "[semantic read digest · seq 2] read secret.txt; 1 lines; safe. "
            "recall_history seq_start=2, seq_end=2 for exact output; "
            "re-read only if the source may have changed.",
        ),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert sum(bool(message.metadata.get("eviction_range")) for message in result.messages) == 1


@pytest.mark.parametrize(
    ("role", "view", "source"),
    [
        (
            MessageRole.ASSISTANT,
            [TextContent("[assistant receipt] RAW SECRET")],
            [TextContent("RAW SECRET")],
        ),
        (
            MessageRole.SYSTEM,
            [TextContent("[notification receipt] RAW SECRET")],
            [TextContent("RAW SECRET")],
        ),
        (
            MessageRole.ASSISTANT,
            [ImageContent("aW1hZ2U=", "image/png"), TextContent("receipt")],
            [ImageContent("aW1hZ2U=", "image/png")],
        ),
        (
            MessageRole.ASSISTANT,
            [ThinkingContent("private reasoning"), TextContent("receipt")],
            [ThinkingContent("private reasoning")],
        ),
    ],
)
def test_view_item_sharing_any_source_content_is_a_boundary(
    role: MessageRole, view: list[ContentBlock], source: list[ContentBlock]
) -> None:
    records = [
        (1, Message(role, view, metadata={"context_evicted": True, "source_seq": 1})),
        (2, _assistant_receipt(2)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages={
            1: Message(role, source),
            2: Message(MessageRole.ASSISTANT, [TextContent("original answer")]),
        },
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)


@pytest.mark.parametrize("kind", ["assistant", "notification", "tool_result"])
def test_spoofed_marker_with_raw_text_is_not_coalesced(kind: str) -> None:
    role = MessageRole.SYSTEM if kind == "notification" else MessageRole.ASSISTANT
    raw = Message(
        role,
        [TextContent("RAW SECRET")],
        metadata={
            "context_evicted": True,
            "eviction_receipt": kind,
            "source_seq": 1,
        },
    )
    records = [(1, raw), (2, _assistant_receipt(2))]
    sources = {
        1: Message(role, [TextContent("RAW SECRET")]),
        2: Message(MessageRole.ASSISTANT, [TextContent("original answer")]),
    }
    if kind == "tool_result":
        call = Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("raw-1", "read", {"path": "secret"}))],
        )
        raw = Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult("raw-1", "RAW SECRET"),
            metadata={
                "context_evicted": True,
                "eviction_receipt": kind,
                "source_seq": 2,
            },
        )
        records = [(1, call), (2, raw), (3, _assistant_receipt(3))]
        sources = {
            1: call,
            2: Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult("raw-1", "RAW SECRET"),
            ),
            3: Message(MessageRole.ASSISTANT, [TextContent("original answer")]),
        }

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages=sources,
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)


def test_inbox_title_beyond_constructor_limit_is_not_coalesced() -> None:
    title = "x" * 81
    records = [
        *_legacy_tool_receipt_pair(
            1,
            f"[workflow result receipt] title={title}; "
            "recall_history seq_start=2, seq_end=2 for exact result",
            name="inbox",
        ),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
        source_messages=_sources_with_tool_result(records, 2, title),
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)


def test_rewritten_tool_argument_sharing_source_content_is_a_boundary() -> None:
    source_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("agent-1", "agent", {"prompt": "RAW PROMPT"}))],
    )
    view_call = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(
                ToolCall("agent-1", "agent", {"prompt": "[receipt] RAW PROMPT"})
            )
        ],
        metadata={"context_evicted": True, "source_seq": 1},
    )
    records = [
        (1, view_call),
        (2, _tool_receipt_pair(1, "agent", "agent-1")[1][1]),
        (3, _assistant_receipt(3)),
    ]

    result = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        source_messages={
            1: source_call,
            2: Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult("agent-1", "original output"),
            ),
            3: Message(MessageRole.ASSISTANT, [TextContent("original answer")]),
        },
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)


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

    first = _evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=2,
        token_counter=lambda message: (
            100
            if any(isinstance(block, ThinkingContent) for block in message.content)
            else 1
        ),
        source_messages=dict(records),
    )
    replay = [
        (int(message.metadata["source_seq"]), message) for message in first.messages
    ]
    second = _evict_messages(
        replay,
        fixed_tokens=0,
        target_tokens=1,
        source_messages=dict(records),
    )

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

    result = _evict_messages(records, fixed_tokens=0, target_tokens=1)

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

    messages = _evict_messages(
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
    result = _evict_messages(records, fixed_tokens=0, target_tokens=1)
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


@pytest.mark.asyncio
async def test_range_receipt_keeps_original_tool_arguments_recallable(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    source_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("agent-1", "agent", {"prompt": "RAW PROMPT"}))],
    )
    source_result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("agent-1", "RAW OUTPUT"),
    )
    source_answer = Message(MessageRole.ASSISTANT, [TextContent("RAW ANSWER")])
    sources = [source_call, source_result, source_answer]
    entries = [store.append_message(message) for message in sources]
    view_call = Message(
        MessageRole.ASSISTANT,
        [
            ToolUseContent(
                ToolCall(
                    "agent-1",
                    "agent",
                    {"prompt": "[agent prompt receipt] recall_history for exact prompt"},
                )
            )
        ],
        metadata={"context_evicted": True, "source_seq": entries[0].seq},
    )
    records = [
        (entries[0].seq, view_call),
        (entries[1].seq, _tool_receipt_pair(1, "agent", "agent-1")[1][1]),
        (entries[2].seq, _assistant_receipt(entries[2].seq)),
    ]
    result = _real_evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        source_messages={
            entry.seq: message
            for entry, message in zip(entries, sources, strict=True)
        },
    )
    assert any(message.metadata.get("eviction_range") for message in result.messages)
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

    assert "RAW PROMPT" in recalled
    assert "RAW OUTPUT" in recalled


def test_range_receipts_are_stable_as_history_grows() -> None:
    initial = [(seq, _assistant_receipt(seq)) for seq in range(1, 7)]
    first = _evict_messages(initial, fixed_tokens=0, target_tokens=1)
    first_range = next(
        message.to_dict()
        for message in first.messages
        if message.metadata.get("eviction_range")
    )
    replay = [
        (int(message.metadata["source_seq"]), message) for message in first.messages
    ]
    replay.extend((seq, _assistant_receipt(seq)) for seq in range(7, 11))

    grown = _evict_messages(replay, fixed_tokens=0, target_tokens=1)

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

    result = _evict_messages(records, fixed_tokens=0, target_tokens=500)

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

    evicted = _evict_messages(
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

    range_evicted = _evict_messages(
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
