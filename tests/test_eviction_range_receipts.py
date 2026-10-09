import json
from pathlib import Path

import pytest

from zeta.agent.tool_results import validated_tool_result
from zeta.context_accounting import message_token_count
from zeta.context_eviction import (
    _assistant_receipt as build_assistant_receipt,
)
from zeta.context_eviction import (
    _collapsed_assistant_receipt,
    _notification_receipt_from_fields,
    _semantic_result_receipt,
    _structured_result_receipt,
    evict_messages,
    eviction_view,
)
from zeta.context_eviction.range_receipts import _is_receipt_kind
from zeta.context_eviction.receipt_constructors import _tool_call_receipt
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
    return build_assistant_receipt(seq, kind)


def _notification_receipt(seq: int) -> Message:
    return _notification_receipt_from_fields(
        role=MessageRole.SYSTEM,
        seq=seq,
        payload={"notifications": [], "original_chars": 2, "sha256": "0" * 16},
        content_digest="0" * 64,
    )


def _tool_receipt_pair(seq: int, name: str, call_id: str) -> list[tuple[int, Message]]:
    call = _tool_call_receipt(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall(call_id, name, {"path": f"{call_id}.txt"}))],
        ),
        seq,
    )
    result = _semantic_result_receipt(
        role=MessageRole.TOOL_RESULT,
        tool_name=name,
        tool_call_id=call_id,
        seq=seq + 1,
        digest="exact receipt",
        content_digest="0" * 64,
    )
    return [(seq, call), (seq + 1, result)]


def _range_text(message: Message) -> str:
    assert message.role is MessageRole.ASSISTANT
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


@pytest.mark.parametrize(
    ("message", "kind", "tool_name"),
    [
        (build_assistant_receipt(1, "assistant text"), "assistant", None),
        (build_assistant_receipt(1, "assistant reasoning"), "assistant", None),
        (_collapsed_assistant_receipt(1, 9, older_read=True), "assistant", None),
        (_collapsed_assistant_receipt(1, 9, older_read=False), "assistant", None),
        (_notification_receipt(1), "notification", None),
        (
            _tool_call_receipt(
                Message(
                    MessageRole.ASSISTANT,
                    [ToolUseContent(ToolCall("call-1", "read", {"path": "a.txt"}))],
                ),
                1,
            ),
            "tool_call",
            None,
        ),
        (
            _semantic_result_receipt(
                role=MessageRole.TOOL_RESULT,
                tool_name="read",
                tool_call_id="call-1",
                seq=1,
                digest="read file; safe",
                content_digest="0" * 64,
            ),
            "tool_result",
            "read",
        ),
        (
            _structured_result_receipt(
                role=MessageRole.TOOL_RESULT,
                receipt_kind="workflow",
                tool_name="inbox",
                tool_call_id="call-1",
                seq=1,
                payload={"action": "list", "messages": []},
                content_digest="0" * 64,
            ),
            "tool_result",
            "inbox",
        ),
        (
            _structured_result_receipt(
                role=MessageRole.TOOL_RESULT,
                receipt_kind="orchestration",
                tool_name="agent",
                tool_call_id="call-1",
                seq=1,
                payload={
                    "call": "call-1",
                    "original_chars": 10,
                    "sha256": "0" * 16,
                    "status": "success",
                    "tool": "agent",
                },
                content_digest="0" * 64,
            ),
            "tool_result",
            "agent",
        ),
    ],
)
def test_receipts_round_trip_through_constructors(
    message: Message, kind: str, tool_name: str | None
) -> None:
    assert _is_receipt_kind(message, 1, kind, tool_name=tool_name)


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

    ranges = [
        message for message in result.messages if message.metadata.get("eviction_range")
    ]
    assert len(ranges) == 1
    receipt = _range_text(ranges[0])
    assert "[evicted range] seq 10-17:" in receipt
    assert "1 tool result (read 1)" in receipt
    assert "5 notifications" in receipt
    assert "1 assistant note" in receipt
    assert receipt.endswith("recall_history seq_start=10, seq_end=17 for exact content")
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
        *_legacy_tool_receipt_pair(1, "[semantic read digest · seq 2] RAW SECRET"),
        (3, _assistant_receipt(3)),
    ]

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        message.metadata.get("eviction_range") for message in result.messages
    )
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

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        message.metadata.get("eviction_range") for message in result.messages
    )
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

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        message.metadata.get("eviction_range") for message in result.messages
    )
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

    result = evict_messages(
        [(1, malicious), (2, _assistant_receipt(2))],
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        message.metadata.get("eviction_range") for message in result.messages
    )
    block = result.messages[0].content[0]
    assert isinstance(block, TextContent)
    assert block.text.endswith("RAW SECRET")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (
            "assistant",
            Message(
                MessageRole.ASSISTANT,
                [TextContent("RAW SECRET")],
                metadata={
                    "context_evicted": True,
                    "eviction_receipt": "assistant",
                    "source_seq": 1,
                },
            ),
        ),
        (
            "notification",
            Message(
                MessageRole.SYSTEM,
                [TextContent("RAW SECRET")],
                metadata={
                    "context_evicted": True,
                    "eviction_receipt": "notification",
                    "source_seq": 1,
                },
            ),
        ),
        (
            "tool_result",
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult("raw-1", "RAW SECRET"),
                metadata={
                    "context_evicted": True,
                    "eviction_receipt": "tool_result",
                    "source_seq": 2,
                },
            ),
        ),
    ],
)
def test_spoofed_marker_with_raw_text_is_not_coalesced(
    kind: str, message: Message
) -> None:
    records = (
        [
            (
                1,
                Message(
                    MessageRole.ASSISTANT,
                    [ToolUseContent(ToolCall("raw-1", "read", {"path": "secret"}))],
                ),
            ),
            (2, message),
            (3, _assistant_receipt(3)),
        ]
        if kind == "tool_result"
        else [(1, message), (2, _assistant_receipt(2))]
    )

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda candidate: (
            1 if candidate.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        candidate.metadata.get("eviction_range") for candidate in result.messages
    )
    assert "RAW SECRET" in json.dumps(
        [candidate.to_dict() for candidate in result.messages]
    )


def test_inbox_title_beyond_constructor_limit_is_not_coalesced() -> None:
    title = "x" * 81
    payload = {
        "action": "list",
        "messages": [
            {
                "claimed": False,
                "done": False,
                "id": "message-1",
                "kind": "question",
                "status": "new",
                "title": title,
            }
        ],
    }
    text = (
        "[workflow result receipt] "
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "; recall_history seq_start=2, seq_end=2 for exact result"
    )
    records = [
        *_legacy_tool_receipt_pair(1, text, name="inbox"),
        (3, _assistant_receipt(3)),
    ]

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda candidate: (
            1 if candidate.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(
        candidate.metadata.get("eviction_range") for candidate in result.messages
    )
    assert title in result.messages[1].tool_result.content


def test_only_generated_receipts_coalesce() -> None:
    generated = _assistant_receipt(1)
    unmarked = Message(
        generated.role,
        list(generated.content),
        metadata={
            key: value
            for key, value in generated.metadata.items()
            if key != "eviction_receipt"
        },
    )
    altered = Message(
        generated.role,
        [TextContent("not the constructor output")],
        metadata=generated.metadata,
    )

    assert _is_receipt_kind(generated, 1, "assistant")
    assert not _is_receipt_kind(unmarked, 1, "assistant")
    assert not _is_receipt_kind(altered, 1, "assistant")

    result = evict_messages(
        [(1, generated), (2, _assistant_receipt(2))],
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )
    assert len(result.messages) == 1
    assert result.messages[0].metadata.get("eviction_range") is True


def test_legacy_view_receipts_are_regenerated_from_source_once() -> None:
    source = Message(MessageRole.ASSISTANT, [TextContent("original " * 200)])
    legacy = Message(
        MessageRole.ASSISTANT,
        [TextContent("an obsolete receipt that must not be trusted")],
        metadata={"context_evicted": True, "source_seq": 1},
    )
    counter = lambda message: (
        9
        if message.metadata.get("eviction_receipt")
        else 10
        if message.metadata.get("context_evicted")
        else 100
    )

    first = evict_messages(
        [(1, legacy)],
        fixed_tokens=0,
        target_tokens=1,
        token_counter=counter,
        source_messages={1: source},
    )
    second = evict_messages(
        [(1, first.messages[0])],
        fixed_tokens=0,
        target_tokens=1,
        token_counter=counter,
        source_messages={1: source},
    )

    assert first.messages[0].metadata["eviction_receipt"] == "assistant"
    assert first.messages[0].to_dict() == second.messages[0].to_dict()
    assert first.items_evicted == 1
    assert second.items_evicted == 0


def test_semantic_digest_excerpts_may_coalesce() -> None:
    records = [
        *_tool_receipt_pair(1, "read", "read-1"),
        *_tool_receipt_pair(3, "read", "read-2"),
    ]

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert len(result.messages) == 1
    assert result.messages[0].metadata.get("eviction_range") is True


def test_reasoning_evicted_raw_text_or_images_never_coalesce() -> None:
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
                [
                    ThinkingContent("more reasoning"),
                    ImageContent("aW1hZ2U=", "image/png"),
                ],
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

    assert not any(
        message.metadata.get("eviction_range") for message in second.messages
    )
    assert [message.to_dict() for message in second.messages] == [
        message.to_dict() for message in first.messages
    ]
    assert any(
        isinstance(block, TextContent) and block.text == "keep this answer"
        for block in second.messages[0].content
    )
    assert any(isinstance(block, ImageContent) for block in second.messages[1].content)


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

    assert (
        sum(bool(message.metadata.get("eviction_range")) for message in result.messages)
        == 2
    )
    kept_user = next(
        message for message in result.messages if message.role is MessageRole.USER
    )
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
    records = [
        (entry.seq, message) for entry, message in zip(entries, originals, strict=True)
    ]
    result = evict_messages(records, fixed_tokens=0, target_tokens=1)
    assert (
        sum(bool(message.metadata.get("eviction_range")) for message in result.messages)
        == 1
    )
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
    assert result.tokens_after == sum(
        message_token_count(message) for message in result.messages
    )
    assert (
        sum(bool(message.metadata.get("eviction_range")) for message in result.messages)
        == 1
    )


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


def test_unmarked_tool_call_with_receipted_results_is_a_range_boundary() -> None:
    records = [
        (
            1,
            Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(ToolCall("raw-read", "read", {"path": "raw.txt"}))],
            ),
        ),
        (
            2,
            _semantic_result_receipt(
                role=MessageRole.TOOL_RESULT,
                tool_name="read",
                tool_call_id="raw-read",
                seq=2,
                digest="exact receipt",
                content_digest="0" * 64,
            ),
        ),
        (3, _assistant_receipt(3)),
    ]

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        token_counter=lambda message: (
            1 if message.metadata.get("eviction_range") else 100
        ),
    )

    assert not any(message.metadata.get("eviction_range") for message in result.messages)


def test_persisted_raw_calls_are_evicted_then_coalesced_and_stable() -> None:
    raw_calls = [
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("read-1", "read", {"path": "raw/" * 100}))],
        ),
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("search-1", "search", {"query": "raw " * 100}))],
        ),
    ]
    records = [
        (1, raw_calls[0]),
        _tool_receipt_pair(1, "read", "read-1")[1],
        (3, raw_calls[1]),
        _tool_receipt_pair(3, "search", "search-1")[1],
    ]
    source_messages = {
        1: raw_calls[0],
        2: Message(
            MessageRole.TOOL_RESULT, tool_result=ToolResult("read-1", "raw output " * 20)
        ),
        3: raw_calls[1],
        4: Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult("search-1", "raw output " * 20),
        ),
    }

    first = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        source_messages=source_messages,
    )
    replay = list(zip(first.source_seqs, first.messages, strict=True))
    second = evict_messages(
        replay,
        fixed_tokens=0,
        target_tokens=1,
        source_messages=source_messages,
    )

    assert len(first.messages) == 1
    assert first.messages[0].metadata.get("eviction_range") is True
    assert first.messages[0].to_dict() == second.messages[0].to_dict()
    assert second.items_evicted == 0


def test_persisted_raw_call_eviction_respects_protection() -> None:
    raw_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("read-1", "read", {"path": "raw.txt"}))],
    )
    persisted_result = _tool_receipt_pair(1, "read", "read-1")[1]
    source_messages = {
        1: raw_call,
        2: Message(
            MessageRole.TOOL_RESULT, tool_result=ToolResult("read-1", "raw output " * 20)
        ),
    }

    result = evict_messages(
        [(1, raw_call), persisted_result],
        fixed_tokens=0,
        target_tokens=1,
        unconsumed_source_seqs={2},
        source_messages=source_messages,
    )

    assert result.messages[0].to_dict() == raw_call.to_dict()
    assert not any(message.metadata.get("eviction_range") for message in result.messages)


def test_persisted_reasoning_eviction_with_raw_text_is_not_replaced() -> None:
    source = Message(
        MessageRole.ASSISTANT,
        [
            ThinkingContent("private reasoning"),
            TextContent("keep this answer"),
            ToolUseContent(ToolCall("read-1", "read", {"path": "raw/" * 100})),
        ],
    )
    persisted = Message(
        MessageRole.ASSISTANT,
        [
            TextContent("keep this answer"),
            ToolUseContent(ToolCall("read-1", "read", {"path": "raw/" * 100})),
            TextContent("[assistant reasoning evicted · seq 1]"),
        ],
        metadata={"context_evicted": True, "source_seq": 1},
    )

    result = evict_messages(
        [(1, persisted)],
        fixed_tokens=0,
        target_tokens=1,
        source_messages={1: source},
    )

    assert result.messages[0].to_dict() == persisted.to_dict()


def test_persisted_raw_call_eviction_respects_size_guard() -> None:
    raw_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("read-1", "read", {"path": "raw.txt"}))],
    )
    persisted_result = _tool_receipt_pair(1, "read", "read-1")[1]
    source_messages = {
        1: raw_call,
        2: Message(
            MessageRole.TOOL_RESULT, tool_result=ToolResult("read-1", "raw output " * 20)
        ),
    }

    result = evict_messages(
        [(1, raw_call), persisted_result],
        fixed_tokens=0,
        target_tokens=1,
        source_messages=source_messages,
        token_counter=lambda message: (
            100 if message.metadata.get("eviction_receipt") == "tool_call" else 1
        ),
    )

    assert result.messages[0].to_dict() == raw_call.to_dict()
    assert not any(message.metadata.get("eviction_range") for message in result.messages)


def test_legacy_regeneration_respects_unconsumed_and_notification_protection() -> None:
    source_result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("read-1", "source output " * 100),
    )
    persisted_result = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("read-1", "old receipt"),
        metadata={"context_evicted": True, "source_seq": 2},
    )
    source_notification = Message(
        MessageRole.SYSTEM,
        [TextContent("source notification " * 100)],
        metadata={"zeta_event": "agent_notifications", "notifications": []},
    )
    persisted_notification = Message(
        MessageRole.SYSTEM,
        [TextContent("old notification receipt")],
        metadata={"context_evicted": True, "source_seq": 3},
    )
    source_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("read-1", "read", {"path": "secret " * 100}))],
    )
    persisted_call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("read-1", "read", {"path": "old receipt"}))],
        metadata={"context_evicted": True, "source_seq": 1},
    )
    source_bash = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("bash-1", "bash", {"command": "secret " * 100}))],
    )
    persisted_bash = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("bash-1", "bash", {"command": "old receipt"}))],
        metadata={"context_evicted": True, "source_seq": 4},
    )
    records = [
        (1, persisted_call),
        (2, persisted_result),
        (3, persisted_notification),
        (4, persisted_bash),
    ]

    result = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=1,
        unconsumed_source_seqs={2},
        source_messages={
            1: source_call,
            2: source_result,
            3: source_notification,
            4: source_bash,
        },
    )

    assert [message.to_dict() for message in result.messages] == [
        message.to_dict() for _, message in records
    ]


def test_legacy_regeneration_keeps_larger_generated_receipt() -> None:
    source = Message(MessageRole.ASSISTANT, [TextContent("large source " * 100)])
    persisted = Message(
        MessageRole.ASSISTANT,
        [TextContent("tiny")],
        metadata={"context_evicted": True, "source_seq": 1},
    )

    result = evict_messages(
        [(1, persisted)],
        fixed_tokens=0,
        target_tokens=1,
        source_messages={1: source},
        token_counter=lambda message: (
            11
            if message.metadata.get("eviction_receipt")
            else 2
            if message.metadata.get("context_evicted")
            else 100
        ),
    )

    assert result.messages[0].to_dict() == persisted.to_dict()
    assert result.items_evicted == 0
