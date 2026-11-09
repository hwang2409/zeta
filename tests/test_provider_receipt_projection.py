"""Eviction receipts reach providers as harness notes, never as assistant turns."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from zeta.context_eviction import evict_messages, eviction_view
from zeta.context_eviction.receipt_constructors import (
    _collapsed_assistant_receipt,
    _notification_receipt_from_fields,
    is_receipt_echo,
    receipt_tokens,
)
from zeta.core.context import CompactionPolicy, ContextAssembler
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic import serialize_request_payload
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import _cache_affinity_json, build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages
from zeta.providers.payload_common import HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER

NOTE_PREFIX = f"{HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER}\n[zeta] context eviction: "
THINKING = ThinkingContent("weighing the options " * 40, signature="sig")
ECHO_TEXT = "[assistant text evicted · seq 99]"


def user(text: str) -> Message:
    return Message(
        MessageRole.USER,
        [TextContent(text)],
        metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
    )


def reply(*blocks: Any) -> Message:
    return Message(MessageRole.ASSISTANT, list(blocks))


def read(call_id: str, path: str) -> ToolUseContent:
    return ToolUseContent(ToolCall(call_id, "read", {"path": path}))


def result(call_id: str, body: str) -> Message:
    return Message(MessageRole.TOOL_RESULT, tool_result=ToolResult(call_id, body))


def notification() -> Message:
    return Message(
        MessageRole.SYSTEM,
        [TextContent("child 7 finished")],
        metadata={"zeta_event": "agent_notifications"},
    )


def echo() -> Message:
    """A persisted model reply that only copies a receipt (the bug's output)."""
    return Message(
        MessageRole.ASSISTANT,
        [THINKING, TextContent(ECHO_TEXT)],
        metadata={
            "response_state": "completed",
            "codex_output_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ECHO_TEXT}],
                }
            ],
        },
    )


def evicted(*messages: Message, passes: int = 1) -> list[Message]:
    """Run production eviction and replay its stored view, like the assembler."""
    records = list(enumerate(messages, start=1))
    for _ in range(passes):
        outcome = evict_messages(records, fixed_tokens=0, target_tokens=0)
        view = eviction_view(records, outcome)
        replayed = ContextAssembler._eviction_view_messages(view)
        records = list(zip((item["seq"] for item in view), replayed, strict=True))
    return [message for _, message in records]


def stored_receipt_texts(messages: list[Message]) -> set[str]:
    return {
        block.text
        for message in messages
        for block in message.content
        if isinstance(block, TextContent)
        and (message.metadata.get("context_evicted") or block.text == ECHO_TEXT)
        and block.text.startswith("[")
        and (
            " evicted · seq " in block.text
            or block.text.startswith("[evicted range]")
            or " collapsed into seq " in block.text
        )
    }


BIG = "line of file output\n" * 300


def row_after_parallel_results() -> tuple[list[Message], int]:
    return [
        *evicted(
            user("start"),
            reply(read("a", "/a"), read("b", "/b")),
            result("a", BIG),
            result("b", BIG + "b"),
            reply(TextContent("both files read " * 30)),
        ),
        user("next"),
    ], 1


def row_before_tool_use() -> tuple[list[Message], int]:
    return [
        *evicted(
            user("start"),
            reply(TextContent("about to call the tool " * 30)),
            reply(read("c", "/c")),
            result("c", BIG),
        ),
        user("next"),
    ], 1


def row_first_message() -> tuple[list[Message], int]:
    return [*evicted(reply(TextContent("opening reply " * 30)), user("start")), user("next")], 1


def row_last_before_notification() -> tuple[list[Message], int]:
    return [
        *evicted(
            user("start"),
            reply(read("d", "/d")),
            result("d", BIG),
            reply(TextContent("waiting for the child " * 30)),
        ),
        notification(),
    ], 1


def row_collapsed_with_orphans() -> tuple[list[Message], int]:
    view = evicted(
        user("start"),
        reply(read("e", "/same")),
        result("e", BIG),
        user("again"),
        reply(read("f", "/same")),
        result("f", BIG),
    )
    # a legacy view keeps the raw result after its collapsed call receipt.
    orphan_view = [
        user("old"),
        _collapsed_assistant_receipt(20, 5, older_read=True),
        result("g", BIG),
    ]
    return [*orphan_view, *view, user("next")], 3


def row_hybrid() -> tuple[list[Message], int]:
    return [
        *evicted(
            user("start"),
            reply(THINKING, TextContent("checking the file"), read("h", "/h")),
            result("h", BIG),
        ),
        user("next"),
    ], 0


def row_thinking_only() -> tuple[list[Message], int]:
    return [*evicted(user("start"), reply(THINKING), user("more")), user("next")], 0


def row_range() -> tuple[list[Message], int]:
    return [
        *evicted(
            user("start"),
            reply(TextContent("first note " * 30)),
            reply(TextContent("second note " * 30)),
            reply(read("i", "/i")),
            result("i", BIG),
            passes=2,
        ),
        user("next"),
    ], 1


def row_legacy_text_receipt() -> tuple[list[Message], int]:
    # stored 2026-10-03..10-08, before receipts carried eviction_receipt.
    text = {
        "content": [{"text": "[assistant text evicted · seq 4]", "type": "text"}],
        "metadata": {"context_evicted": True, "source_seq": 4},
        "role": "assistant",
    }
    collapsed = {
        "content": [{"text": "[older duplicate read collapsed into seq 9]", "type": "text"}],
        "metadata": {"collapsed_into_seq": 9, "context_evicted": True, "source_seq": 6},
        "role": "assistant",
    }
    return [
        user("start"),
        Message.from_dict(text),
        user("more"),
        Message.from_dict(collapsed),
        user("next"),
    ], 2


def row_echo() -> tuple[list[Message], int]:
    return [user("before"), echo(), user("after")], 0


ROWS: dict[str, Callable[[], tuple[list[Message], int]]] = {
    "after-parallel-results": row_after_parallel_results,
    "before-tool-use": row_before_tool_use,
    "first-message": row_first_message,
    "last-before-notification": row_last_before_notification,
    "collapsed-with-orphans": row_collapsed_with_orphans,
    "hybrid": row_hybrid,
    "thinking-only": row_thinking_only,
    "range": row_range,
    "legacy-text-receipt": row_legacy_text_receipt,
    "echo": row_echo,
}


def anthropic_turns(messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
    payload = build_messages_payload(
        messages, [], model="claude-test", max_tokens=4096, thinking_budget=1024
    )
    joined: list[dict[str, Any]] = []
    for turn in payload["messages"]:
        content = turn["content"]
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if joined and joined[-1]["role"] == turn["role"]:
            joined[-1]["content"].extend(blocks)
        else:
            joined.append({"role": turn["role"], "content": list(blocks)})
    system = " ".join(block.get("text", "") for block in payload.get("system", []))
    return joined, system


def wire_view(provider: str, messages: list[Message]) -> tuple[list[str], list[str], str]:
    """Check provider pairing; return assistant texts, user texts, system text."""
    assistant: list[str] = []
    users: list[str] = []
    if provider == "anthropic":
        turns, system = anthropic_turns(messages)
        for index, turn in enumerate(turns):
            texts = [b["text"] for b in turn["content"] if b["type"] == "text"]
            (assistant if turn["role"] == "assistant" else users).extend(texts)
            kinds = [block["type"] for block in turn["content"]]
            if turn["role"] == "user" and "tool_result" in kinds:
                first_other = next(
                    (i for i, kind in enumerate(kinds) if kind != "tool_result"),
                    len(kinds),
                )
                assert "tool_result" not in kinds[first_other:], turn
            calls = [b["id"] for b in turn["content"] if b["type"] == "tool_use"]
            if calls:
                following = turns[index + 1]
                assert following["role"] == "user"
                ids = [b.get("tool_use_id") for b in following["content"][: len(calls)]]
                assert sorted(ids) == sorted(calls), following
        return assistant, users, system
    if provider == "codex":
        payload = build_responses_payload(messages, [], model="gpt-test")
        items = payload["input"]
        for index, item in enumerate(items):
            if item.get("type") == "function_call":
                outputs = [
                    later
                    for later in items[index + 1 :]
                    if later.get("type") == "function_call_output"
                    and later["call_id"] == item["call_id"]
                ]
                assert len(outputs) == 1, item
            if item.get("type") == "function_call_output":
                assert any(
                    earlier.get("type") == "function_call"
                    and earlier["call_id"] == item["call_id"]
                    for earlier in items[:index]
                ), item
            if item.get("role") in ("assistant", "user"):
                texts = [b["text"] for b in item["content"] if "text" in b]
                (assistant if item["role"] == "assistant" else users).extend(texts)
        return assistant, users, payload["instructions"]
    rows = build_ollama_messages(messages)
    open_calls: set[str] = set()
    for row in rows:
        if row["role"] == "assistant":
            assistant.append(row["content"])
            open_calls = {call["function"]["name"] for call in row.get("tool_calls", [])}
        elif row["role"] == "tool":
            assert row["tool_name"] in open_calls, row
        else:
            open_calls = set()
            if row["role"] == "user":
                users.append(row["content"])
    system = " ".join(row["content"] for row in rows if row["role"] == "system")
    return assistant, users, system


@pytest.mark.parametrize("row", list(ROWS))
@pytest.mark.parametrize("provider", ["anthropic", "codex", "ollama"])
def test_receipts_become_harness_notes_with_pairing_intact(provider: str, row: str) -> None:
    messages, expected_notes = ROWS[row]()
    receipts = stored_receipt_texts(messages)
    assert receipts, "the row must contain a real stored receipt or echo"

    assistant, users, system = wire_view(provider, messages)

    leaked = [text for text in assistant if any(receipt in text for receipt in receipts)]
    assert leaked == [], f"{provider}/{row}: receipt text sent as an assistant turn"
    assert not any(receipt in system for receipt in receipts)
    assert not any("[assistant reasoning evicted" in text for text in assistant)
    notes = [text for text in users if text.startswith(NOTE_PREFIX)]
    assert len(notes) == expected_notes, users
    assert all(any(receipt in note for receipt in receipts) for note in notes)
    assert ECHO_TEXT not in "".join(users), "an echo must be dropped, not turned into a note"
    if row == "hybrid":
        assert any("checking the file" in text for text in assistant)


@pytest.mark.parametrize("provider", ["anthropic", "codex", "ollama"])
def test_head_notification_receipt_stays_out_of_system_prompt(provider: str) -> None:
    receipt = _notification_receipt_from_fields(
        role=MessageRole.SYSTEM,
        seq=2,
        payload={"notifications": [{"kind": "child_idle", "child_instance_id": "c1"}]},
        content_digest="abc",
    )
    messages = [Message(MessageRole.SYSTEM, [TextContent("you are zeta")]), receipt, user("hi")]

    _, users, system = wire_view(provider, messages)

    assert "notification receipt" not in system
    assert "you are zeta" in system
    assert sum(text.startswith(NOTE_PREFIX) for text in users) == 1


def history() -> list[Message]:
    view, _ = row_collapsed_with_orphans()
    return [
        Message(MessageRole.SYSTEM, [TextContent("you are zeta")]),
        *row_first_message()[0][:-1],
        *row_after_parallel_results()[0],
        *row_hybrid()[0],
        *view,
        echo(),
        user("latest question"),
    ]


def serialized(provider: str, messages: list[Message]) -> tuple[bytes, list[bytes]]:
    """Whole request bytes, plus each wire turn with cache_control removed."""
    if provider == "anthropic":
        payload = build_messages_payload(
            messages, [], model="claude-test", max_tokens=4096, thinking_budget=1024
        )
        whole = serialize_request_payload(payload)
        turns = payload["messages"]
    elif provider == "codex":
        payload = build_responses_payload(messages, [], model="gpt-test")
        affinity = _cache_affinity_json(payload, messages, "gpt-test")
        whole = json.dumps([payload, affinity], sort_keys=True).encode()
        turns = payload["input"]
    else:
        turns = build_ollama_messages(messages)
        whole = json.dumps(turns, sort_keys=True).encode()

    def strip_cache(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip_cache(v) for k, v in value.items() if k != "cache_control"}
        if isinstance(value, list):
            return [strip_cache(v) for v in value]
        return value

    return whole, [json.dumps(strip_cache(turn), sort_keys=True).encode() for turn in turns]


@pytest.mark.parametrize("provider", ["anthropic", "codex", "ollama"])
def test_projection_is_deterministic_and_keeps_the_prefix(provider: str) -> None:
    first, first_turns = serialized(provider, history())
    second, _ = serialized(provider, history())
    assert first == second

    longer = [*history(), reply(TextContent("an answer")), user("follow-up")]
    _, longer_turns = serialized(provider, longer)
    # the builder moves anthropic cache_control to the newest blocks each
    # request (strip_cache above); every other prefix byte must stay the same.
    assert longer_turns[: len(first_turns)] == first_turns


@pytest.mark.asyncio
async def test_building_payloads_changes_no_stored_row_or_message(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    try:
        store.append_message(user("inspect the files"))
        for index in range(6):
            store.append_message(reply(THINKING, read(f"r{index}", f"/f{index}")))
            store.append_message(result(f"r{index}", BIG * 8))
            store.append_message(reply(TextContent(f"file {index} looks fine " * 40)))
            store.append_message(user(f"next file {index}"))
        store.append_message(echo())
        store.append_message(user("latest"))
        context = await ContextAssembler(
            store, token_budget=12_000, retained_tail=1, compaction_policy=CompactionPolicy()
        ).assemble_context()
        assert any(m.metadata.get("context_evicted") for m in context.messages)
        # history() adds the hybrid and collapsed shapes the assembler did not make here.
        messages = [*context.messages, *history()[1:]]
        rows_before = [entry.to_dict() for entry in store.replay()]
        file_before = store.path.read_bytes()
        messages_before = [message.to_dict() for message in messages]
        identities = [id(message) for message in messages]

        build_messages_payload(messages, [], model="claude-test", max_tokens=4096, thinking_budget=1024)
        build_responses_payload(messages, [], model="gpt-test")
        build_ollama_messages(messages)

        # the TUI and zeta serve read these rows; the projection must not touch them.
        assert [entry.to_dict() for entry in store.replay()] == rows_before
        assert store.path.read_bytes() == file_before
        assert [message.to_dict() for message in messages] == messages_before
        assert [id(message) for message in messages] == identities
    finally:
        store.close()


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("[assistant text evicted · seq 5]", 1),
        ("  [assistant text evicted · seq 1474][assistant reasoning evicted · seq 1516]\n", 2),
        ("[duplicate result collapsed into seq 9]", 1),
        (
            (
                "[evicted range] seq 3-8: 2 assistant notes, 1 user request: fix it]; "
                "recall_history seq_start=3, seq_end=8 for exact content"
            ),
            1,
        ),
        ("done. [assistant text evicted · seq 5]", None),
        ("[assistant text evicted · seq 5] done.", None),
        ("[assistant text evicted · seq 05]", None),
        ("[user text evicted · seq 5]", None),
        (
            "[evicted range] seq 3-8: notes; recall_history seq_start=3, seq_end=9 for exact content",
            None,
        ),
        (
            "[evicted range] seq 3-8: a\nb; recall_history seq_start=3, seq_end=8 for exact content",
            None,
        ),
        ("[evicted range] seq 3-8: " + "x" * 440 + "; recall_history seq_start=3, seq_end=8 for exact content", None),
        ("", None),
    ],
)
def test_receipt_grammar_accepts_only_whole_receipt_text(text: str, tokens: int | None) -> None:
    parsed = receipt_tokens(text)
    assert (None if parsed is None else len(parsed)) == tokens
    assert is_receipt_echo(reply(THINKING, TextContent(text))) is (tokens is not None)
    assert not is_receipt_echo(reply(TextContent(text), read("x", "/x")))
    assert not is_receipt_echo(
        Message(MessageRole.ASSISTANT, [TextContent(text)], metadata={"context_evicted": True})
    )
