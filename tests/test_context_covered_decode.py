from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from zeta.context_eviction import valid_eviction_view
from zeta.core.context import ContextAssembler
from zeta.core.context_parts.types import _ContextItem
from zeta.core.store import ConversationEntry, ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def msg(role: MessageRole, text: str, metadata: dict | None = None) -> Message:
    return Message(role, [TextContent(text)], metadata=metadata)


def corpus() -> list[ConversationEntry]:
    failed = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("failed-call", "run", {}))],
        metadata={"turn_failed": True},
    )
    values = {
        1: msg(MessageRole.ASSISTANT, "covered"),
        2: msg(MessageRole.ASSISTANT, "pending", {"zeta.pending_prompt_id": "pending-1"}),
        3: msg(MessageRole.ASSISTANT, "covered-third"),
        4: failed,
        5: Message(MessageRole.TOOL_RESULT, [], tool_result=ToolResult("failed-call", "result")),
        6: msg(MessageRole.ASSISTANT, "pending-step", {"zeta.step_delivery": {"notification_ids": ["notice-1"]}}),
        7: msg(MessageRole.USER, "tail"),
        8: msg(MessageRole.ASSISTANT, "uncovered"),
    }
    entries = [
        ConversationEntry(seq=n, id=str(n), parent_id=None, lane="main", type="message", data={"message": value.to_dict()})
        for n, value in values.items()
    ]
    entries.append(ConversationEntry(
        seq=9, id="old-marker", parent_id=None, lane="main", type="compaction",
        data={"kind": "evict", "source_seq_start": 1, "source_seq_end": 3, "view": [{"seq": 1, "message": values[1].to_dict()}]},
    ))
    entries.append(ConversationEntry(seq=10, id="marker", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 1, "source_seq_end": 3, "replaces": ["old-marker"], "view": [{"seq": 1, "message": values[1].to_dict()}, {"seq": 3, "message": values[3].to_dict()}]}))
    entries.append(ConversationEntry(seq=11, id="overlap", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 3, "source_seq_end": 4, "view": [{"seq": 3, "message": values[3].to_dict()}, {"seq": 4, "message": values[4].to_dict()}]}))
    entries.append(ConversationEntry(seq=12, id="step-marker", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 6, "source_seq_end": 6, "view": [{"seq": 6, "message": values[6].to_dict()}]}))
    entries.append(ConversationEntry(seq=13, id="reordered", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 2, "source_seq_end": 2, "view": [{"seq": 2, "message": values[2].to_dict()}]}))
    entries.append(ConversationEntry(seq=14, id="invalid", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 20, "source_seq_end": 21, "view": [{"seq": 99, "message": values[1].to_dict()}]}))
    return entries


def reference_items(assembler: ContextAssembler, entries: list[ConversationEntry]):
    all_markers = [entry for entry in entries if entry.type == "compaction"]
    superseded = {replaced for marker in all_markers for replaced in marker.data.get("replaces", [])}
    markers = [entry for entry in all_markers if entry.id not in superseded]
    active = {entry.seq for entry in entries if entry.type in {"message", "compaction"}}
    markers = [marker for marker in markers if marker.data.get("kind") != "evict" or valid_eviction_view(marker, active)]
    ranges = [(marker.data["source_seq_start"], marker.data["source_seq_end"]) for marker in markers]
    by_start = {marker.data["source_seq_start"]: marker for marker in markers}
    items = []
    emitted = set()
    failed_ids = set()
    for entry in entries:
        marker = by_start.get(entry.seq)
        if marker is not None:
            items.extend(assembler._marker_items(marker, active, entries))
            emitted.add(marker.id)
        if entry.type != "message":
            continue
        message = Message.from_dict(entry.data["message"])
        metadata = message.metadata
        if metadata.get("zeta.pending_prompt_id") == "pending-1":
            continue
        delivery = metadata.get("zeta.step_delivery")
        if isinstance(delivery, dict) and "notice-1" in delivery.get("notification_ids", ()):
            continue
        if message.metadata.get("turn_failed"):
            failed_ids.update(block.tool_call.id for block in message.content if isinstance(block, ToolUseContent))
            continue
        if any(start <= entry.seq <= end for start, end in ranges):
            continue
        if message.role is MessageRole.TOOL_RESULT and message.tool_result is not None and message.tool_result.tool_call_id in failed_ids:
            continue
        items.append(_ContextItem(entry, message))
    for marker in markers:
        if marker.id not in emitted:
            items.extend(assembler._marker_items(marker, active, entries))
    return items


def project(store: ConversationStore, entries: list[ConversationEntry]):
    assembler = ContextAssembler(store, token_counter=lambda message: len(message.content))
    items = assembler._visible_items(entries)
    assembler._track_token_counts(items)
    return [item.message.to_dict() for item in items], assembler._count([item.message for item in items])


def test_covered_decode_matches_reference_projection_and_tokens(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    try:
        store.delivery_fold.active_prompts["pending-1"] = object()
        store.delivery_fold.active_notifications["notice-1"] = object()
        entries = corpus()
        projected, tokens = project(store, entries)
        reference_assembler = ContextAssembler(store, token_counter=lambda message: len(message.content))
        reference = reference_items(reference_assembler, entries)
        assert projected == [item.message.to_dict() for item in reference]
        assert tokens == sum(len(item.message.content) for item in reference)
    finally:
        store.close()


def test_covered_decode_skips_ordinary_rows(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    try:
        store.delivery_fold.active_prompts["pending-1"] = object()
        store.delivery_fold.active_notifications["notice-1"] = object()
        entries = corpus()
        with patch.object(Message, "from_dict", wraps=Message.from_dict) as decode:
            project(store, entries)
        # test-contract: seam _visible_items is the narrow projection seam under test
        assert decode.call_count == 12
    finally:
        store.close()


def test_covered_odd_metadata_keeps_decoder_validation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    try:
        entry = ConversationEntry(seq=1, id="1", parent_id=None, lane="main", type="message", data={"message": msg(MessageRole.ASSISTANT, "odd").to_dict()})
        entry.data["message"]["metadata"] = []
        marker = ConversationEntry(seq=2, id="marker", parent_id=None, lane="main", type="compaction", data={"kind": "evict", "source_seq_start": 1, "source_seq_end": 1, "view": [{"seq": 1, "message": msg(MessageRole.ASSISTANT, "odd").to_dict()}]})
        try:
            ContextAssembler(store)._visible_items([entry, marker])
        except ValueError as exc:
            # test-contract: seam validates malformed persisted metadata
            assert str(exc) == "message metadata must be an object"
        else:
            raise AssertionError("odd metadata must reach Message.from_dict")
    finally:
        store.close()
