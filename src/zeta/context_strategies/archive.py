"""Durable, branch-local VISTA context archives."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import Message, MessageRole, TextContent, ToolUseContent
from . import ContextTelemetry


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One visible context block with its source sequence range."""

    entry: ConversationEntry | None
    message: Message
    source_start: int | None
    source_end: int | None
    fixed: bool = False


def context_blocks(items: Sequence[Any]) -> list[ContextBlock]:
    """Adapt assembler items without coupling strategy modules to core.context."""

    blocks: list[ContextBlock] = []
    for item in items:
        entry = item.entry
        message = item.message
        from_compaction = entry is None or entry.type == "compaction"
        blocks.append(
            ContextBlock(
                entry,
                message,
                (
                    message.metadata.get("source_seq_start")
                    if from_compaction
                    else entry.seq
                ),
                (
                    message.metadata.get("source_seq_end")
                    if from_compaction
                    else entry.seq
                ),
                item.fixed,
            )
        )
    return blocks


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    archive_id: str
    seq_start: int
    seq_end: int
    tokens: int


def _message_entries(entries: Sequence[ConversationEntry]) -> list[ConversationEntry]:
    return [entry for entry in entries if entry.type == "message"]


def _pair_spans(entries: Sequence[ConversationEntry]) -> list[tuple[int, int]]:
    calls: dict[str, int] = {}
    grouped: dict[int, list[int]] = {}
    for entry in _message_entries(entries):
        message = Message.from_dict(entry.data["message"])
        for block in message.content:
            if isinstance(block, ToolUseContent):
                calls[block.tool_call.id] = entry.seq
                grouped.setdefault(entry.seq, []).append(entry.seq)
        if message.tool_result is not None:
            call_seq = calls.get(message.tool_result.tool_call_id)
            if call_seq is not None:
                grouped.setdefault(call_seq, [call_seq]).append(entry.seq)
    return [(min(seqs), max(seqs)) for seqs in grouped.values()]


def snap_range(
    entries: Sequence[ConversationEntry], seq_start: int, seq_end: int
) -> tuple[int, int]:
    """Snap a sequence range outwards until no tool call/result pair is split."""

    if type(seq_start) is not int or type(seq_end) is not int:
        raise ValueError("seq_start and seq_end must be integers")
    if seq_start < 1 or seq_end < seq_start:
        raise ValueError("invalid sequence range")
    start, end = seq_start, seq_end
    changed = True
    spans = _pair_spans(entries)
    while changed:
        changed = False
        for pair_start, pair_end in spans:
            if pair_start <= end and pair_end >= start and (
                pair_start < start or pair_end > end
            ):
                start, end = min(start, pair_start), max(end, pair_end)
                changed = True
    return start, end


def _latest_user_seq(entries: Sequence[ConversationEntry]) -> int | None:
    for entry in reversed(_message_entries(entries)):
        if Message.from_dict(entry.data["message"]).role is MessageRole.USER:
            return entry.seq
    return None


def _validate_target(
    entries: Sequence[ConversationEntry],
    start: int,
    end: int,
    *,
    retained_tail: int | None,
) -> list[ConversationEntry]:
    selected = [entry for entry in _message_entries(entries) if start <= entry.seq <= end]
    if not selected:
        raise ValueError("range contains no messages on the active branch")
    latest_user = _latest_user_seq(entries)
    if latest_user is not None and start <= latest_user <= end:
        raise ValueError("range cannot include the latest user turn")
    if retained_tail is not None:
        tail = _message_entries(entries)[-retained_tail:]
        if any(start <= entry.seq <= end for entry in tail):
            raise ValueError("range overlaps the retained tail")
    return selected


def _estimate_tokens(entries: Sequence[ConversationEntry]) -> int:
    chars = sum(
        len(json.dumps(entry.data["message"], ensure_ascii=False, separators=(",", ":")))
        for entry in entries
    )
    return max(1, (chars + 3) // 4)


def archive_context(
    store: ConversationStore,
    *,
    seq_start: int,
    seq_end: int,
    note: str | None = None,
    retained_tail: int = 8,
) -> ArchiveResult:
    """Persist an archive decision without modifying original messages."""

    if note is not None and (type(note) is not str or len(note) > 500):
        raise ValueError("note must be a string of at most 500 characters")
    entries = store.replay()
    start, end = snap_range(entries, seq_start, seq_end)
    selected = _validate_target(entries, start, end, retained_tail=retained_tail)
    number = 1 + sum(entry.type == "context_archive" for entry in entries)
    archive_id = f"A{number}"
    tokens = _estimate_tokens(selected)
    data: dict[str, Any] = {
        "archive_id": archive_id,
        "source_seq_start": start,
        "source_seq_end": end,
        "tokens": tokens,
    }
    if note:
        data["note"] = note
    store._append_row("context_archive", data)
    ContextTelemetry().emit(
        "archive", kind="archive", range=[start, end], tokens=tokens
    )
    return ArchiveResult(archive_id, start, end, tokens)


def restore_context(store: ConversationStore, *, archive_id: str) -> ArchiveResult:
    """Persist restoration of an active archive on this branch."""

    if type(archive_id) is not str or not archive_id:
        raise ValueError("archive_id must be a nonempty string")
    entries = store.replay()
    restored = {
        entry.data.get("archive_id")
        for entry in entries
        if entry.type == "context_restore"
    }
    archive = next(
        (
            entry
            for entry in reversed(entries)
            if entry.type == "context_archive"
            and entry.data.get("archive_id") == archive_id
            and archive_id not in restored
        ),
        None,
    )
    if archive is None:
        raise ValueError(f"archive {archive_id!r} is not active on this branch")
    data = archive.data
    store._append_row("context_restore", {"archive_id": archive_id})
    result = ArchiveResult(
        archive_id,
        data["source_seq_start"],
        data["source_seq_end"],
        data["tokens"],
    )
    ContextTelemetry().emit(
        "restore",
        kind="restore",
        range=[result.seq_start, result.seq_end],
        tokens=result.tokens,
    )
    return result


def render_archive(archive: ConversationEntry) -> ContextBlock:
    """Render one archive decision as a bounded assistant placeholder."""

    start = archive.data["source_seq_start"]
    end = archive.data["source_seq_end"]
    note = archive.data.get("note")
    note_part = "" if not note else f" · {note}"
    archive_id = archive.data["archive_id"]
    placeholder = (
        f"[archived #{archive_id} seq {start}–{end} · "
        f"~{archive.data['tokens']} tok{note_part} · "
        f'context_restore("{archive_id}")]'
    )
    message = Message(
        MessageRole.ASSISTANT,
        [TextContent(placeholder)],
        metadata={
            "context_archive": archive_id,
            "context_strategy_fixed": True,
            "source_seq_start": start,
            "source_seq_end": end,
        },
    )
    return ContextBlock(archive, message, start, end, fixed=True)
