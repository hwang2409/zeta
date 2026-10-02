"""Ordered projection of durable context-hiding decisions."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

from ..core.store import ConversationEntry
from ..protocol.types import Message
from .archive import ContextBlock, render_archive
from .edit import render_edit
from .evict import EVICTION_KIND, eviction_entries

CompactionRenderer = Callable[[ConversationEntry], Sequence[ContextBlock]]


def apply_persisted_decisions(
    entries: Sequence[ConversationEntry],
    blocks: Sequence[ContextBlock],
    *,
    strategies: Collection[str],
    render_compaction: CompactionRenderer,
) -> list[ContextBlock]:
    """Project decisions in log order, so the later overlapping decision wins.

    Every decision replaces its whole overlapping visible block span. Restoring an
    archive removes only that archive decision from this replay; an overlapping
    edit, eviction, or summary compaction continues to hide its source range.
    """

    markers = [entry for entry in entries if entry.type == "compaction"]
    superseded = {
        marker_id for marker in markers for marker_id in marker.data.get("replaces", [])
    }
    restored = {
        entry.data.get("archive_id")
        for entry in entries
        if entry.type == "context_restore"
    }
    decisions = [
        entry
        for entry in entries
        if (
            entry.type == "compaction"
            and entry.id not in superseded
            or entry.type == "context_archive"
            and "archive" in strategies
            and entry.data.get("archive_id") not in restored
            or entry.type == "context_replace"
            and "edit" in strategies
        )
    ]

    result = list(blocks)
    for decision in decisions:
        start = decision.data["source_seq_start"]
        end = decision.data["source_seq_end"]
        affected = [
            index
            for index, block in enumerate(result)
            if block.source_start is not None
            and block.source_end is not None
            and block.source_start <= end
            and block.source_end >= start
        ]
        replacements = _render_decision(decision, render_compaction)
        if not affected:
            if decision.type == "compaction":
                result.extend(replacements)
            continue
        result[min(affected) : max(affected) + 1] = replacements
    return result


def _render_decision(
    decision: ConversationEntry,
    render_compaction: CompactionRenderer,
) -> list[ContextBlock]:
    if decision.type == "context_archive":
        return [render_archive(decision)]
    if decision.type == "context_replace":
        return [render_edit(decision)]
    if decision.data.get("kind") != EVICTION_KIND:
        return list(render_compaction(decision))
    blocks = [
        ContextBlock(
            entry,
            Message.from_dict(entry.data["message"]),
            entry.seq,
            entry.seq,
        )
        for entry in eviction_entries(decision)
    ]
    pinned = decision.data.get("pinned_message")
    if pinned is not None:
        blocks.append(
            ContextBlock(
                decision,
                Message.from_dict(pinned),
                decision.data["source_seq_start"],
                decision.data["source_seq_end"],
            )
        )
    return blocks
