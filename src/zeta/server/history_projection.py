"""Immutable session-history projections shared by serve read workers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..core.store import ConversationStore
from ..protocol.types import is_passive_harness_message
from .protocol import MAX_REQUEST_ID_BYTES, FrameCodec

HISTORY_PAGE_ENTRIES = 128
Generation = tuple[int, bytes]


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    groups: tuple[tuple[Any, ...], ...]
    approval_calls: Mapping[str, object]


@dataclass
class _CachedProjection:
    generation: Generation
    projection: HistoryProjection


_cache_creation_lock = threading.Lock()


def _generation(store: ConversationStore) -> Generation:
    return (
        getattr(store, "_log_offset", 0),
        getattr(store, "_entry_prefix_digest", b""),
    )


def _cache(store: ConversationStore) -> _CachedProjection | None:
    return getattr(store, "_serve_history_projection", None)


def projection(store: ConversationStore) -> HistoryProjection:
    """Return one immutable projection for the store's validated generation."""
    cached = _cache(store)
    generation = _generation(store)
    if cached is not None and cached.generation == generation:
        return cached.projection

    # The store is loop-owned, but its validated snapshots are read here by the
    # serve worker. Retry if an append publishes a newer generation mid-copy.
    while True:
        generation = _generation(store)
        branch = store.active_branch_snapshot()
        if generation == _generation(store):
            break

    approval_calls: dict[str, object] = {}
    groups: list[list[Any]] = [[]]
    for entry in branch:
        if entry.type == "message":
            approval_calls.update(
                {
                    request["request_id"]: request["tool_call"]
                    for request in entry.data.get("approval_requests", [])
                }
            )
        if entry.type == "message" and is_passive_harness_message(
            entry.data["message"]
        ):
            continue
        if entry.type in {"message", "notification"}:
            groups[-1].append(entry)
            groups.append([])
        elif entry.type in {"approval_resolution", "approval_history"}:
            groups[-1].append(entry)
    built = HistoryProjection(
        tuple(tuple(group) for group in groups), MappingProxyType(approval_calls)
    )
    with _cache_creation_lock:
        cached = _cache(store)
        if cached is None or cached.generation != generation:
            store._serve_history_projection = _CachedProjection(generation, built)
            return built
        return cached.projection


def page(
    projection: HistoryProjection,
    offset: int,
    row_builder: Callable[[Any, Mapping[str, object]], list[dict[str, object]]],
    *,
    page_limit: int = HISTORY_PAGE_ENTRIES,
) -> dict[str, object]:
    groups = projection.groups
    entry_count = len(groups) - 1
    if offset > entry_count:
        return {"messages": [], "next_offset": None}
    rows: list[dict[str, object]] = []
    codec = FrameCodec()
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    stop = min(offset + page_limit, entry_count)
    indexes = range(offset, stop)
    if stop == entry_count:
        indexes = range(offset, stop + 1)
    next_offset = offset
    for group_index in indexes:
        group = [
            row
            for entry in groups[group_index]
            for row in row_builder(entry, projection.approval_calls)
        ]
        candidate_offset = min(group_index + 1, entry_count)
        candidate = {
            "messages": [*rows, *group],
            "next_offset": candidate_offset if candidate_offset < entry_count else None,
        }
        if not codec.response_fits(request_id, candidate) and rows:
            break
        if not codec.response_fits(request_id, candidate):
            for row in group:
                if "content" in row:
                    row["content"] = [
                        {"type": "text", "text": "[history row too large; truncated]"}
                    ]
                    row["tool_result"] = None
                if "notification" in row:
                    row["notification"] = None
        rows.extend(group)
        next_offset = candidate_offset
    return {
        "messages": rows,
        "next_offset": next_offset if next_offset < entry_count else None,
    }
