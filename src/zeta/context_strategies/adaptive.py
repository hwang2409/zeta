"""Adaptive retained-tail fitting shared by compaction strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import ceil
from typing import Any, Protocol

from ..protocol.types import (
    Message,
    MessageRole,
    ToolResult,
    ToolUseContent,
    flatten_tool_content,
)
from .evict import EvictionResult, evict_messages, eviction_view

IMAGE_TOKEN_ESTIMATE = 1024


def message_token_count(message: Message) -> int:
    """Estimate message tokens without charging base64 as text."""

    value = message.to_dict()
    image_count = 0
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image":
                content[index] = {
                    key: item for key, item in block.items() if key != "data"
                }
                image_count += 1
    tool_result = value.get("tool_result")
    if isinstance(tool_result, dict):
        blocks = tool_result.get("content_blocks")
        if isinstance(blocks, list):
            tool_result["content_blocks"] = [
                {key: item for key, item in block.items() if key != "data"}
                if isinstance(block, dict) and block.get("type") == "image"
                else block
                for block in blocks
            ]
            image_count += sum(
                isinstance(block, dict) and block.get("type") == "image"
                for block in blocks
            )
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return max(1, ceil(len(encoded) / 4) + image_count * IMAGE_TOKEN_ESTIMATE)


def message_digest(messages: Sequence[Message]) -> str:
    encoded = json.dumps(
        [message.to_dict() for message in messages],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ContextItem(Protocol):
    """Minimal visible-item shape needed by adaptive fitting."""

    entry: Any
    message: Message
    fixed: bool


@dataclass(frozen=True, slots=True)
class EvictionPlan:
    source_start: int
    source_end: int
    replaces: list[str]
    pinned_message: Message | None
    view: list[dict]
    result: EvictionResult


def plan_eviction(
    items: Sequence[ContextItem],
    system_messages: Sequence[Message],
    latest_user: int | None,
    token_budget: int,
    token_counter: Callable[[Message], int],
) -> EvictionPlan | None:
    """Plan whole-view eviction while keeping strategy blocks and latest user fixed."""

    pinned = items[latest_user] if latest_user is not None else None
    candidates = [
        item
        for index, item in enumerate(items)
        if not item.message.metadata.get("context_strategy_fixed")
        and index != latest_user
    ]
    if not candidates:
        return None
    start, end, replaces = compaction_source(candidates, pinned)
    records = strategy_records(candidates, start)
    fixed = [
        *system_messages,
        *(
            item.message
            for item in items
            if item.message.metadata.get("context_strategy_fixed")
        ),
        *([] if pinned is None else [pinned.message]),
    ]
    result = evict_messages(
        records,
        fixed_tokens=sum(token_counter(message) for message in fixed),
        target_tokens=max(1, int(token_budget * 0.7)),
        token_counter=token_counter,
    )
    if not result.reached_target or not result.items_evicted:
        return None
    return EvictionPlan(
        start,
        end,
        replaces,
        None if pinned is None else pinned.message,
        eviction_view(records, result),
        result,
    )


def has_only_compacted_prefix(
    items: Sequence[ContextItem], boundary: int, latest_user: int | None
) -> bool:
    """Return whether a pre-tail view contains markers but no raw history."""

    prefix = items[:boundary]
    has_uncompacted = any(
        not item.fixed
        and item.entry is not None
        and not item.entry.data.get("eviction_marker_id")
        and index != latest_user
        for index, item in enumerate(prefix)
    )
    has_compacted = any(
        item.fixed
        or item.entry is not None
        and item.entry.data.get("eviction_marker_id")
        for item in prefix
    )
    return not has_uncompacted and has_compacted


def replace_item_messages(
    items: Sequence[ContextItem], messages: Sequence[Message]
) -> list[ContextItem]:
    """Copy context items while substituting request-only message views."""

    return [replace(item, message=message) for item, message in zip(items, messages)]


def latest_user_index(items: Sequence[ContextItem]) -> int | None:
    return next(
        (
            index
            for index in range(len(items) - 1, -1, -1)
            if items[index].message.role is MessageRole.USER
        ),
        None,
    )


def committed_messages(
    items: Sequence[ContextItem],
    boundary: int,
    system_messages: Sequence[Message],
    latest_user: int | None,
) -> list[Message]:
    committed = [item.message for item in items if item.fixed]
    committed.extend(
        item.message
        for index, item in enumerate(items)
        if not item.fixed and (index >= boundary or index == latest_user)
    )
    return [*system_messages, *committed]


def shrink_tail_boundary(
    items: Sequence[ContextItem],
    boundary: int,
    system_messages: Sequence[Message],
    target: int,
    latest_user: int | None,
    token_counter: Callable[[Message], int],
) -> int:
    minimum = _minimum_tail_boundary(items)
    for candidate in range(boundary + 1, minimum + 1):
        if not _is_valid_tail_boundary(items, candidate):
            continue
        messages = committed_messages(items, candidate, system_messages, latest_user)
        if sum(token_counter(message) for message in messages) <= target:
            return candidate
    return minimum


def truncate_tool_results(
    messages: Sequence[Message],
    target: int,
    result_seqs: Mapping[int, int],
    token_counter: Callable[[Message], int],
) -> list[Message] | None:
    def count(values: Sequence[Message]) -> int:
        return sum(token_counter(message) for message in values)

    if count(messages) <= target:
        return list(messages)
    result = list(messages)
    candidates: list[tuple[int, int, int, str]] = []
    for index, message in enumerate(messages):
        tool_result = message.tool_result
        seq = result_seqs.get(id(message))
        if tool_result is None or seq is None:
            continue
        content = (
            flatten_tool_content(tool_result.content_blocks, detailed_images=True)
            if tool_result.content_blocks is not None
            else tool_result.content
        )
        candidates.append((-len(content), seq, index, content))
    for _, seq, index, content in sorted(candidates):
        if count(result) <= target:
            break
        low, high = 0, len(content)
        best: Message | None = None
        while low <= high:
            shown = (low + high) // 2
            replacement = _truncated_tool_message(result[index], content, shown, seq)
            proposed = [*result[:index], replacement, *result[index + 1 :]]
            if count(proposed) <= target:
                best = replacement
                low = shown + 1
            else:
                high = shown - 1
        result[index] = best or _truncated_tool_message(result[index], content, 0, seq)
    return result if count(result) <= target else None


def compaction_source(
    candidates: Sequence[ContextItem],
    pinned_user: ContextItem | None,
) -> tuple[int, int, list[str]]:
    source_ranges: list[tuple[int, int]] = []
    replaces: list[str] = []
    for item in (*candidates, pinned_user):
        if item is None or item.entry is None:
            continue
        entry = item.entry
        eviction_marker_id = entry.data.get("eviction_marker_id")
        if isinstance(eviction_marker_id, str):
            source_ranges.append(
                (
                    entry.data["eviction_source_seq_start"],
                    entry.data["eviction_source_seq_end"],
                )
            )
            if eviction_marker_id not in replaces:
                replaces.append(eviction_marker_id)
        elif entry.type == "compaction":
            source_ranges.append(
                (entry.data["source_seq_start"], entry.data["source_seq_end"])
            )
            if entry.id not in replaces:
                replaces.append(entry.id)
        else:
            source_ranges.append((entry.seq, entry.seq))
    return (
        min(start for start, _ in source_ranges),
        max(end for _, end in source_ranges),
        replaces,
    )


def strategy_records(
    candidates: Sequence[ContextItem], source_start: int
) -> list[tuple[int, Message]]:
    return [
        (
            item.message.metadata.get(
                "source_seq",
                item.message.metadata.get(
                    "source_seq_start",
                    item.entry.seq if item.entry is not None else source_start,
                ),
            ),
            item.message,
        )
        for item in candidates
    ]


def _minimum_tail_boundary(items: Sequence[ContextItem]) -> int:
    if not items:
        return 0
    last = len(items) - 1
    message = items[last].message
    if message.role is not MessageRole.TOOL_RESULT or message.tool_result is None:
        return last
    call_id = message.tool_result.tool_call_id
    for index in range(last - 1, -1, -1):
        if any(
            isinstance(block, ToolUseContent) and block.tool_call.id == call_id
            for block in items[index].message.content
        ):
            return index
    return last


def _is_valid_tail_boundary(items: Sequence[ContextItem], boundary: int) -> bool:
    call_indexes: dict[str, int] = {}
    for index, item in enumerate(items):
        for block in item.message.content:
            if isinstance(block, ToolUseContent):
                call_indexes[block.tool_call.id] = index
    return all(
        item.message.tool_result is None
        or call_indexes.get(item.message.tool_result.tool_call_id, -1) >= boundary
        for item in items[boundary:]
    )


def _truncated_tool_message(
    message: Message,
    content: str,
    shown: int,
    seq: int,
) -> Message:
    head_size = (shown + 1) // 2
    tail_size = shown - head_size
    excerpt = content[:head_size]
    if tail_size:
        excerpt += "\n…\n" + content[len(content) - tail_size :]
    marker = (
        f"[output truncated for context: showed {shown} of {len(content)} chars; "
        f"full output is in the session log at seq {seq}]"
    )
    excerpt = f"{excerpt}\n{marker}" if excerpt else marker
    original = message.tool_result
    if original is None:
        return message
    return Message(
        message.role,
        [],
        tool_result=ToolResult(
            original.tool_call_id,
            excerpt,
            is_error=original.is_error,
            is_canceled=original.is_canceled,
        ),
        metadata=dict(message.metadata),
    )
