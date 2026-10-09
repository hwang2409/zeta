"""Deterministic range receipts for old per-item eviction receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..protocol.types import (
    ContentBlock,
    ImageContent,
    Message,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)

_RANGE_LIMIT = 440
_MAX_TOOL_KINDS = 12


@dataclass(frozen=True, slots=True)
class RangeCandidate:
    """A complete replacement view and the source sequences it coalesces."""

    records: list[tuple[int, Message]]
    coalesced_source_seqs: frozenset[int]


@dataclass(frozen=True, slots=True)
class _ReceiptUnit:
    records: tuple[tuple[int, Message], ...]
    kind: str
    tool_names: tuple[str, ...] = ()


def range_receipt_candidate(
    records: Sequence[tuple[int, Message]],
    *,
    source_messages: Mapping[int, Message],
    allows: Callable[[int], bool],
    is_smaller: Callable[[Sequence[Message], Sequence[Message]], bool],
) -> RangeCandidate:
    """Coalesce maximal runs that contain no content from their source rows.

    A provider tool call and all its receipted results are one unit. Unchanged
    tool-use blocks are structural pairing, not removable raw content; rewritten
    argument values must not retain their original payload. The source rows stay
    available through ``recall_history`` after the whole unit is coalesced.
    """

    output: list[tuple[int, Message]] = []
    coalesced: set[int] = set()
    run: list[_ReceiptUnit] = []
    index = 0

    def flush() -> None:
        flattened = [record for unit in run for record in unit.records]
        if len(run) >= 2:
            start, end = flattened[0][0], flattened[-1][0]
            receipt = _range_receipt(start, end, run)
            if is_smaller([message for _, message in flattened], [receipt]):
                output.append((start, receipt))
                coalesced.update(seq for seq, _ in flattened)
            else:
                output.extend(flattened)
        else:
            output.extend(flattened)
        run.clear()

    while index < len(records):
        unit, next_index = _receipt_unit(
            records, index, source_messages=source_messages, allows=allows
        )
        if unit is None:
            flush()
            output.append(records[index])
            index += 1
            continue
        if run and unit.records[0][0] != run[-1].records[-1][0] + 1:
            flush()
        run.append(unit)
        index = next_index
    flush()
    return RangeCandidate(output, frozenset(coalesced))


def _receipt_unit(
    records: Sequence[tuple[int, Message]],
    index: int,
    *,
    source_messages: Mapping[int, Message],
    allows: Callable[[int], bool],
) -> tuple[_ReceiptUnit | None, int]:
    seq, message = records[index]
    source = source_messages.get(seq)
    if (
        not allows(seq)
        or message.metadata.get("eviction_range")
        or message.role is MessageRole.USER
        or source is None
        or source.role is MessageRole.USER
        or _shares_source_content(message, source)
    ):
        return None, index + 1

    calls = [
        block.tool_call
        for block in message.content
        if isinstance(block, ToolUseContent)
    ]
    if calls:
        expected = {call.id: call.name for call in calls}
        grouped = [records[index]]
        names: list[str] = []
        cursor = index + 1
        while cursor < len(records):
            result_seq, result_message = records[cursor]
            result = result_message.tool_result
            if result is None or result.tool_call_id not in expected:
                break
            result_source = source_messages.get(result_seq)
            if (
                not allows(result_seq)
                or result_source is None
                or result_message.role is MessageRole.USER
                or result_source.role is MessageRole.USER
                or _shares_source_content(result_message, result_source)
            ):
                return None, index + 1
            grouped.append(records[cursor])
            names.append(expected.pop(result.tool_call_id))
            cursor += 1
        if expected:
            return None, index + 1
        return _ReceiptUnit(tuple(grouped), "tool", tuple(names)), cursor

    if message.tool_result is not None or any(
        isinstance(block, ToolUseContent) for block in message.content
    ):
        return None, index + 1
    kind = "notification" if source.role is MessageRole.SYSTEM else "assistant"
    return _ReceiptUnit((records[index],), kind), index + 1


def _shares_source_content(view: Message, source: Message) -> bool:
    """Return whether provider-visible source content survives in the view."""

    for source_block in source.content:
        if isinstance(source_block, ToolUseContent):
            view_call = next(
                (
                    block.tool_call
                    for block in view.content
                    if isinstance(block, ToolUseContent)
                    and block.tool_call.id == source_block.tool_call.id
                ),
                None,
            )
            if view_call is None:
                continue
            for key, source_value in source_block.tool_call.arguments.items():
                if (
                    key in view_call.arguments
                    and view_call.arguments[key] != source_value
                    and _payload_survives(source_value, view_call.arguments[key])
                ):
                    return True
            continue
        if any(
            not isinstance(view_block, ToolUseContent)
            and _block_payload_survives(source_block, view_block)
            for view_block in view.content
        ):
            return True

    source_result = source.tool_result
    view_result = view.tool_result
    if source_result is None or view_result is None:
        return False
    if _payload_survives(source_result.content, view_result.content):
        return True
    if (
        source_result.content_blocks is not None
        and view_result.content_blocks is not None
        and any(
            _tool_block_payload_survives(source_block, view_block)
            for source_block in source_result.content_blocks
            for view_block in view_result.content_blocks
        )
    ):
        return True
    return (
        source_result.structured_content is not None
        and view_result.structured_content is not None
        and _payload_survives(
            source_result.structured_content, view_result.structured_content
        )
    )


def _block_payload_survives(source: ContentBlock, view: ContentBlock) -> bool:
    if type(source) is not type(view):
        return False
    if isinstance(source, (TextContent, ThinkingContent)):
        assert isinstance(view, (TextContent, ThinkingContent))
        return _payload_survives(source.text, view.text)
    if isinstance(source, (ImageContent, RedactedThinkingContent)):
        assert isinstance(view, (ImageContent, RedactedThinkingContent))
        return _payload_survives(source.data, view.data)
    return False


def _tool_block_payload_survives(
    source: Mapping[str, object], view: Mapping[str, object]
) -> bool:
    if source.get("type") != view.get("type"):
        return False
    for key in ("text", "data"):
        if key in source and key in view:
            return _payload_survives(source[key], view[key])
    source_resource = source.get("resource")
    view_resource = view.get("resource")
    if isinstance(source_resource, Mapping) and isinstance(view_resource, Mapping):
        for key in ("text", "blob"):
            if key in source_resource and key in view_resource:
                return _payload_survives(source_resource[key], view_resource[key])
    return source == view


def _payload_survives(source: object, view: object) -> bool:
    if isinstance(source, str) and isinstance(view, str):
        return bool(source) and source in view
    return source == view


def _range_receipt(start: int, end: int, units: Sequence[_ReceiptUnit]) -> Message:
    kinds = Counter(unit.kind for unit in units)
    tools = Counter(name for unit in units for name in unit.tool_names)
    parts: list[str] = []
    if tools:
        sorted_tools = sorted(tools.items())
        shown = sorted_tools[:_MAX_TOOL_KINDS]
        breakdown = ", ".join(f"{name} {count}" for name, count in shown)
        if len(shown) < len(sorted_tools):
            breakdown += (
                f", other {sum(count for _, count in sorted_tools[len(shown) :])}"
            )
        count = sum(tools.values())
        parts.append(
            f"{count} tool {'result' if count == 1 else 'results'} ({breakdown})"
        )
    if kinds["notification"]:
        count = kinds["notification"]
        parts.append(f"{count} {'notification' if count == 1 else 'notifications'}")
    if kinds["assistant"]:
        count = kinds["assistant"]
        parts.append(f"{count} assistant {'note' if count == 1 else 'notes'}")
    summary = ", ".join(parts)
    suffix = f"; recall_history seq_start={start}, seq_end={end} for exact content"
    text = f"[evicted range] seq {start}-{end}: {summary}{suffix}"
    if len(text) > _RANGE_LIMIT:
        text = f"[evicted range] seq {start}-{end}: {len(units)} receipt items{suffix}"
    return Message(
        MessageRole.ASSISTANT,
        [TextContent(text)],
        metadata={
            "context_evicted": True,
            "eviction_range": True,
            "source_seq": start,
            "range_seq_end": end,
        },
    )
