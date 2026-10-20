"""Deterministic range receipts for old per-item eviction receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..protocol.types import Message, MessageRole, TextContent, ToolUseContent
from .receipt_constructors import RECEIPT_FIELDS_METADATA, RECEIPT_KIND_METADATA

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
    allows: Callable[[int], bool],
    is_smaller: Callable[[Sequence[tuple[int, Message]], Sequence[Message]], bool],
    coarse: bool = False,
    compacted_tool_calls: bool = False,
    stop_after_fold: Callable[[Sequence[tuple[int, Message]], Sequence[Message]], bool]
    | None = None,
) -> RangeCandidate:
    """Coalesce receipt exchanges, or globally fold eligible old records."""

    if coarse:
        return _coarse_candidate(
            records,
            allows=allows,
            is_smaller=is_smaller,
            compacted_tool_calls=compacted_tool_calls,
        )

    output: list[tuple[int, Message]] = []
    coalesced: set[int] = set()
    run: list[_ReceiptUnit] = []
    index = 0

    def flush() -> bool:
        flattened = [record for unit in run for record in unit.records]
        folded: list[Message] | None = None
        if len(run) >= 2:
            start, end = flattened[0][0], flattened[-1][0]
            receipt = _range_receipt(start, end, run)
            if is_smaller(flattened, [receipt]):
                folded = [receipt]
                output.append((start, receipt))
                coalesced.update(seq for seq, _ in flattened)
            else:
                output.extend(flattened)
        else:
            output.extend(flattened)
        run.clear()
        return (
            folded is not None
            and stop_after_fold is not None
            and stop_after_fold(flattened, folded)
        )

    while index < len(records):
        unit, next_index = _receipt_unit(
            records, index, allows, compacted_tool_calls=compacted_tool_calls
        )
        if unit is None:
            if flush():
                output.extend(records[index:])
                return RangeCandidate(output, frozenset(coalesced))
            output.append(records[index])
            index += 1
            continue
        if run and unit.records[0][0] != run[-1].records[-1][0] + 1 and flush():
            output.extend(records[index:])
            return RangeCandidate(output, frozenset(coalesced))
        run.append(unit)
        index = next_index
    flush()
    return RangeCandidate(output, frozenset(coalesced))


def _coarse_candidate(
    records: Sequence[tuple[int, Message]],
    *,
    allows: Callable[[int], bool],
    is_smaller: Callable[[Sequence[tuple[int, Message]], Sequence[Message]], bool],
    compacted_tool_calls: bool,
) -> RangeCandidate:
    units: list[_ReceiptUnit] = []
    unit_indexes: set[int] = set()
    index = 0
    while index < len(records):
        unit, next_index = _coarse_receipt_unit(
            records, index, allows, compacted_tool_calls=compacted_tool_calls
        )
        if unit is not None:
            units.append(unit)
            unit_indexes.update(range(index, next_index))
        index = next_index

    if len(units) < 2 or not any(unit.kind != "user" for unit in units):
        return RangeCandidate(list(records), frozenset())
    flattened = [record for unit in units for record in unit.records]
    start = min(_unit_start(unit) for unit in units)
    end = max(_unit_end(unit) for unit in units)
    receipt = _range_receipt(start, end, units, include_user_metadata=True)
    if not is_smaller(flattened, [receipt]):
        return RangeCandidate(list(records), frozenset())
    output = [
        record for index, record in enumerate(records) if index not in unit_indexes
    ]
    output.append((start, receipt))
    output.sort(key=lambda record: record[0])
    return RangeCandidate(
        output,
        frozenset(seq for seq, _ in flattened),
    )


def _unit_start(unit: _ReceiptUnit) -> int:
    return min(seq for seq, _ in unit.records)


def _unit_end(unit: _ReceiptUnit) -> int:
    end = max(seq for seq, _ in unit.records)
    for seq, message in unit.records:
        if message.metadata.get("eviction_range"):
            range_end = message.metadata.get("range_seq_end")
            if type(range_end) is int:
                end = max(end, range_end)
    return end


def _coarse_receipt_unit(
    records: Sequence[tuple[int, Message]],
    index: int,
    allows: Callable[[int], bool],
    *,
    compacted_tool_calls: bool,
) -> tuple[_ReceiptUnit | None, int]:
    seq, message = records[index]
    if not allows(seq):
        return None, index + 1
    if message.role is MessageRole.USER:
        return _ReceiptUnit((records[index],), "user"), index + 1
    if message.metadata.get("eviction_range"):
        return _ReceiptUnit((records[index],), "range"), index + 1
    return _receipt_unit(
        records, index, allows, compacted_tool_calls=compacted_tool_calls
    )


def _receipt_unit(
    records: Sequence[tuple[int, Message]],
    index: int,
    allows: Callable[[int], bool],
    *,
    compacted_tool_calls: bool,
) -> tuple[_ReceiptUnit | None, int]:
    seq, message = records[index]
    if (
        not allows(seq)
        or message.metadata.get("eviction_range")
        or message.metadata.get("eviction_view_invalid")
    ):
        return None, index + 1
    if message.role is MessageRole.USER:
        return None, index + 1

    calls = [
        block.tool_call
        for block in message.content
        if isinstance(block, ToolUseContent)
    ]
    if calls:
        if any(not isinstance(block, ToolUseContent) for block in message.content):
            if not compacted_tool_calls or not _is_compacted_tool_call_receipt(
                message, seq
            ):
                return None, index + 1
        elif not _is_receipt_kind(message, seq, "tool_call"):
            return None, index + 1
        expected = {call.id: call.name for call in calls}
        grouped = [records[index]]
        names: list[str] = []
        cursor = index + 1
        while cursor < len(records):
            result_seq, result_message = records[cursor]
            result = result_message.tool_result
            if result is None or result.tool_call_id not in expected:
                break
            if not allows(result_seq) or not _is_receipt_kind(
                result_message,
                result_seq,
                "tool_result",
                tool_name=expected[result.tool_call_id],
            ):
                return None, index + 1
            grouped.append(records[cursor])
            names.append(expected.pop(result.tool_call_id))
            cursor += 1
        if expected:
            return None, index + 1
        return _ReceiptUnit(tuple(grouped), "tool", tuple(names)), cursor

    for kind in ("assistant", "notification"):
        if _is_receipt_kind(message, seq, kind):
            return _ReceiptUnit((records[index],), kind), index + 1
    return None, index + 1


def _is_compacted_tool_call_receipt(message: Message, seq: int) -> bool:
    marker = f"[assistant reasoning evicted · seq {seq}]"
    if (
        not message.content
        or not isinstance(message.content[-1], TextContent)
        or message.content[-1].text != marker
    ):
        return False
    calls = [
        block.tool_call
        for block in message.content[:-1]
        if isinstance(block, ToolUseContent)
    ]
    return (
        message.role is MessageRole.ASSISTANT
        and message.tool_result is None
        and message.metadata == {"context_evicted": True, "source_seq": seq}
        and len(calls) == len(message.content) - 1
        and bool(calls)
        and all(
            call.arguments == {"eviction_receipt": {"source_seq": seq}}
            for call in calls
        )
    )


def _is_receipt_kind(
    message: Message, seq: int, kind: str, *, tool_name: str | None = None
) -> bool:
    marked = message.metadata.get(RECEIPT_KIND_METADATA)
    if marked != kind:
        return False
    if message.metadata.get("source_seq") != seq:
        return False
    if kind == "assistant":
        rendered = _round_trip_assistant(message, seq)
    elif kind == "notification":
        rendered = _round_trip_notification(message, seq)
    elif kind == "tool_call":
        rendered = _round_trip_tool_call(message, seq)
    else:
        rendered = _round_trip_tool_result(message, seq, tool_name)
    return rendered is not None and rendered.to_dict() == message.to_dict()


def _round_trip_assistant(message: Message, seq: int) -> Message | None:
    from .receipt_constructors import _assistant_receipt, _collapsed_assistant_receipt

    fields = message.metadata.get(RECEIPT_FIELDS_METADATA)
    if not isinstance(fields, Mapping):
        return None
    kind = fields.get("kind")
    if isinstance(kind, str):
        return _assistant_receipt(
            seq,
            kind,
            role=message.role,
            metadata=_base_metadata(message, "collapsed_into_seq"),
        )
    collapsed_into = fields.get("collapsed_into_seq")
    older_read = fields.get("older_read")
    if type(collapsed_into) is not int or type(older_read) is not bool:
        return None
    return _collapsed_assistant_receipt(seq, collapsed_into, older_read=older_read)


def _round_trip_notification(message: Message, seq: int) -> Message | None:
    from .receipt_constructors import _notification_receipt_from_fields

    fields = message.metadata.get(RECEIPT_FIELDS_METADATA)
    payload = fields.get("payload") if isinstance(fields, Mapping) else None
    digest = message.metadata.get("eviction_content_digest")
    if not isinstance(payload, Mapping) or not isinstance(digest, str):
        return None
    return _notification_receipt_from_fields(
        role=message.role, seq=seq, payload=payload, content_digest=digest
    )


def _round_trip_tool_call(message: Message, seq: int) -> Message | None:
    from .receipt_constructors import _tool_call_receipt

    return _tool_call_receipt(message, seq, metadata=_base_metadata(message))


def _round_trip_tool_result(
    message: Message, seq: int, tool_name: str | None
) -> Message | None:
    from .receipt_constructors import (
        _semantic_result_receipt,
        _structured_result_receipt,
    )

    result = message.tool_result
    fields = message.metadata.get(RECEIPT_FIELDS_METADATA)
    digest = message.metadata.get("eviction_content_digest")
    if (
        result is None
        or not isinstance(fields, Mapping)
        or not isinstance(digest, str)
        or fields.get("tool_name") != tool_name
        or tool_name is None
    ):
        return None
    base = _base_metadata(message, "eviction_content_digest")
    semantic_digest = fields.get("digest")
    if isinstance(semantic_digest, str):
        return _semantic_result_receipt(
            role=message.role,
            tool_name=tool_name,
            tool_call_id=result.tool_call_id,
            seq=seq,
            digest=semantic_digest,
            content_digest=digest,
            is_error=result.is_error,
            is_canceled=result.is_canceled,
            metadata=base,
        )
    receipt_kind = fields.get("receipt_kind")
    payload = fields.get("payload")
    if receipt_kind not in {"orchestration", "workflow"} or not isinstance(
        payload, Mapping
    ):
        return None
    return _structured_result_receipt(
        role=message.role,
        receipt_kind=receipt_kind,
        tool_name=tool_name,
        tool_call_id=result.tool_call_id,
        seq=seq,
        payload=payload,
        content_digest=digest,
        is_error=result.is_error,
        is_canceled=result.is_canceled,
        metadata=base,
    )


def _base_metadata(message: Message, *extra_owned: str) -> dict[str, object]:
    owned = {
        "context_evicted",
        RECEIPT_KIND_METADATA,
        RECEIPT_FIELDS_METADATA,
        "source_seq",
        *extra_owned,
    }
    return {key: value for key, value in message.metadata.items() if key not in owned}


def _range_receipt(
    start: int,
    end: int,
    units: Sequence[_ReceiptUnit],
    *,
    include_user_metadata: bool = False,
) -> Message:
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
    user_count = 0
    users: list[str] = []
    for unit in units:
        if unit.kind == "user":
            user_count += 1
            users.extend(_user_excerpts(unit.records[0][1]))
        elif unit.kind == "range":
            message = unit.records[0][1]
            count = message.metadata.get("user_count")
            excerpts = message.metadata.get("user_excerpts")
            if type(count) is int and count >= 0:
                user_count += count
            if isinstance(excerpts, list):
                users.extend(
                    excerpt for excerpt in excerpts if isinstance(excerpt, str)
                )
    suffix = f"; recall_history seq_start={start}, seq_end={end} for exact content"
    if user_count:
        excerpts = [
            excerpt[:77] + "..." if len(excerpt) > 80 else excerpt for excerpt in users
        ]
        while excerpts:
            user_summary = (
                f"{user_count} user {'request' if user_count == 1 else 'requests'}: "
                + "; ".join(excerpts)
            )
            summary = ", ".join([*parts, user_summary])
            text = f"[evicted range] seq {start}-{end}: {summary}{suffix}"
            if len(text) <= _RANGE_LIMIT:
                break
            excerpts.pop(0)
        else:
            label = "request" if user_count == 1 else "requests"
            summary = ", ".join([*parts, f"{user_count} user {label}"])
            text = f"[evicted range] seq {start}-{end}: {summary}{suffix}"
    else:
        summary = ", ".join(parts)
        text = f"[evicted range] seq {start}-{end}: {summary}{suffix}"
    if len(text) > _RANGE_LIMIT:
        text = f"[evicted range] seq {start}-{end}: {len(units)} receipt items{suffix}"
    metadata = {
        "context_evicted": True,
        RECEIPT_KIND_METADATA: "range",
        "eviction_range": True,
        "source_seq": start,
        "range_seq_end": end,
    }
    if include_user_metadata:
        metadata.update(
            user_count=user_count,
            user_excerpts=excerpts if user_count else [],
        )
    return Message(MessageRole.ASSISTANT, [TextContent(text)], metadata=metadata)


def _user_excerpts(message: Message) -> list[str]:
    return [
        excerpt[:77] + "..."
        if len(excerpt := " ".join(block.text.split())) > 80
        else excerpt
        for block in message.content
        if isinstance(block, TextContent) and block.text.strip()
    ]
