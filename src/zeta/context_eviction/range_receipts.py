"""Deterministic range receipts for old per-item eviction receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..protocol.types import Message, MessageRole, TextContent, ToolUseContent

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
) -> RangeCandidate:
    """Coalesce maximal runs, while existing ranges permanently break later runs.

    A provider tool call and all its receipted results are one unit. This removes
    the complete exchange or leaves it intact, so no provider sees an orphan.
    """

    output: list[tuple[int, Message]] = []
    coalesced: set[int] = set()
    run: list[_ReceiptUnit] = []
    index = 0

    def flush() -> None:
        if len(run) < 2:
            for unit in run:
                output.extend(unit.records)
        else:
            flattened = [record for unit in run for record in unit.records]
            start, end = flattened[0][0], flattened[-1][0]
            output.append((start, _range_receipt(start, end, run)))
            coalesced.update(seq for seq, _ in flattened)
        run.clear()

    while index < len(records):
        unit, next_index = _receipt_unit(records, index, allows)
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
    allows: Callable[[int], bool],
) -> tuple[_ReceiptUnit | None, int]:
    seq, message = records[index]
    if not allows(seq) or message.metadata.get("eviction_range"):
        return None, index + 1
    if message.role is MessageRole.USER:
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
            if not allows(result_seq) or not result_message.metadata.get("context_evicted"):
                return None, index + 1
            grouped.append(records[cursor])
            names.append(expected.pop(result.tool_call_id))
            cursor += 1
        if expected:
            return None, index + 1
        return _ReceiptUnit(tuple(grouped), "tool", tuple(names)), cursor

    if message.tool_result is not None or not message.metadata.get("context_evicted"):
        return None, index + 1
    text = "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
    kind = "notification" if "[notification receipt]" in text else "assistant"
    return _ReceiptUnit((records[index],), kind), index + 1


def _range_receipt(start: int, end: int, units: Sequence[_ReceiptUnit]) -> Message:
    kinds = Counter(unit.kind for unit in units)
    tools = Counter(name for unit in units for name in unit.tool_names)
    parts: list[str] = []
    if tools:
        sorted_tools = sorted(tools.items())
        shown = sorted_tools[:_MAX_TOOL_KINDS]
        breakdown = ", ".join(f"{name} {count}" for name, count in shown)
        if len(shown) < len(sorted_tools):
            breakdown += f", other {sum(count for _, count in sorted_tools[len(shown):])}"
        count = sum(tools.values())
        parts.append(f"{count} tool {'result' if count == 1 else 'results'} ({breakdown})")
    if kinds["notification"]:
        count = kinds["notification"]
        parts.append(f"{count} {'notification' if count == 1 else 'notifications'}")
    if kinds["assistant"]:
        count = kinds["assistant"]
        parts.append(f"{count} assistant {'note' if count == 1 else 'notes'}")
    summary = ", ".join(parts)
    suffix = (
        f"; recall_history seq_start={start}, seq_end={end} for exact content"
    )
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
