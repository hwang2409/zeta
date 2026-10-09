"""Deterministic range receipts for old per-item eviction receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..protocol.types import Message, MessageRole, TextContent, ToolUseContent

_RANGE_LIMIT = 440
_MAX_TOOL_KINDS = 12
RECEIPT_KIND_METADATA = "eviction_receipt"


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
    is_smaller: Callable[[Sequence[Message], Sequence[Message]], bool],
) -> RangeCandidate:
    """Coalesce smaller maximal runs; existing ranges permanently break runs.

    A provider tool call and all its receipted results are one unit. This removes
    the complete exchange or leaves it intact, so no provider sees an orphan.
    Each run is accepted separately through the caller's shared size policy.
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
        if any(
            not isinstance(block, ToolUseContent)
            and not (
                isinstance(block, TextContent)
                and _is_generated_assistant_receipt(block.text, message, seq)
            )
            for block in message.content
        ):
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
                result_message, result_seq, "tool_result"
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


def _is_receipt_kind(message: Message, seq: int, kind: str) -> bool:
    marked = message.metadata.get(RECEIPT_KIND_METADATA)
    if marked is not None:
        return (
            marked == kind
            and message.metadata.get("source_seq") == seq
            and (
                (
                    kind == "tool_result"
                    and message.tool_result is not None
                    and not message.content
                    and message.tool_result.content_blocks is None
                    and message.tool_result.structured_content is None
                )
                or (
                    kind != "tool_result"
                    and message.tool_result is None
                    and len(message.content) == 1
                    and isinstance(message.content[0], TextContent)
                )
            )
        )
    if (
        not message.metadata.get("context_evicted")
        or message.metadata.get("source_seq") != seq
    ):
        return False
    if kind == "tool_result":
        result = message.tool_result
        if (
            result is None
            or message.content
            or result.content_blocks is not None
            or result.structured_content is not None
            or "eviction_content_digest" not in message.metadata
        ):
            return False
        return (
            f" · seq {seq}]" in result.content
            and result.content.startswith("[semantic ")
        ) or (
            result.content.startswith(
                ("[orchestration result receipt] ", "[workflow result receipt] ")
            )
            and f"seq_start={seq}, seq_end={seq} for exact result" in result.content
        )
    if (
        message.tool_result is not None
        or len(message.content) != 1
        or not isinstance(message.content[0], TextContent)
    ):
        return False
    text = message.content[0].text
    if kind == "notification":
        return (
            "eviction_content_digest" in message.metadata
            and text.startswith("[notification receipt] ")
            and f"seq_start={seq}, seq_end={seq} for exact notification" in text
        )
    if _is_generated_assistant_receipt(text, message, seq):
        return True
    collapsed_into = message.metadata.get("collapsed_into_seq")
    return type(collapsed_into) is int and text in {
        f"[older duplicate read collapsed into seq {collapsed_into}]",
        f"[duplicate result collapsed into seq {collapsed_into}]",
    }


def _is_generated_assistant_receipt(text: str, message: Message, seq: int) -> bool:
    return message.metadata.get("source_seq") == seq and text in {
        f"[assistant text evicted · seq {seq}]",
        f"[assistant reasoning evicted · seq {seq}]",
    }


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
            RECEIPT_KIND_METADATA: "range",
            "eviction_range": True,
            "source_seq": start,
            "range_seq_end": end,
        },
    )
