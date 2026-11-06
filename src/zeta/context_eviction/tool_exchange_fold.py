"""Atomic receipt conversion and folding for complete tool exchanges."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

from ..protocol.types import Message, ToolCall, ToolUseContent
from .range_receipts import range_receipt_candidate
from .receipt_constructors import _generic_result_receipt, _tool_call_receipt


def fold_consumed_tool_exchanges(
    records: Sequence[tuple[int, Message]],
    *,
    allows: Callable[[int], bool],
    token_counter: Callable[[Message], int],
) -> tuple[list[tuple[int, Message]], set[int]]:
    """Receiptize complete old exchanges only when their final fold is smaller."""
    messages = [message for _, message in records]
    calls = _tool_calls(messages)
    result_indexes: dict[str, list[int]] = defaultdict(list)
    for index, message in enumerate(messages):
        if message.tool_result is not None:
            result_indexes[message.tool_result.tool_call_id].append(index)

    converted = list(records)
    converted_seqs: set[int] = set()
    for call_index, (call_seq, message) in enumerate(records):
        tool_calls = [
            block.tool_call
            for block in message.content
            if isinstance(block, ToolUseContent)
        ]
        if not tool_calls:
            continue
        indexes = [call_index]
        for call in tool_calls:
            matches = result_indexes.get(call.id, ())
            if not matches:
                indexes = []
                break
            indexes.extend(matches)
        if not indexes or not all(allows(records[index][0]) for index in indexes):
            continue
        if not message.metadata.get("context_evicted"):
            converted[call_index] = (
                call_seq,
                _tool_call_receipt(message, call_seq),
            )
            converted_seqs.add(call_seq)
        for index in indexes[1:]:
            seq, result_message = records[index]
            result = result_message.tool_result
            call = calls.get(result.tool_call_id) if result is not None else None
            if call is not None and not result_message.metadata.get("context_evicted"):
                converted[index] = (
                    seq,
                    _generic_result_receipt(result_message, call, seq),
                )
                converted_seqs.add(seq)

    def smaller(
        originals: Sequence[tuple[int, Message]], candidates: Sequence[Message]
    ) -> bool:
        return sum(map(token_counter, candidates)) < sum(
            token_counter(message) for _, message in originals
        )

    fine = range_receipt_candidate(converted, allows=allows, is_smaller=smaller)
    coarse = range_receipt_candidate(
        fine.records,
        allows=allows,
        is_smaller=smaller,
        coarse=True,
    )
    candidate = coarse if coarse.coalesced_source_seqs else fine
    if not candidate.coalesced_source_seqs or not smaller(
        records, [message for _, message in candidate.records]
    ):
        return list(records), set()
    return candidate.records, converted_seqs | set(candidate.coalesced_source_seqs)


def _tool_calls(messages: Sequence[Message]) -> dict[str, ToolCall]:
    return {
        block.tool_call.id: block.tool_call
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
