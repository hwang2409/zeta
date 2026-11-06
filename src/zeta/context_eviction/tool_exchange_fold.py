"""Atomic receipt conversion and folding for complete tool exchanges."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence

from ..protocol.types import Message, ToolCall, ToolUseContent
from .range_receipts import range_receipt_candidate
from .receipt_constructors import _generic_result_receipt, _tool_call_receipt

Record = tuple[int, Message]


def fold_consumed_tool_exchanges(
    records: Sequence[Record],
    *,
    allows: Callable[[int], bool],
    token_counter: Callable[[Message], int],
    target_tokens: int,
) -> tuple[list[Record], set[int]]:
    """Fold the oldest complete exchanges until the message target is reached."""
    messages = [message for _, message in records]
    calls = _tool_calls(messages)
    result_indexes: dict[str, list[int]] = defaultdict(list)
    for index, message in enumerate(messages):
        if message.tool_result is not None:
            result_indexes[message.tool_result.tool_call_id].append(index)

    token_cache: dict[int, tuple[Message, int]] = {}

    def count(message: Message) -> int:
        cached = token_cache.get(id(message))
        if cached is None:
            cached = (message, token_counter(message))
            token_cache[id(message)] = cached
        return cached[1]

    original_tokens = sum(map(count, messages))
    converted = list(records)
    converted_seqs: set[int] = set()
    best = list(records)
    best_seqs: set[int] = set()
    best_tokens = original_tokens

    for indexes in _exchange_indexes(records, result_indexes, allows):
        converted_seqs.update(
            _convert_exchange(records, converted, indexes=indexes, calls=calls)
        )
        candidate, coalesced = _fold_candidate(
            converted,
            selected_until=max(records[index][0] for index in indexes),
            allows=allows,
            count=count,
        )
        candidate_tokens = sum(count(message) for _, message in candidate)
        if candidate_tokens >= original_tokens:
            continue
        best = candidate
        best_tokens = candidate_tokens
        best_seqs = converted_seqs | coalesced
        if best_tokens <= target_tokens:
            break

    if best_tokens >= original_tokens:
        return list(records), set()
    return best, best_seqs


def _exchange_indexes(
    records: Sequence[Record],
    result_indexes: dict[str, list[int]],
    allows: Callable[[int], bool],
) -> Iterator[list[int]]:
    for call_index, (_, message) in enumerate(records):
        call_ids = [
            block.tool_call.id
            for block in message.content
            if isinstance(block, ToolUseContent)
        ]
        indexes = _complete_exchange_indexes(call_index, call_ids, result_indexes)
        if indexes and all(allows(records[index][0]) for index in indexes):
            yield indexes


def _complete_exchange_indexes(
    call_index: int,
    call_ids: Sequence[str],
    result_indexes: dict[str, list[int]],
) -> list[int]:
    if not call_ids or any(call_id not in result_indexes for call_id in call_ids):
        return []
    return [
        call_index,
        *(index for call_id in call_ids for index in result_indexes[call_id]),
    ]


def _convert_exchange(
    records: Sequence[Record],
    converted: list[Record],
    *,
    indexes: Sequence[int],
    calls: dict[str, ToolCall],
) -> set[int]:
    changed: set[int] = set()
    for index in indexes:
        seq, message = records[index]
        replacement = _exchange_receipt(message, seq, calls)
        if replacement is not message:
            converted[index] = (seq, replacement)
            changed.add(seq)
    return changed


def _exchange_receipt(
    message: Message, seq: int, calls: dict[str, ToolCall]
) -> Message:
    if message.metadata.get("context_evicted"):
        return message
    if message.tool_result is None:
        return _tool_call_receipt(message, seq)
    call = calls.get(message.tool_result.tool_call_id)
    if call is None:
        return message
    return _generic_result_receipt(message, call, seq)


def _fold_candidate(
    records: Sequence[Record],
    *,
    selected_until: int,
    allows: Callable[[int], bool],
    count: Callable[[Message], int],
) -> tuple[list[Record], set[int]]:
    def selected(seq: int) -> bool:
        return seq <= selected_until and allows(seq)

    def smaller(originals: Sequence[Record], candidates: Sequence[Message]) -> bool:
        return sum(map(count, candidates)) < sum(
            count(message) for _, message in originals
        )

    fine = range_receipt_candidate(records, allows=selected, is_smaller=smaller)
    coarse = range_receipt_candidate(
        fine.records,
        allows=selected,
        is_smaller=smaller,
        coarse=True,
    )
    candidate = coarse if coarse.coalesced_source_seqs else fine
    return candidate.records, set(
        fine.coalesced_source_seqs | coarse.coalesced_source_seqs
    )


def _tool_calls(messages: Sequence[Message]) -> dict[str, ToolCall]:
    return {
        block.tool_call.id: block.tool_call
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
