from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ..protocol.types import Message
from .summaries import SummaryCache, SummaryRequest
from .summary_receipts import decorate_receipt, receipt_summary_request


def estimate_summary_reserve(
    messages: Sequence[Message],
    *,
    current_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> int:
    """Return ready decoration growth beyond existing target headroom."""
    ready_extra = 0
    for message in messages:
        request = receipt_summary_request(message, source_messages)
        if request is None or (summary := summary_cache.get(request.key)) is None:
            continue
        decorated = decorate_receipt(message, summary)
        source_tokens = sum(
            token_counter(source)
            for seq, source in source_messages.items()
            if request.key.seq_start <= seq <= request.key.seq_end
        )
        decorated_tokens = token_counter(decorated)
        if decorated_tokens < source_tokens:
            ready_extra += decorated_tokens - token_counter(message)
    return min(
        max(0, ready_extra - (target_tokens - current_tokens)),
        target_tokens // 10,
    )


def finalize_summaries(
    messages: Sequence[Message],
    *,
    reached_target: bool,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> tuple[list[Message], int, tuple[SummaryRequest, ...]]:
    """Decorate ready receipts in order without exceeding the original target."""
    finalized = list(messages)
    running_total = fixed_tokens + sum(map(token_counter, finalized))
    misses: list[SummaryRequest] = []
    for index, message in enumerate(finalized):
        request = receipt_summary_request(message, source_messages)
        if request is None:
            continue
        summary = summary_cache.get(request.key)
        if summary is None:
            if request.exact_range:
                misses.append(request)
            continue
        decorated = decorate_receipt(message, summary)
        original_tokens = sum(
            token_counter(source)
            for seq, source in source_messages.items()
            if request.key.seq_start <= seq <= request.key.seq_end
        )
        decorated_tokens = token_counter(decorated)
        next_total = running_total - token_counter(message) + decorated_tokens
        if decorated_tokens < original_tokens and reached_target and next_total <= target_tokens:
            finalized[index] = decorated
            running_total = next_total
    return finalized, running_total, tuple(misses)
