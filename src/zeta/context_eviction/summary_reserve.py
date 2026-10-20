from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

from ..protocol.types import Message
from . import (
    EvictionResult,
    _evict_deterministic,
    _finalize_summaries,
    estimated_tokens,
)
from .summaries import SummaryCache
from .summary_receipts import receipt_summary_request


def evict_messages(records: Sequence[tuple[int, Message]], *, fixed_tokens: int, target_tokens: int, token_counter: Callable[[Message], int] | None = None, unconsumed_source_seqs: Collection[int] = (), source_messages: Mapping[int, Message] | None = None, summary_cache: SummaryCache | None = None, summaries_enabled: bool = False) -> EvictionResult:
    if token_counter is None:
        token_counter = estimated_tokens
    result = _evict_deterministic(records, fixed_tokens=fixed_tokens, target_tokens=target_tokens, token_counter=token_counter, unconsumed_source_seqs=unconsumed_source_seqs, source_messages=source_messages, summary_cache=summary_cache, summaries_enabled=summaries_enabled)
    reserve = 0
    if summaries_enabled and summary_cache is not None and result.reached_target:
        sources = source_messages or {}
        ready_count = sum(receipt_summary_request(message, sources) is not None and summary_cache.get(receipt_summary_request(message, sources).key) is not None for message in result.messages)
        reserve = target_tokens // 10 if ready_count else 0
        if reserve:
            result = _evict_deterministic(records, fixed_tokens=fixed_tokens, target_tokens=target_tokens - reserve, token_counter=token_counter, unconsumed_source_seqs=unconsumed_source_seqs, source_messages=source_messages, summary_cache=summary_cache, summaries_enabled=summaries_enabled)
    result = EvictionResult(result.messages, result.items_evicted, result.tokens_before, result.tokens_after, result.reached_target if not reserve else True, result.source_seqs, result.summary_misses, reserve)
    return _finalize_summaries(result, fixed_tokens=fixed_tokens, target_tokens=target_tokens, token_counter=token_counter, source_messages=source_messages, summary_cache=summary_cache, summaries_enabled=summaries_enabled)
