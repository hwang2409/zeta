from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

from ..protocol.types import Message
from .summaries import SummaryCache
from .summary_receipts import decorate_receipt, receipt_summary_request


def evict_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int] | None = None,
    unconsumed_source_seqs: Collection[int] = (),
    source_messages: Mapping[int, Message] | None = None,
    summary_cache: SummaryCache | None = None,
    summaries_enabled: bool = False,
):
    from . import (
        EvictionResult,
        _evict_deterministic,
        _finalize_summaries,
        estimated_tokens,
    )

    if token_counter is None:
        token_counter = estimated_tokens
    result = _evict_deterministic(
        records,
        fixed_tokens=fixed_tokens,
        target_tokens=target_tokens,
        token_counter=token_counter,
        unconsumed_source_seqs=unconsumed_source_seqs,
        source_messages=source_messages,
        summary_cache=summary_cache,
        summaries_enabled=summaries_enabled,
    )
    pass_target = target_tokens
    if summaries_enabled and summary_cache is not None and result.reached_target:
        ready_extra = 0
        sources = source_messages or {}
        for message in result.messages:
            request = receipt_summary_request(message, sources)
            if request is None or (summary := summary_cache.get(request.key)) is None:
                continue
            decorated = decorate_receipt(message, summary)
            source_tokens = sum(
                token_counter(source)
                for seq, source in sources.items()
                if request.key.seq_start <= seq <= request.key.seq_end
            )
            if token_counter(decorated) < source_tokens:
                ready_extra += token_counter(decorated) - token_counter(message)
        reserve = min(
            max(0, ready_extra - (target_tokens - result.tokens_after)),
            target_tokens // 10,
        )
        if reserve:
            pass_target = target_tokens - reserve
            result = _evict_deterministic(
                records,
                fixed_tokens=fixed_tokens,
                target_tokens=pass_target,
                token_counter=token_counter,
                unconsumed_source_seqs=unconsumed_source_seqs,
                source_messages=source_messages,
                summary_cache=summary_cache,
                summaries_enabled=summaries_enabled,
            )
    result = EvictionResult(
        result.messages,
        result.items_evicted,
        result.tokens_before,
        result.tokens_after,
        result.reached_target if pass_target == target_tokens else True,
        result.source_seqs,
        result.summary_misses,
    )
    return _finalize_summaries(
        result,
        fixed_tokens=fixed_tokens,
        target_tokens=target_tokens,
        token_counter=token_counter,
        source_messages=source_messages,
        summary_cache=summary_cache,
        summaries_enabled=summaries_enabled,
    )
