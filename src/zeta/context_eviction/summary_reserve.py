from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..protocol.types import Message
from .summaries import SummaryCache, SummaryRequest
from .summary_receipts import decorate_receipt, receipt_summary_request


def evict_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int],
    unconsumed_source_seqs: Collection[int],
    source_messages: Mapping[int, Message] | None,
    summary_cache: SummaryCache | None,
    summaries_enabled: bool,
    deterministic_evict: Callable[..., Any],
) -> Any:
    """Run deterministic eviction and at most one summary-reserve pass."""
    arguments = {
        "fixed_tokens": fixed_tokens,
        "token_counter": token_counter,
        "unconsumed_source_seqs": unconsumed_source_seqs,
        "source_messages": source_messages,
        "summary_cache": summary_cache,
        "summaries_enabled": summaries_enabled,
    }
    result = deterministic_evict(records, target_tokens=target_tokens, **arguments)
    first_pass_tokens = result.tokens_after
    reserve = 0
    summaries_ready = (
        summaries_enabled
        and summary_cache is not None
        and source_messages is not None
        and result.reached_target
    )
    if summaries_ready:
        reserve = estimate_summary_reserve(
            result.messages,
            result.tokens_after,
            target_tokens,
            token_counter,
            source_messages,
            summary_cache,
        )
        if reserve:
            result = deterministic_evict(
                records,
                target_tokens=max(0, first_pass_tokens - target_tokens // 10),
                summary_target_tokens=target_tokens,
                **arguments,
            )
            reserve = estimate_summary_reserve(
                result.messages,
                first_pass_tokens,
                target_tokens,
                token_counter,
                source_messages,
                summary_cache,
            )
    reached_target = result.reached_target if not reserve else True
    misses = result.summary_misses
    messages, tokens_after = result.messages, result.tokens_after
    if summaries_enabled and summary_cache is not None and source_messages is not None:
        messages, tokens_after, misses = finalize_summaries(
            messages,
            reached_target=reached_target,
            fixed_tokens=fixed_tokens,
            target_tokens=target_tokens,
            token_counter=token_counter,
            source_messages=source_messages,
            summary_cache=summary_cache,
        )
    return replace(
        result,
        messages=messages,
        tokens_after=tokens_after,
        reached_target=reached_target,
        summary_misses=misses,
        reserve_tokens=reserve,
    )


def ready_decoration_growth(
    messages: Sequence[Message],
    *,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> int:
    """Return growth from ready decorations that are smaller than their sources."""
    growth = 0
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
            growth += decorated_tokens - token_counter(message)
    return growth


class ReserveFoldTracker:
    """Track the projected decorated total while range folds change topology."""

    def __init__(
        self,
        messages: Sequence[Message],
        *,
        fixed_tokens: int,
        target_tokens: int,
        cap_tokens: int,
        token_counter: Callable[[Message], int],
        source_messages: Mapping[int, Message],
        summary_cache: SummaryCache,
    ) -> None:
        self._token_counter = token_counter
        self._source_messages = source_messages
        self._summary_cache = summary_cache
        self._target_tokens = target_tokens
        self._cap_tokens = cap_tokens
        self._deterministic_tokens = fixed_tokens + sum(map(token_counter, messages))
        self._decoration_growth = ready_decoration_growth(
            messages,
            token_counter=token_counter,
            source_messages=source_messages,
            summary_cache=summary_cache,
        )
        self.done = False

    def __call__(
        self,
        originals: Sequence[tuple[int, Message]],
        candidates: Sequence[Message],
    ) -> bool:
        original_messages = [message for _, message in originals]
        self._deterministic_tokens += sum(map(self._token_counter, candidates)) - sum(
            map(self._token_counter, original_messages)
        )
        self._decoration_growth += ready_decoration_growth(
            candidates,
            token_counter=self._token_counter,
            source_messages=self._source_messages,
            summary_cache=self._summary_cache,
        ) - ready_decoration_growth(
            original_messages,
            token_counter=self._token_counter,
            source_messages=self._source_messages,
            summary_cache=self._summary_cache,
        )
        self.done = (
            self._deterministic_tokens + self._decoration_growth <= self._target_tokens
            or self._deterministic_tokens <= self._cap_tokens
        )
        return self.done


def estimate_summary_reserve(
    messages: Sequence[Message],
    current_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> int:
    """Return ready decoration growth beyond existing target headroom."""
    ready_extra = ready_decoration_growth(
        messages,
        token_counter=token_counter,
        source_messages=source_messages,
        summary_cache=summary_cache,
    )
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
        if (
            decorated_tokens < original_tokens
            and reached_target
            and next_total <= target_tokens
        ):
            finalized[index] = decorated
            running_total = next_total
    return finalized, running_total, tuple(misses)
