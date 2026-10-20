from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..protocol.types import Message
from .summaries import SummaryCache, SummaryRequest
from .summary_receipts import (
    decorate_receipt,
    receipt_summary_request,
    summary_decoration_fits_source,
)


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
                target_tokens=max(0, first_pass_tokens - reserve),
                summary_target_tokens=target_tokens,
                **arguments,
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


@dataclass(frozen=True, slots=True)
class _ReadyDecoration:
    index: int
    message: Message
    request: SummaryRequest
    decorated: Message
    message_tokens: int
    decorated_tokens: int

    @property
    def growth(self) -> int:
        return self.decorated_tokens - self.message_tokens


def _summary_priority(item: _ReadyDecoration) -> tuple[int, int, int, int]:
    """Prioritize useful live-orchestrator receipts, newest first per tier."""
    kind = item.request.key.kind
    is_coarse = kind == "range" and "user_count" in item.message.metadata
    tier = (
        0
        if kind == "range" and not is_coarse
        else 1
        if kind in {"notification", "assistant"}
        else 2
    )
    return (tier, -item.request.key.seq_end, -item.request.key.seq_start, -item.index)


def _ready_decorations(
    messages: Sequence[Message],
    *,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> tuple[_ReadyDecoration, ...]:
    ready: list[_ReadyDecoration] = []
    for index, message in enumerate(messages):
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
        if summary_decoration_fits_source(
            request.key.kind,
            decorated_tokens=decorated_tokens,
            source_tokens=source_tokens,
        ):
            ready.append(
                _ReadyDecoration(
                    index,
                    message,
                    request,
                    decorated,
                    token_counter(message),
                    decorated_tokens,
                )
            )
    return tuple(sorted(ready, key=_summary_priority))


def ready_decoration_growth(
    messages: Sequence[Message],
    *,
    token_counter: Callable[[Message], int],
    source_messages: Mapping[int, Message],
    summary_cache: SummaryCache,
) -> int:
    """Return prioritized ready-decoration growth allowed by the source rules."""
    return sum(
        item.growth
        for item in _ready_decorations(
            messages,
            token_counter=token_counter,
            source_messages=source_messages,
            summary_cache=summary_cache,
        )
    )


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

    def has_ready_decoration(self, message: Message) -> bool:
        return bool(
            _ready_decorations(
                [message],
                token_counter=self._token_counter,
                source_messages=self._source_messages,
                summary_cache=self._summary_cache,
            )
        )

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
    """Return prioritized decoration growth beyond existing target headroom."""
    ready_extra = sum(
        item.growth
        for item in _ready_decorations(
            messages,
            token_counter=token_counter,
            source_messages=source_messages,
            summary_cache=summary_cache,
        )
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
    """Decorate newest ready receipts first without exceeding the original target."""
    finalized = list(messages)
    running_total = fixed_tokens + sum(map(token_counter, finalized))
    misses = [
        request
        for message in finalized
        if (request := receipt_summary_request(message, source_messages)) is not None
        and request.exact_range
        and summary_cache.get(request.key) is None
    ]
    ready = _ready_decorations(
        finalized,
        token_counter=token_counter,
        source_messages=source_messages,
        summary_cache=summary_cache,
    )
    for item in ready:
        next_total = running_total + item.growth
        if reached_target and next_total <= target_tokens:
            finalized[item.index] = item.decorated
            running_total = next_total
    return finalized, running_total, tuple(misses)
