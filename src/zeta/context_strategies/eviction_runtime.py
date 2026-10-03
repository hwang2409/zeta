"""Context-assembler integration for deterministic eviction strategies."""

from __future__ import annotations

from typing import Any

from .adaptive import (
    plan_eviction,
    plan_eviction2,
    replace_item_messages,
    truncate_tool_results,
)
from .evict import EVICTION2_KIND, EVICTION2_SUM_KIND


async def apply_eviction(
    assembler: Any,
    items: list[Any],
    system_messages: list[Any],
    latest_user: int | None,
    branch_id: str | None,
    backend: Any,
    *,
    max_source_tokens: int,
    stale_error: type[Exception],
) -> Any | None:
    """Persist and assemble one eviction plan, or return ``None``."""

    strategy = next(
        (flag for flag in ("evict2sum", "evict2", "evict") if flag in assembler.strategies),
        None,
    )
    if strategy is None:
        return None
    kind = {
        "evict": "eviction",
        "evict2": EVICTION2_KIND,
        "evict2sum": EVICTION2_SUM_KIND,
    }[strategy]
    if strategy == "evict":
        eviction = plan_eviction(
            items,
            system_messages,
            latest_user,
            assembler.token_budget,
            assembler.token_counter,
        )
    else:
        eviction = plan_eviction2(
            items,
            system_messages,
            latest_user,
            assembler.token_budget,
            assembler.token_counter,
            recall_enabled="recall" in assembler.strategies,
        )
    if eviction is None:
        return None

    summary = (
        "[deterministic eviction view]"
        if strategy == "evict"
        else "[deterministic semantic eviction view]"
    )
    if kind == EVICTION2_SUM_KIND and eviction.assistant_messages:
        summary = await assembler.compaction_policy.summarize_chunked(
            eviction.assistant_messages,
            backend=backend or assembler.backend,
            system_prompt=system_messages[0] if system_messages else None,
            max_source_tokens=max_source_tokens,
            on_success=assembler.on_completion_success,
            on_usage=assembler.record_usage,
            on_telemetry=assembler._record_compaction_telemetry,
        )
    try:
        assembler.store.append_compaction_marker(
            summary,
            eviction.source_start,
            eviction.source_end,
            replaces=eviction.replaces,
            pinned_message=eviction.pinned_message,
            expected_parent_id=branch_id,
            kind=kind,
            view=eviction.view,
        )
    except ValueError as exc:
        raise stale_error("active branch changed during eviction") from exc

    proposed_items = assembler._visible_items(assembler.store.replay())
    proposed_messages = [
        *system_messages,
        *(item.message for item in proposed_items),
    ]
    result_seqs = {
        id(item.message): item.entry.seq
        for item in proposed_items
        if item.entry is not None and item.message.tool_result is not None
    }
    truncated = truncate_tool_results(
        proposed_messages,
        assembler.token_budget,
        result_seqs,
        assembler.token_counter,
    )
    request_messages = proposed_messages if truncated is None else truncated
    request_items = proposed_items
    if truncated is not None:
        request_items = replace_item_messages(
            proposed_items, truncated[len(system_messages) :]
        )
    proposed = assembler._context(
        assembler._with_strategy_tail(request_messages, request_items),
        True,
    )
    assembler._experiment_telemetry.emit(
        "context_strategy",
        kind=strategy,
        range=[eviction.source_start, eviction.source_end],
        items_folded=eviction.result.items_folded,
        items_evicted=eviction.result.items_evicted,
        tokens_before=eviction.result.tokens_before,
        tokens_after=eviction.result.tokens_after,
    )
    assembler._provider_token_total = None
    assembler.last_context = proposed
    assembler._emit_request_telemetry(proposed)
    return proposed
