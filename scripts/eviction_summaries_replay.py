#!/usr/bin/env python3
"""Run a read-only feature-off/on eviction-summary replay."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from zeta.context_accounting import message_token_count
from zeta.context_eviction import (
    TARGET_RATIO,
    decorate_receipt,
    evict_messages,
    eviction_summary_candidates,
    normalize_receipt,
)
from zeta.context_eviction.summaries import (
    EvictionSummaryWorker,
    SummaryKey,
    SummaryResponse,
    trim_summary,
    valid_summary,
)
from zeta.context_eviction.summary_provider import build_summary_provider
from zeta.context_eviction.summary_receipts import receipt_summary_request
from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.models.catalog import provider_for_model
from zeta.protocol.types import Message
from zeta.providers.factory import build_backend


class _MemoryCache:
    def __init__(self) -> None:
        self.records: dict[str, str] = {}
        self.loaded = True

    async def load(self) -> None:
        return

    def get(self, key: SummaryKey) -> str | None:
        return self.records.get(key.as_string())

    async def put(self, key: SummaryKey, response: SummaryResponse | str) -> bool:
        summary = trim_summary(response if isinstance(response, str) else response.text)
        if not valid_summary(summary):
            return False
        self.records[key.as_string()] = summary
        return True


def _provider(model: str, *, offline: bool):
    if offline:

        async def fake(prompt: str) -> SummaryResponse:
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            return SummaryResponse(f"offline factual summary {digest}", 0, 0)

        return fake
    backend, _ = build_backend(
        provider_for_model(model),
        model,
        home=Path.home() / ".zeta",
        require_credentials=True,
        stall_retries=0,
    )

    return build_summary_provider(backend)


def _inputs(
    session_dir: Path,
) -> tuple[list[tuple[int, Message]], dict[int, Message], int]:
    store = ConversationStore(
        session_dir.parent,
        session_id=session_dir.name,
        _read_only=True,
        _must_exist=True,
    )
    branch = store.replay()
    metadata = json.loads((session_dir / "meta.json").read_text())
    budget = int(metadata.get("compaction_budget") or 200_000)
    assembler = ContextAssembler(store, token_budget=budget)
    items = assembler._visible_items(branch)  # replay tool: use production projection
    records = [
        (int(item.message.metadata.get("source_seq", item.entry.seq)), item.message)
        for item in items
        if item.entry is not None
    ]
    sources = {
        entry.seq: Message.from_dict(entry.data["message"])
        for entry in branch
        if entry.type == "message"
    }
    return records, sources, max(1, int(budget * TARGET_RATIO))


def _view(result: Any) -> dict[str, Any]:
    ranges = sorted(
        [
            [message.metadata["source_seq"], message.metadata["range_seq_end"]]
            for message in result.messages
            if message.metadata.get("eviction_range")
        ]
    )
    return {
        "selected_seqs": list(result.source_seqs),
        "range_bounds": ranges,
        "tokens_after": result.tokens_after,
        "reached_target": result.reached_target,
    }


def _ready_decoration_growth(
    messages: list[Message], sources: dict[int, Message], cache: _MemoryCache
) -> int:
    growth = 0
    for message in messages:
        plain = normalize_receipt(message)
        request = receipt_summary_request(plain, sources)
        if request is None or (summary := cache.get(request.key)) is None:
            continue
        decorated = decorate_receipt(plain, summary)
        source_tokens = sum(
            message_token_count(source)
            for seq, source in sources.items()
            if request.key.seq_start <= seq <= request.key.seq_end
        )
        if message_token_count(decorated) < source_tokens:
            growth += message_token_count(decorated) - message_token_count(plain)
    return growth


def _largest_new_range_saving(off: Any, on: Any) -> int:
    off_ranges = {
        (message.metadata["source_seq"], message.metadata["range_seq_end"])
        for message in off.messages
        if message.metadata.get("eviction_range")
    }
    largest = 0
    for message in on.messages:
        if not message.metadata.get("eviction_range"):
            continue
        bounds = (message.metadata["source_seq"], message.metadata["range_seq_end"])
        if bounds in off_ranges:
            continue
        original_tokens = sum(
            message_token_count(original)
            for seq, original in zip(off.source_seqs, off.messages, strict=True)
            if bounds[0] <= seq <= bounds[1]
        )
        largest = max(largest, original_tokens - message_token_count(message))
    return largest


async def _enabled_pass(
    worker: EvictionSummaryWorker,
    records: list[tuple[int, Message]],
    sources: dict[int, Message],
    target: int,
) -> Any:
    while worker.queue:
        await worker.run_once()
    return evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=target,
        source_messages=sources,
        summary_cache=worker.cache,
        summaries_enabled=True,
    )


async def _converge_enabled_view(
    worker: EvictionSummaryWorker,
    records: list[tuple[int, Message]],
    sources: dict[int, Message],
    target: int,
) -> Any:
    on = await _enabled_pass(worker, records, sources, target)
    for _ in range(3):
        for message in on.messages:
            if (request := receipt_summary_request(message, sources)) is not None:
                worker.admit(request)
        for request in on.summary_misses:
            worker.admit(request)
        next_on = await _enabled_pass(worker, records, sources, target)
        if next_on.source_seqs == on.source_seqs and next_on.tokens_after == on.tokens_after:
            return next_on
        on = next_on
    return on


async def replay(session_dir: Path, model: str, offline: bool) -> dict[str, Any]:
    records, sources, target = _inputs(session_dir)
    off = evict_messages(
        records, fixed_tokens=0, target_tokens=target, source_messages=sources
    )
    cache = _MemoryCache()
    worker = EvictionSummaryWorker(cache, _provider(model, offline=offline))
    for request in eviction_summary_candidates(records):
        worker.admit(request)
    on = await _converge_enabled_view(worker, records, sources, target)
    await worker.close()
    requests = [
        (message, request)
        for message in on.messages
        if (request := receipt_summary_request(message, sources)) is not None
    ]
    ready = [request for _, request in requests if cache.get(request.key) is not None]
    summary_count = sum(
        message.metadata.get("summary_data") is not None for message in on.messages
    )
    eligible_receipts = len(requests)
    off_view, on_view = _view(off), _view(on)
    largest_single_fold_saving = _largest_new_range_saving(off, on)
    deterministic_tokens = sum(
        message_token_count(normalize_receipt(message)) for message in on.messages
    )
    ready_growth = _ready_decoration_growth(on.messages, sources, cache)
    return {
        "session_dir": str(session_dir),
        "target_tokens": target,
        "model": model,
        "offline": offline,
        "feature_off": off_view,
        "feature_on": on_view,
        "off_selected_count": len(off_view["selected_seqs"]),
        "on_selected_count": len(on_view["selected_seqs"]),
        "extra_evicted_tokens": off.tokens_after - deterministic_tokens,
        "reserve_tokens": on.reserve_tokens,
        "largest_single_fold_saving": largest_single_fold_saving,
        "reserve_bound": on.reserve_tokens + largest_single_fold_saving,
        "tokens_vs_target": {
            "off": off.tokens_after - target,
            "on": on.tokens_after - target,
        },
        "shown_rate": summary_count / eligible_receipts if eligible_receipts else 1.0,
        "cache_ready_rate": len(ready) / eligible_receipts if eligible_receipts else 1.0,
        "queue_drops": worker.drops,
        "token_totals": {"off": off.tokens_after, "on": on.tokens_after},
        "eligible_receipts": eligible_receipts,
        "summary_count": summary_count,
        "deterministic_tokens_after": deterministic_tokens,
        "decoration_headroom": target - deterministic_tokens,
        "ready_decoration_growth": ready_growth,
        "shown_decoration_growth": on.tokens_after - deterministic_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(replay(args.session_dir, args.model, args.offline))
    if (
        result["feature_on"]["reached_target"]
        and result["feature_on"]["tokens_after"] > result["target_tokens"]
    ):
        raise SystemExit("summary-enabled eviction exceeded target")
    if result["extra_evicted_tokens"] > result["reserve_bound"]:
        raise SystemExit("summary reserve exceeded the final-fold bound")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
