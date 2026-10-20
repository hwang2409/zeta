#!/usr/bin/env python3
"""Run a read-only feature-off/on eviction-summary replay."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from zeta.context_eviction import evict_messages, eviction_summary_candidates
from zeta.context_eviction.summaries import (
    EvictionSummaryWorker,
    SummaryKey,
    SummaryResponse,
)
from zeta.context_eviction.summary_receipts import receipt_summary_request
from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.memory.provider import complete_reconciliation, use_response_byte_limit
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
        self.records[key.as_string()] = (
            response if isinstance(response, str) else response.text
        )
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

    async def complete(prompt: str) -> SummaryResponse:
        with use_response_byte_limit(512):
            response = await complete_reconciliation(backend, prompt)
        return SummaryResponse(
            response.text,
            sum(
                int(response.usage.get(key, 0) or 0)
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            ),
            int(response.usage.get("output_tokens", 0) or 0),
        )

    return complete


def _inputs(session_dir: Path) -> tuple[list[tuple[int, Message]], dict[int, Message], int]:
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
    return records, sources, max(1, int(budget * 0.55))


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


async def replay(session_dir: Path, model: str, offline: bool) -> dict[str, Any]:
    records, sources, target = _inputs(session_dir)
    off = evict_messages(
        records, fixed_tokens=0, target_tokens=target, source_messages=sources
    )
    cache = _MemoryCache()
    worker = EvictionSummaryWorker(cache, _provider(model, offline=offline))
    for request in eviction_summary_candidates(records):
        worker.admit(request)
    while worker.queue:
        await worker.run_once()
    first_on = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=target,
        source_messages=sources,
        summary_cache=cache,  # type: ignore[arg-type]
        summaries_enabled=True,
    )
    for request in first_on.summary_misses:
        worker.admit(request)
    while worker.queue:
        await worker.run_once()
    on = evict_messages(
        records,
        fixed_tokens=0,
        target_tokens=target,
        source_messages=sources,
        summary_cache=cache,  # type: ignore[arg-type]
        summaries_enabled=True,
    )
    await worker.close()
    requests = [
        request
        for message in on.messages
        if (request := receipt_summary_request(message, sources)) is not None
    ]
    summary_count = sum(
        message.metadata.get("summary_data") is not None for message in on.messages
    )
    eligible_receipts = len(requests)
    off_view, on_view = _view(off), _view(on)
    return {
        "session_dir": str(session_dir),
        "model": model,
        "offline": offline,
        "feature_off": off_view,
        "feature_on": on_view,
        "selected_seqs_identical": off_view["selected_seqs"] == on_view["selected_seqs"],
        "range_bounds_identical": off_view["range_bounds"] == on_view["range_bounds"],
        "summary_readiness_rate": (
            summary_count / eligible_receipts if eligible_receipts else 1.0
        ),
        "queue_drops": worker.drops,
        "token_totals": {"off": off.tokens_after, "on": on.tokens_after},
        "summary_count": summary_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(replay(args.session_dir, args.model, args.offline))
    if not result["selected_seqs_identical"] or not result["range_bounds_identical"]:
        raise SystemExit("eviction selection changed with summaries enabled")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
