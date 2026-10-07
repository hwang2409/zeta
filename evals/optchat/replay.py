"""Read-only replay comparison for the OptChat context-view experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from evals.optchat.eviction_replay import (
    _message_token_count,
    _percentile,
    _selected_logs,
    _signatures,
    active_message_records,
    aggregate,
    replay_records,
)
from evals.optchat.strategy import OptChatView, map_zeta_message
from zeta.protocol.types import Message, MessageRole

OPTCHAT_MARKS = (50_000, 80_000, 100_000)
PHOEBE_SESSION = "638f0ce63c9d4749bf1a86d607f4f187"
EXTRA_SESSION = "0720dd12611c4356a51a28df5d3bb342"
SONNET_INPUT_PER_M = 3.0
SONNET_CACHE_READ_PER_M = 0.30
SONNET_WRITE_5M_PER_M = 3.75
SONNET_WRITE_1H_PER_M = 6.0


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    body: bytes
    optchat_breakpoints: tuple[int, ...]
    zeta_breakpoints: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    strategy: str
    logs: int
    requests: int
    mean_bytes: float
    p95_bytes: float
    mean_estimated_tokens: float
    p95_estimated_tokens: float
    prefix_share: float
    optchat_breakpoint_share: float
    zeta_breakpoint_share: float
    optchat_writes_per_100: float
    zeta_writes_per_100: float
    optchat_cost_5m_per_100: float
    optchat_cost_1h_per_100: float
    zeta_cost_5m_per_100: float
    zeta_cost_1h_per_100: float
    compactor_calls_per_message: float
    compactor_overshoot_rate: float
    later_reference_events: int
    reference_in_view: int
    reference_one_zoom: int
    reference_deeper: int
    reference_leaf_only: int


def replay_optchat(
    records: Sequence[tuple[int, Message]],
    *,
    view_bytes: int,
    recent_verbatim: int = 0,
) -> tuple[list[RequestSnapshot], dict[str, int | float]]:
    view = OptChatView(view_bytes=view_bytes, recent_verbatim=recent_verbatim)
    requests: list[RequestSnapshot] = []
    mapped_ids: dict[int, list[int]] = defaultdict(list)
    signatures_by_seq = {seq: _signatures(message) for seq, message in records}
    signature_sources: dict[tuple[str, str], list[int]] = defaultdict(list)
    reference_counts: Counter[str] = Counter()

    for seq, message in records:
        mapped = map_zeta_message(message)
        if message.role is MessageRole.USER:
            rendered = view.render()
            raw = "\n".join(f"{kind}: {text}" for kind, text in mapped).encode()
            body = rendered + b"\n<new-message>\n" + raw
            opt_marks = tuple(_line_mark(rendered, mark) for mark in OPTCHAT_MARKS)
            opt_marks = tuple(mark for mark in opt_marks if mark > 0) + (len(body),)
            requests.append(
                RequestSnapshot(
                    body=body,
                    optchat_breakpoints=tuple(dict.fromkeys(opt_marks)),
                    zeta_breakpoints=(len(rendered), len(body)),
                )
            )
            _record_references(
                seq,
                signatures_by_seq[seq],
                signature_sources,
                mapped_ids,
                view,
                reference_counts,
            )
        for kind, text in mapped:
            message_id = view.append(kind, text, source_seq=seq)
            mapped_ids[seq].append(message_id)
        for kind, values in signatures_by_seq[seq].items():
            for value in values:
                signature_sources[(kind, value)].append(seq)

    stats = view.stats
    return requests, {
        "messages": len(view.messages),
        "model_calls": stats.model_calls,
        "oversize_nodes": stats.oversize_nodes,
        "nodes": stats.nodes,
        "reference_events": sum(reference_counts.values()),
        "reference_in_view": reference_counts["view"],
        "reference_one_zoom": reference_counts["one"],
        "reference_deeper": reference_counts["deeper"],
        "reference_leaf_only": reference_counts["leaf"],
    }


def _record_references(
    current_seq: int,
    current: dict[str, set[str]],
    sources: dict[tuple[str, str], list[int]],
    mapped_ids: dict[int, list[int]],
    view: OptChatView,
    counts: Counter[str],
) -> None:
    seen_sources: set[int] = set()
    for kind, values in current.items():
        for value in values:
            for source_seq in sources.get((kind, value), ()):
                if source_seq >= current_seq or source_seq in seen_sources:
                    continue
                seen_sources.add(source_seq)
                ids = mapped_ids.get(source_seq, ())
                depths = [view.reference_depth(message_id, (value,)) for message_id in ids]
                concrete = [depth for depth in depths if depth is not None]
                if 0 in concrete:
                    counts["view"] += 1
                elif 1 in concrete:
                    counts["one"] += 1
                elif concrete:
                    counts["deeper"] += 1
                else:
                    counts["leaf"] += 1


def _line_mark(view: bytes, target: int) -> int:
    if target >= len(view):
        return 0
    position = view.rfind(b"\n", 0, target + 1)
    return max(position + 1, 0)


def _common_prefix(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    chunk = 8192
    while index + chunk <= limit and left[index : index + chunk] == right[index : index + chunk]:
        index += chunk
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _cache_metrics(
    requests: Sequence[RequestSnapshot], breakpoint_field: str
) -> tuple[int, int, int]:
    total = sum(len(request.body) for request in requests)
    raw_prefix = 0
    cached = 0
    for previous, current in pairwise(requests):
        common = _common_prefix(previous.body, current.body)
        raw_prefix += common
        old_marks = getattr(previous, breakpoint_field)
        new_marks = set(getattr(current, breakpoint_field))
        cached += max(
            (mark for mark in old_marks if mark <= common and mark in new_marks),
            default=0,
        )
    return total, raw_prefix, cached


def summarize_optchat(
    name: str,
    replayed: Sequence[tuple[list[RequestSnapshot], dict[str, int | float]]],
) -> StrategyMetrics:
    requests = [request for snapshots, _ in replayed for request in snapshots]
    sizes = [len(request.body) for request in requests]
    total, raw_prefix, opt_cached = _cache_metrics(requests, "optchat_breakpoints")
    _, _, zeta_cached = _cache_metrics(requests, "zeta_breakpoints")
    count = len(requests)
    messages = sum(int(stats["messages"]) for _, stats in replayed)
    calls = sum(int(stats["model_calls"]) for _, stats in replayed)
    nodes = sum(int(stats["nodes"]) for _, stats in replayed)
    oversize = sum(int(stats["oversize_nodes"]) for _, stats in replayed)
    return StrategyMetrics(
        strategy=name,
        logs=len(replayed),
        requests=count,
        mean_bytes=statistics.mean(sizes) if sizes else 0.0,
        p95_bytes=_percentile(sizes, 0.95),
        mean_estimated_tokens=statistics.mean(size / 4 for size in sizes) if sizes else 0.0,
        p95_estimated_tokens=_percentile([size // 4 for size in sizes], 0.95),
        prefix_share=raw_prefix / total if total else 0.0,
        optchat_breakpoint_share=opt_cached / total if total else 0.0,
        zeta_breakpoint_share=zeta_cached / total if total else 0.0,
        optchat_writes_per_100=_writes_per_100(total, opt_cached, count),
        zeta_writes_per_100=_writes_per_100(total, zeta_cached, count),
        optchat_cost_5m_per_100=_cost_per_100(total, opt_cached, count, SONNET_WRITE_5M_PER_M),
        optchat_cost_1h_per_100=_cost_per_100(total, opt_cached, count, SONNET_WRITE_1H_PER_M),
        zeta_cost_5m_per_100=_cost_per_100(total, zeta_cached, count, SONNET_WRITE_5M_PER_M),
        zeta_cost_1h_per_100=_cost_per_100(total, zeta_cached, count, SONNET_WRITE_1H_PER_M),
        compactor_calls_per_message=calls / messages if messages else 0.0,
        compactor_overshoot_rate=oversize / nodes if nodes else 0.0,
        later_reference_events=sum(int(stats["reference_events"]) for _, stats in replayed),
        reference_in_view=sum(int(stats["reference_in_view"]) for _, stats in replayed),
        reference_one_zoom=sum(int(stats["reference_one_zoom"]) for _, stats in replayed),
        reference_deeper=sum(int(stats["reference_deeper"]) for _, stats in replayed),
        reference_leaf_only=sum(int(stats["reference_leaf_only"]) for _, stats in replayed),
    )


def _writes_per_100(total_bytes: int, cached_bytes: int, requests: int) -> float:
    if not requests:
        return 0.0
    return ((total_bytes - cached_bytes) / 4) * 100 / requests


def _cost_per_100(
    total_bytes: int,
    cached_bytes: int,
    requests: int,
    write_price: float,
) -> float:
    if not requests:
        return 0.0
    reads = cached_bytes / 4
    writes = (total_bytes - cached_bytes) / 4
    return (reads * SONNET_CACHE_READ_PER_M + writes * write_price) / 1_000_000 * 100 / requests


def _metadata(path: Path) -> dict[str, Any]:
    records = active_message_records(path)
    is_agent = path.parent.parent.name == "agents"
    return {
        "id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
        "session": path.parent.parent.parent.name if is_agent else path.parent.name,
        "agent": path.parent.name if is_agent else None,
        "bytes": path.stat().st_size,
        "messages": len(records),
        "estimated_tokens": sum(_message_token_count(message) for _, message in records),
    }


def selected_logs(root: Path, limit: int = 12) -> list[Path]:
    paths = _selected_logs(root, limit, include_session=None)
    extra = root / EXTRA_SESSION / "conversation.jsonl"
    if extra.exists() and extra not in paths:
        paths.append(extra)
    phoebe = root / PHOEBE_SESSION / "conversation.jsonl"
    if phoebe.exists() and not any(PHOEBE_SESSION in path.parts for path in paths):
        paths.append(phoebe)
    return paths


def _current_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    session_results = []
    fixed_results = []
    for path in paths:
        records = active_message_records(path)
        session = _metadata(path)["session"]
        session_cap = 1_000_000 if session == PHOEBE_SESSION else 200_000
        session_results.append(replay_records(records, session_cap))
        fixed_results.append(replay_records(records, 200_000))
    return [
        {"strategy": "current-session-cap", "by_cap": aggregate(session_results)},
        {"strategy": "current-200k", "by_cap": aggregate(fixed_results)},
    ]


def run(paths: Sequence[Path]) -> dict[str, Any]:
    records = [(path, active_message_records(path)) for path in paths]
    variants = [
        ("optchat-128kb", 128_000, 0),
        ("optchat-256kb", 256_000, 0),
        ("optchat-128kb-recent-8", 128_000, 8),
    ]
    opt_rows = []
    for name, budget, recent in variants:
        replayed = [
            replay_optchat(log_records, view_bytes=budget, recent_verbatim=recent)
            for _, log_records in records
        ]
        opt_rows.append(asdict(summarize_optchat(name, replayed)))
    return {
        "revision": os.environ.get("OPTCHAT_REVISION", "unknown"),
        "logs": [_metadata(path) for path in paths],
        "current": _current_rows(paths),
        "optchat": opt_rows,
        "assumptions": {
            "token_estimate": "utf8 bytes / 4 for rendered OptChat requests",
            "recent_variant": "summary view plus 8 duplicate verbatim log items",
            "cost_model": {
                "anthropic_base_input_per_m": SONNET_INPUT_PER_M,
                "cache_read_per_m": SONNET_CACHE_READ_PER_M,
                "write_5m_per_m": SONNET_WRITE_5M_PER_M,
                "write_1h_per_m": SONNET_WRITE_1H_PER_M,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, default=Path.home() / ".zeta/sessions")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(selected_logs(args.sessions_root, args.limit))
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
