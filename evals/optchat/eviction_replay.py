"""Read-only offline replay for deterministic context eviction.

The scanner snapshots each JSONL file size and opens it only in binary read mode.
It does not construct ConversationStore or SessionManager. Replay uses the real
``evict_messages`` implementation and production token counter and ratios.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zeta.context_eviction import HYSTERESIS_RATIO, TARGET_RATIO, evict_messages
from zeta.core.context import ContextAssembler, _message_token_count
from zeta.core.store import ConversationEntry
from zeta.protocol.types import Message, MessageRole, TextContent, ToolUseContent

DEFAULT_CAPS = (64_000, 100_000, 150_000, 200_000, 300_000, 400_000, 1_000_000)
_PATH_RE = re.compile(r"(?<![\w.-])(?:[~./]|[A-Za-z0-9_-]+/)[A-Za-z0-9_./-]+")
_IDENTIFIER_RE = re.compile(r"\b(?=[A-Za-z0-9_.:-]{8,}\b)(?=[A-Za-z0-9_.:-]*[0-9_.:-])[A-Za-z][A-Za-z0-9_.:-]*")


@dataclass(frozen=True)
class ReferenceMetrics:
    evicted_items: int
    later_referenced_items: int
    path_matches: int
    command_matches: int
    identifier_matches: int


@dataclass(frozen=True)
class CapMetrics:
    cap: int
    requests: int
    fitted_requests: int
    unfit_requests: int
    evictions: int
    mean_tokens: float
    p50_tokens: float
    p95_tokens: float
    max_tokens: int
    total_input_tokens: int
    unchanged_prefix_fraction: float
    cached_token_share: float
    references: ReferenceMetrics
    request_token_counts: tuple[int, ...] = ()
    unchanged_prefix_count: int = 0
    comparable_request_count: int = 0
    cached_prefix_tokens: int = 0


def bounded_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete object rows up to the file size observed at open."""

    with open(path, "rb") as handle:
        remaining = os.fstat(handle.fileno()).st_size
        while remaining > 0:
            line = handle.readline(remaining)
            if not line:
                return
            remaining -= len(line)
            if not line.endswith(b"\n"):
                return
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                yield row


def active_message_records(path: Path) -> list[tuple[int, Message]]:
    """Read the final active branch directly, without locks, writes, or repair."""

    entries: list[ConversationEntry] = []
    for row in bounded_json_rows(path):
        if row.get("type") == "header":
            continue
        try:
            entries.append(ConversationEntry.from_dict(row))
        except (KeyError, TypeError, ValueError):
            continue
    if not entries:
        return []
    by_id = {entry.id: entry for entry in entries}
    current: ConversationEntry | None = entries[-1]
    branch: list[ConversationEntry] = []
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        branch.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None
    branch.reverse()
    records = []
    for entry in branch:
        if entry.type != "message":
            continue
        try:
            records.append((entry.seq, Message.from_dict(entry.data["message"])))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _encoded(message: Message) -> bytes:
    return json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":")).encode()


def _prefix_tokens(
    previous: Sequence[Message],
    current: Sequence[Message],
    *,
    token_count: Callable[[Message], int] = _message_token_count,
    encode: Callable[[Message], bytes] = _encoded,
) -> tuple[int, bool]:
    total = 0
    matched = 0
    for old, new in zip(previous, current):
        if encode(old) != encode(new):
            break
        total += token_count(new)
        matched += 1
    return total, matched == len(previous)


def _message_text(message: Message) -> str:
    values: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            values.append(block.text)
        elif isinstance(block, ToolUseContent):
            values.append(json.dumps(block.tool_call.arguments, sort_keys=True))
    if message.tool_result is not None:
        values.append(message.tool_result.content)
    return "\n".join(values)


def _signatures(message: Message) -> dict[str, set[str]]:
    text = _message_text(message)
    paths = {match.group(0).rstrip(".,:;)") for match in _PATH_RE.finditer(text)}
    identifiers = {match.group(0).casefold() for match in _IDENTIFIER_RE.finditer(text)}
    commands: set[str] = set()
    for block in message.content:
        if not isinstance(block, ToolUseContent) or block.tool_call.name != "bash":
            continue
        command = block.tool_call.arguments.get("command", block.tool_call.arguments.get("cmd"))
        if isinstance(command, str) and command.strip():
            commands.add(" ".join(command.split()))
    return {"path": paths, "command": commands, "identifier": identifiers}


def _later_references(
    records: Sequence[tuple[int, Message]], evicted_at: dict[int, int]
) -> ReferenceMetrics:
    signatures_by_seq: dict[int, dict[str, set[str]]] = {}
    last_seen: dict[str, dict[str, int]] = {
        "path": {},
        "command": {},
        "identifier": {},
    }
    for seq, message in records:
        signatures = _signatures(message)
        signatures_by_seq[seq] = signatures
        for kind, values in signatures.items():
            for value in values:
                last_seen[kind][value] = seq
    counts = Counter()
    referenced = 0
    for source_seq, request_seq in evicted_at.items():
        source = signatures_by_seq.get(source_seq, {})
        matched_kinds = {
            kind
            for kind in ("path", "command", "identifier")
            if any(last_seen[kind].get(value, 0) > request_seq for value in source.get(kind, ()))
        }
        if matched_kinds:
            referenced += 1
            counts.update(matched_kinds)
    return ReferenceMetrics(
        evicted_items=len(evicted_at),
        later_referenced_items=referenced,
        path_matches=counts["path"],
        command_matches=counts["command"],
        identifier_matches=counts["identifier"],
    )


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _identity_cached[Value](
    function: Callable[[Message], Value],
) -> Callable[[Message], Value]:
    """Cache an immutable message calculation without depending on hashing."""

    cache: dict[int, tuple[Message, Value]] = {}

    def cached(message: Message) -> Value:
        result = cache.get(id(message))
        if result is None or result[0] is not message:
            result = (message, function(message))
            cache[id(message)] = result
        return result[1]

    return cached


def _replay_assembler(
    token_count: Callable[[Message], int],
) -> ContextAssembler:
    """Build a store-free production fitter with cached message token counts."""

    # Construction is deliberately bypassed: replay needs only the pure fitting
    # method and must not construct or access a ConversationStore.
    assembler = object.__new__(ContextAssembler)
    assembler.token_counter = token_count
    return assembler


def _fit_tool_results(
    assembler: ContextAssembler,
    records: Sequence[tuple[int, Message]],
    cap: int,
) -> list[tuple[int, Message]] | None:
    """Apply the production assembler's deterministic tool-result fitting."""

    messages = [message for _, message in records]
    fitted = assembler._truncate_tool_results(
        messages,
        cap,
        {
            id(message): seq
            for seq, message in records
            if message.tool_result is not None
        },
    )
    if fitted is None:
        return None
    return [
        (seq, message)
        for (seq, _), message in zip(records, fitted, strict=True)
    ]


def replay_records(records: Sequence[tuple[int, Message]], cap: int) -> CapMetrics:
    """Replay request boundaries through production eviction and result fitting."""

    visible: list[tuple[int, Message]] = []
    previous: list[Message] = []
    token_count = _identity_cached(_message_token_count)
    encode = _identity_cached(_encoded)
    assembler = _replay_assembler(token_count)
    request_tokens: list[int] = []
    cache_tokens = 0
    unchanged = 0
    comparisons = 0
    evictions = 0
    evicted_at: dict[int, int] = {}
    previous_eviction_end: int | None = None

    for seq, message in records:
        visible.append((seq, message))
        if message.role is not MessageRole.USER:
            continue
        total = sum(token_count(item) for _, item in visible)
        if total > cap:
            growth = (
                sum(
                    token_count(item)
                    for item_seq, item in visible
                    if previous_eviction_end is not None and item_seq > previous_eviction_end
                )
                if previous_eviction_end is not None
                else cap
            )
            if previous_eviction_end is None or growth >= max(1, int(cap * HYSTERESIS_RATIO)):
                candidates = visible[:-1]
                result = evict_messages(
                    candidates,
                    fixed_tokens=token_count(message),
                    target_tokens=max(1, int(cap * TARGET_RATIO)),
                    token_counter=token_count,
                )
                if result.items_evicted:
                    changed = [
                        item_seq
                        for (item_seq, old), new in zip(candidates, result.messages, strict=True)
                        if encode(old) != encode(new)
                    ]
                    for item_seq in changed:
                        evicted_at.setdefault(item_seq, seq)
                    visible = [
                        (item_seq, item)
                        for (item_seq, _), item in zip(candidates, result.messages, strict=True)
                    ] + [visible[-1]]
                    previous_eviction_end = seq
                    evictions += 1
        assembled = visible
        if sum(token_count(item) for _, item in visible) > cap:
            fitted = _fit_tool_results(assembler, visible, cap)
            if fitted is None:
                previous = []
                continue
            assembled = fitted
        current = [item for _, item in assembled]
        tokens = sum(token_count(item) for item in current)
        request_tokens.append(tokens)
        if previous:
            prefix, is_unchanged = _prefix_tokens(
                previous,
                current,
                token_count=token_count,
                encode=encode,
            )
            cache_tokens += prefix
            unchanged += is_unchanged
            comparisons += 1
        previous = current

    total_input = sum(request_tokens)
    return CapMetrics(
        cap=cap,
        requests=sum(
            message.role is MessageRole.USER for _, message in records
        ),
        fitted_requests=len(request_tokens),
        unfit_requests=(
            sum(message.role is MessageRole.USER for _, message in records)
            - len(request_tokens)
        ),
        evictions=evictions,
        mean_tokens=statistics.mean(request_tokens) if request_tokens else 0.0,
        p50_tokens=_percentile(request_tokens, 0.50),
        p95_tokens=_percentile(request_tokens, 0.95),
        max_tokens=max(request_tokens, default=0),
        total_input_tokens=total_input,
        unchanged_prefix_fraction=unchanged / comparisons if comparisons else 0.0,
        cached_token_share=cache_tokens / total_input if total_input else 0.0,
        references=_later_references(records, evicted_at),
        request_token_counts=tuple(request_tokens),
        unchanged_prefix_count=unchanged,
        comparable_request_count=comparisons,
        cached_prefix_tokens=cache_tokens,
    )


def replay_file(path: Path, caps: Iterable[int] = DEFAULT_CAPS) -> list[CapMetrics]:
    records = active_message_records(path)
    return [replay_records(records, cap) for cap in caps]


def aggregate(results: Sequence[CapMetrics]) -> list[dict[str, Any]]:
    """Combine per-log metrics, weighting request and item rates correctly."""

    grouped: dict[int, list[CapMetrics]] = {}
    for result in results:
        grouped.setdefault(result.cap, []).append(result)
    rows = []
    for cap, items in sorted(grouped.items()):
        requests = sum(item.requests for item in items)
        fitted_requests = sum(item.fitted_requests for item in items)
        request_tokens = [token for item in items for token in item.request_token_counts]
        comparisons = sum(item.comparable_request_count for item in items)
        evicted = sum(item.references.evicted_items for item in items)
        total = sum(item.total_input_tokens for item in items)
        rows.append(
            {
                "cap": cap,
                "logs": len(items),
                "requests": requests,
                "fitted_requests": fitted_requests,
                "unfit_requests": sum(item.unfit_requests for item in items),
                "evictions": sum(item.evictions for item in items),
                "mean_tokens": total / fitted_requests if fitted_requests else 0.0,
                "p50_tokens": _percentile(request_tokens, 0.50),
                "p95_tokens": _percentile(request_tokens, 0.95),
                "max_tokens": max((item.max_tokens for item in items), default=0),
                "total_input_tokens": total,
                "unchanged_prefix_fraction": (
                    sum(item.unchanged_prefix_count for item in items) / comparisons
                    if comparisons
                    else 0.0
                ),
                "cached_token_share": (
                    sum(item.cached_prefix_tokens for item in items) / total if total else 0.0
                ),
                "evicted_items": evicted,
                "later_referenced_items": sum(item.references.later_referenced_items for item in items),
                "later_reference_rate": (
                    sum(item.references.later_referenced_items for item in items) / evicted
                    if evicted
                    else 0.0
                ),
                "path_matches": sum(item.references.path_matches for item in items),
                "command_matches": sum(item.references.command_matches for item in items),
                "identifier_matches": sum(item.references.identifier_matches for item in items),
            }
        )
    return rows


def _history_tokens(path: Path) -> int:
    return sum(
        _message_token_count(message)
        for _, message in active_message_records(path)
    )


def _selected_logs(root: Path, limit: int, include_session: str | None) -> list[Path]:
    paths = sorted(
        root.glob("**/conversation.jsonl"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    # Stored size is a cheap first-stage bound, but abandoned branches can make
    # it overstate active history. Rank a wider candidate set by active tokens.
    candidates = paths[: max(limit * 3, limit)]
    candidates.sort(key=_history_tokens, reverse=True)
    selected = candidates[:limit]
    if include_session:
        for path in paths:
            if path.parent.name == include_session and path not in selected:
                selected.append(path)
                break
    return selected


def _replay_cap(payload: tuple[Path, int]) -> CapMetrics:
    path, cap = payload
    return replay_records(active_message_records(path), cap)


def _log_metadata(path: Path) -> dict[str, Any]:
    records = active_message_records(path)
    is_agent = path.parent.parent.name == "agents"
    return {
        "id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
        "session": path.parent.parent.parent.name if is_agent else path.parent.name,
        "agent": path.parent.name if is_agent else None,
        "messages": len(records),
        "estimated_tokens": sum(
            _message_token_count(message) for _, message in records
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, default=Path.home() / ".zeta/sessions")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--include-session", default="c482b3a4d1884ec7962d894f691bec7d")
    parser.add_argument("--caps", default=",".join(map(str, DEFAULT_CAPS)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args(argv)
    caps = tuple(int(value) for value in args.caps.split(","))
    if args.concurrency < 1:
        raise SystemExit("concurrency must be positive")
    paths = _selected_logs(args.sessions_root, args.limit, args.include_session)
    log_rows = [_log_metadata(path) for path in paths]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        results = list(
            executor.map(
                _replay_cap,
                ((path, cap) for path in paths for cap in caps),
            )
        )
    output = {"caps": aggregate(results), "logs": log_rows}
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
