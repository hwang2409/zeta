"""Experimental, opt-in context strategy helpers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import Message, TextContent, ToolUseContent

KNOWN_STRATEGIES = frozenset({"recall", "budget", "archive", "edit", "nudge", "fold", "evict"})
RECALL_DEFAULT_MAX_CHARS = 8_000
RECALL_HARD_MAX_CHARS = 20_000


def context_strategies(value: str | None = None) -> frozenset[str]:
    """Return recognized comma-separated strategy flags from the environment."""

    raw = os.environ.get("ZETA_CONTEXT_STRATEGY", "") if value is None else value
    return frozenset(
        flag
        for part in raw.split(",")
        if (flag := part.strip().casefold()) in KNOWN_STRATEGIES
    )


class ContextTelemetry:
    """Best-effort JSONL telemetry kept separate from the session log."""

    def __init__(self, path: str | None = None) -> None:
        configured = os.environ.get("ZETA_CONTEXT_TELEMETRY") if path is None else path
        self.path = Path(configured).expanduser() if configured else None

    def emit(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **data,
        }
        encoded = (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
        except OSError:
            # Experimental telemetry must never break a model request.
            return


def active_compacted_ranges(
    entries: Sequence[ConversationEntry],
) -> list[tuple[int, int]]:
    """Return effective compaction ranges on this already-active branch."""

    markers = [entry for entry in entries if entry.type == "compaction"]
    superseded: set[str] = set()
    for marker in markers:
        superseded.update(marker.data.get("replaces", []))
    return [
        (marker.data["source_seq_start"], marker.data["source_seq_end"])
        for marker in markers
        if marker.id not in superseded
    ]


def _is_covered(seq: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= seq <= end for start, end in ranges)


def _message_record(entry: ConversationEntry) -> dict[str, Any]:
    message = Message.from_dict(entry.data["message"])
    record: dict[str, Any] = {"role": message.role.value}
    text = "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
    if text:
        record["text"] = text
    calls = [
        {
            "id": block.tool_call.id,
            "name": block.tool_call.name,
            "args": block.tool_call.arguments,
        }
        for block in message.content
        if isinstance(block, ToolUseContent)
    ]
    if calls:
        record["tool_calls"] = calls
    if message.tool_result is not None:
        result: dict[str, Any] = {
            "tool_call_id": message.tool_result.tool_call_id,
            "content": message.tool_result.content,
            "is_error": message.tool_result.is_error,
        }
        if message.tool_result.content_blocks is not None:
            result["content_blocks"] = message.tool_result.content_blocks
        if message.tool_result.structured_content is not None:
            result["structured"] = message.tool_result.structured_content
        record["tool_result"] = result
    return record


def _render_entry(entry: ConversationEntry) -> str:
    return f"seq {entry.seq} " + json.dumps(
        _message_record(entry), ensure_ascii=False, separators=(",", ":")
    )


def _truncate_with_hint(text: str, max_chars: int, hint: str) -> str:
    if len(text) <= max_chars:
        return text
    suffix = f"\n[truncated; {hint}]"
    if len(suffix) >= max_chars:
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def _render_range(
    entries: Sequence[ConversationEntry], max_chars: int, seq_end: int
) -> str:
    lines = [_render_entry(entry) for entry in entries]
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return body
    included: list[str] = []
    for entry, line in zip(entries, lines, strict=True):
        hint = f"continue with seq_start={entry.seq}, seq_end={seq_end}"
        suffix = f"\n[truncated; {hint}]"
        candidate = "\n".join([*included, line, suffix])
        if len(candidate) > max_chars:
            if included:
                return "\n".join([*included, suffix])
            retry_suffix = (
                "\n[truncated; "
                f"retry seq_start={entry.seq}, seq_end={entry.seq} "
                "with a larger max_chars]"
            )
            if len(retry_suffix) >= max_chars:
                return retry_suffix[:max_chars]
            return line[: max_chars - len(retry_suffix)] + retry_suffix
        included.append(line)
    return body


def recall_history(
    store: ConversationStore,
    *,
    query: str | None = None,
    seq_start: int | None = None,
    seq_end: int | None = None,
    max_chars: int = RECALL_DEFAULT_MAX_CHARS,
) -> tuple[str, str]:
    """Render covered messages from the active branch without mutating it."""

    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    max_chars = min(max_chars, RECALL_HARD_MAX_CHARS)
    branch = store.replay()
    ranges = active_compacted_ranges(branch)
    restored_archives = {
        entry.data.get("archive_id")
        for entry in branch
        if entry.type == "context_restore"
    }
    ranges.extend(
        (entry.data["source_seq_start"], entry.data["source_seq_end"])
        for entry in branch
        if entry.type == "context_replace"
        or (
            entry.type == "context_archive"
            and entry.data.get("archive_id") not in restored_archives
        )
    )
    hidden = [
        entry
        for entry in branch
        if entry.type == "message" and _is_covered(entry.seq, ranges)
    ]
    if seq_start is not None or seq_end is not None:
        if type(seq_start) is not int or type(seq_end) is not int:
            raise ValueError("seq_start and seq_end must be provided together")
        if seq_start < 1 or seq_end < seq_start:
            raise ValueError("invalid sequence range")
        selected = [entry for entry in hidden if seq_start <= entry.seq <= seq_end]
        if not selected:
            return "No compacted messages in that range on the active branch.", "range"
        return _render_range(selected, max_chars, seq_end), "range"
    if query is not None and query.strip():
        folded = query.casefold().strip()
        tokens = re.findall(r"\w+", folded)
        matches: list[tuple[int, ConversationEntry, str]] = []
        for entry in hidden:
            rendered = _render_entry(entry)
            searchable = rendered.casefold()
            score = (10 if folded in searchable else 0) + sum(
                searchable.count(token) for token in tokens
            )
            if score:
                snippet = rendered if len(rendered) <= 320 else f"{rendered[:317]}..."
                matches.append((score, entry, snippet))
        matches.sort(key=lambda item: (-item[0], item[1].seq))
        body = "\n".join(item[2] for item in matches[:20])
        if not body:
            body = "No matching compacted messages on the active branch."
        return _truncate_with_hint(body, max_chars, "refine the query for more matches"), "query"
    raise ValueError("provide query or both seq_start and seq_end")


def budget_readout(
    items: Sequence[tuple[str, str, int]],
    *,
    estimated_tokens: int,
    budget: int,
    compactions: int,
) -> str:
    """Build a compact, bounded VISTA-style tail readout."""

    remaining = max(0.0, 100.0 * (budget - estimated_tokens) / budget)
    largest = sorted(items, key=lambda item: item[2], reverse=True)[:5]
    details = ", ".join(
        f"{seq} {label} ~{tokens}t" for seq, label, tokens in largest
    ) or "none"
    text = (
        f"[context budget] estimated {estimated_tokens:,} / {budget:,} tokens; "
        f"{remaining:.0f}% remaining; compactions {compactions}; "
        f"largest visible: {details}"
    )
    return text[:600]
