"""Deterministic fallback summaries for conversation compaction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FALLBACK_SUMMARY_PREFIX = "[automatic fallback summary: model returned no summary]"
_FALLBACK_TRUNCATION_MARKER = "[fallback summary truncated]"


def _bounded_summary(lines: list[str], max_chars: int) -> str:
    """Join fallback lines without exceeding the caller's output bound."""

    if max_chars < 0:
        raise ValueError("fallback summary bound must not be negative")
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    suffix = f"\n{_FALLBACK_TRUNCATION_MARKER}"
    body_start = f"{FALLBACK_SUMMARY_PREFIX}\n"
    body_chars = max_chars - len(body_start) - len(suffix)
    if body_chars < 0:
        return FALLBACK_SUMMARY_PREFIX[:max_chars]
    body = "\n".join(lines[1:])[:body_chars]
    return f"{body_start}{body}{suffix}"


def fallback_summary(source: str, *, max_chars: int) -> str:
    """Build a deterministic fallback within the exact output character bound."""

    lines = [FALLBACK_SUMMARY_PREFIX]
    try:
        rows = json.loads(source)
    except (json.JSONDecodeError, TypeError):
        digest = hashlib.sha256(source.encode()).hexdigest()
        return _bounded_summary(
            [FALLBACK_SUMMARY_PREFIX, f"source_sha256={digest}"], max_chars
        )
    rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            lines.append(f"summary: {str(row)[:400]}")
            continue
        role = str(row.get("role", "unknown"))[:100]
        content = row.get("content")
        blocks = content if isinstance(content, list) else []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, Mapping) and type(block.get("text")) is str
        )
        nonempty = [line for line in text.splitlines() if line.strip()]
        if role == "user":
            lines.append(f"user: {text[:400]}")
        elif nonempty:
            excerpt = nonempty[0][:200]
            if len(nonempty) > 1:
                excerpt += f" … {nonempty[-1][:200]}"
            lines.append(f"{role}: {excerpt}")
        for block in blocks:
            call = block.get("tool_call") if isinstance(block, Mapping) else None
            if isinstance(call, Mapping):
                args = json.dumps(call.get("arguments", {}), sort_keys=True)
                digest = hashlib.sha256(args.encode()).hexdigest()[:16]
                name = str(call.get("name", "unknown"))[:400]
                lines.append(f"tool: {name} args_sha256={digest}")
    return _bounded_summary(lines, max_chars)
