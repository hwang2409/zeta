"""Deterministic fallback summaries for conversation compaction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FALLBACK_SUMMARY_PREFIX = "[automatic fallback summary: model returned no summary]"


def fallback_summary(source: str) -> str:
    """Build a bounded fallback from a serialized compaction source range."""

    lines = [FALLBACK_SUMMARY_PREFIX]
    try:
        rows = json.loads(source)
    except (json.JSONDecodeError, TypeError):
        digest = hashlib.sha256(source.encode()).hexdigest()
        return f"{FALLBACK_SUMMARY_PREFIX}\nsource_sha256={digest}"
    rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            lines.append(f"summary: {str(row)[:400]}")
            continue
        role = str(row.get("role", "unknown"))
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
                lines.append(
                    f"tool: {call.get('name', 'unknown')} args_sha256={digest}"
                )
        if sum(map(len, lines)) > 3_000:
            lines.append("[fallback summary truncated]")
            break
    return "\n".join(lines)
