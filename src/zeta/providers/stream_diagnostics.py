"""Bounded, local-only diagnostics for salvaged provider streams."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STREAM_DIAGNOSTICS_MAX_BYTES = 1024 * 1024
_CAUSE_MAX_BYTES = 300
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:bearer|api[-_ ]?key|(?:access|refresh|auth(?:entication)?)?[-_ ]?token)"
    r"\b(?:\s*[:=]\s*|\s+)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def sanitize_cause(cause: str) -> str:
    """Return a short cause message with credential-like values removed."""

    excerpt = cause[: _CAUSE_MAX_BYTES * 4]
    excerpt = _SECRET_PATTERN.sub("[redacted]", excerpt)
    return " ".join(excerpt.split())[:_CAUSE_MAX_BYTES]


class StreamDiagnostics:
    """Build and write at most one diagnostic record for a stream."""

    def __init__(
        self,
        path: Path,
        *,
        headers: Mapping[str, str],
        model: str | None,
        started_at: float,
    ) -> None:
        self.path = path
        self.headers = headers
        self.model = model
        self.started_at = started_at
        self.written = False

    def record(
        self,
        cause: str,
        *,
        bytes_received: int,
        sse_events_received: int,
        last_event_at: float,
        open_blocks: int,
        closed_blocks: int,
        stop_reason: str | None,
    ) -> None:
        if self.written:
            return
        now = time.monotonic()
        request_id = self.headers.get("request-id") or self.headers.get("x-request-id")
        response_model = (
            self.headers.get("model")
            or self.headers.get("x-model")
            or self.headers.get("anthropic-model")
            or self.model
        )
        write_stream_diagnostic(
            self.path,
            {
                "timestamp": time.time(),
                "cause": cause,
                "stream_age_seconds": max(0.0, now - self.started_at),
                "bytes_received": bytes_received,
                "sse_events_received": sse_events_received,
                "idle_gap_seconds": max(0.0, now - last_event_at),
                "open_blocks": open_blocks,
                "closed_blocks": closed_blocks,
                "stop_reason": stop_reason,
                "request_id": request_id,
                "model": response_model,
            },
        )
        self.written = True


def write_stream_diagnostic(
    path: Path,
    record: Mapping[str, Any],
    *,
    max_bytes: int | None = None,
) -> None:
    """Append one bounded record without affecting stream handling."""

    try:
        cap = STREAM_DIAGNOSTICS_MAX_BYTES if max_bytes is None else max_bytes
        encoded = _encode_record(record, cap)
        if not encoded:
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        if path.exists() and path.stat().st_size + len(encoded) > cap:
            rotated = path.with_name(f"{path.name}.1")
            rotated.unlink(missing_ok=True)
            path.replace(rotated)
        with path.open("ab") as handle:
            handle.write(encoded)
        os.chmod(path, 0o600)
    except OSError:
        return


def _encode_record(record: Mapping[str, Any], cap: int) -> bytes:
    record = dict(record)
    if isinstance(record.get("cause"), str):
        record["cause"] = sanitize_cause(record["cause"])
    bounded = {
        key: _bound_value(value, cap)
        for key, value in record.items()
    }
    encoded = _json_line(bounded)
    while len(encoded) > cap:
        text_fields = [
            key for key, value in bounded.items() if isinstance(value, str) and value
        ]
        if not text_fields:
            break
        key = max(text_fields, key=lambda item: len(str(bounded[item]).encode()))
        value = str(bounded[key])
        bounded[key] = _truncate(value, max(0, len(value.encode()) // 2))
        encoded = _json_line(bounded)
    if len(encoded) <= cap:
        return encoded
    fallback = {"cause": _truncate(str(record.get("cause", "")), max(0, cap // 4))}
    encoded = _json_line(fallback)
    return encoded if len(encoded) <= cap else b""


def _bound_value(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, min(_CAUSE_MAX_BYTES, cap))
    return value


def _truncate(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
