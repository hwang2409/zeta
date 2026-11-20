"""Shared secret detection and redaction for persisted project evidence."""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN (?P<label>(?:RSA |EC |OPENSSH )?PRIVATE KEY)-----"
        r".*?-----END (?P=label)-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)

# Lookahead copies match at every start position, so a long match (such as
# `key=\S+` swallowing a second key) cannot hide a match that starts inside it.
_OVERLAPPING_SECRET_PATTERNS = tuple(
    re.compile("(?=(" + pattern.pattern + "))", pattern.flags)
    for pattern in _SECRET_PATTERNS
)

REDACTED_SECRET = "[secret omitted]"


def contains_secret(value: str) -> bool:
    """Return whether text matches a shared secret pattern."""

    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def redact_secrets(value: str) -> str:
    """Replace complete secret-bearing matches while retaining safe context."""

    # Match on the original text and merge overlapping spans, so one pattern
    # never matches inside another pattern's marker or leftover fragment.
    spans = sorted(
        match.span(1)
        for pattern in _OVERLAPPING_SECRET_PATTERNS
        for match in pattern.finditer(value)
        if match.end(1) > match.start(1)
    )
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(value[cursor:start])
        parts.append(REDACTED_SECRET)
        cursor = end
    parts.append(value[cursor:])
    return "".join(parts)


def redact_secret_values(value: Any) -> Any:
    """Recursively redact strings in JSON-shaped data."""

    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [redact_secret_values(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secret_values(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_secret_values(item) for key, item in value.items()}
    return value
