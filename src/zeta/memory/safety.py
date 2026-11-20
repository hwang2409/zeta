"""Shared secret detection and redaction for persisted project evidence."""

from __future__ import annotations

import re
from typing import Any

_KEY_PREFIX = r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*"
# Token patterns cannot hide a match of their own kind inside one match, so a
# plain scan finds all of them. Key matches can (`a=x;password=y`), so keys
# are found by prefix and each one covers to the end of its `\S+` run.
_TOKEN_PATTERNS = (
    re.compile(
        r"-----BEGIN (?P<label>(?:RSA |EC |OPENSSH )?PRIVATE KEY)-----"
        r".*?-----END (?P=label)-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk[-_]|ghp_|github_pat_)[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
    # JWT: base64url header (`{"` encodes to `eyJ`), payload, and signature.
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}"),
    # Only the token after `Bearer ` is masked, so the auth scheme stays readable.
    # shortcut: exactly one whitespace after Bearer, widen if multi-space headers show up.
    re.compile(r"(?<=\bBearer\s)[A-Za-z0-9._~+/-]{16,}=*", re.IGNORECASE),
)
_SECRET_PATTERNS = (*_TOKEN_PATTERNS, re.compile(_KEY_PREFIX + r"\S+", re.IGNORECASE))
_KEY_START = re.compile(_KEY_PREFIX + r"(?=\S)", re.IGNORECASE)
_RUN = re.compile(r"\S+")

REDACTED_SECRET = "[secret omitted]"


def contains_secret(value: str) -> bool:
    """Return whether text matches a shared secret pattern."""

    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _secret_spans(value: str) -> list[tuple[int, int]]:
    spans = [
        match.span() for pattern in _TOKEN_PATTERNS for match in pattern.finditer(value)
    ]
    run_end = -1
    for match in _KEY_START.finditer(value):
        if match.end() >= run_end:  # keys inside one run share its end
            run_end = _RUN.match(value, match.end()).end()
        spans.append((match.start(), run_end))
    return spans


def redact_secrets(value: str) -> str:
    """Replace complete secret-bearing matches while retaining safe context."""

    # Match on the original text and merge overlapping spans, so one pattern
    # never matches inside another pattern's marker or leftover fragment.
    spans = sorted(_secret_spans(value))
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
