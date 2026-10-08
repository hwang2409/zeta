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

REDACTED_SECRET = "[secret omitted]"


def contains_secret(value: str) -> bool:
    """Return whether text matches a shared secret pattern."""

    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def redact_secrets(value: str) -> str:
    """Replace complete secret-bearing matches while retaining safe context."""

    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED_SECRET, redacted)
    return redacted


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
