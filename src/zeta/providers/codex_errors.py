"""Compatibility imports for shared Codex errors."""

from ..codex import (
    CodexAuthError,
    CodexBackendError,
    CodexHTTPError,
    CodexStreamError,
)

__all__ = [
    "CodexAuthError",
    "CodexBackendError",
    "CodexHTTPError",
    "CodexStreamError",
]
