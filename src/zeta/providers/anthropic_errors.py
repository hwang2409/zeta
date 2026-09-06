"""Errors and HTTP response classification for the Anthropic provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .auth import error_body_excerpt
from .transport import retry_after_seconds


class AnthropicBackendError(RuntimeError):
    """Base class for errors that the agent loop can report as backend errors."""

    code = "backend_error"


class AnthropicAuthError(AnthropicBackendError):
    """Raised when Claude subscription credentials are missing or invalid."""

    code = "auth_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnthropicHTTPError(AnthropicBackendError):
    """Raised when Anthropic returns an unsuccessful HTTP response."""

    code = "http_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


class AnthropicStreamError(AnthropicBackendError):
    """Raised when an Anthropic SSE stream is invalid or ends early."""

    code = "stream_error"

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_reason: str | None = None,
        is_stall: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_reason = retry_reason
        self.is_stall = is_stall


def http_error(
    status_code: int,
    body: bytes,
    headers: Mapping[str, str] | None = None,
) -> AnthropicBackendError:
    """Classify an Anthropic HTTP response without exposing its body."""

    message = error_body_excerpt(body) or "request failed"
    error_type = AnthropicAuthError if status_code in {401, 403} else AnthropicHTTPError
    retry_after = retry_after_seconds(headers)
    retryable = _is_overloaded_body(body)
    if error_type is AnthropicAuthError:
        return error_type(
            f"Anthropic HTTP {status_code}: {message}", status_code=status_code
        )
    return error_type(
        f"Anthropic HTTP {status_code}: {message}",
        status_code=status_code,
        retry_after=retry_after,
        retryable=retryable,
    )


def _is_overloaded_body(body: bytes) -> bool:
    try:
        payload: Any = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    detail = payload.get("error")
    return isinstance(detail, Mapping) and detail.get("type") == "overloaded_error"


__all__ = [
    "AnthropicAuthError",
    "AnthropicBackendError",
    "AnthropicHTTPError",
    "AnthropicStreamError",
    "http_error",
]
