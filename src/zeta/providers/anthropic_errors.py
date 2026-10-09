"""Errors and HTTP response classification for the Anthropic provider."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .auth import error_body_excerpt
from .transport import retry_after_seconds

_ANTHROPIC_PROMPT_TOKENS = re.compile(
    r"\bprompt is too long:\s*([0-9][0-9,]*) tokens?\s*>\s*"
    r"[0-9][0-9,]* maximum\b",
    re.IGNORECASE,
)


def _provider_prompt_tokens(message: str) -> int | None:
    match = _ANTHROPIC_PROMPT_TOKENS.search(message)
    return int(match.group(1).replace(",", "")) if match is not None else None


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
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        retry_reason: str | None = None,
        is_stall: bool = False,
    ) -> None:
        super().__init__(message)
        if type(code) is str and code:
            self.code = {
                "authentication_error": "auth_error",
                "permission_error": "permission_denied",
                "not_found_error": "model_not_found",
            }.get(code, code)
            if code == "invalid_request_error" and "prompt is too long" in message.lower():
                self.code = "context_length_exceeded"
        self.status_code = status_code if type(status_code) is int else None
        self.retryable = retryable
        self.retry_reason = retry_reason
        self.is_stall = is_stall
        self.provider_prompt_tokens = (
            _provider_prompt_tokens(message)
            if self.code == "context_length_exceeded"
            else None
        )


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
    error = error_type(
        f"Anthropic HTTP {status_code}: {message}",
        status_code=status_code,
        retry_after=retry_after,
        retryable=retryable,
    )
    if status_code == 400 and _is_context_overflow_body(body):
        error.code = "context_length_exceeded"
        error.provider_prompt_tokens = _provider_prompt_tokens(message)
    return error


def _is_context_overflow_body(body: bytes) -> bool:
    try:
        payload: Any = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    detail = payload.get("error")
    return (
        isinstance(detail, Mapping)
        and detail.get("type") == "invalid_request_error"
        and type(detail.get("message")) is str
        and "prompt is too long" in detail["message"].lower()
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
