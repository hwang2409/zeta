"""The closed ``zeta.error.code`` of a failed span.

The code is read only from machine fields (``code``, ``retry_reason``, ``status_code``) and
the exception class, never from a message: provider messages can carry request content.
"""

from __future__ import annotations

import asyncio

# zeta error codes and retry reasons (providers/*, protocol ErrorInfo codes), plus the
# provider codes zeta passes through: codex/OpenAI codes and Anthropic error types.
_KNOWN = {
    "invalid_request_error": "invalid_request",
    "rate_limit_error": "rate_limit",
    "rate_limit_exceeded": "rate_limit",
    "usage_limit_reached": "rate_limit",
    "subscription_limit": "rate_limit",
    "insufficient_quota": "rate_limit",
    "overloaded_error": "overloaded",
    "server_is_overloaded": "overloaded",
    "slow_down": "overloaded",
    "context_length_exceeded": "context_length",
    "auth_error": "auth",
    "authentication_error": "auth",
    "permission_denied": "auth",
    "permission_error": "auth",
    "server_error": "server_error",
    "api_error": "server_error",
    "stream_stall": "timeout",
    "timeout": "timeout",
    "account_lock_timeout": "timeout",
    "stream_error": "stream_error",
    "stream_inconsistent": "stream_error",
    "transport_error": "network",
    "model_not_found": "model_not_found",
    "not_found_error": "model_not_found",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "max_turns": "turn_limit",
}
_STATUS = {400: "invalid_request", 401: "auth", 403: "auth", 404: "model_not_found", 408: "timeout", 429: "rate_limit", 504: "timeout", 529: "overloaded"}
ERROR_CODES = frozenset({*_KNOWN.values(), *_STATUS.values(), "cancelled", "network", "tool_error", "unknown"})


def error_code(error: object) -> str:
    """Map an exception, exception class, or ``ErrorInfo``-like object onto ``ERROR_CODES``."""
    kind = error if isinstance(error, type) else type(error)
    if issubclass(kind, (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt)):
        return "cancelled"
    if not isinstance(error, type):
        for field in ("code", "retry_reason"):
            value = getattr(error, field, None)
            if isinstance(value, str) and value in _KNOWN:
                return _KNOWN[value]
        status = getattr(error, "status_code", None)
        if type(status) is int:
            return _STATUS.get(status, "server_error" if status >= 500 else "unknown")
    if issubclass(kind, TimeoutError):
        return "timeout"
    if issubclass(kind, ConnectionError):
        return "network"
    return "unknown"
