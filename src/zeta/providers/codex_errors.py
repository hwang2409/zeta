"""Errors raised by the Codex provider."""

from __future__ import annotations


class CodexBackendError(RuntimeError):
    """Base class for errors that the agent loop can report."""

    code = "backend_error"


class CodexAuthError(CodexBackendError):
    """Raised when ChatGPT subscription credentials are missing or invalid."""

    code = "auth_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CodexHTTPError(CodexBackendError):
    """Raised when the ChatGPT backend returns an unsuccessful response."""

    code = "http_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class CodexStreamError(CodexBackendError):
    """Raised when a Responses SSE stream violates its lifecycle contract."""

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


__all__ = [
    "CodexAuthError",
    "CodexBackendError",
    "CodexHTTPError",
    "CodexStreamError",
]
