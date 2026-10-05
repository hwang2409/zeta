"""Neutral access point for shared Codex authentication and request plumbing."""

from .providers.codex import (
    CODEX_API_URL,
    DEFAULT_CODEX_MODEL,
    CodexCredentialStore,
    extract_account_id,
)
from .providers.codex_errors import (
    CodexAuthError,
    CodexBackendError,
    CodexHTTPError,
    CodexStreamError,
)


def codex_request_headers(access_token: str) -> dict[str, str]:
    return {
        "accept": "text/event-stream",
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": extract_account_id(access_token),
        "content-type": "application/json",
        "originator": "zeta",
        "openai-beta": "responses=experimental",
        "user-agent": "zeta/0.1",
    }


__all__ = [
    "CODEX_API_URL",
    "DEFAULT_CODEX_MODEL",
    "CodexAuthError",
    "CodexBackendError",
    "CodexCredentialStore",
    "CodexHTTPError",
    "CodexStreamError",
    "codex_request_headers",
]
