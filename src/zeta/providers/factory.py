"""Build a completion backend for a network provider.

The agent tool lets a child run on a different model than its parent, so backend
construction has to be reachable from outside the TUI, where it used to live.

Only the real providers are built here. The offline "fake" provider lives in the
TUI layer, which this layer must not import, so tui.app wraps this factory and
handles that case itself.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.session import env_home
from ..model_catalog import provider_for_model
from ..types import CompletionBackend
from .anthropic import (
    AnthropicApiKeyCredential,
    AnthropicAuthError,
    AnthropicBackend,
    AnthropicCredential,
    AnthropicCredentialStore,
)
from .auth import OAuthCredentialStore
from .codex import CodexBackend, CodexCredentialStore
from .transport import DEFAULT_STREAM_STALL_RETRIES, DEFAULT_STREAM_STALL_SECONDS

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.4"

# Opt-in gate for ZETA-87 API-key auth: a bare ANTHROPIC_API_KEY must never,
# on its own, change how an interactive session authenticates (plenty of
# developers have it exported for unrelated tools), so a session only uses it
# once this is also set explicitly.
API_KEY_OPT_IN_VAR = "ZETA_ALLOW_API_KEY"


def credential_store(
    provider: str,
    *,
    home: str | Path | None = None,
) -> OAuthCredentialStore | None:
    """Return the provider's OAuth store, or None when it needs no credentials."""

    auth_home = Path(home) if home is not None else env_home()
    if provider == "claude":
        return AnthropicCredentialStore(auth_home / "anthropic-oauth.json")
    if provider == "codex":
        return CodexCredentialStore(auth_home / "codex-oauth.json")
    return None


def _anthropic_credential(auth_home: Path) -> AnthropicCredential:
    """Pick Claude's credential source, defaulting to subscription OAuth.

    API-key auth is opt-in only: it exists for evaluation/automation (see
    docs/design.md ZETA-87), never as an implicit fallback for an interactive
    session that merely has ANTHROPIC_API_KEY set in its environment.
    """

    if os.environ.get(API_KEY_OPT_IN_VAR) != "1":
        return AnthropicCredentialStore(auth_home / "anthropic-oauth.json")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnthropicAuthError(
            f"{API_KEY_OPT_IN_VAR}=1 is set but ANTHROPIC_API_KEY is empty; "
            "either set ANTHROPIC_API_KEY to use API-key auth, or unset "
            f"{API_KEY_OPT_IN_VAR} and run `zeta login` for subscription OAuth"
        )
    return AnthropicApiKeyCredential(api_key)


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
    require_credentials: bool = False,
) -> tuple[CompletionBackend, str]:
    """Build a network provider backend and report the model it settled on."""

    auth_home = Path(home) if home is not None else env_home()
    stall_kwargs = {
        "stall_seconds": (
            DEFAULT_STREAM_STALL_SECONDS if stall_seconds is None else stall_seconds
        ),
        "stall_retries": (
            DEFAULT_STREAM_STALL_RETRIES if stall_retries is None else stall_retries
        ),
    }
    if provider == "claude":
        credential = _anthropic_credential(auth_home)
        selected_model = model or DEFAULT_CLAUDE_MODEL
        if require_credentials:
            _require_credentials(credential)
        return AnthropicBackend(
            model=selected_model,
            token_store=credential,
            diagnostics_path=auth_home / "logs" / "stream-diagnostics.jsonl",
            **stall_kwargs,
        ), selected_model
    if provider == "codex":
        credential = CodexCredentialStore(auth_home / "codex-oauth.json")
        if require_credentials:
            _require_credentials(credential)
        selected_model = model or DEFAULT_CODEX_MODEL
        return CodexBackend(
            model=selected_model,
            token_store=credential,
            **stall_kwargs,
        ), selected_model
    raise ValueError(f"unsupported provider: {provider}")


def _require_credentials(credential: AnthropicCredential | OAuthCredentialStore) -> None:
    """Check local login availability without refreshing or sending a request."""
    if (
        isinstance(credential, OAuthCredentialStore)
        and credential.read() is None
        and credential.bootstrap() is None
    ):
        raise credential.auth_error_type(
            f"no {credential.provider_label} OAuth login found; log in first"
        )


__all__ = [
    "API_KEY_OPT_IN_VAR",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_MODEL",
    "build_backend",
    "credential_store",
    "provider_for_model",
]
