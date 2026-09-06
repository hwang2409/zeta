"""Build a completion backend for a network provider.

The agent tool lets a child run on a different model than its parent, so backend
construction has to be reachable from outside the TUI, where it used to live.

Only the real providers are built here. The offline "fake" provider lives in the
TUI layer, which this layer must not import, so tui.app wraps this factory and
handles that case itself.
"""

from __future__ import annotations

from pathlib import Path

from ..core.session import env_home
from ..model_catalog import provider_for_model
from ..types import CompletionBackend
from .anthropic import AnthropicBackend, AnthropicCredentialStore
from .auth import OAuthCredentialStore
from .codex import CodexBackend, CodexCredentialStore
from .transport import DEFAULT_STREAM_STALL_RETRIES, DEFAULT_STREAM_STALL_SECONDS

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.4"


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


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
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
        selected_model = model or DEFAULT_CLAUDE_MODEL
        return AnthropicBackend(
            model=selected_model,
            token_store=AnthropicCredentialStore(auth_home / "anthropic-oauth.json"),
            **stall_kwargs,
        ), selected_model
    if provider == "codex":
        selected_model = model or DEFAULT_CODEX_MODEL
        return CodexBackend(
            model=selected_model,
            token_store=CodexCredentialStore(auth_home / "codex-oauth.json"),
            **stall_kwargs,
        ), selected_model
    raise ValueError(f"unsupported provider: {provider}")


__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_MODEL",
    "build_backend",
    "credential_store",
    "provider_for_model",
]
