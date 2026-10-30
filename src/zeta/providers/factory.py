"""Build a completion backend for a network provider.

The agent tool lets a child run on a different model than its parent, so backend
construction has to be reachable from outside the TUI, where it used to live.

Only real providers are built here. Test backends are injected at the runtime
composition seam and are not part of this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..accounts import AccountCredentials, AccountId, AccountStore
from ..codex import CodexCredentialDriver
from ..config.settings import load_settings
from ..core.session import account_home, env_home
from ..core.slash import context_window
from ..models.catalog import provider_for_model
from ..protocol.types import CompletionBackend
from .anthropic import (
    AnthropicApiKeyCredential,
    AnthropicAuthError,
    AnthropicBackend,
    AnthropicCredential,
    AnthropicCredentialDriver,
)
from .codex import DEFAULT_CODEX_MODEL, CodexBackend
from .ollama import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_API_URL,
    OllamaBackend,
    model_context_window,
)
from .transport import DEFAULT_STREAM_STALL_RETRIES, DEFAULT_STREAM_STALL_SECONDS

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"

# Opt-in gate for ZETA-87 API-key auth: a bare ANTHROPIC_API_KEY must never,
# on its own, change how an interactive session authenticates (plenty of
# developers have it exported for unrelated tools), so a session only uses it
# once this is also set explicitly.
API_KEY_OPT_IN_VAR = "ZETA_ALLOW_API_KEY"


def credential_store(
    provider: str,
    *,
    home: str | Path | None = None,
    account: AccountId | None = None,
) -> AccountCredentials | None:
    """Return the account credential adapter, or None for credential-free providers."""

    if provider not in {"claude", "codex"}:
        return None
    auth_home = account_home(home)
    registry = AccountStore(auth_home)
    canonical = "anthropic" if provider == "claude" else provider
    if account is not None and account.provider != canonical:
        raise ValueError("account provider does not match")
    selected = account or registry.resolve_account(provider)
    driver = (
        AnthropicCredentialDriver()
        if provider == "claude"
        else CodexCredentialDriver()
    )
    return AccountCredentials(registry, selected, driver)


def _anthropic_credential(
    runtime_home: Path, account: AccountId | None = None
) -> AnthropicCredential:
    """Pick Claude's credential source, defaulting to subscription OAuth.

    API-key auth is opt-in only: it exists for evaluation (see
    docs/design.md ZETA-87), never as an implicit fallback for an interactive
    session that merely has ANTHROPIC_API_KEY set in its environment.
    """

    if os.environ.get(API_KEY_OPT_IN_VAR) != "1":
        credential = credential_store("claude", home=runtime_home, account=account)
        assert credential is not None
        return credential
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
    ollama_base_url: str | None = None,
    token_budget: int | None = None,
    account: AccountId | None = None,
) -> tuple[CompletionBackend, str]:
    """Build a network provider backend and report the model it settled on."""

    runtime_home = Path(home) if home is not None else env_home()
    stall_kwargs = {
        "stall_seconds": (
            DEFAULT_STREAM_STALL_SECONDS if stall_seconds is None else stall_seconds
        ),
        "stall_retries": (
            DEFAULT_STREAM_STALL_RETRIES if stall_retries is None else stall_retries
        ),
    }
    if provider == "claude":
        credential = _anthropic_credential(runtime_home, account)
        selected_model = model or DEFAULT_CLAUDE_MODEL
        if require_credentials:
            _require_credentials(credential)
        return AnthropicBackend(
            model=selected_model,
            token_store=credential,
            diagnostics_path=runtime_home / "logs" / "stream-diagnostics.jsonl",
            **stall_kwargs,
        ), selected_model
    if provider == "codex":
        credential = credential_store("codex", home=runtime_home, account=account)
        assert credential is not None
        if require_credentials:
            _require_credentials(credential)
        selected_model = model or DEFAULT_CODEX_MODEL
        return CodexBackend(
            model=selected_model,
            token_store=credential,
            diagnostics_path=runtime_home / "logs" / "stream-diagnostics.jsonl",
            **stall_kwargs,
        ), selected_model
    if provider == "ollama":
        selected_model = model or DEFAULT_OLLAMA_MODEL
        configured_url = load_settings(home=runtime_home).settings.ollama_base_url
        base_url = (
            ollama_base_url
            or os.environ.get("ZETA_OLLAMA_BASE_URL")
            or configured_url
            or OLLAMA_API_URL
        )
        window = context_window("ollama", selected_model)
        if window is None:
            window = model_context_window(selected_model, base_url)
        requested_budget = token_budget or window
        effective_budget = min(requested_budget, window)
        return OllamaBackend(
            model=selected_model,
            base_url=base_url,
            stall_seconds=stall_kwargs["stall_seconds"],
            stall_retries=stall_kwargs["stall_retries"],
            token_budget=effective_budget,
            model_context_window=window,
        ), selected_model
    raise ValueError(f"unsupported provider: {provider}")


def _require_credentials(
    credential: AnthropicCredential | AccountCredentials,
) -> None:
    """Check local login availability without refreshing or sending a request."""
    if (
        isinstance(credential, AccountCredentials)
        and credential.read() is None
        and credential.bootstrap() is None
    ):
        raise credential.login_required_error()


__all__ = [
    "API_KEY_OPT_IN_VAR",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "build_backend",
    "credential_store",
    "provider_for_model",
]
