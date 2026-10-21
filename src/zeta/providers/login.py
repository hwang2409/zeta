"""Shared provider wiring for CLI and native-client OAuth login."""

from pathlib import Path
from urllib.parse import urlsplit

from ..core.login_flow import LoginProvider
from .anthropic import (
    DEFAULT_REDIRECT_URI as ANTHROPIC_REDIRECT_URI,
)
from .anthropic import (
    AnthropicCredentialStore,
)
from .anthropic import (
    build_authorization_url as build_anthropic_authorization_url,
)
from .anthropic import (
    exchange_authorization_code as exchange_anthropic_authorization_code,
)
from .auth import OAuthTokens, build_pkce_parameters
from .codex import (
    DEFAULT_CODEX_REDIRECT_URI,
    CodexCredentialStore,
    extract_account_id,
)
from .codex import (
    build_authorization_url as build_codex_authorization_url,
)
from .codex import (
    exchange_authorization_code as exchange_codex_authorization_code,
)


def pkce_values() -> tuple[str, str, str]:
    parameters = build_pkce_parameters()
    return parameters.verifier, parameters.challenge, parameters.state


def _no_handle(tokens: OAuthTokens) -> str | None:
    del tokens
    return None


def _redirect_contract(redirect_uri: str) -> tuple[int, str]:
    parsed = urlsplit(redirect_uri)
    if parsed.port is None or not parsed.path:
        raise ValueError(f"invalid login redirect URI: {redirect_uri}")
    return parsed.port, parsed.path


def build_login_provider(provider: str, home: Path) -> LoginProvider[OAuthTokens]:
    if provider == "anthropic":
        callback_port, callback_path = _redirect_contract(ANTHROPIC_REDIRECT_URI)
        return LoginProvider(
            name="anthropic",
            callback_port=callback_port,
            callback_path=callback_path,
            build_authorization_url=build_anthropic_authorization_url,
            exchange_authorization_code=exchange_anthropic_authorization_code,
            credential_store=AnthropicCredentialStore(home / "anthropic-oauth.json"),
            token_handle=_no_handle,
        )
    if provider == "codex":
        callback_port, callback_path = _redirect_contract(DEFAULT_CODEX_REDIRECT_URI)
        return LoginProvider(
            name="codex",
            callback_port=callback_port,
            callback_path=callback_path,
            build_authorization_url=build_codex_authorization_url,
            exchange_authorization_code=exchange_codex_authorization_code,
            credential_store=CodexCredentialStore(home / "codex-oauth.json"),
            token_handle=lambda tokens: extract_account_id(tokens.access_token),
        )
    raise ValueError(f"unsupported login provider: {provider}")
