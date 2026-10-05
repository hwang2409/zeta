"""Shared Codex authentication and request primitives.

This neutral module is used by the Codex provider and side-call tools. It imports
neither package so request authentication has one owner without crossing seams.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .core.session import env_home
from .oauth import OAuthCredentialStore, OAuthTokens


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
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        retry_reason: str | None = None,
        is_stall: bool = False,
    ) -> None:
        super().__init__(message)
        if type(code) is str and code:
            self.code = code
        self.status_code = status_code if type(status_code) is int else None
        self.retryable = retryable
        self.retry_reason = retry_reason
        self.is_stall = is_stall


CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_SCOPES = "openid profile email offline_access"
DEFAULT_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
JWT_AUTH_CLAIM = "https://api.openai.com/auth"


def build_authorization_url(
    state: str,
    code_challenge: str,
    redirect_uri: str = DEFAULT_CODEX_REDIRECT_URI,
) -> str:
    """Build the ChatGPT plan OAuth PKCE authorization URL."""

    if not state or not code_challenge or not redirect_uri:
        raise CodexAuthError("Codex OAuth PKCE parameters are incomplete")
    params = {
        "client_id": CODEX_CLIENT_ID,
        "response_type": "code",
        "scope": CODEX_OAUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": "codex_cli_rs",
        "redirect_uri": redirect_uri,
    }
    return f"{CODEX_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_authorization_code(
    client: httpx.AsyncClient,
    code: str,
    state: str,
    code_verifier: str,
    redirect_uri: str,
    *,
    token_url: str = CODEX_TOKEN_URL,
) -> OAuthTokens:
    """Exchange a ChatGPT plan authorization code for OAuth tokens."""

    if not code or not state or not code_verifier or not redirect_uri:
        raise CodexAuthError("Codex OAuth code exchange parameters are incomplete")
    try:
        response = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise CodexAuthError("Codex OAuth code exchange failed") from exc
    if response.status_code >= 400:
        raise CodexHTTPError(
            f"Codex OAuth code exchange failed with HTTP {response.status_code}"
        )
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CodexAuthError("Codex OAuth code response is invalid") from exc
    if not isinstance(value, Mapping):
        raise CodexAuthError("Codex OAuth code response is invalid")
    access = _first_string(value, "access_token", "accessToken", "access")
    refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
    expires_in = value.get("expires_in", value.get("expiresIn"))
    if not access or not refresh or type(expires_in) not in {int, float}:
        raise CodexAuthError("Codex OAuth code response is invalid")
    return OAuthTokens(access, refresh, time.time() + float(expires_in) - 300)


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


def _extract_codex_tokens(value: Any) -> OAuthTokens:
    if not isinstance(value, Mapping):
        raise TypeError("Codex credentials are not an object")
    nested = value.get("tokens")
    if isinstance(nested, Mapping):
        value = nested
    access = _first_string(value, "access_token", "accessToken", "access")
    refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
    if not access or not refresh:
        raise ValueError("Codex credentials are incomplete")
    payload = _jwt_payload(access)
    expiry = payload.get("exp")
    expires_at = (
        float(expiry)
        if type(expiry) in {int, float} and math.isfinite(float(expiry))
        else 0.0
    )
    return OAuthTokens(access, refresh, expires_at)


class CodexCredentialStore(OAuthCredentialStore):
    """Owns zeta's Codex OAuth file and reads ~/.codex/auth.json only to bootstrap."""

    auth_error_type = CodexAuthError
    http_error_type = CodexHTTPError
    provider_label = "Codex"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        codex_auth: str | Path | None = None,
        token_url: str = CODEX_TOKEN_URL,
    ) -> None:
        super().__init__(path or env_home() / "codex-oauth.json", token_url=token_url)
        if codex_auth is not None:
            self.codex_auth = Path(codex_auth)
        elif codex_home := os.environ.get("CODEX_HOME"):
            self.codex_auth = Path(codex_home) / "auth.json"
        elif "ZETA_HOME" not in os.environ:
            self.codex_auth = Path.home() / ".codex" / "auth.json"
        else:
            self.codex_auth = None

    def bootstrap(self) -> OAuthTokens | None:
        if self.codex_auth is None or not self.codex_auth.exists():
            return None
        try:
            with self.codex_auth.open() as handle:
                return _extract_codex_tokens(json.load(handle))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CodexAuthError("Codex credentials could not be read") from exc

    async def refresh(
        self, refresh_token: str, client: httpx.AsyncClient
    ) -> OAuthTokens:
        try:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
                headers={"accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError("Codex OAuth token refresh failed") from exc
        if response.status_code >= 400:
            raise CodexHTTPError(
                f"Codex OAuth token refresh failed with HTTP {response.status_code}"
            )
        try:
            value = response.json()
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CodexAuthError("Codex OAuth token response is invalid") from exc
        if not isinstance(value, Mapping):
            raise CodexAuthError("Codex OAuth token response is invalid")
        access = _first_string(value, "access_token", "accessToken", "access")
        refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
        expires_in = value.get("expires_in", value.get("expiresIn"))
        refresh_keys = ("refresh_token", "refreshToken", "refresh")
        refresh_present = any(key in value for key in refresh_keys)
        if (
            not access
            or type(expires_in) not in {int, float}
            or (refresh_present and not refresh)
        ):
            raise CodexAuthError("Codex OAuth token response is invalid")
        return OAuthTokens(
            access,
            refresh or refresh_token,
            time.time() + float(expires_in) - 300,
        )


def extract_account_id(access_token: str) -> str:
    """Derive the ChatGPT account id from the access token claim."""

    try:
        payload = _jwt_payload(access_token)
        account = payload[JWT_AUTH_CLAIM]["chatgpt_account_id"]
    except (
        KeyError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        binascii.Error,
        json.JSONDecodeError,
    ):
        raise CodexAuthError("Codex access token has no ChatGPT account id") from None
    if type(account) is not str or not account:
        raise CodexAuthError("Codex access token has no ChatGPT account id")
    return account


def _jwt_payload(access_token: str) -> Mapping[str, Any]:
    parts = access_token.split(".")
    if len(parts) != 3:
        raise ValueError
    encoded = parts[1] + "=" * (-len(parts[1]) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError
    return payload


def codex_request_headers(access_token: str) -> dict[str, str]:
    """Build the common headers for a Codex Responses request."""

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
    "DEFAULT_CODEX_REDIRECT_URI",
    "CodexAuthError",
    "CodexBackendError",
    "CodexCredentialStore",
    "CodexHTTPError",
    "CodexStreamError",
    "build_authorization_url",
    "codex_request_headers",
    "exchange_authorization_code",
    "extract_account_id",
]
