"""OAuth 2.1 authorization-code + PKCE flow for MCP HTTP servers.

Follows the MCP authorization spec (draft 2025-06-18): the client discovers
the protected-resource metadata, then the authorization-server metadata,
optionally registers dynamically, then runs an authorization-code + PKCE
S256 flow against a localhost redirect. Tokens land in
``~/.zeta/mcp-tokens`` via :mod:`.oauth_store`; the PKCE verifier is never
persisted.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx

from .oauth_store import MCPOAuthToken, save_token

logger = logging.getLogger(__name__)

OAUTH_LISTEN_TIMEOUT_SECONDS = 300.0
DEFAULT_CLIENT_NAME = "zeta"
DEFAULT_REDIRECT_PATH = "/callback"


class MCPOAuthError(RuntimeError):
    """Raised when the OAuth flow fails."""


class MCPOAuthStateError(MCPOAuthError):
    """Raised when the redirect state parameter does not match."""


BrowserOpener = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AuthServerMetadata:
    """Subset of RFC 8414 metadata this client uses."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    scopes_supported: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProtectedResourceMetadata:
    """Result of the /.well-known/oauth-protected-resource probe."""

    resource: str
    authorization_server: str | None


async def discover_protected_resource(
    server_url: str,
    *,
    http_client: httpx.AsyncClient,
    resource_metadata_url: str | None = None,
) -> ProtectedResourceMetadata:
    """Probe the MCP server for RFC 9728 protected-resource metadata."""

    parsed = urlparse(server_url)
    default_url = urljoin(
        f"{parsed.scheme}://{parsed.netloc}",
        "/.well-known/oauth-protected-resource",
    )
    metadata_url = resource_metadata_url or default_url
    try:
        response = await http_client.get(metadata_url, timeout=10.0)
    except httpx.HTTPError as exc:
        logger.debug("MCP OAuth: protected-resource probe failed: %s", exc)
        return ProtectedResourceMetadata(server_url, None)
    if response.status_code >= 400:
        return ProtectedResourceMetadata(server_url, None)
    try:
        data = response.json()
    except ValueError:
        return ProtectedResourceMetadata(server_url, None)
    if type(data) is not dict:
        return ProtectedResourceMetadata(server_url, None)
    resource_value = data.get("resource")
    resource = resource_value if type(resource_value) is str else server_url
    servers = data.get("authorization_servers")
    server: str | None = None
    if type(servers) is list and servers:
        first = servers[0]
        if type(first) is str:
            server = first
    return ProtectedResourceMetadata(resource, server)


async def discover_auth_server(
    authorization_server_url: str,
    *,
    http_client: httpx.AsyncClient,
) -> AuthServerMetadata:
    """Fetch RFC 8414 authorization-server metadata."""

    parsed = urlparse(authorization_server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    metadata_url = urljoin(base, "/.well-known/oauth-authorization-server")
    try:
        response = await http_client.get(metadata_url, timeout=10.0)
    except httpx.HTTPError as exc:
        raise MCPOAuthError(
            f"MCP OAuth: could not reach authorization-server metadata: {exc}"
        ) from exc
    if response.status_code >= 400:
        raise MCPOAuthError(
            f"MCP OAuth: authorization-server metadata returned HTTP "
            f"{response.status_code}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise MCPOAuthError(
            "MCP OAuth: authorization-server metadata was not JSON"
        ) from exc
    if type(data) is not dict:
        raise MCPOAuthError(
            "MCP OAuth: authorization-server metadata was not an object"
        )
    issuer = data.get("issuer")
    auth_endpoint = data.get("authorization_endpoint")
    token_endpoint = data.get("token_endpoint")
    if type(issuer) is not str or type(auth_endpoint) is not str or type(token_endpoint) is not str:
        raise MCPOAuthError(
            "MCP OAuth: metadata missing issuer, authorization_endpoint, "
            "or token_endpoint"
        )
    methods = data.get("code_challenge_methods_supported")
    if type(methods) is list and "S256" not in methods:
        raise MCPOAuthError(
            "MCP OAuth: authorization server does not advertise PKCE S256"
        )
    registration = data.get("registration_endpoint")
    scopes = data.get("scopes_supported") or ()
    scope_tuple: tuple[str, ...] = ()
    if type(scopes) is list:
        scope_tuple = tuple(item for item in scopes if type(item) is str)
    return AuthServerMetadata(
        issuer=issuer,
        authorization_endpoint=auth_endpoint,
        token_endpoint=token_endpoint,
        registration_endpoint=registration if type(registration) is str else None,
        scopes_supported=scope_tuple,
    )


async def register_client(
    metadata: AuthServerMetadata,
    redirect_uri: str,
    *,
    http_client: httpx.AsyncClient,
    client_name: str = DEFAULT_CLIENT_NAME,
) -> tuple[str, str | None]:
    """Register a public client dynamically (RFC 7591) at the auth server."""

    if metadata.registration_endpoint is None:
        raise MCPOAuthError(
            "MCP OAuth: authorization server has no registration_endpoint "
            "and no client is preregistered"
        )
    payload = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    try:
        response = await http_client.post(
            metadata.registration_endpoint, json=payload, timeout=10.0
        )
    except httpx.HTTPError as exc:
        raise MCPOAuthError(
            f"MCP OAuth: dynamic client registration failed: {exc}"
        ) from exc
    if response.status_code >= 400:
        raise MCPOAuthError(
            f"MCP OAuth: dynamic client registration returned HTTP "
            f"{response.status_code}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise MCPOAuthError(
            "MCP OAuth: registration response was not JSON"
        ) from exc
    client_id = data.get("client_id")
    if type(client_id) is not str or not client_id:
        raise MCPOAuthError(
            "MCP OAuth: registration response missing client_id"
        )
    secret = data.get("client_secret")
    return client_id, secret if type(secret) is str else None


def generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge) for PKCE S256."""

    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_authorization_url(
    metadata: AuthServerMetadata,
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    resource: str,
    scope: str | None = None,
) -> str:
    """Build the authorization request URL with PKCE + resource params."""

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if scope:
        params["scope"] = scope
    separator = "&" if "?" in metadata.authorization_endpoint else "?"
    return f"{metadata.authorization_endpoint}{separator}{urlencode(params)}"


@dataclass(slots=True)
class _RedirectPayload:
    code: str | None = None
    state: str | None = None
    error: str | None = None


async def wait_for_redirect(
    server: asyncio.base_events.Server,
    payload: _RedirectPayload,
    signal: asyncio.Event,
    *,
    timeout: float = OAUTH_LISTEN_TIMEOUT_SECONDS,
) -> _RedirectPayload:
    """Block until the browser redirect fires (or the timeout expires)."""

    try:
        await asyncio.wait_for(signal.wait(), timeout=timeout)
    except TimeoutError as exc:
        raise MCPOAuthError(
            f"MCP OAuth: no redirect received after {timeout:.0f}s"
        ) from exc
    finally:
        server.close()
        try:
            await server.wait_closed()
        except (OSError, RuntimeError):
            pass
    return payload


async def start_redirect_listener(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    path: str = DEFAULT_REDIRECT_PATH,
) -> tuple[asyncio.base_events.Server, str, _RedirectPayload, asyncio.Event]:
    """Bind a localhost TCP listener that captures one redirect query string.

    Returns the server handle, the fully qualified redirect URI, the payload
    that receives the captured code/state, and an event set once the redirect
    fires.
    """

    payload = _RedirectPayload()
    signal = asyncio.Event()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
        except (ConnectionError, asyncio.IncompleteReadError):
            writer.close()
            return
        try:
            method, target, _version = request_line.decode("latin-1").split(" ", 2)
        except ValueError:
            _write_response(writer, 400, "bad request")
            await _drain_and_close(writer)
            return
        while True:
            try:
                header = await reader.readline()
            except (ConnectionError, asyncio.IncompleteReadError):
                header = b""
            if header in {b"", b"\r\n", b"\n"}:
                break
        if method.upper() != "GET":
            _write_response(writer, 405, "method not allowed")
            await _drain_and_close(writer)
            return
        parsed = urlparse(target)
        if parsed.path != path:
            _write_response(writer, 404, "not found")
            await _drain_and_close(writer)
            return
        params = parse_qs(parsed.query)
        payload.code = _first_param(params, "code")
        payload.state = _first_param(params, "state")
        payload.error = _first_param(params, "error")
        message = (
            "Authorization received. You can close this tab and return to zeta."
            if payload.error is None
            else f"Authorization failed: {payload.error}"
        )
        _write_response(writer, 200, message)
        await _drain_and_close(writer)
        signal.set()

    server = await asyncio.start_server(handle, host=host, port=port)
    sockets = server.sockets or ()
    if not sockets:
        server.close()
        raise MCPOAuthError("MCP OAuth: could not bind localhost listener")
    bound_port = sockets[0].getsockname()[1]
    redirect_uri = f"http://{host}:{bound_port}{path}"
    return server, redirect_uri, payload, signal


def _first_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    value = values[0]
    return value or None


def _write_response(
    writer: asyncio.StreamWriter,
    status: int,
    body: str,
) -> None:
    payload = body.encode("utf-8")
    status_line = f"HTTP/1.1 {status} OK\r\n"
    headers = (
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    )
    writer.write(status_line.encode("ascii") + headers.encode("ascii") + payload)


async def _drain_and_close(writer: asyncio.StreamWriter) -> None:
    try:
        await writer.drain()
    except (ConnectionError, RuntimeError):
        pass
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, RuntimeError):
        pass


async def exchange_code_for_token(
    metadata: AuthServerMetadata,
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    client_id: str,
    client_secret: str | None,
    resource: str,
    http_client: httpx.AsyncClient,
) -> dict[str, object]:
    """Exchange the authorization code for a token response."""

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": resource,
    }
    if client_secret is not None:
        data["client_secret"] = client_secret
    return await _post_token_request(metadata.token_endpoint, data, http_client)


async def refresh_access_token(
    *,
    token_endpoint: str,
    refresh_token: str,
    client_id: str,
    client_secret: str | None,
    resource: str,
    http_client: httpx.AsyncClient,
) -> dict[str, object]:
    """Exchange a refresh token for a fresh access token."""

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "resource": resource,
    }
    if client_secret is not None:
        data["client_secret"] = client_secret
    return await _post_token_request(token_endpoint, data, http_client)


async def _post_token_request(
    endpoint: str,
    data: dict[str, str],
    http_client: httpx.AsyncClient,
) -> dict[str, object]:
    try:
        response = await http_client.post(
            endpoint,
            data=data,
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise MCPOAuthError(f"MCP OAuth: token request failed: {exc}") from exc
    if response.status_code >= 400:
        raise MCPOAuthError(
            f"MCP OAuth: token endpoint returned HTTP {response.status_code}"
        )
    try:
        value = response.json()
    except ValueError as exc:
        raise MCPOAuthError("MCP OAuth: token response was not JSON") from exc
    if type(value) is not dict:
        raise MCPOAuthError("MCP OAuth: token response was not an object")
    return value


def token_from_response(
    response: dict[str, object],
    *,
    fallback_refresh_token: str | None,
    metadata: AuthServerMetadata,
    client_id: str,
    client_secret: str | None,
    redirect_uri: str,
    resource: str,
    now: float | None = None,
) -> MCPOAuthToken:
    """Fold a token endpoint response into the persistable token bundle."""

    access = response.get("access_token")
    if type(access) is not str or not access:
        raise MCPOAuthError("MCP OAuth: token response missing access_token")
    token_type = response.get("token_type") or "Bearer"
    if type(token_type) is not str:
        raise MCPOAuthError("MCP OAuth: token_type must be a string")
    refresh = response.get("refresh_token")
    refresh_value: str | None
    if type(refresh) is str and refresh:
        refresh_value = refresh
    else:
        refresh_value = fallback_refresh_token
    expires_in = response.get("expires_in")
    expires_at: float | None = None
    if type(expires_in) in {int, float}:
        expires_at = (now if now is not None else time.time()) + float(expires_in)  # type: ignore[arg-type]
    scope_value = response.get("scope")
    return MCPOAuthToken(
        access_token=access,
        refresh_token=refresh_value,
        expires_at=expires_at,
        token_type=token_type,
        scope=scope_value if type(scope_value) is str else None,
        authorization_server=metadata.issuer,
        token_endpoint=metadata.token_endpoint,
        authorization_endpoint=metadata.authorization_endpoint,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        resource=resource,
    )


async def authorize(
    *,
    server_name: str,
    server_url: str,
    home: str | None = None,
    resource_metadata_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    browser_opener: BrowserOpener | None = None,
    listen_timeout: float = OAUTH_LISTEN_TIMEOUT_SECONDS,
) -> MCPOAuthToken:
    """Drive the full browser flow and persist the resulting token."""

    owns_client = http_client is None
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=10.0)
    listener: asyncio.base_events.Server | None = None
    try:
        resource_meta = await discover_protected_resource(
            server_url,
            http_client=http_client,
            resource_metadata_url=resource_metadata_url,
        )
        auth_server = resource_meta.authorization_server or server_url
        metadata = await discover_auth_server(auth_server, http_client=http_client)
        (
            listener,
            redirect_uri,
            payload,
            signal,
        ) = await start_redirect_listener()
        client_id, client_secret = await register_client(
            metadata, redirect_uri, http_client=http_client
        )
        verifier, challenge = generate_pkce()
        state = secrets.token_urlsafe(24)
        scope = " ".join(metadata.scopes_supported) if metadata.scopes_supported else None
        url = build_authorization_url(
            metadata,
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=challenge,
            resource=resource_meta.resource,
            scope=scope,
        )
        opener = browser_opener or _default_browser_opener
        try:
            opener(url)
        except Exception as exc:  # noqa: BLE001 - defensive: report but keep waiting
            logger.warning("MCP OAuth: browser opener failed for %s: %s", server_name, exc)
        outcome = await wait_for_redirect(
            listener, payload, signal, timeout=listen_timeout
        )
        listener = None
        if outcome.error:
            raise MCPOAuthError(f"MCP OAuth: authorization denied: {outcome.error}")
        if outcome.state != state:
            raise MCPOAuthStateError(
                "MCP OAuth: authorization redirect state did not match"
            )
        if not outcome.code:
            raise MCPOAuthError("MCP OAuth: authorization redirect missing code")
        response = await exchange_code_for_token(
            metadata,
            code=outcome.code,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
            client_id=client_id,
            client_secret=client_secret,
            resource=resource_meta.resource,
            http_client=http_client,
        )
        token = token_from_response(
            response,
            fallback_refresh_token=None,
            metadata=metadata,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            resource=resource_meta.resource,
        )
        save_token(server_name, token, home=home)
        return token
    finally:
        if listener is not None:
            listener.close()
            try:
                await listener.wait_closed()
            except (OSError, RuntimeError):
                pass
        if owns_client:
            await http_client.aclose()


def _default_browser_opener(url: str) -> None:
    if not webbrowser.open(url, new=1, autoraise=True):
        logger.info("MCP OAuth: open this URL to authorize: %s", url)


__all__ = [
    "OAUTH_LISTEN_TIMEOUT_SECONDS",
    "AuthServerMetadata",
    "BrowserOpener",
    "MCPOAuthError",
    "MCPOAuthStateError",
    "ProtectedResourceMetadata",
    "authorize",
    "build_authorization_url",
    "discover_auth_server",
    "discover_protected_resource",
    "exchange_code_for_token",
    "generate_pkce",
    "refresh_access_token",
    "register_client",
    "start_redirect_listener",
    "token_from_response",
    "wait_for_redirect",
]
