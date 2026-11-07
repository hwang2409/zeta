"""Tests for the MCP OAuth flow, resources attachment, and token hygiene."""

from __future__ import annotations

import asyncio
import inspect
import json
import multiprocessing
import os
import socket
import stat
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp import (
    MCPServerConfig,
    StreamableHTTPMCPClient,
    load_mcp_config,
    oauth_store,
)
from zeta.mcp.client import MCPHTTPError, MCPResource, MCPResourceContent
from zeta.mcp.commands import parse_add_command
from zeta.mcp.http import MAX_RESPONSE_BYTES
from zeta.mcp.management import MCPManagementService
from zeta.mcp.oauth import (
    AuthServerMetadata,
    MCPOAuthError,
    MCPOAuthStateError,
    ProtectedResourceMetadata,
    authorize,
    build_authorization_url,
    discover_auth_server,
    discover_protected_resource,
    generate_pkce,
    refresh_access_token,
)
from zeta.mcp.oauth_store import (
    CredentialSnapshot,
    MCPCredentialStore,
    MCPOAuthToken,
    SaveIfCurrentResult,
    redact_token,
    token_is_expired,
    token_state,
    token_store_path,
)
from zeta.mcp.resources import (
    MCPResourceError,
    fetch_resource,
    format_resource_list,
    list_resources,
)
from zeta.protocol.types import MessageOrigin, TextContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools._spill import SpillStore


async def _fire_redirect(url: str) -> None:
    """Send a raw HTTP GET to the OAuth redirect listener without httpx."""

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    reader, writer = await asyncio.open_connection(host, port)
    request = (
        f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(request)
    await writer.drain()
    try:
        await reader.read(-1)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, RuntimeError):
            pass


class _FakeAuthServer:
    """Handles every well-known + token endpoint the OAuth flow uses."""

    def __init__(
        self,
        *,
        server_url: str = "https://mcp.test",
        resource: str | None = None,
        authorization_server: str = "https://auth.test",
        refresh_error_status: int | None = None,
    ) -> None:
        self.server_url = server_url
        self.resource = resource or server_url
        self.authorization_server = authorization_server
        self.refresh_error_status = refresh_error_status
        self.registrations: list[dict[str, object]] = []
        self.token_requests: list[dict[str, str]] = []
        self.issued_access_tokens: list[str] = []
        self.rejected_states: list[str] = []
        self.requests: list[str] = []
        self.next_access_token = "access-token-1"
        self.next_refresh_token = "refresh-token-1"
        self.expected_code_verifier: str | None = None
        self.last_code_challenge: str | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        parsed = urlparse(str(request.url))
        path = parsed.path
        if path == "/.well-known/oauth-protected-resource":
            return httpx.Response(
                200,
                json={
                    "resource": self.resource,
                    "authorization_servers": [self.authorization_server],
                },
                request=request,
            )
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": self.authorization_server,
                    "authorization_endpoint": f"{self.authorization_server}/authorize",
                    "token_endpoint": f"{self.authorization_server}/token",
                    "registration_endpoint": f"{self.authorization_server}/register",
                    "code_challenge_methods_supported": ["S256"],
                    "scopes_supported": ["mcp:tools"],
                },
                request=request,
            )
        if path == "/register" and request.method == "POST":
            payload = json.loads(request.content)
            self.registrations.append(payload)
            return httpx.Response(
                200,
                json={"client_id": "fake-client", "client_secret": None},
                request=request,
            )
        if path == "/token" and request.method == "POST":
            form = {
                key: value[0]
                for key, value in parse_qs(request.content.decode("utf-8")).items()
            }
            self.token_requests.append(form)
            grant = form.get("grant_type")
            if grant == "refresh_token" and self.refresh_error_status is not None:
                return httpx.Response(
                    self.refresh_error_status,
                    json={"error": "invalid_grant"},
                    request=request,
                )
            access = self.next_access_token
            self.issued_access_tokens.append(access)
            self.next_access_token = (
                f"access-token-{len(self.issued_access_tokens) + 1}"
            )
            return httpx.Response(
                200,
                json={
                    "access_token": access,
                    "refresh_token": self.next_refresh_token,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "scope": form.get("scope", ""),
                },
                request=request,
            )
        return httpx.Response(404, json={"error": "not found"}, request=request)


def _monkey_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "zeta-home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    return home


def _write_token(name: str, home: Path, **overrides: object) -> MCPOAuthToken:
    token = MCPOAuthToken(
        access_token=overrides.get("access_token", "access-token-1"),
        refresh_token=overrides.get("refresh_token", "refresh-token-1"),
        expires_at=overrides.get("expires_at"),
        token_type=overrides.get("token_type", "Bearer"),
        scope=overrides.get("scope", "mcp:tools"),
        authorization_server=overrides.get("authorization_server", "https://auth.test"),
        token_endpoint=overrides.get("token_endpoint", "https://auth.test/token"),
        authorization_endpoint=overrides.get(
            "authorization_endpoint", "https://auth.test/authorize"
        ),
        client_id=overrides.get("client_id", "fake-client"),
        client_secret=overrides.get("client_secret"),
        redirect_uri=overrides.get("redirect_uri", "http://127.0.0.1:0/callback"),
        resource=overrides.get("resource", "https://mcp.test/resource"),
        refresh_error=overrides.get("refresh_error"),
    )
    path = token_store_path(name, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(token)
    values.pop("client_secret", None)
    path.write_text(json.dumps(values))
    path.chmod(0o600)
    return token


def _write_legacy_token(name: str, home: Path, *, client_secret: str) -> MCPOAuthToken:
    token = MCPOAuthToken(
        access_token="stale-token",
        refresh_token="refresh-token-1",
        expires_at=None,
        token_type="Bearer",
        scope="mcp:tools",
        authorization_server="https://auth.test",
        token_endpoint="https://auth.test/token",
        authorization_endpoint="https://auth.test/authorize",
        client_id="registered-client",
        client_secret=client_secret,
        redirect_uri="http://127.0.0.1:8000/callback",
        resource="https://mcp.test/rpc",
    )
    path = token_store_path(name, str(home))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(token)))
    return token


def test_pkce_generates_verifier_and_challenge() -> None:
    verifier, challenge = generate_pkce()
    assert len(verifier) >= 43
    assert challenge != verifier
    other_verifier, other_challenge = generate_pkce()
    assert verifier != other_verifier
    assert challenge != other_challenge


@pytest.mark.asyncio
async def test_token_store_round_trips_with_secure_permissions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = _write_token("srv", home)
    store = MCPCredentialStore(home=home)
    saved = await store.save_login("srv", token)
    loaded = await store.load("srv")
    assert loaded == saved
    file_mode = token_store_path("srv", str(home)).stat().st_mode
    assert stat.S_IMODE(file_mode) == 0o600


@pytest.mark.asyncio
async def test_new_token_bundle_does_not_persist_client_secret(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = _write_token("srv", home, client_secret="configured-secret")
    store = MCPCredentialStore(home=home)
    await store.save_login("srv", token)

    payload = json.loads(token_store_path("srv", str(home)).read_text())
    assert "client_secret" not in payload["token"]
    loaded = await store.load("srv")
    assert loaded.token is not None
    assert loaded.token.client_secret is None


@pytest.mark.asyncio
async def test_delete_token_commits_tombstone(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_token("srv", home)
    store = MCPCredentialStore(home=home)
    deleted = await store.delete("srv")
    assert await store.load("srv") == deleted
    assert deleted.generation == 1
    assert deleted.token is None


def test_token_state_reflects_expiry_and_refresh_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fresh = _write_token("fresh", home, expires_at=1e12)
    expired = _write_token("expired", home, expires_at=0.0)
    broken = _write_token("broken", home, expires_at=1e12, refresh_error="denied")
    assert token_state(fresh) == "authorized"
    assert token_state(expired) == "expired"
    assert token_state(broken) == "refresh-failed"
    assert token_state(None) == "unauthorized"
    assert not token_is_expired(fresh)
    assert token_is_expired(expired)


def test_redact_token_masks_secrets(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = _write_token("srv", home, refresh_token="r", client_secret="s")
    redacted = redact_token(token)
    assert redacted.access_token == "<redacted>"
    assert redacted.refresh_token == "<redacted>"
    assert redacted.client_secret == "<redacted>"
    assert redacted.token_endpoint == token.token_endpoint


@pytest.mark.asyncio
async def test_full_browser_flow_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))

    async def _do_redirect(url: str) -> None:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        state = params["state"][0]
        redirect_uri = params["redirect_uri"][0]
        assert params["code_challenge_method"][0] == "S256"
        assert params["resource"][0] == auth.resource
        auth.last_code_challenge = params["code_challenge"][0]
        await _fire_redirect(f"{redirect_uri}?code=auth-code&state={state}")

    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()

    def open_browser(url: str) -> None:
        assert threading.get_ident() != loop_thread_id
        asyncio.run_coroutine_threadsafe(_do_redirect(url), loop)

    token = await authorize(
        server_name="live",
        server_url=auth.server_url,
        http_client=http_client,
        browser_opener=open_browser,
    )
    await refresh_access_token(
        token_endpoint=token.token_endpoint,
        refresh_token=token.refresh_token or "",
        client_id=token.client_id,
        client_secret=token.client_secret,
        resource=token.resource,
        http_client=http_client,
    )
    await http_client.aclose()

    assert token.access_token == "access-token-1"
    assert token.refresh_token == "refresh-token-1"
    assert not token_store_path("live", home).exists()
    assert auth.registrations, "dynamic registration should have fired"
    token_form = auth.token_requests[0]
    assert token_form["grant_type"] == "authorization_code"
    assert token_form["code"] == "auth-code"
    assert token_form["resource"] == auth.resource
    assert token_form["code_verifier"], (
        "PKCE verifier must be present in the token request"
    )
    assert auth.token_requests[1]["grant_type"] == "refresh_token"
    assert auth.token_requests[1]["resource"] == auth.resource


@pytest.mark.asyncio
async def test_preregistered_client_uses_fixed_callback_scopes_and_extra_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        callback_port = available.getsockname()[1]

    async def redirect(url: str) -> None:
        params = parse_qs(urlparse(url).query)
        assert params["client_id"] == ["registered-client"]
        assert params["scope"] == ["read calendar"]
        assert params["access_type"] == ["offline"]
        assert params["prompt"] == ["consent"]
        redirect_uri = params["redirect_uri"][0]
        assert urlparse(redirect_uri).port == callback_port
        await _fire_redirect(
            f"{redirect_uri}?code=registered-code&state={params['state'][0]}"
        )

    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()
    browser_threads: list[int] = []

    def open_browser(url: str) -> None:
        browser_threads.append(threading.get_ident())
        asyncio.run_coroutine_threadsafe(redirect(url), loop)

    token = await authorize(
        server_name="live",
        server_url=auth.server_url,
        client_id="registered-client",
        client_secret="configured-secret",
        callback_port=callback_port,
        scopes=("read", "calendar"),
        authorization_params={"access_type": "offline", "prompt": "consent"},
        http_client=http_client,
        browser_opener=open_browser,
    )
    await http_client.aclose()

    assert browser_threads and all(
        thread != loop_thread_id for thread in browser_threads
    )
    assert not auth.registrations
    assert token.client_id == "registered-client"
    assert auth.token_requests == [
        {
            "grant_type": "authorization_code",
            "code": "registered-code",
            "redirect_uri": f"http://127.0.0.1:{callback_port}/callback",
            "client_id": "registered-client",
            "code_verifier": auth.token_requests[0]["code_verifier"],
            "resource": auth.resource,
            "client_secret": "configured-secret",
        }
    ]


@pytest.mark.parametrize(
    "reserved",
    [
        "response_type",
        "client_id",
        "redirect_uri",
        "state",
        "code_challenge",
        "code_challenge_method",
        "scope",
        "resource",
        "State",
        "REDIRECT_URI",
    ],
)
def test_config_rejects_reserved_authorization_params(
    tmp_path: Path, reserved: str
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "bad": {
                        "transport": "streamable-http",
                        "url": "https://mcp.test",
                        "auth": {
                            "type": "oauth",
                            "authorization_params": {reserved: "override"},
                        },
                    }
                }
            }
        )
    )

    config = load_mcp_config(path)

    assert "bad" in config.malformed_servers
    assert config.malformed_servers["bad"].malformed_reason == (
        f"auth.authorization_params must not override reserved parameters: {reserved}"
    )


@pytest.mark.asyncio
async def test_discovery_overrides_replace_default_probes() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        if url == "https://metadata.test/protected":
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.test/rpc",
                    "authorization_servers": ["https://login.test"],
                },
                request=request,
            )
        if url == "https://login.test/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://login.test",
                    "authorization_endpoint": "https://login.test/authorize",
                    "token_endpoint": "https://login.test/token",
                    "code_challenge_methods_supported": ["S256"],
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resource = await discover_protected_resource(
        "https://mcp.test/rpc",
        resource_metadata_url="https://metadata.test/protected",
        http_client=client,
    )
    metadata = await discover_auth_server("https://login.test", http_client=client)
    await client.aclose()

    assert resource.authorization_server == "https://login.test"
    assert metadata.issuer == "https://login.test"
    assert requests == [
        "https://metadata.test/protected",
        "https://login.test/.well-known/oauth-authorization-server",
    ]


@pytest.mark.asyncio
async def test_authorization_server_path_is_appended_to_well_known_url() -> None:
    requested_url = ""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_url
        requested_url = str(request.url)
        return httpx.Response(
            200,
            json={
                "issuer": "https://login.example/oauth",
                "authorization_endpoint": "https://login.example/oauth/authorize",
                "token_endpoint": "https://login.example/oauth/token",
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        metadata = await discover_auth_server(
            "https://login.example/oauth", http_client=client
        )
    finally:
        await client.aclose()

    assert requested_url == (
        "https://login.example/.well-known/oauth-authorization-server/oauth"
    )
    assert metadata.issuer == "https://login.example/oauth"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource", "server_url"),
    [
        ("https://host", "https://host/mcp"),
        ("https://host/", "https://host/mcp"),
        ("HTTPS://HOST", "https://host/mcp"),
        ("https://host:443", "https://host/mcp"),
        ("https://[2001:db8::1]:8443", "https://[2001:db8::1]:8443/mcp"),
        ("https://xn--fa-hia.de", "https://faß.de/mcp"),
        ("https://faß.de", "https://xn--fa-hia.de/mcp"),
        ("https://host./", "https://host./mcp"),
    ],
)
async def test_protected_resource_accepts_canonical_parent(
    resource: str, server_url: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resource": resource}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        metadata = await discover_protected_resource(server_url, http_client=client)

    assert metadata.resource == resource


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_url",
    [
        "https://host/a/../other",
        "https://host/%2e%2e/other",
        "https://host/a%2fb",
        "https://host/a//b",
        "https://host/a\\b",
    ],
)
async def test_ambiguous_server_url_keeps_exact_resource_match(
    server_url: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resource": server_url}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        metadata = await discover_protected_resource(server_url, http_client=client)

    assert metadata.resource == server_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource", "server_url"),
    [
        ("https://fass.de", "https://faß.de/mcp"),
        ("https://faß.de", "https://fass.de/mcp"),
        ("https://other", "https://host/mcp"),
        ("http://host", "https://host/mcp"),
        ("https://host:8443", "https://host/mcp"),
        ("https://host/other", "https://host/mcp"),
        ("https://host/m", "https://host/mcp"),
        ("https://host/a/../other", "https://host/a/../other/mcp"),
        ("https://host/a", "https://host/a/../other"),
        ("https://host/%2e%2e", "https://host/mcp"),
        ("https://host", "https://host/%2E%2E/mcp"),
        ("https://host/a%2fb", "https://host/a%2fb/mcp"),
        ("https://host", "https://host/a%5Cb/mcp"),
        ("https://host/a\\b", "https://host/a\\b/mcp"),
        ("https://host", "https://host/a\\b/mcp"),
        ("https://host/a//b", "https://host/a//b/mcp"),
        ("https://host", "https://host/a//b/mcp"),
        ("https://user@host", "https://host/mcp"),
        ("https://host?query=1", "https://host/mcp"),
        ("https://host#fragment", "https://host/mcp"),
        ("https://host.evil", "https://host/mcp"),
        ("https://evil.host", "https://host/mcp"),
        ("https://host.", "https://host/mcp"),
    ],
)
async def test_protected_resource_rejects_non_parent_or_ambiguous_url(
    resource: str, server_url: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resource": resource}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError):
            await discover_protected_resource(server_url, http_client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("path_result", ["404", "html", "missing-resource"])
async def test_protected_resource_path_probe_falls_back_to_root(
    path_result: str,
) -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/mcp"):
            if path_result == "404":
                return httpx.Response(404, request=request)
            if path_result == "html":
                return httpx.Response(200, text="<html>", request=request)
            return httpx.Response(200, json={}, request=request)
        return httpx.Response(200, json={"resource": "https://host"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        metadata = await discover_protected_resource(
            "https://host/mcp", http_client=client
        )

    assert metadata.resource == "https://host"
    assert requests == [
        "https://host/.well-known/oauth-protected-resource/mcp",
        "https://host/.well-known/oauth-protected-resource",
    ]


@pytest.mark.asyncio
async def test_protected_resource_root_redirect_is_unavailable_and_not_followed() -> (
    None
):
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/mcp"):
            return httpx.Response(404, request=request)
        return httpx.Response(
            302, headers={"location": "https://other.test/metadata"}, request=request
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        metadata = await discover_protected_resource(
            "https://host/mcp", http_client=client
        )

    assert metadata == ProtectedResourceMetadata("https://host/mcp", None)
    assert requests == [
        "https://host/.well-known/oauth-protected-resource/mcp",
        "https://host/.well-known/oauth-protected-resource",
    ]


@pytest.mark.asyncio
async def test_invalid_path_resource_is_strict_and_does_not_fall_back() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, json={"resource": "https://other"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError, match="resource does not match"):
            await discover_protected_resource("https://host/mcp", http_client=client)

    assert requests == ["https://host/.well-known/oauth-protected-resource/mcp"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(302), "returned HTTP 302"),
        (httpx.Response(200, text="html"), "was not JSON"),
        (httpx.Response(200, json=[]), "was not an object"),
        (httpx.Response(200, json={}), "missing resource"),
    ],
)
async def test_configured_resource_metadata_url_remains_strict(
    response: httpx.Response, message: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError, match=message):
            await discover_protected_resource(
                "https://host/mcp",
                resource_metadata_url="https://metadata.test/resource",
                http_client=client,
            )


@pytest.mark.asyncio
async def test_default_protected_resource_probe_keeps_invalid_response_fallback() -> (
    None
):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not JSON", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        metadata = await discover_protected_resource(
            "https://mcp.test/rpc", http_client=client
        )
    finally:
        await client.aclose()

    assert metadata.resource == "https://mcp.test/rpc"
    assert metadata.authorization_server is None


@pytest.mark.asyncio
@pytest.mark.parametrize("inconsistency", ["resource", "issuer", "endpoint"])
async def test_discovery_rejects_inconsistent_override_metadata(
    inconsistency: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.test":
            resource = (
                "https://other.test"
                if inconsistency == "resource"
                else "https://mcp.test/rpc"
            )
            return httpx.Response(
                200,
                json={"resource": resource},
                request=request,
            )
        issuer = (
            "https://other.test" if inconsistency == "issuer" else "https://login.test"
        )
        endpoint = (
            "http://remote.test/token"
            if inconsistency == "endpoint"
            else "https://login.test/token"
        )
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": "https://login.test/authorize",
                "token_endpoint": endpoint,
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        if inconsistency == "resource":
            with pytest.raises(MCPOAuthError, match="resource does not match"):
                await discover_protected_resource(
                    "https://mcp.test/rpc",
                    resource_metadata_url="https://metadata.test/protected",
                    http_client=client,
                )
        else:
            expected = (
                "issuer does not match"
                if inconsistency == "issuer"
                else "must use https"
            )
            with pytest.raises(MCPOAuthError, match=expected):
                await discover_auth_server("https://login.test", http_client=client)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "unsafe_url"),
    [
        ("resource", "https://mcp.test/\x1b[2J"),
        ("authorization_server", "https://login.test/\u0085bad"),
        ("issuer", "https://login.test/ bad"),
        ("authorization_endpoint", "https://login.test/\x1b[2Jauthorize"),
        ("token_endpoint", "https://login.test/\tdtoken"),
        ("registration_endpoint", "https://login.test/\ndregister"),
    ],
)
async def test_discovery_rejects_unsafe_metadata_urls(
    field: str,
    unsafe_url: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resource_url = "https://mcp.test/rpc"
    authorization_server_url = "https://login.test"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.test":
            metadata: dict[str, object] = {
                "resource": resource_url,
                "authorization_servers": [authorization_server_url],
            }
            if field == "resource":
                metadata["resource"] = unsafe_url
            elif field == "authorization_server":
                metadata["authorization_servers"] = [unsafe_url]
            return httpx.Response(200, json=metadata, request=request)
        metadata = {
            "issuer": authorization_server_url,
            "authorization_endpoint": "https://login.test/authorize",
            "token_endpoint": "https://login.test/token",
            "registration_endpoint": "https://login.test/register",
        }
        metadata[field] = unsafe_url
        return httpx.Response(200, json=metadata, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError) as exc_info:
            if field in {"resource", "authorization_server"}:
                await discover_protected_resource(
                    resource_url,
                    resource_metadata_url="https://metadata.test/protected",
                    http_client=client,
                )
            else:
                await discover_auth_server(authorization_server_url, http_client=client)

    label = "authorization server" if field == "authorization_server" else field
    assert str(exc_info.value) == (
        f"MCP OAuth: {label} must not contain control or whitespace characters"
    )
    assert unsafe_url not in capsys.readouterr().err
    assert unsafe_url not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "resource",
        "authorization_server",
        "issuer",
        "authorization_endpoint",
        "token_endpoint",
        "registration_endpoint",
    ],
)
async def test_discovery_rejects_oversized_metadata_urls(
    field: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resource_url = "https://mcp.test/rpc"
    authorization_server_url = "https://login.test"
    oversized = "https://login.test/" + "x" * 2_100

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.test":
            metadata: dict[str, object] = {
                "resource": resource_url,
                "authorization_servers": [authorization_server_url],
            }
            if field == "resource":
                metadata["resource"] = oversized
            elif field == "authorization_server":
                metadata["authorization_servers"] = [oversized]
            return httpx.Response(200, json=metadata, request=request)
        metadata = {
            "issuer": authorization_server_url,
            "authorization_endpoint": "https://login.test/authorize",
            "token_endpoint": "https://login.test/token",
            "registration_endpoint": "https://login.test/register",
        }
        metadata[field] = oversized
        return httpx.Response(200, json=metadata, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError) as exc_info:
            if field in {"resource", "authorization_server"}:
                await discover_protected_resource(
                    resource_url,
                    resource_metadata_url="https://metadata.test/protected",
                    http_client=client,
                )
            else:
                await discover_auth_server(authorization_server_url, http_client=client)

    label = "authorization server" if field == "authorization_server" else field
    assert str(exc_info.value) == f"MCP OAuth: {label} must be at most 2048 characters"
    assert oversized not in capsys.readouterr().err
    assert oversized not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["resource", "issuer"])
async def test_metadata_mismatch_error_bounds_server_value(field: str) -> None:
    server_value = "https://other.test/" + "x" * 1_024
    expected_value = server_value[:256]

    async def handler(request: httpx.Request) -> httpx.Response:
        if field == "resource":
            return httpx.Response(
                200,
                json={"resource": server_value},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "issuer": server_value,
                "authorization_endpoint": "https://login.test/authorize",
                "token_endpoint": "https://login.test/token",
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthError) as exc_info:
            if field == "resource":
                await discover_protected_resource(
                    "https://mcp.test/rpc",
                    resource_metadata_url="https://metadata.test/protected",
                    http_client=client,
                )
            else:
                await discover_auth_server("https://login.test", http_client=client)

    expected = (
        "MCP OAuth: protected-resource metadata resource does not match "
        f"the MCP server URL ({expected_value!r} != 'https://mcp.test/rpc')"
        if field == "resource"
        else "MCP OAuth: authorization-server metadata issuer does not match "
        f"the authorization server URL ({expected_value!r} != 'https://login.test')"
    )
    assert str(exc_info.value) == expected


@pytest.mark.parametrize("reserved", ["state", "State", "REDIRECT_URI"])
def test_authorization_url_defensively_rejects_reserved_params(
    reserved: str,
) -> None:
    metadata = AuthServerMetadata(
        issuer="https://auth.test",
        authorization_endpoint="https://auth.test/authorize",
        token_endpoint="https://auth.test/token",
        registration_endpoint=None,
        scopes_supported=(),
    )
    with pytest.raises(MCPOAuthError) as exc_info:
        build_authorization_url(
            metadata,
            client_id="client",
            redirect_uri="http://127.0.0.1:8888/callback",
            state="expected",
            code_challenge="challenge",
            resource="https://mcp.test",
            authorization_params={reserved: "attacker"},
        )
    assert str(exc_info.value) == (
        "MCP OAuth: authorization parameters cannot override reserved parameters: "
        f"{reserved}"
    )


@pytest.mark.asyncio
async def test_authorize_rejects_mismatched_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))

    async def _do_redirect(url: str) -> None:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        redirect_uri = params["redirect_uri"][0]
        await _fire_redirect(f"{redirect_uri}?code=auth-code&state=WRONG")

    loop = asyncio.get_running_loop()

    def open_browser(url: str) -> None:
        asyncio.run_coroutine_threadsafe(_do_redirect(url), loop)

    with pytest.raises(MCPOAuthStateError):
        await authorize(
            server_name="live",
            server_url=auth.server_url,
            http_client=http_client,
            browser_opener=open_browser,
        )
    assert not token_store_path("live", home).exists()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_authorize_surfaces_error_from_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))

    async def _do_redirect(url: str) -> None:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        redirect_uri = params["redirect_uri"][0]
        state = params["state"][0]
        await _fire_redirect(f"{redirect_uri}?error=access_denied&state={state}")

    loop = asyncio.get_running_loop()

    def open_browser(url: str) -> None:
        asyncio.run_coroutine_threadsafe(_do_redirect(url), loop)

    with pytest.raises(MCPOAuthError, match="access_denied"):
        await authorize(
            server_name="live",
            server_url=auth.server_url,
            http_client=http_client,
            browser_opener=open_browser,
        )
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_type", "message"),
    [
        ("valid", None, None),
        ("wrong-state", MCPOAuthStateError, "state did not match"),
        ("wrong-redirect", MCPOAuthError, "does not match the redirect URI"),
        ("duplicate-code", MCPOAuthError, "exactly one code parameter"),
        ("error", MCPOAuthError, "access_denied: user declined"),
    ],
)
async def test_headless_callback_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    error_type: type[Exception] | None,
    message: str | None,
) -> None:
    from zeta.mcp import oauth as oauth_module

    _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))
    authorization_url = ""

    def capture_print(value: str, *, file: object) -> None:
        nonlocal authorization_url
        del file
        authorization_url = value.split("\n", 1)[1]

    def callback_input(_prompt: str) -> str:
        params = parse_qs(urlparse(authorization_url).query)
        redirect_uri = params["redirect_uri"][0]
        state = params["state"][0]
        if response == "wrong-state":
            state = "wrong"
        if response == "wrong-redirect":
            redirect_uri = "http://127.0.0.1:9/wrong"
        if response == "error":
            return (
                f"{redirect_uri}?error=access_denied&"
                f"error_description=user+declined&state={state}"
            )
        code = (
            "headless-code&code=second-code"
            if response == "duplicate-code"
            else "headless-code"
        )
        return f"{redirect_uri}?code={code}&state={state}"

    monkeypatch.setattr(oauth_module, "print", capture_print, raising=False)
    monkeypatch.setattr("builtins.input", callback_input)

    if error_type is None:
        token = await authorize(
            server_name="live",
            server_url=auth.server_url,
            client_id="registered-client",
            no_browser=True,
            http_client=http_client,
        )
        assert token.access_token == "access-token-1"
        assert [request["code"] for request in auth.token_requests] == ["headless-code"]
    else:
        with pytest.raises(error_type, match=message):
            await authorize(
                server_name="live",
                server_url=auth.server_url,
                client_id="registered-client",
                no_browser=True,
                http_client=http_client,
            )
        assert auth.token_requests == []
    await http_client.aclose()


def test_callback_error_is_safe_for_terminal() -> None:
    from zeta.mcp import oauth as oauth_module

    unsafe_error = "bad\x1b[2J\r\n" + "x" * 300
    unsafe_description = "user\x00denied\x85" + "y" * 300
    unsafe_error_uri = "https://attacker.test/\x1b[2J"
    callback_url = "http://127.0.0.1:8888/callback?" + urlencode(
        {
            "error": unsafe_error,
            "error_description": unsafe_description,
            "error_uri": unsafe_error_uri,
            "state": "expected",
        }
    )

    with pytest.raises(MCPOAuthError) as exc_info:
        oauth_module._validate_callback_url(
            callback_url,
            redirect_uri="http://127.0.0.1:8888/callback",
            expected_state="expected",
        )

    safe_error = ("bad[2J" + "x" * 300)[:256]
    safe_description = ("userdenied" + "y" * 300)[:256]
    assert str(exc_info.value) == (
        f"MCP OAuth: authorization denied: {safe_error}: {safe_description}"
    )
    assert unsafe_error_uri not in str(exc_info.value)
    assert not any(
        ord(character) < 32 or 127 <= ord(character) <= 159
        for character in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_http_client_refreshes_on_401_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    initial = _write_token("live", home, access_token="stale-token")

    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        url = str(request.url)
        if url.endswith("/token"):
            form = {
                key: value[0]
                for key, value in parse_qs(request.content.decode("utf-8")).items()
            }
            assert form["grant_type"] == "refresh_token"
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "refresh-token-2",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
                request=request,
            )
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        if body["method"] == "tools/call":
            call_count += 1
            auth_header = request.headers.get("authorization", "")
            if call_count == 1:
                assert auth_header == "Bearer stale-token"
                return httpx.Response(401, text="expired", request=request)
            assert auth_header == "Bearer fresh-token"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": False,
                    },
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport_client, home=str(home))
    await client.connect()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["content"][0]["text"] == "ok"
    assert call_count == 2
    persisted = (await MCPCredentialStore(home=home).load("live")).token
    assert persisted is not None and persisted.access_token == "fresh-token"
    assert persisted.refresh_token == "refresh-token-2"
    assert persisted.refresh_error is None
    del initial
    await client.close()


@pytest.mark.asyncio
async def test_refresh_uses_client_secret_resolved_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    monkeypatch.setenv("MCP_CLIENT_SECRET", "environment-secret")
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "live": {
                        "transport": "streamable-http",
                        "url": "https://mcp.test/rpc",
                        "auth": {
                            "type": "oauth",
                            "client_id": "registered-client",
                            "client_secret": "${MCP_CLIENT_SECRET}",
                        },
                    }
                }
            }
        )
    )
    config = load_mcp_config(config_path).configured_servers["live"]
    _write_token(
        "live",
        home,
        access_token="stale-token",
        client_id="registered-client",
        resource="https://mcp.test/rpc",
    )
    requests: list[dict[str, list[str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(
            200,
            json={"access_token": "fresh-token", "token_type": "Bearer"},
            request=request,
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    current = await MCPCredentialStore(home=home).load("live")
    assert current.token is not None
    assert await client._refresh_token_once(MCPHTTPError(401, "expired"), current)

    assert requests[0]["client_secret"] == ["environment-secret"]
    assert (
        "client_secret"
        not in json.loads(token_store_path("live", str(home)).read_text())["token"]
    )
    await client.close()


@pytest.mark.asyncio
async def test_legacy_stored_secret_refreshes_and_rewrite_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    token = _write_legacy_token("live", home, client_secret="legacy-stored-secret")
    requests: list[dict[str, list[str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(
            200,
            json={"access_token": "fresh-token", "token_type": "Bearer"},
            request=request,
        )

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    current = await MCPCredentialStore(home=home).load("live")
    assert current.token == token
    assert await client._refresh_token_once(MCPHTTPError(401, "expired"), current)

    assert requests[0]["client_secret"] == ["legacy-stored-secret"]
    assert (
        "client_secret"
        not in json.loads(token_store_path("live", str(home)).read_text())["token"]
    )
    await client.close()


@pytest.mark.asyncio
async def test_http_client_reports_refresh_failure_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, access_token="stale-token")

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/token"):
            return httpx.Response(400, json={"error": "invalid_grant"}, request=request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        return httpx.Response(401, text="expired", request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport_client, home=str(home))
    await client.connect()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "MCP OAuth token refresh failed" in text
    assert "run /mcp auth live" in text
    persisted = (await MCPCredentialStore(home=home).load("live")).token
    assert persisted is not None
    assert persisted.refresh_error is None
    await client.close()


@pytest.mark.asyncio
async def test_missing_refresh_token_surfaces_auth_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, access_token="stale-token", refresh_token=None)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        return httpx.Response(401, text="expired", request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport_client, home=str(home))
    await client.connect()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["isError"] is True
    assert "run /mcp auth live" in result["content"][0]["text"]
    await client.close()


@pytest.mark.asyncio
async def test_config_rejects_oauth_on_stdio(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "bad": {
                        "transport": "stdio",
                        "command": "server",
                        "auth": {"type": "oauth"},
                    }
                }
            }
        )
    )
    config = load_mcp_config(path)
    assert "bad" in config.malformed_servers
    assert "streamable-http" in config.malformed_servers["bad"].malformed_reason


@pytest.mark.asyncio
async def test_config_rejects_oauth_with_inline_token(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "bad": {
                        "transport": "streamable-http",
                        "url": "https://mcp.test",
                        "auth": {"type": "oauth", "token": "leaked"},
                    }
                }
            }
        )
    )
    config = load_mcp_config(path)
    assert "bad" in config.malformed_servers


def test_parse_add_command_supports_oauth_flag() -> None:
    config = parse_add_command(["live", "--http", "https://mcp.test", "--oauth"])
    assert config.auth_type == "oauth"
    assert config.url == "https://mcp.test"


@pytest.mark.asyncio
async def test_resources_list_and_attach_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_mcp import _FakeClient

    class _ResourceClient(_FakeClient):
        async def list_resources(self) -> list[MCPResource]:
            return [
                MCPResource(uri="mcp://doc/1", name="doc1", mime_type="text/plain"),
                MCPResource(uri="mcp://doc/2", name="doc2"),
            ]

        async def read_resource(self, uri: str) -> str:
            return f"payload for {uri}"

    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module

    def build_client(config: MCPServerConfig) -> _ResourceClient:
        return _ResourceClient(config)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty()
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add live --http https://mcp.example")

    listing = await loop.slash_mcp("resources live")
    assert "mcp://doc/1" in listing
    assert "mime=text/plain" in listing

    attachment = await loop.slash_mcp("resources live mcp://doc/1")
    from zeta.model_input import ModelInputEnvelope

    assert isinstance(attachment, ModelInputEnvelope)
    assert "[mcp-resource: live:mcp://doc/1" in attachment.text
    assert "payload for mcp://doc/1" in attachment.text
    await loop.close()


@pytest.mark.asyncio
async def test_mcp_large_text_resource_spills(tmp_path: Path) -> None:
    payload = "large text resource\n" * 20_000

    class _BulkyClient:
        config = MCPServerConfig("srv", "streamable-http", url="https://mcp.test")

        async def read_resource(self, uri: str) -> str:
            del uri
            return payload

        async def list_resources(self) -> list[MCPResource]:
            return []

    spill = SpillStore()
    try:
        attachment = await fetch_resource(
            _BulkyClient(),  # type: ignore[arg-type]
            server="srv",
            uri="mcp://big",
            spill_store=spill,
        )
        assert attachment.text != payload
        assert attachment.spill_paths
        assert attachment.spill_paths[0].read_text() == payload
        assert str(attachment.spill_paths[0]) in attachment.labeled_text
    finally:
        spill.close()


@pytest.mark.asyncio
async def test_mcp_large_blob_resource_spills(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 2_000

    class _BlobClient:
        config = MCPServerConfig("srv", "streamable-http", url="https://mcp.test")

        async def read_resource(self, uri: str) -> tuple[MCPResourceContent, ...]:
            del uri
            return (MCPResourceContent(payload, "application/octet-stream"),)

        async def list_resources(self) -> list[MCPResource]:
            return []

    spill = SpillStore()
    try:
        attachment = await fetch_resource(
            _BlobClient(),  # type: ignore[arg-type]
            server="srv",
            uri="mcp://blob",
            spill_store=spill,
        )
        assert attachment.spill_paths[0].read_bytes() == payload
        assert "application/octet-stream" in attachment.labeled_text
        assert str(attachment.spill_paths[0]) in attachment.labeled_text
    finally:
        spill.close()


@pytest.mark.asyncio
async def test_list_resources_wraps_server_errors() -> None:
    class _BrokenClient:
        config = MCPServerConfig("srv", "streamable-http", url="https://mcp.test")

        async def list_resources(self) -> list[MCPResource]:
            raise RuntimeError("no resources capability")

    client = _BrokenClient()
    with pytest.raises(MCPResourceError, match="no resources capability"):
        await list_resources(client, server="srv")  # type: ignore[arg-type]


def test_format_resource_list_renders_metadata() -> None:
    rendered = format_resource_list(
        "srv",
        [
            MCPResource(
                uri="mcp://a",
                name="a",
                description="alpha",
                mime_type="text/markdown",
            )
        ],
    )
    assert "srv: 1 resources" in rendered
    assert "mcp://a" in rendered
    assert "mime=text/markdown" in rendered
    assert "description=alpha" in rendered


@pytest.mark.asyncio
async def test_tokens_do_not_appear_in_conversation_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    token = _write_token(
        "live",
        home,
        access_token="SECRET-ACCESS-TOKEN-DO-NOT-LOG",
        refresh_token="SECRET-REFRESH-TOKEN-DO-NOT-LOG",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                    },
                },
                request=request,
            )
        if body["method"] == "tools/list":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": []}},
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport_client, home=str(home))
    await client.connect()
    await client.list_tools()
    await client.close()
    await transport_client.aclose()

    project = tmp_path / "proj"
    project.mkdir()
    store = ConversationStore(project)
    backend = FakeBackend([ScriptedTurn([TextContent("hello")])])
    loop = AgentLoop(
        backend, store, skip_mcp_mount=True, skill_catalog=SkillCatalog.empty()
    )
    events = [
        event
        async for event in loop.run_turn("please answer", origin=MessageOrigin.USER)
    ]
    await loop.close()

    haystack = ""
    for session_file in store.session_dir.rglob("*.jsonl"):
        haystack += session_file.read_text()
    haystack += json.dumps([event.type.value for event in events])
    assert token.access_token not in haystack
    assert token.refresh_token not in haystack


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["status", "auth live", "logout live"])
async def test_slash_mcp_constructs_management_service_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "live": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example",
                        "auth": {"type": "oauth"},
                    }
                }
            }
        )
    )
    token = _write_token("live", home, expires_at=1e12)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module
    from zeta.mcp import oauth as oauth_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    async def fake_authorize(**kwargs: object) -> MCPOAuthToken:
        return token

    # test-contract: seam avoid network while exercising the public slash command
    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    # test-contract: seam avoid browser consent during the public auth command
    monkeypatch.setattr(oauth_module, "authorize", fake_authorize)
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty()
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    loop_thread_id = threading.get_ident()
    resolve_threads: list[int] = []
    original_resolve = Path.resolve

    def checked_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == project:
            resolve_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread_id
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", checked_resolve)
    try:
        await loop.slash_mcp(command)
        assert resolve_threads
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_status_shows_oauth_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, expires_at=1e12)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    # test-contract: seam avoid network while exercising the public slash command
    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty()
    )
    loop.set_mcp_scope(home=home, project_dir=project)

    await loop.slash_mcp("add live --http https://mcp.example --oauth")
    status = await loop.slash_mcp("")
    assert "auth: oauth (authorized)" in status
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_auth_runs_flow_and_persists_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module
    from zeta.mcp import oauth as oauth_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)

    class _RecordingAuthorize:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def __call__(self, **kwargs: object) -> MCPOAuthToken:
            self.calls.append(kwargs)
            return _write_token(str(kwargs["server_name"]), home, expires_at=1e12)

    recorder = _RecordingAuthorize()
    monkeypatch.setattr(oauth_module, "authorize", recorder)
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "live": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example",
                        "auth": {
                            "type": "oauth",
                            "authorization_server_url": "https://login.example",
                            "resource_metadata_url": "https://mcp.example/metadata",
                            "authorization_params": {"access_type": "offline"},
                        },
                    }
                }
            }
        )
    )

    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(project),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=home, project_dir=project)

    output = await loop.slash_mcp("auth live")
    assert "auth: oauth (authorized)" in output
    assert recorder.calls, "authorize must be invoked"
    assert recorder.calls[0] == {
        "server_name": "live",
        "server_url": "https://mcp.example",
        "client_id": None,
        "client_secret": None,
        "callback_port": 0,
        "scopes": None,
        "authorization_server_url": "https://login.example",
        "resource_metadata_url": "https://mcp.example/metadata",
        "authorization_params": {"access_type": "offline"},
        "no_browser": False,
    }
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_auth_reports_flow_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module
    from zeta.mcp import oauth as oauth_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)

    async def boom(**kwargs: object) -> MCPOAuthToken:
        raise MCPOAuthError("consent denied")

    monkeypatch.setattr(oauth_module, "authorize", boom)
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty()
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add live --http https://mcp.example --oauth")

    output = await loop.slash_mcp("auth live")
    assert "mcp error: consent denied" in output
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_auth_rejects_stdio_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    from zeta.mcp import mount as mount_module

    class _NoopClient:
        def __init__(self, config: MCPServerConfig) -> None:
            self.config = config
            self.protocol_version = "2025-06-18"
            self.capabilities = {"tools": {}}

        async def connect(self) -> None: ...
        async def list_tools(self) -> list:
            return []

        # test-contract: seam minimum MCP client protocol fake
        async def close(self) -> None: ...

    # test-contract: seam inject the minimum MCP client protocol fake
    monkeypatch.setattr(
        mount_module, "_build_client", lambda config: _NoopClient(config)
    )
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty()
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add local --stdio command")

    output = await loop.slash_mcp("auth local")
    assert "requires an HTTP MCP server" in output
    await loop.close()


@pytest.mark.asyncio
async def test_mcp_http_large_response_spills_not_errors() -> None:
    payload_text = "x" * (MAX_RESPONSE_BYTES * 2)
    chunks_yielded = 0
    spill = SpillStore()

    async def payload(request_id: int):
        nonlocal chunks_yielded
        encoded = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"contents": [{"text": payload_text}]},
            }
        ).encode()
        for start in range(0, len(encoded), 50_000):
            chunks_yielded += 1
            yield encoded[start : start + 50_000]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        if body["method"] == "resources/read":
            return httpx.Response(
                200,
                content=payload(body["id"]),
                headers={"content-type": "application/json"},
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig("live", "streamable-http", url="https://mcp.test/rpc")
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, spill_store=spill)
    try:
        await client.connect()
        contents = await client.read_resource("mcp://big")
        assert contents[0].data == payload_text
        assert chunks_yielded > MAX_RESPONSE_BYTES // 50_000
    finally:
        await client.close()
        spill.close()


@pytest.mark.asyncio
async def test_http_client_single_flights_concurrent_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent 401s must trigger exactly one refresh grant."""

    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, access_token="stale-token")

    refresh_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_count
        url = str(request.url)
        if url.endswith("/token"):
            refresh_count += 1
            await asyncio.sleep(0.05)
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "refresh-token-2",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
                request=request,
            )
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        if body["method"] == "tools/call":
            auth_header = request.headers.get("authorization", "")
            if auth_header == "Bearer stale-token":
                return httpx.Response(401, text="expired", request=request)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": False,
                    },
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    await client.connect()

    results = await asyncio.gather(
        client.call_tool("echo", {}, AbortSignal()),
        client.call_tool("echo", {}, AbortSignal()),
    )
    for result in results:
        assert result["content"][0]["text"] == "ok"
    assert refresh_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_http_get_notification_refreshes_on_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, access_token="stale-token")
    get_auth: list[str] = []
    refresh_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_count
        if str(request.url).endswith("/token"):
            refresh_count += 1
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "refresh-token-2",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
                request=request,
            )
        assert request.method == "GET"
        auth = request.headers.get("authorization", "")
        get_auth.append(auth)
        return httpx.Response(
            401 if auth == "Bearer stale-token" and len(get_auth) < 3 else 405,
            text="expired",
            request=request,
        )

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    client._session_id = "session-1"
    await client._listen_notifications()

    assert get_auth == ["Bearer stale-token", "Bearer fresh-token"]
    assert refresh_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_http_notification_refreshes_on_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_send_notification` must refresh on 401 and retry with the fresh token."""

    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("live", home, access_token="stale-token")

    notification_calls: list[str] = []
    refresh_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_count
        url = str(request.url)
        if url.endswith("/token"):
            refresh_count += 1
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "refresh-token-2",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
                request=request,
            )
        body = json.loads(request.content)
        if "id" not in body:
            auth = request.headers.get("authorization", "")
            notification_calls.append(auth)
            if auth == "Bearer stale-token":
                return httpx.Response(401, text="expired", request=request)
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig(
        "live",
        "streamable-http",
        url="https://mcp.test/rpc",
        auth_type="oauth",
    )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    await client.connect()
    assert refresh_count == 1
    assert notification_calls == ["Bearer stale-token", "Bearer fresh-token"]
    await client.close()


@pytest.mark.asyncio
async def test_credential_reads_run_off_loop_through_call_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    _write_token("live", home, access_token="ordered-token")
    loop_thread_id = threading.get_ident()
    credential_path = token_store_path("live", home)
    observed_threads: list[int] = []
    original_open = os.open

    def checked_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if str(path) == str(credential_path):
            observed_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread_id
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("zeta.mcp.oauth_store.os.open", checked_open)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": "ok"}]},
            },
            request=request,
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test", auth_type="oauth"
        ),
        client=transport,
        home=str(home),
    )
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["content"][0]["text"] == "ok"
    assert observed_threads
    await client.close()
    await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "reason", "text_fragment"),
    [
        ("malformed", "malformed", "is malformed"),
        ("oversized", "oversized", "is too large"),
        ("symlink", "unsafe_file", "is unsafe"),
    ],
)
async def test_call_tool_maps_unsafe_credential_records_without_http(
    tmp_path: Path,
    record: str,
    reason: str,
    text_fragment: str,
) -> None:
    home = tmp_path / "home"
    path = token_store_path("live", home)
    path.parent.mkdir(parents=True)
    if record == "malformed":
        path.write_text("{broken")
    elif record == "oversized":
        path.write_text(json.dumps({"access_token": "x" * (70 * 1024)}))
    else:
        target = tmp_path / "other-token"
        target.write_text("{}")
        path.symlink_to(target)
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(500, request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test", auth_type="oauth"
        ),
        client=transport,
        home=str(home),
    )
    result = await client.call_tool("echo", {}, AbortSignal())
    error = result["structuredContent"]["credential_store_error"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == error["text"]
    assert error == {
        "code": "mcp_credential_store",
        "reason": reason,
        "server": "live",
        "credential_file": str(path.absolute()),
        "operation": "load",
        "recovery": "move_aside_then_auth",
        "text": error["text"],
    }
    assert text_fragment in error["text"]
    assert attempts == []
    await client.close()
    await transport.aclose()


@pytest.mark.asyncio
async def test_idle_client_does_not_send_token_after_public_logout(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    token = _write_token("live", home, access_token="deleted-token")
    manager = MCPManagementService(home=home, project_dir=project)
    await manager.add("live", scope="user", url="https://mcp.test", oauth=True)
    attempts: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers.get("authorization"))
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": "unexpected"}]},
            },
            request=request,
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test", auth_type="oauth"
        ),
        client=transport,
        home=str(home),
    )
    outcome = manager.logout("live", scope="user")
    if inspect.isawaitable(outcome):
        await outcome
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["isError"] is True
    assert token.access_token not in result["content"][0]["text"]
    assert attempts == []
    await client.close()
    await transport.aclose()


@pytest.mark.asyncio
async def test_connect_rereads_credentials_for_notification_post_and_get(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_token("live", home, access_token="first-token")
    requests: list[tuple[str, str | None]] = []
    get_seen = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        requests.append((request.method, auth))
        if request.method == "GET":
            get_seen.set()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"",
                request=request,
            )
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            _write_token("live", home, access_token="second-token")
            return httpx.Response(
                200,
                headers={"mcp-session-id": "session"},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                request=request,
            )
        _write_token("live", home, access_token="third-token")
        return httpx.Response(202, request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test", auth_type="oauth"
        ),
        client=transport,
        home=str(home),
    )
    await client.connect()
    await asyncio.wait_for(get_seen.wait(), timeout=10)
    assert requests[:3] == [
        ("POST", "Bearer first-token"),
        ("POST", "Bearer second-token"),
        ("GET", "Bearer third-token"),
    ]
    await client.close()
    await transport.aclose()


@pytest.mark.asyncio
async def test_two_clients_refresh_with_generation_cas_and_fresh_retry(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_token("live", home, access_token="stale-token")
    exchange_count = 0
    both_exchanges = asyncio.Event()
    first_retry = asyncio.Event()
    tool_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exchange_count
        if request.url.path == "/token":
            exchange_count += 1
            index = exchange_count
            if index == 1:
                await asyncio.wait_for(both_exchanges.wait(), timeout=10)
                access = "winner-token"
            else:
                both_exchanges.set()
                await asyncio.wait_for(first_retry.wait(), timeout=10)
                access = "loser-token"
            return httpx.Response(
                200,
                json={"access_token": access, "token_type": "Bearer"},
                request=request,
            )
        auth = request.headers.get("authorization")
        tool_headers.append(auth)
        if auth == "Bearer stale-token":
            return httpx.Response(401, text="expired", request=request)
        if auth == "Bearer winner-token":
            first_retry.set()
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                },
                request=request,
            )
        return httpx.Response(500, request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = MCPServerConfig(
        "live", "streamable-http", url="https://mcp.test/rpc", auth_type="oauth"
    )
    clients = [
        StreamableHTTPMCPClient(config, client=transport, home=str(home))
        for _ in range(2)
    ]
    results = await asyncio.gather(
        *(client.call_tool("echo", {}, AbortSignal()) for client in clients)
    )
    assert [result["content"][0]["text"] for result in results] == ["ok", "ok"]
    assert tool_headers.count("Bearer winner-token") == 2
    assert "Bearer loser-token" not in tool_headers
    stored = await MCPCredentialStore(home=home).load("live")
    assert stored.generation == 1
    assert stored.token is not None and stored.token.access_token == "winner-token"
    for client in clients:
        await client.close()
    await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "reason"),
    [("candidate_oversized", "candidate_oversized"), ("durability", "durability")],
)
async def test_refresh_commit_failure_never_sends_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    reason: str,
) -> None:
    home = tmp_path / "home"
    _write_token("live", home, access_token="stale-token")
    candidate = (
        "x" * (70 * 1024) if failure == "candidate_oversized" else "candidate-token"
    )
    candidate_requests: list[str] = []
    original_fsync = os.fsync

    if failure == "durability":

        def fail_directory_fsync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("injected directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr("zeta.mcp.oauth_store.os.fsync", fail_directory_fsync)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(
                200,
                json={"access_token": candidate, "token_type": "Bearer"},
                request=request,
            )
        auth = request.headers.get("authorization")
        if auth == f"Bearer {candidate}":
            candidate_requests.append(auth)
        return httpx.Response(401, text="expired", request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test/rpc", auth_type="oauth"
        ),
        client=transport,
        home=str(home),
    )
    result = await client.call_tool("echo", {}, AbortSignal())
    error = result["structuredContent"]["credential_store_error"]
    assert error["reason"] == reason
    assert error["operation"] == "save_if_current"
    assert candidate_requests == []
    await client.close()
    await transport.aclose()


class _CredentialRaceStore(MCPCredentialStore):
    """Gate one worker transaction while public MCP operations keep running."""

    def __init__(
        self,
        *,
        home: Path,
        loop_thread_id: int,
        gate: str,
        load_number: int = 0,
    ) -> None:
        super().__init__(home=home)
        self.loop_thread_id = loop_thread_id
        self.gate = gate
        self.load_number = load_number
        self.loads = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block(self) -> None:
        assert threading.get_ident() != self.loop_thread_id
        self.entered.set()
        assert self.release.wait(timeout=5), "credential race gate timed out"

    def _load_transaction(self, name: str, operation: str) -> CredentialSnapshot:
        self.loads += 1
        selected = self.loads == self.load_number
        if self.gate == "before_load" and selected:
            self._block()
        snapshot = super()._load_transaction(name, operation)  # type: ignore[arg-type]
        if self.gate == "after_load" and selected:
            self._block()
        return snapshot

    def _save_if_current_transaction(
        self,
        name: str,
        expected_generation: int,
        replacement: MCPOAuthToken,
    ) -> SaveIfCurrentResult:
        result = super()._save_if_current_transaction(
            name, expected_generation, replacement
        )
        if self.gate == "after_save_if_current":
            self._block()
        return result

    def _save_login_transaction(
        self, name: str, token: MCPOAuthToken
    ) -> CredentialSnapshot:
        snapshot = super()._save_login_transaction(name, token)
        if self.gate == "after_save_login":
            self._block()
        return snapshot

    def _delete_transaction(self, name: str) -> CredentialSnapshot:
        snapshot = super()._delete_transaction(name)
        if self.gate == "after_delete":
            self._block()
        return snapshot


async def _wait_for_thread_gate(store: _CredentialRaceStore) -> None:
    entered = await asyncio.wait_for(
        asyncio.to_thread(store.entered.wait, 10), timeout=11
    )
    assert entered, "credential race gate was not reached"


async def _oauth_manager(home: Path, project: Path) -> MCPManagementService:
    project.mkdir(exist_ok=True)
    manager = MCPManagementService(home=home, project_dir=project)
    await manager.add("live", scope="user", url="https://mcp.test/rpc", oauth=True)
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["committed", "stale"])
@pytest.mark.parametrize("window", ["before_result_publication", "before_retry_read"])
async def test_logout_wins_every_refresh_result_to_retry_window(
    tmp_path: Path,
    disposition: str,
    window: str,
) -> None:
    home = tmp_path / "home"
    original = _write_token("live", home, access_token="stale-token")
    manager = await _oauth_manager(home, tmp_path / "project")
    race = _CredentialRaceProcess(
        home,
        gate=(
            "after_save_if_current"
            if window == "before_result_publication"
            else "before_load"
        ),
        load_number=3,
    )
    try:
        await race.request(401, "expired")
        token_request = await race.receive()
        assert token_request[:2] == ("request", "/token")
        if disposition == "stale":
            winner = replace(original, access_token="winner-token")
            await MCPCredentialStore(home=home).save_login("live", winner)
        race.connection.send(
            (200, {"access_token": "candidate-token", "token_type": "Bearer"})
        )
        await race.gate()
        await manager.logout("live", scope="user")
        race.release()
        result = await race.result()

        assert result["isError"] is True
        assert race.headers == ["Bearer stale-token"]
        authority = await MCPCredentialStore(home=home).load("live")
        assert authority.token is None
    finally:
        await race.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window", "replace_during_failure"),
    [
        ("before_failed_reload", False),
        ("after_failed_reload", True),
    ],
)
async def test_logout_wins_failed_refresh_reload_windows(
    tmp_path: Path,
    window: str,
    replace_during_failure: bool,
) -> None:
    home = tmp_path / "home"
    original = _write_token("live", home, access_token="stale-token")
    manager = await _oauth_manager(home, tmp_path / "project")
    race = _CredentialRaceProcess(
        home,
        gate="before_load" if window == "before_failed_reload" else "after_load",
        load_number=3,
    )
    try:
        await race.request(401, "expired")
        token_request = await race.receive()
        assert token_request[:2] == ("request", "/token")
        if replace_during_failure:
            changed = replace(original, access_token="changed-token")
            await MCPCredentialStore(home=home).save_login("live", changed)
        race.connection.send((400, {"error": "invalid_grant"}))
        await race.gate()
        await manager.logout("live", scope="user")
        race.release()
        result = await race.result()

        assert result["isError"] is True
        assert race.headers == ["Bearer stale-token"]
        authority = await MCPCredentialStore(home=home).load("live")
        assert authority.token is None
    finally:
        await race.close()


@pytest.mark.asyncio
async def test_failed_refresh_adopts_changed_durable_token(tmp_path: Path) -> None:
    home = tmp_path / "home"
    original = _write_token("live", home, access_token="stale-token")
    race = _CredentialRaceProcess(home, gate="none")
    try:
        await race.request(401, "expired")
        token_request = await race.receive()
        assert token_request[:2] == ("request", "/token")
        changed = replace(original, access_token="changed-token")
        await MCPCredentialStore(home=home).save_login("live", changed)
        race.connection.send((400, {"error": "invalid_grant"}))
        retry = await race.receive()
        assert retry[:3] == ("request", "/rpc", "Bearer changed-token")
        body = json.loads(retry[3])
        race.connection.send(
            (
                200,
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": "changed"}]},
                },
            )
        )
        result = await race.result()

        assert result["content"][0]["text"] == "changed"
        assert race.headers == ["Bearer stale-token"]
    finally:
        await race.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("last_commit", ["logout", "login"])
async def test_public_login_and_logout_follow_both_exclusive_lock_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_commit: str,
) -> None:
    home = tmp_path / "home"
    original = _write_token("live", home, access_token="old-token")
    manager = await _oauth_manager(home, tmp_path / "project")
    candidate = replace(original, access_token="new-login-token")

    async def fake_authorize(**_kwargs: object) -> MCPOAuthToken:
        return candidate

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    gate = "after_save_login" if last_commit == "logout" else "after_delete"
    store = _CredentialRaceStore(
        home=home,
        loop_thread_id=threading.get_ident(),
        gate=gate,
    )
    manager._credential_store = store

    first = (
        asyncio.create_task(manager.login("live", scope="user", no_browser=True))
        if last_commit == "logout"
        else asyncio.create_task(manager.logout("live", scope="user"))
    )
    await _wait_for_thread_gate(store)
    second = (
        asyncio.create_task(manager.logout("live", scope="user"))
        if last_commit == "logout"
        else asyncio.create_task(manager.login("live", scope="user", no_browser=True))
    )
    await asyncio.wait_for(second, timeout=10)
    store.release.set()
    await asyncio.wait_for(first, timeout=10)

    authority = await MCPCredentialStore(home=home).load("live")
    if last_commit == "logout":
        assert authority.token is None
    else:
        assert authority.token is not None
        assert authority.token.access_token == "new-login-token"


@pytest.mark.asyncio
async def test_colliding_server_names_keep_separate_credentials_and_logout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    manager = MCPManagementService(home=home, project_dir=tmp_path / "project")
    (tmp_path / "project").mkdir()
    names = ("team/server", "team_server")
    for name in names:
        await manager.add(
            name,
            scope="user",
            url=f"https://{name.replace('/', '-')}.test/rpc",
            oauth=True,
        )

    async def fake_authorize(*, server_name: str, server_url: str, **kwargs: object):
        token = MCPOAuthToken(
            access_token=f"token-for-{server_name}",
            refresh_token="refresh-token",
            expires_at=None,
            token_type="Bearer",
            scope=None,
            authorization_server="https://auth.test",
            token_endpoint="https://auth.test/token",
            authorization_endpoint="https://auth.test/authorize",
            client_id="client",
            client_secret=None,
            redirect_uri="http://127.0.0.1/callback",
            resource=server_url,
        )
        legacy_save = getattr(oauth_store, "save_token", None)
        if legacy_save is not None:
            legacy_save(server_name, token, home=kwargs.get("home"))
        return token

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    for name in names:
        await manager.login(name, scope="user", no_browser=True)

    seen: dict[str, list[str | None]] = {name: [] for name in names}

    def client_for(name: str) -> tuple[StreamableHTTPMCPClient, httpx.AsyncClient]:
        async def handler(request: httpx.Request) -> httpx.Response:
            seen[name].append(request.headers.get("authorization"))
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": name}]},
                },
                request=request,
            )

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return (
            StreamableHTTPMCPClient(
                MCPServerConfig(
                    name,
                    "streamable-http",
                    url=f"https://{name.replace('/', '-')}.test/rpc",
                    auth_type="oauth",
                ),
                client=transport,
                home=str(home),
            ),
            transport,
        )

    pairs = {name: client_for(name) for name in names}
    try:
        for name, (client, _transport) in pairs.items():
            result = await client.call_tool("echo", {}, AbortSignal())
            assert result["content"][0]["text"] == name
            assert seen[name] == [f"Bearer token-for-{name}"]

        logout = manager.logout("team/server", scope="user")
        if inspect.isawaitable(logout):
            await logout
        removed = await pairs["team/server"][0].call_tool("echo", {}, AbortSignal())
        retained = await pairs["team_server"][0].call_tool("echo", {}, AbortSignal())
        assert removed["isError"] is True
        assert retained["content"][0]["text"] == "team_server"
        assert seen["team/server"] == ["Bearer token-for-team/server"]
        assert seen["team_server"] == [
            "Bearer token-for-team_server",
            "Bearer token-for-team_server",
        ]
    finally:
        for client, transport in pairs.values():
            await client.close()
            await transport.aclose()


class _ProcessCredentialRaceStore(MCPCredentialStore):
    def __init__(
        self,
        *,
        home: Path,
        loop_thread_id: int,
        gate: str,
        load_number: int,
        connection: Any,
    ) -> None:
        super().__init__(home=home)
        self.loop_thread_id = loop_thread_id
        self.gate = gate
        self.load_number = load_number
        self.connection = connection
        self.loads = 0

    def _block(self) -> None:
        assert threading.get_ident() != self.loop_thread_id
        self.connection.send(("gate", self.gate))
        assert self.connection.recv() == "release"

    def _load_transaction(self, name: str, operation: str) -> CredentialSnapshot:
        self.loads += 1
        selected = self.loads == self.load_number
        if self.gate == "before_load" and selected:
            self._block()
        snapshot = super()._load_transaction(name, operation)  # type: ignore[arg-type]
        if self.gate == "after_load" and selected:
            self._block()
        return snapshot

    def _save_if_current_transaction(
        self,
        name: str,
        expected_generation: int,
        replacement: MCPOAuthToken,
    ) -> SaveIfCurrentResult:
        result = super()._save_if_current_transaction(
            name, expected_generation, replacement
        )
        if self.gate == "after_save_if_current":
            self._block()
        return result


async def _credential_race_child(
    home_text: str,
    gate: str,
    load_number: int,
    connection: Any,
) -> None:
    home = Path(home_text)
    store = _ProcessCredentialRaceStore(
        home=home,
        loop_thread_id=threading.get_ident(),
        gate=gate,
        load_number=load_number,
        connection=connection,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        connection.send(
            (
                "request",
                request.url.path,
                request.headers.get("authorization"),
                bytes(request.content),
            )
        )
        status, payload = await asyncio.to_thread(connection.recv)
        if isinstance(payload, dict):
            return httpx.Response(status, json=payload, request=request)
        return httpx.Response(status, text=payload, request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "live", "streamable-http", url="https://mcp.test/rpc", auth_type="oauth"
        ),
        client=transport,
        home=home_text,
    )
    client._credential_store = store
    try:
        result = await client.call_tool("echo", {}, AbortSignal())
        connection.send(("result", result))
    finally:
        await client.close()
        await transport.aclose()
        connection.close()


def _credential_race_process_entry(
    home_text: str,
    gate: str,
    load_number: int,
    connection: Any,
) -> None:
    asyncio.run(_credential_race_child(home_text, gate, load_number, connection))


class _CredentialRaceProcess:
    def __init__(self, home: Path, *, gate: str, load_number: int = 0) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(
            target=_credential_race_process_entry,
            args=(str(home), gate, load_number, child),
        )
        self.process.start()
        child.close()
        self.headers: list[str | None] = []

    async def receive(self) -> tuple[Any, ...]:
        message = await asyncio.wait_for(
            asyncio.to_thread(self.connection.recv), timeout=10
        )
        assert isinstance(message, tuple)
        return message

    async def request(self, status: int, payload: object) -> tuple[Any, ...]:
        message = await self.receive()
        assert message[0] == "request"
        self.headers.append(message[2])
        self.connection.send((status, payload))
        return message

    async def gate(self) -> None:
        message = await self.receive()
        assert message[0] == "gate"

    def release(self) -> None:
        self.connection.send("release")

    async def result(self) -> dict[str, object]:
        while True:
            message = await self.receive()
            if message[0] == "result":
                return message[1]
            assert message[0] == "request"
            self.headers.append(message[2])
            body = json.loads(message[3])
            self.connection.send(
                (
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"content": [{"type": "text", "text": "unexpected"}]},
                    },
                )
            )

    async def close(self) -> None:
        await asyncio.to_thread(self.process.join, 10)
        if self.process.is_alive():
            self.process.kill()
            await asyncio.to_thread(self.process.join, 5)
            raise AssertionError("credential race child did not exit")
        assert self.process.exitcode == 0
        self.process.close()
        self.connection.close()
