"""Tests for the MCP OAuth flow, resources attachment, and token hygiene."""

from __future__ import annotations

import asyncio
import json
import socket
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp import (
    MCPServerConfig,
    StreamableHTTPMCPClient,
    load_mcp_config,
)
from zeta.mcp.client import MCPHTTPError, MCPResource, MCPResourceContent
from zeta.mcp.commands import parse_add_command
from zeta.mcp.http import MAX_RESPONSE_BYTES
from zeta.mcp.oauth import (
    AuthServerMetadata,
    MCPOAuthError,
    MCPOAuthStateError,
    authorize,
    build_authorization_url,
    discover_auth_server,
    discover_protected_resource,
    generate_pkce,
)
from zeta.mcp.oauth_store import (
    MCPOAuthToken,
    delete_token,
    load_token,
    redact_token,
    save_token,
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
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Connection: close\r\n\r\n"
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
            self.next_access_token = f"access-token-{len(self.issued_access_tokens) + 1}"
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
    save_token(name, token, home=str(home))
    return token


def _write_legacy_token(
    name: str, home: Path, *, client_secret: str
) -> MCPOAuthToken:
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


def test_token_store_round_trips_with_secure_permissions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = _write_token("srv", home)
    loaded = load_token("srv", home=str(home))
    assert loaded is not None
    assert loaded.access_token == token.access_token
    file_mode = token_store_path("srv", str(home)).stat().st_mode
    assert stat.S_IMODE(file_mode) == 0o600


def test_new_token_bundle_does_not_persist_client_secret(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_token("srv", home, client_secret="configured-secret")

    payload = json.loads(token_store_path("srv", str(home)).read_text())

    assert "client_secret" not in payload
    loaded = load_token("srv", home=str(home))
    assert loaded is not None
    assert loaded.client_secret is None


def test_delete_token_removes_the_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_token("srv", home)
    assert load_token("srv", home=str(home)) is not None
    delete_token("srv", home=str(home))
    assert load_token("srv", home=str(home)) is None


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

    def open_browser(url: str) -> None:
        asyncio.create_task(_do_redirect(url))

    token = await authorize(
        server_name="live",
        server_url=auth.server_url,
        home=str(home),
        http_client=http_client,
        browser_opener=open_browser,
    )
    await http_client.aclose()

    assert token.access_token == "access-token-1"
    assert token.refresh_token == "refresh-token-1"
    stored = load_token("live", home=str(home))
    assert stored is not None
    assert stored.access_token == token.access_token
    assert auth.registrations, "dynamic registration should have fired"
    token_form = auth.token_requests[0]
    assert token_form["grant_type"] == "authorization_code"
    assert token_form["code"] == "auth-code"
    assert token_form["resource"] == auth.resource
    assert token_form["code_verifier"], "PKCE verifier must be present in the token request"


@pytest.mark.asyncio
async def test_preregistered_client_uses_fixed_callback_scopes_and_extra_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
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

    def open_browser(url: str) -> None:
        asyncio.create_task(redirect(url))

    token = await authorize(
        server_name="live",
        server_url=auth.server_url,
        home=str(home),
        client_id="registered-client",
        client_secret="configured-secret",
        callback_port=callback_port,
        scopes=("read", "calendar"),
        authorization_params={"access_type": "offline", "prompt": "consent"},
        http_client=http_client,
        browser_opener=open_browser,
    )
    await http_client.aclose()

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
    assert reserved in config.malformed_servers["bad"].malformed_reason
    assert "reserved" in config.malformed_servers["bad"].malformed_reason


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
    metadata = await discover_auth_server(
        "https://login.test", http_client=client
    )
    await client.aclose()

    assert resource.authorization_server == "https://login.test"
    assert metadata.issuer == "https://login.test"
    assert requests == [
        "https://metadata.test/protected",
        "https://login.test/.well-known/oauth-authorization-server",
    ]


@pytest.mark.asyncio
async def test_default_protected_resource_probe_keeps_invalid_response_fallback() -> None:
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
            "https://other.test"
            if inconsistency == "issuer"
            else "https://login.test"
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
            expected = "issuer does not match" if inconsistency == "issuer" else "must use https"
            with pytest.raises(MCPOAuthError, match=expected):
                await discover_auth_server("https://login.test", http_client=client)
    finally:
        await client.aclose()


def test_authorization_url_defensively_rejects_reserved_params() -> None:
    metadata = AuthServerMetadata(
        issuer="https://auth.test",
        authorization_endpoint="https://auth.test/authorize",
        token_endpoint="https://auth.test/token",
        registration_endpoint=None,
        scopes_supported=(),
    )
    with pytest.raises(MCPOAuthError, match="reserved parameters: state"):
        build_authorization_url(
            metadata,
            client_id="client",
            redirect_uri="http://127.0.0.1:8888/callback",
            state="expected",
            code_challenge="challenge",
            resource="https://mcp.test",
            authorization_params={"state": "attacker"},
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

    def open_browser(url: str) -> None:
        asyncio.create_task(_do_redirect(url))

    with pytest.raises(MCPOAuthStateError):
        await authorize(
            server_name="live",
            server_url=auth.server_url,
            home=str(home),
            http_client=http_client,
            browser_opener=open_browser,
        )
    assert load_token("live", home=str(home)) is None
    await http_client.aclose()


@pytest.mark.asyncio
async def test_authorize_surfaces_error_from_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    auth = _FakeAuthServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(auth.handle))

    async def _do_redirect(url: str) -> None:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        redirect_uri = params["redirect_uri"][0]
        state = params["state"][0]
        await _fire_redirect(f"{redirect_uri}?error=access_denied&state={state}")

    def open_browser(url: str) -> None:
        asyncio.create_task(_do_redirect(url))

    with pytest.raises(MCPOAuthError, match="access_denied"):
        await authorize(
            server_name="live",
            server_url=auth.server_url,
            home=str(home),
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

    home = _monkey_home(monkeypatch, tmp_path)
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
            home=str(home),
            client_id="registered-client",
            no_browser=True,
            http_client=http_client,
        )
        assert token.access_token == "access-token-1"
        assert [request["code"] for request in auth.token_requests] == [
            "headless-code"
        ]
    else:
        with pytest.raises(error_type, match=message):
            await authorize(
                server_name="live",
                server_url=auth.server_url,
                home=str(home),
                client_id="registered-client",
                no_browser=True,
                http_client=http_client,
            )
        assert auth.token_requests == []
    await http_client.aclose()


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
    persisted = load_token("live", home=str(home))
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
    current = client._current_token
    assert current is not None
    assert await client._refresh_token_once(MCPHTTPError(401, "expired"), current)

    assert requests[0]["client_secret"] == ["environment-secret"]
    assert "client_secret" not in json.loads(
        token_store_path("live", str(home)).read_text()
    )
    await client.close()


@pytest.mark.asyncio
async def test_legacy_stored_secret_refreshes_and_rewrite_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    token = _write_legacy_token(
        "live", home, client_secret="legacy-stored-secret"
    )
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
    current = client._current_token
    assert current == token
    assert await client._refresh_token_once(MCPHTTPError(401, "expired"), current)

    assert requests[0]["client_secret"] == ["legacy-stored-secret"]
    assert "client_secret" not in json.loads(
        token_store_path("live", str(home)).read_text()
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
            return httpx.Response(
                400, json={"error": "invalid_grant"}, request=request
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
    persisted = load_token("live", home=str(home))
    assert persisted is not None
    assert persisted.refresh_error is not None
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
    assert (
        "streamable-http"
        in config.malformed_servers["bad"].malformed_reason
    )


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
    loop = AgentLoop(FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty())
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
    loop = AgentLoop(backend, store, skip_mcp_mount=True, skill_catalog=SkillCatalog.empty())
    events = [event async for event in loop.run_turn("please answer", origin=MessageOrigin.USER)]
    await loop.close()

    haystack = ""
    for session_file in store.session_dir.rglob("*.jsonl"):
        haystack += session_file.read_text()
    haystack += json.dumps(
        [event.type.value for event in events]
    )
    assert token.access_token not in haystack
    assert token.refresh_token not in haystack


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

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty())
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
    from zeta.mcp import commands as commands_module
    from zeta.mcp import mount as mount_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)

    class _RecordingAuthorize:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def __call__(self, **kwargs: object) -> MCPOAuthToken:
            self.calls.append(kwargs)
            token = _write_token(
                str(kwargs["server_name"]),
                Path(str(kwargs["home"])) if kwargs.get("home") else home,
                expires_at=1e12,
            )
            return token

    recorder = _RecordingAuthorize()
    monkeypatch.setattr(commands_module, "authorize", recorder)
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
        "home": str(home),
        "client_id": None,
        "client_secret": None,
        "callback_port": 0,
        "scopes": None,
        "authorization_server_url": "https://login.example",
        "resource_metadata_url": "https://mcp.example/metadata",
        "authorization_params": {"access_type": "offline"},
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
    from zeta.mcp import commands as commands_module
    from zeta.mcp import mount as mount_module

    async def fake_connect_and_list(client: Any) -> list:
        return []

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)

    async def boom(**kwargs: object) -> MCPOAuthToken:
        raise MCPOAuthError("consent denied")

    monkeypatch.setattr(commands_module, "authorize", boom)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty())
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
        async def close(self) -> None: ...

    monkeypatch.setattr(mount_module, "_build_client", lambda config: _NoopClient(config))
    loop = AgentLoop(FakeBackend([]), ConversationStore(project), skill_catalog=SkillCatalog.empty())
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
