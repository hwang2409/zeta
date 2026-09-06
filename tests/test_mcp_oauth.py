"""Tests for the MCP OAuth flow, resources attachment, and token hygiene."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from zeta.core.abort import AbortSignal
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.mcp import (
    MCPServerConfig,
    StreamableHTTPMCPClient,
    load_mcp_config,
)
from zeta.mcp.client import MCPHTTPError, MCPResource
from zeta.mcp.commands import parse_add_command
from zeta.mcp.http import MAX_RESPONSE_BYTES
from zeta.mcp.oauth import (
    MCPOAuthError,
    MCPOAuthStateError,
    authorize,
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
    MCPResourceTooLargeError,
    fetch_resource,
    format_resource_list,
    list_resources,
)
from zeta.types import TextContent


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
        resource: str = "https://mcp.test/resource",
        authorization_server: str = "https://auth.test",
        refresh_error_status: int | None = None,
    ) -> None:
        self.server_url = server_url
        self.resource = resource
        self.authorization_server = authorization_server
        self.refresh_error_status = refresh_error_status
        self.registrations: list[dict[str, object]] = []
        self.token_requests: list[dict[str, str]] = []
        self.issued_access_tokens: list[str] = []
        self.rejected_states: list[str] = []
        self.next_access_token = "access-token-1"
        self.next_refresh_token = "refresh-token-1"
        self.expected_code_verifier: str | None = None
        self.last_code_challenge: str | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
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
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add live --http https://mcp.example")

    listing = await loop.slash_mcp("resources live")
    assert "mcp://doc/1" in listing
    assert "mime=text/plain" in listing

    attachment = await loop.slash_mcp("resources live mcp://doc/1")
    from zeta.mcp.prompt_commands import SlashModelInput

    assert isinstance(attachment, SlashModelInput)
    assert "[mcp-resource: live:mcp://doc/1" in attachment.text
    assert "payload for mcp://doc/1" in attachment.text
    await loop.close()


@pytest.mark.asyncio
async def test_resource_attach_rejects_too_large_payload() -> None:
    class _BulkyClient:
        config = MCPServerConfig("srv", "streamable-http", url="https://mcp.test")

        async def read_resource(self, uri: str) -> str:
            del uri
            return "x" * 300_000

        async def list_resources(self) -> list[MCPResource]:
            return []

    client = _BulkyClient()  # type: ignore[assignment]
    with pytest.raises(MCPResourceTooLargeError, match="bytes"):
        await fetch_resource(client, server="srv", uri="mcp://big")  # type: ignore[arg-type]


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
    loop = AgentLoop(backend, store, skip_mcp_mount=True)
    events = [event async for event in loop.run_turn("please answer")]
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
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
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

    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add live --http https://mcp.example --oauth")

    output = await loop.slash_mcp("auth live")
    assert "auth: oauth (authorized)" in output
    assert recorder.calls, "authorize must be invoked"
    assert recorder.calls[0]["server_name"] == "live"
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
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
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
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.slash_mcp("add local --stdio command")

    output = await loop.slash_mcp("auth local")
    assert "requires an HTTP MCP server" in output
    await loop.close()


@pytest.mark.asyncio
async def test_http_client_aborts_oversized_streamed_body() -> None:
    """A server that streams past the transport cap is aborted early."""

    chunk_size = 50_000
    total_chunks = 20  # 1MB total, transport cap is 400KB
    chunks_yielded = 0

    async def payload():
        nonlocal chunks_yielded
        for _ in range(total_chunks):
            chunks_yielded += 1
            yield b"x" * chunk_size

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
                content=payload(),
                headers={"content-type": "application/json"},
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig("live", "streamable-http", url="https://mcp.test/rpc")
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport)
    await client.connect()
    with pytest.raises(MCPHTTPError, match="too large"):
        await client.read_resource("mcp://big")
    max_expected_chunks = (MAX_RESPONSE_BYTES // chunk_size) + 2
    assert chunks_yielded <= max_expected_chunks
    assert chunks_yielded < total_chunks
    await client.close()


@pytest.mark.asyncio
async def test_http_client_rejects_oversized_content_length() -> None:
    """Content-Length declared above the cap is rejected before body reads."""

    body_read = False

    async def payload():
        nonlocal body_read
        body_read = True
        yield b"x"

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
            oversize = str(MAX_RESPONSE_BYTES + 1)
            return httpx.Response(
                200,
                content=payload(),
                headers={
                    "content-type": "application/json",
                    "content-length": oversize,
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    config = MCPServerConfig("live", "streamable-http", url="https://mcp.test/rpc")
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport)
    await client.connect()
    with pytest.raises(MCPHTTPError, match="too large"):
        await client.read_resource("mcp://big")
    assert body_read is False
    await client.close()


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
