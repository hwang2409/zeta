import json
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.test_mcp_tool_filter import FilterClient
from zeta.mcp import MCPServerConfig, StreamableHTTPMCPClient
from zeta.mcp.oauth_store import MCPOAuthToken, save_token
from zeta.server.runtime import ServerRuntime


@pytest.mark.asyncio
async def test_serve_mounts_mcp_server_tool_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "mcp.json").write_text(json.dumps({"servers": {"google": {
        "transport": "stdio",
        "command": "unused",
        "allowed_tools": ["read_*"],
    }}}))
    clients: list[FilterClient] = []

    def build(config: MCPServerConfig) -> FilterClient:
        client = FilterClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr("zeta.mcp.mount._build_client", build)
    runtime = ServerRuntime(
        home,
        cwd=tmp_path,
        provider="codex",
        backend_factory=lambda *_args, **_kwargs: (FakeBackend([]), "test-model"),
    )
    try:
        await runtime.create_session()
        await runtime.loop.ensure_mcp_servers()
        assert clients
        assert runtime.loop.tool_registry.registered_names >= {"google__read_file"}
        assert "google__write_file" not in runtime.loop.tool_registry.registered_names
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_serve_mounts_already_authorized_oauth_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "mail": {
                        "transport": "streamable-http",
                        "url": "https://mcp.test/rpc",
                        "auth": {"type": "oauth"},
                    }
                }
            }
        )
    )
    token = MCPOAuthToken(
        access_token="stored-access-token",
        refresh_token="stored-refresh-token",
        expires_at=None,
        token_type="Bearer",
        scope="read",
        authorization_server="https://auth.test",
        token_endpoint="https://auth.test/token",
        authorization_endpoint="https://auth.test/authorize",
        client_id="client",
        client_secret=None,
        redirect_uri="http://127.0.0.1:8000/callback",
        resource="https://mcp.test/rpc",
    )
    save_token("mail", token, home=home)
    mounted: list[StreamableHTTPMCPClient] = []

    async def connect_and_list(client: StreamableHTTPMCPClient) -> list:
        mounted.append(client)
        assert client._current_token is not None
        assert client._current_token.access_token == "stored-access-token"
        return []

    monkeypatch.setattr("zeta.mcp.mount._connect_and_list", connect_and_list)
    runtime = ServerRuntime(
        home,
        cwd=tmp_path,
        provider="codex",
        backend_factory=lambda *_args, **_kwargs: (FakeBackend([]), "test-model"),
    )
    try:
        await runtime.create_session()
        await runtime.loop.ensure_mcp_servers()
        assert len(mounted) == 1
        assert mounted[0].config.auth_type == "oauth"
    finally:
        await runtime.close()
