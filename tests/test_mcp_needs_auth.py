"""MCP OAuth needs-auth state: a rejected refresh stops later mount attempts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tests.support.storage_homes import initialize_storage_home
from zeta.mcp import StreamableHTTPMCPClient, load_mcp_config
from zeta.mcp import mount as mount_module
from zeta.mcp.client import MCPAuthRequiredError
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.management import MCPManagementService
from zeta.mcp.mount import mount_mcp_servers
from zeta.mcp.oauth_store import MCPCredentialStore, MCPOAuthToken, token_store_path
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry

_TOKEN = MCPOAuthToken(
    access_token="stale-token",
    refresh_token="refresh-token-1",
    expires_at=None,
    token_type="Bearer",
    scope="mcp:tools",
    authorization_server="https://auth.test",
    token_endpoint="https://auth.test/token",
    authorization_endpoint="https://auth.test/authorize",
    client_id="fake-client",
    client_secret=None,
    redirect_uri="http://127.0.0.1:0/callback",
    resource="https://mcp.test/rpc",
)


class _Server:
    """Fake MCP server plus token endpoint; only `fresh-token` is accepted."""

    def __init__(self, refresh_status: int, refresh_body: dict[str, str]) -> None:
        self.refresh_status = refresh_status
        self.refresh_body = refresh_body
        self.requests: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url.path}")
        if request.url.path == "/token":
            return httpx.Response(
                self.refresh_status, json=self.refresh_body, request=request
            )
        if request.headers.get("authorization") != "Bearer fresh-token":
            return httpx.Response(401, text="expired", request=request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        result: dict[str, object] = (
            {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}
            if body["method"] == "initialize"
            else {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
        )
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
            request=request,
        )


async def _start(
    tmp_path: Path,
    home: Path,
    server: _Server | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], str]:
    """Mount the configured servers once, like one TUI start; return notices.

    With no fake server, the production client factory runs unchanged.
    """
    if server is not None:
        handle = server.handle

        def build(config: MCPServerConfig) -> StreamableHTTPMCPClient:
            transport = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            return StreamableHTTPMCPClient(config, client=transport, home=str(home))

        # test-contract: seam the client factory seam; it routes HTTP to the fake
        monkeypatch.setattr(mount_module, "_build_client", build)
    (tmp_path / "tools").mkdir(exist_ok=True)
    registry = ToolRegistry(
        tmp_path / "tools", register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    notices: list[str] = []
    try:
        mount = await mount_mcp_servers(
            registry,
            load_mcp_config(home / "mcp.json"),
            notice_sink=notices.append,
            home=str(home),
        )
        try:
            return notices, mount.statuses["slack"].state
        finally:
            await mount.close()
    finally:
        await registry.close()


async def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    initialize_storage_home(home)
    (tmp_path / "project").mkdir()
    manager = MCPManagementService(home=home, project_dir=tmp_path / "project")
    await manager.add("slack", scope="user", url="https://mcp.test/rpc", oauth=True)
    await MCPCredentialStore(home=home).save_login("slack", _TOKEN)
    return home


@pytest.mark.asyncio
async def test_refresh_401_skips_later_mounts_until_mcp_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    home = await _home(tmp_path, monkeypatch)
    server = _Server(401, {"error": "invalid_client"})

    await _start(tmp_path, home, server, monkeypatch)
    assert "POST /token" in server.requests

    server.requests.clear()
    caplog.clear()
    notices, state = await _start(tmp_path, home, server, monkeypatch)
    assert server.requests == [], "a needs-auth server must not be contacted"
    assert "failed to mount" not in caplog.text
    assert notices == ["mcp · slack needs reauthorization: /mcp auth slack"]
    assert state == "failed"

    async def fake_authorize(**_kwargs: object) -> MCPOAuthToken:
        return replace(_TOKEN, access_token="fresh-token")

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    manager = MCPManagementService(home=home, project_dir=tmp_path / "project")
    await manager.login("slack", scope="user", no_browser=True)

    notices, state = await _start(tmp_path, home, server, monkeypatch)
    assert state == "mounted"
    assert "mcp · slack mounted (1 tools)" in notices


@pytest.mark.asyncio
async def test_refresh_server_error_does_not_set_needs_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = await _home(tmp_path, monkeypatch)
    server = _Server(503, {"error": "temporarily_unavailable"})

    await _start(tmp_path, home, server, monkeypatch)
    server.requests.clear()
    await _start(tmp_path, home, server, monkeypatch)

    assert "POST /token" in server.requests
    token = (await MCPCredentialStore(home=home).load("slack")).token
    assert token is not None and token.refresh_error is None


@pytest.mark.asyncio
async def test_never_authorized_server_shows_one_notice_without_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    home = await _home(tmp_path, monkeypatch)
    await MCPCredentialStore(home=home).delete("slack")

    # the real ~/.zeta shape: oauth server configured, no token file; production client
    notices, state = await _start(tmp_path, home, None, monkeypatch)

    assert notices == ["mcp · slack needs authorization: /mcp auth slack"]
    assert "failed to mount" not in caplog.text
    assert state == "failed"


@pytest.mark.asyncio
async def test_needs_auth_write_waits_for_the_store_lock_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = await _home(tmp_path, monkeypatch)
    lock_path = token_store_path("slack", home).with_suffix(".lock")
    loop = asyncio.get_running_loop()
    held = os.open(lock_path, os.O_RDWR)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            # another process holds the credential lock while the marker is written
            fcntl.flock(held, fcntl.LOCK_EX)
            loop.call_later(0.3, fcntl.flock, held, fcntl.LOCK_UN)
            return httpx.Response(401, json={"error": "invalid_client"})
        return httpx.Response(401, text="expired", request=request)

    config = load_mcp_config(home / "mcp.json").configured_servers["slack"]
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    beat = asyncio.create_task(heartbeat())
    try:
        with pytest.raises(MCPAuthRequiredError):
            await asyncio.wait_for(client.connect(), 10)
        assert ticks >= 10, "the event loop stalled while the store lock was held"
        token = (await MCPCredentialStore(home=home).load("slack")).token
        assert token is not None
        assert token.refresh_error == "MCP OAuth: token endpoint returned HTTP 401"
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        await client.close()
        os.close(held)
