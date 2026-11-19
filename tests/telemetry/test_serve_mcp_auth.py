"""Headless MCP auth via real serve RPC and the real loopback callback listener."""

import json
import logging
import sys
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.telemetry.test_serve_propagation import Wire, _server
from tests.telemetry.test_serve_propagation import (
    traced as traced,  # noqa: PLC0414 - pytest fixture
)
from tests.test_mcp_oauth import _FakeAuthServer, _fire_redirect
from zeta.mcp.oauth_store import MCPCredentialStore


@pytest.mark.asyncio
async def test_mcp_headless_auth_url_is_start_only_and_logout_wins_races(
    tmp_path, monkeypatch, traced, caplog, capsys,
):
    telemetry, memory = traced
    wire = Wire(_server(tmp_path, "unix", telemetry, monkeypatch), "unix")
    provider = _FakeAuthServer()
    provider.next_access_token = "CANARY_mcp_access_2d2"
    provider.next_refresh_token = "CANARY_mcp_refresh_2d2"
    client_secret = "CANARY_mcp_client_2d2"
    log_secret = "CANARY_mcp_log_2d2"
    real_client = httpx.AsyncClient

    async def handle(request):
        logging.getLogger("httpx").warning("private diagnostic: %s", log_secret)
        return await provider.handle(request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    caplog.set_level(logging.DEBUG)
    await wire.open()

    async def call(method, params=None):
        return json.loads(await wire.rpc(method, method, params))

    async def state(method):
        return (await call(method, {"name": "example"}))["result"]["data"]

    async def terminal():
        while (status := await state("mcp_auth_status")) == {"state": "pending"}:
            pass
        return status

    urls = []
    store = MCPCredentialStore(home=tmp_path)
    try:
        await call("hello", {"protocol_version": "1.2", "features": ["otel", "mcp_management_v1"]})
        listed = (await call("mcp_server_list"))["result"]
        result = await call("mcp_server_add", {"revision": listed["revision"], "mutation": {
            "name": "example", "scope": "user", "url": provider.server_url, "oauth": True,
            "client_id": "serve-test", "client_secret": client_secret,
        }})
        assert "result" in result

        async def start():
            pending = await state("mcp_auth_start")
            assert pending["state"] == "pending"
            url = pending["authorization_url"]
            urls.append(url)
            assert await state("mcp_auth_status") == {"state": "pending"}
            query = parse_qs(urlparse(url).query)
            return f"{query['redirect_uri'][0]}?code=test-code&state={query['state'][0]}"

        await start()
        assert (await call("mcp_auth_start", {"name": "example"}))["error"]["code"] == -32004
        assert await state("mcp_auth_cancel") == {"state": "cancelled"}
        assert (await store.load("example")).token is None

        redirect = await start()
        await await_or_fail(_fire_redirect(redirect), what="OAuth loopback callback")
        assert await await_or_fail(terminal(), what="credential publication") == {"state": "succeeded"}
        assert (await store.load("example")).token is not None
        assert await state("mcp_auth_logout") == {"state": "unauthorized"}
        assert (await store.load("example")).token is None

        # A logout by another owner after start must fence even a first login with no token.
        redirect = await start()
        await store.delete("example")
        await await_or_fail(_fire_redirect(redirect), what="stale OAuth loopback callback")
        assert await await_or_fail(terminal(), what="stale OAuth publication") == {"state": "failed"}
        assert (await store.load("example")).token is None
    finally:
        await close_serve(wire.server, wire.writer)

    assert provider.token_requests[0]["client_secret"] == client_secret
    frames = b"".join(wire.received).decode()
    assert frames.count('"authorization_url"') == len(urls) == 3
    capture = capsys.readouterr()
    diagnostics = capture.out + capture.err + caplog.text + "".join(
        span.to_json() for span in memory.get_finished_spans()
    )
    for url in urls:
        assert url not in diagnostics
        assert parse_qs(urlparse(url).query)["state"][0] not in diagnostics
    for secret in (client_secret, "CANARY_mcp_access_2d2", "CANARY_mcp_refresh_2d2", log_secret):
        assert secret not in frames + diagnostics


@pytest.mark.asyncio
async def test_mcp_stdio_probe_never_exposes_child_stderr_or_tool_text(tmp_path, traced, monkeypatch, caplog, capsys):
    telemetry, memory = traced
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    wire = Wire(_server(tmp_path, "unix", telemetry, monkeypatch), "unix")
    secret = "CANARY_mcp_stdio_2d2"
    script = '''
import json, os, sys
secret = os.environ["MCP_CANARY"]
print(secret, file=sys.stderr, flush=True)
for line in sys.stdin:
    rpc = json.loads(line)
    method = rpc["method"]
    if method == "initialize":
        data = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {}}
    elif method == "tools/list":
        data = {"tools": [{"name": secret, "description": secret, "inputSchema": {}}]}
    else:
        assert method == "notifications/initialized"
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": data}), flush=True)
'''
    caplog.set_level(logging.DEBUG)
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel", "mcp_management_v1"]})
        listed = json.loads(await wire.rpc("list", "mcp_server_list"))["result"]
        added = json.loads(await wire.rpc("add", "mcp_server_add", {"revision": listed["revision"], "mutation": {
            "name": "example", "scope": "user", "command": sys.executable, "args": ["-c", script],
            "env": {"MCP_CANARY": secret},
        }}))
        assert "result" in added
        reply = json.loads(await wire.rpc("test", "mcp_server_test", {"name": "example"}))
        assert reply["result"] == {"data": {"status": "ok", "tool_count": 1}}
    finally:
        await close_serve(wire.server, wire.writer)
    assert not (tmp_path / "mcp-logs").exists()
    capture = capsys.readouterr()
    assert secret not in b"".join(wire.received).decode() + caplog.text + capture.err + "".join(
        span.to_json() for span in memory.get_finished_spans()
    )
