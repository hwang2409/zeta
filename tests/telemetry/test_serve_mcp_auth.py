"""Headless MCP auth via real serve RPC and the real loopback callback listener."""

import asyncio
import json
import logging
import sys
from functools import partial
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.telemetry.test_serve_propagation import Wire, _server
from tests.telemetry.test_serve_propagation import (
    traced as traced,  # noqa: PLC0414 - pytest fixture
)
from tests.test_mcp_oauth import _FakeAuthServer, _fire_redirect
from zeta.mcp import oauth
from zeta.mcp.envelope import MCPCallLimits
from zeta.mcp.oauth_store import MCPCredentialStore
from zeta.server.handlers.resources import mcp_management
from zeta.skills.catalog import SkillCatalog
from zeta.tools.registry import ToolRegistry


async def _launch(url):
    reply = await await_or_fail(_fire_redirect(url), what="OAuth browser launch")
    assert reply.startswith(b"HTTP/1.1 302 ")
    return reply.split(b"\r\nLocation: ", 1)[1].split(b"\r\n", 1)[0].decode("ascii")


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

        async def start(*, open_browser=True):
            pending = await state("mcp_auth_start")
            assert pending["state"] == "pending"
            url = pending["authorization_url"]
            urls.append(url)
            assert await state("mcp_auth_status") == {"state": "pending"}
            if not open_browser:
                return None
            query = parse_qs(urlparse(await _launch(url)).query)
            return f"{query['redirect_uri'][0]}?code=test-code&state={query['state'][0]}"

        await start(open_browser=False)
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
    for secret in (client_secret, "CANARY_mcp_access_2d2", "CANARY_mcp_refresh_2d2", log_secret):
        assert secret not in frames + diagnostics


@pytest.mark.parametrize(("resource_source", "outcome"), [
    ("published-path", "succeeded"),
    ("configured-path", "failed"),
    ("fallback", "succeeded"),
    ("published-path", "cancelled"),
    ("published-path", "expired"),
])
@pytest.mark.asyncio
async def test_mcp_headless_auth_launch_keeps_provider_url_private(
    tmp_path, monkeypatch, traced, caplog, capsys, resource_source, outcome,
):
    telemetry, memory = traced
    wire = Wire(_server(tmp_path, "unix", telemetry, monkeypatch), "unix")
    secret = "CANARY_mcp_url_path_2d2"
    server_url = f"https://mcp.linear.test/mcp/{secret}"
    resource = "https://mcp.linear.test/mcp" if resource_source == "published-path" else server_url
    provider = _FakeAuthServer(
        server_url=server_url,
        resource=resource,
        authorization_server=server_url if resource_source == "fallback" else "https://auth.test",
    )
    endpoint = "https://auth.test/authorize"
    if resource_source == "configured-path":
        endpoint += "?provider=CANARY_mcp_provider_2d2"
    if outcome == "expired":
        monkeypatch.setattr(oauth, "authorize", partial(oauth.authorize, listen_timeout=0))
    real_client = httpx.AsyncClient

    async def handle(request):
        logging.getLogger("httpx").debug("request: %s", request.url)
        path = request.url.path
        if path.startswith("/.well-known/oauth-protected-resource"):
            if resource_source == "fallback":
                return httpx.Response(404)
            return httpx.Response(200, json={
                "resource": resource, "authorization_servers": [provider.authorization_server],
            })
        if path.startswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(200, json={
                "issuer": provider.authorization_server,
                "authorization_endpoint": endpoint,
                "token_endpoint": "https://auth.test/token",
            })
        return await provider.handle(request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    caplog.set_level(logging.DEBUG)
    await wire.open()
    bound = []
    start_server = asyncio.start_server

    async def capture_listener(*args, **kwargs):
        listener = await start_server(*args, **kwargs)
        bound.extend(sock.getsockname() for sock in listener.sockets)
        return listener

    monkeypatch.setattr(asyncio, "start_server", capture_listener)

    async def rpc(method, params=None):
        return json.loads(await wire.rpc(method, method, params))["result"]

    async def terminal():
        while (status := await rpc("mcp_auth_status", {"name": "example"}))["data"]["state"] == "pending":
            pass
        return status["data"]

    location = None
    state = None
    try:
        await rpc("hello", {"protocol_version": "1.2", "features": ["otel", "mcp_management_v1"]})
        listed = await rpc("mcp_server_list")
        await rpc("mcp_server_add", {"revision": listed["revision"], "mutation": {
            "name": "example", "scope": "user", "url": server_url, "oauth": True, "client_id": "serve-test",
        }})
        pending = (await rpc("mcp_auth_start", {"name": "example"}))["data"]
        assert pending["state"] == "pending"
        launch_url = pending["authorization_url"]
        launch = urlparse(launch_url)
        assert launch.scheme == "http" and launch.hostname == "127.0.0.1"
        assert launch.path.startswith("/launch/") and not launch.query
        assert bound == [("127.0.0.1", launch.port)]
        if outcome == "cancelled":
            assert (await rpc("mcp_auth_cancel", {"name": "example"}))["data"] == {"state": outcome}
            with pytest.raises(ConnectionRefusedError):
                await await_or_fail(_fire_redirect(launch_url), what="cancelled OAuth launch")
        elif outcome == "expired":
            assert await await_or_fail(terminal(), what="OAuth launch expiry") == {"state": "failed"}
            with pytest.raises(ConnectionRefusedError):
                await await_or_fail(_fire_redirect(launch_url), what="expired OAuth launch")
        else:
            location = await _launch(launch_url)
            assert location.startswith(endpoint + ("&" if "?" in endpoint else "?"))
            query = parse_qs(urlparse(location).query)
            assert query["resource"] == [resource]
            redirect_uri, = query["redirect_uri"]
            assert urlparse(redirect_uri).netloc == launch.netloc
            state, = query["state"]
            replay = await await_or_fail(_fire_redirect(launch_url), what="OAuth launch replay")
            assert replay.startswith(b"HTTP/1.1 404 ") and b"Location:" not in replay

            # An upstream error can reflect the full URL. Serve returns only the failed state.
            params = {"state": state}
            if outcome == "failed":
                params.update(error="access_denied", error_description=location)
            else:
                params["code"] = "test-code"
            callback_url = f"{redirect_uri}?{urlencode(params)}"
            await await_or_fail(_fire_redirect(callback_url), what="OAuth loopback callback")
            assert await await_or_fail(terminal(), what="OAuth terminal state") == {"state": outcome}
            if outcome == "succeeded":
                [token_request] = provider.token_requests
                assert token_request["resource"] == resource
    finally:
        await close_serve(wire.server, wire.writer)

    capture = capsys.readouterr()
    observed = b"".join(wire.received).decode() + capture.out + capture.err + caplog.text + "".join(
        span.to_json() for span in memory.get_finished_spans()
    )
    for private in (secret, endpoint, "CANARY_mcp_provider_2d2", location, state):
        if private is not None:
            assert private not in observed


@pytest.mark.parametrize("hang_on", ["initialize", "tools/list"])
@pytest.mark.asyncio
async def test_mcp_stdio_probe_times_out_and_reaps_child(tmp_path, traced, monkeypatch, hang_on):
    telemetry, _ = traced
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty(),
        mcp_limits=MCPCallLimits(deadline_seconds=1),
    )
    wire = Wire(_server(tmp_path, "unix", telemetry, monkeypatch), "unix")
    # test-contract: seam idle serve has no session registry; inject limits while driving the real RPC and child.
    monkeypatch.setattr(mcp_management, "_registry", lambda _client: registry)
    processes = []
    spawn = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)
    script = '''
import json, sys
for line in sys.stdin:
    rpc = json.loads(line)
    if rpc["method"] == "initialize" and sys.argv[1] != "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {}}
        print(json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": result}), flush=True)
'''
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel", "mcp_management_v1"]})
        listed = json.loads(await wire.rpc("list", "mcp_server_list"))["result"]
        added = json.loads(await wire.rpc("add", "mcp_server_add", {"revision": listed["revision"], "mutation": {
            "name": "example", "scope": "user", "command": sys.executable, "args": ["-c", script, hang_on],
        }}))
        assert "result" in added
        reply = json.loads(await wire.rpc("test", "mcp_server_test", {"name": "example"}))
        assert reply["result"] == {"data": {"status": "failed", "tool_count": 0}}
        [process] = processes
        assert process.returncode is not None
    finally:
        # A regression must not leave the fixture child behind.
        for process in processes:
            if process.returncode is None:
                process.kill()
                await await_or_fail(process.wait(), what="reap probe fixture")
        await close_serve(wire.server, wire.writer)
        await registry.close()


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
