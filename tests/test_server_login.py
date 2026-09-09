"""Native login RPCs exercise the real listener without contacting OAuth servers."""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from tests.test_server import _close, _connect, _request
from zeta.providers import anthropic
from zeta.providers import login as providers
from zeta.providers.auth import OAuthTokens
from zeta.providers.factory import credential_store
from zeta.server import ZetaServer
from zeta.server.login import REQUESTS


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("ZETA_ALLOW_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "_keychain_claude_tokens", lambda: None)
    # A missed stub must fail locally, before any HTTP request leaves the test.
    async def forbidden(*args, **kwargs):
        raise AssertionError("real OAuth exchange attempted")
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", forbidden)
    monkeypatch.setattr(providers, "exchange_codex_authorization_code", forbidden)


async def rpc(reader, writer, method, **params):
    return (await _request(reader, writer, method, method, params))[-1]


async def callback(url, params):
    parsed = urlsplit(url)
    reader, writer = await asyncio.open_connection("127.0.0.1", parsed.port)
    writer.write(f"GET {parsed.path}?{urlencode(params)} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
    await writer.drain()
    status = int((await reader.readline()).split()[1])
    await reader.read()
    writer.close()
    await writer.wait_closed()
    return status


async def connect(tmp_path, provider="claude", version="1.1"):
    server = ZetaServer(home=tmp_path / "zeta", port=0, provider=provider)
    reader, writer = await _connect(server)
    hello = await rpc(reader, writer, "hello", protocol_version=version)
    return server, reader, writer, hello["result"]


@pytest.mark.parametrize("server_provider,version", [("fake", "1.1"), ("claude", "1.0")])
async def test_login_is_hidden_and_rejected_for_fake_and_legacy(tmp_path, server_provider, version):
    server, reader, writer, hello = await connect(tmp_path, server_provider, version)
    try:
        assert not set(REQUESTS) & set(hello["capabilities"]["requests"])
        for method in REQUESTS:
            result = await rpc(reader, writer, method, provider="claude")
            assert result["error"]["code"] == -32601
        assert not server._client.logins.tasks
    finally:
        await _close(server, writer)


@pytest.mark.parametrize("provider", ["claude", "codex"])
async def test_rpc_login_persists_synthetic_exchange_and_reports_presence(tmp_path, monkeypatch, provider):
    expected = OAuthTokens("access", "refresh", 4_000_000_000)
    captured = {}
    def build(state, challenge, redirect):
        captured.update(state=state, challenge=challenge, redirect=redirect)
        return "https://authorize.invalid/?" + urlencode({"state": state})
    async def exchange(client, code, state, verifier, redirect):
        assert code == "synthetic-code"
        assert state == captured["state"]
        assert verifier and verifier != captured["challenge"]
        assert redirect == captured["redirect"]
        return expected
    suffix = "anthropic" if provider == "claude" else "codex"
    monkeypatch.setattr(providers, f"build_{suffix}_authorization_url", build)
    monkeypatch.setattr(providers, f"exchange_{suffix}_authorization_code", exchange)
    monkeypatch.setattr(providers, "extract_account_id", lambda _: "test-account")
    server, reader, writer, hello = await connect(tmp_path)
    try:
        assert set(REQUESTS) <= set(hello["capabilities"]["requests"])
        rows = (await rpc(reader, writer, "login_providers"))["result"]["providers"]
        assert rows == [{"provider": p, "credentials_present": False} for p in ("claude", "codex")]
        result = (await rpc(reader, writer, "login_start", provider=provider))["result"]
        assert result["state"] == "pending"
        assert parse_qs(urlsplit(result["authorization_url"]).query)["state"] == [captured["state"]]
        duplicate = await rpc(reader, writer, "login_start", provider=provider)
        assert duplicate["error"]["data"]["code"] == "login_in_progress"
        # Both providers may wait at once, with independent cancellation.
        other = "codex" if provider == "claude" else "claude"
        assert (await rpc(reader, writer, "login_start", provider=other))["result"]["state"] == "pending"
        assert (await rpc(reader, writer, "login_cancel", provider=other))["result"]["state"] == "cancelled"
        assert await callback(captured["redirect"], {"code": "synthetic-code", "state": captured["state"]}) == 200
        await asyncio.wait_for(server._client.logins.tasks[provider], 3)
        assert (await rpc(reader, writer, "login_status", provider=provider))["result"] == {"state": "succeeded"}
        assert credential_store(provider, home=server.home).read() == expected
        rows = (await rpc(reader, writer, "login_providers"))["result"]["providers"]
        assert next(row for row in rows if row["provider"] == provider)["credentials_present"]
        assert (await rpc(reader, writer, "status"))["result"]["state"] == "idle"
    finally:
        await _close(server, writer)


@pytest.mark.parametrize("outcome", ["cancel", "disconnect", "timeout", "exchange_timeout", "state", "denied", "exchange", "start"])
async def test_login_failures_release_listener_and_allow_retry(tmp_path, monkeypatch, outcome):
    captured = {}
    def build(state, challenge, redirect):
        captured.update(state=state, redirect=redirect)
        if outcome == "start":
            raise OSError("private detail")
        return "https://authorize.invalid/"
    async def exchange(*args):
        if outcome == "exchange_timeout":
            await asyncio.Event().wait()
        raise RuntimeError("SECRET_TOKEN")
    monkeypatch.setattr(providers, "build_anthropic_authorization_url", build)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    server, reader, writer, _ = await connect(tmp_path)
    manager = server._client.logins
    if outcome in {"timeout", "exchange_timeout"}:
        manager.timeout = 0.3
    try:
        started = (await rpc(reader, writer, "login_start", provider="claude"))["result"]
        if outcome in {"state", "denied", "exchange", "exchange_timeout"}:
            params = {"code": "synthetic", "state": captured["state"]}
            if outcome == "state":
                params["state"] = "wrong"
            if outcome == "denied":
                params["error"] = "access_denied"
            assert await callback(captured["redirect"], params) == (400 if outcome in {"state", "denied"} else 200)
        if outcome == "cancel":
            result = (await rpc(reader, writer, "login_cancel", provider="claude"))["result"]
            assert result == {"state": "cancelled"}
        elif outcome == "disconnect":
            await _close(server, writer)
            assert not manager.tasks
        else:
            result = await asyncio.wait_for(manager.tasks["claude"], 3)
            expected = "login_timeout" if "timeout" in outcome else "login_callback_error" if outcome in {"state", "denied"} else "login_failed"
            assert result["error"]["code"] == expected
            assert "SECRET" not in str(result)
            if outcome == "start":
                assert started == result
        # Cleanup has completed before status reports a terminal state.
        port = urlsplit(captured["redirect"]).port
        listener = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", port)
        listener.close()
        await listener.wait_closed()
        assert credential_store("claude", home=server.home).read() is None
        if outcome != "disconnect":
            monkeypatch.setattr(providers, "build_anthropic_authorization_url", lambda *args: "https://authorize.invalid/")
            retry = (await rpc(reader, writer, "login_start", provider="claude"))["result"]
            assert retry["state"] == "pending"
    finally:
        await _close(server, writer)


async def test_unknown_provider_rejected_and_idle_cancel_is_safe(tmp_path):
    server, reader, writer, _ = await connect(tmp_path)
    try:
        for method in ("login_start", "login_status", "login_cancel"):
            for provider in ("fake", "unknown", ["claude"], None):
                assert (await rpc(reader, writer, method, provider=provider))["error"]["code"] == -32602
        assert (await rpc(reader, writer, "login_cancel", provider="claude"))["result"] == {"state": "idle"}
    finally:
        await _close(server, writer)


@pytest.mark.parametrize("version,provider_error,background,status,expected", [
    ("1.1", True, False, None, "codex"),
    ("1.1", True, False, 401, "codex"),
    ("1.1", False, False, None, None),
    ("1.1", True, True, None, None),
    ("1.0", True, False, None, None),
])
async def test_login_action_uses_failed_provider_before_model_revert(tmp_path, version, provider_error, background, status, expected):
    from tests.test_server import _event
    from zeta.core.fake import FakeBackend
    from zeta.server import model_selection
    from zeta.types import ErrorInfo, StreamEvent, StreamEventType

    server = ZetaServer(home=tmp_path / "zeta", port=0, provider="claude", model="claude-sonnet-4-6",
                        backend_factory=lambda provider, model, home: (FakeBackend([]), model))
    reader, writer = await _connect(server)
    try:
        await rpc(reader, writer, "hello", protocol_version=version)
        created = await rpc(reader, writer, "new_session")
        assert "result" in created, created
        model_selection.apply(server.runtime, "gpt-5.4", "ask")
        await server._client._event(
            StreamEvent(StreamEventType.ERROR, error=ErrorInfo(
                "http_error" if status else "auth_error", "Not a string classification hint", status, provider_error)),
            session_id=server.runtime.session_id, background=background,
        )
        event = await _event(reader, "error")
        assert event["data"].get("login_provider") == expected
        if expected:
            assert server.runtime.metadata.provider == "claude"
    finally:
        await _close(server, writer)


async def test_cancel_during_success_cleanup_waits_for_listener_close(tmp_path, monkeypatch):
    import threading

    from zeta.core import login_flow

    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    create = login_flow._create_redirect_server
    captured = {}

    def create_server(handler):
        server = create(handler)
        shutdown = server.shutdown
        def slow_shutdown():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            shutdown()
        server.shutdown = slow_shutdown
        return server

    def build(state, challenge, redirect):
        captured.update(state=state, redirect=redirect)
        return "https://authorize.invalid/"

    async def exchange(*args):
        return OAuthTokens("access", "refresh", 4_000_000_000)

    monkeypatch.setattr(login_flow, "_create_redirect_server", create_server)
    monkeypatch.setattr(providers, "build_anthropic_authorization_url", build)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    server, reader, writer, _ = await connect(tmp_path)
    try:
        await rpc(reader, writer, "login_start", provider="claude")
        await callback(captured["redirect"], {"code": "synthetic", "state": captured["state"]})
        await asyncio.wait_for(entered.wait(), 3)
        task = server._client.logins.tasks["claude"]
        # Force cancellation at exactly the cleanup boundary, then exercise
        # the RPC's second cancellation path while cleanup is still pending.
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert (await rpc(reader, writer, "login_cancel", provider="claude"))["result"] == {"state": "cancelled"}
        listener = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", urlsplit(captured["redirect"]).port)
        listener.close()
        await listener.wait_closed()
    finally:
        release.set()
        await _close(server, writer)
