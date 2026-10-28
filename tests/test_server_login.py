"""Native login RPCs exercise the real listener without contacting OAuth servers."""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import replace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from tests.support.login import block_loop_until_publication_decision_expires
from tests.test_server import _close, _connect, _request
from zeta.accounts import AccountStore
from zeta.core import login_flow
from zeta.core.login_flow import LoginPortInUseError
from zeta.providers import anthropic
from zeta.providers import login as providers
from zeta.providers.auth import OAuthTokens
from zeta.providers.factory import credential_store
from zeta.server import ZetaServer
from zeta.server import login as server_login
from zeta.server.login import REQUESTS, Logins


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("ZETA_ALLOW_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "_keychain_claude_tokens", lambda: None)
    real_build_login_provider = server_login.build_login_provider
    monkeypatch.setattr(
        server_login,
        "build_login_provider",
        lambda provider, home: replace(
            real_build_login_provider(provider, home), callback_port=0
        ),
    )
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


@pytest.mark.parametrize("server_provider,version", [("claude", "1.0")])
async def test_login_is_hidden_and_rejected_for_legacy_protocol(tmp_path, server_provider, version):
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
        expected_path = "/callback" if provider == "claude" else "/auth/callback"
        parsed_redirect = urlsplit(captured["redirect"])
        assert parsed_redirect.path == expected_path
        assert parsed_redirect.port is not None and parsed_redirect.port > 0
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
        saved = await asyncio.to_thread(credential_store, provider, home=server.home)
        assert await asyncio.to_thread(saved.read) == expected
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
        saved = await asyncio.to_thread(credential_store, "claude", home=server.home)
        assert await asyncio.to_thread(saved.read) is None
        if outcome != "disconnect":
            monkeypatch.setattr(providers, "build_anthropic_authorization_url", lambda *args: "https://authorize.invalid/")
            retry = (await rpc(reader, writer, "login_start", provider="claude"))["result"]
            assert retry["state"] == "pending"
    finally:
        await _close(server, writer)


async def test_serve_login_publication_does_not_block_event_loop(
    tmp_path, monkeypatch
):
    expected = OAuthTokens("access", "refresh", 4_000_000_000)
    callback_ran = threading.Event()
    exchange_started = threading.Event()
    release_lock = threading.Event()
    lock_ready = threading.Event()
    callback_before_release: list[bool] = []
    captured = {}

    def build(state, _challenge, redirect):
        captured.update(state=state, redirect=redirect)
        return "https://authorize.invalid/"

    async def exchange(*_args):
        exchange_started.set()
        asyncio.get_running_loop().call_soon(callback_ran.set)
        return expected

    monkeypatch.setattr(providers, "build_anthropic_authorization_url", build)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    provider = replace(
        await asyncio.to_thread(
            providers.build_login_provider, "anthropic", tmp_path / "zeta"
        ),
        callback_port=0,
    )
    monkeypatch.setattr(server_login, "build_login_provider", lambda *_args: provider)
    monkeypatch.setattr(server_login, "LOGIN_PUBLICATION_TIMEOUT_SECONDS", 0.05)

    def hold_lifecycle_lock() -> None:
        with provider.credential_store.store.lifecycle_guard(
            provider.credential_store.account
        ):
            lock_ready.set()
            if not exchange_started.wait(timeout=5):
                callback_before_release.append(False)
                return
            callback_before_release.append(callback_ran.wait(timeout=1))
            if callback_before_release[-1]:
                release_lock.wait()

    holder = threading.Thread(target=hold_lifecycle_lock)
    holder.start()
    assert lock_ready.wait(timeout=5)
    manager = Logins(tmp_path / "zeta")
    url = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(manager._run("claude", url))
    try:
        await asyncio.wait_for(url, 3)
        assert await callback(
            captured["redirect"], {"code": "synthetic", "state": captured["state"]}
        ) == 200
        result = await asyncio.wait_for(task, 3)
    finally:
        release_lock.set()
        holder.join(timeout=5)

    assert not holder.is_alive()
    assert callback_before_release == [True]
    assert result == {
        "state": "failed",
        "error": {
            "code": "login_lock_timeout",
            "message": "Sign-in could not save credentials because the account is busy. Please try again.",
        },
    }


async def test_serve_login_decision_expiry_reports_failure_and_releases_locks(
    tmp_path, monkeypatch
):
    home = tmp_path / "zeta"
    expected = OAuthTokens("access", "refresh", 4_000_000_000)
    captured = {}
    loop = asyncio.get_running_loop()
    block_loop, loop_blocked = block_loop_until_publication_decision_expires(
        monkeypatch, loop
    )

    def build(state, _challenge, redirect):
        captured.update(state=state, redirect=redirect)
        return "https://authorize.invalid/"

    async def exchange(*_args):
        return expected

    provider = replace(
        await asyncio.to_thread(providers.build_login_provider, "anthropic", home),
        callback_port=0,
        build_authorization_url=build,
        exchange_authorization_code=exchange,
    )
    real_save = provider.credential_store.save

    def observed_save(*args, **kwargs):
        loop.call_soon_threadsafe(block_loop)
        assert loop_blocked.wait(timeout=1)
        return real_save(*args, **kwargs)

    monkeypatch.setattr(provider.credential_store, "save", observed_save)
    monkeypatch.setattr(server_login, "build_login_provider", lambda *_args: provider)
    monkeypatch.setattr(login_flow, "_PUBLICATION_DECISION_TIMEOUT_SECONDS", 0.05)
    server, reader, writer, _ = await connect(tmp_path)
    try:
        assert (await rpc(reader, writer, "login_start", provider="claude"))["result"]["state"] == "pending"
        await callback(
            captured["redirect"],
            {"code": "synthetic", "state": captured["state"]},
        )
        result = await asyncio.wait_for(server._client.logins.tasks["claude"], 3)

        assert result == {
            "state": "failed",
            "error": {
                "code": "login_save_failed",
                "message": "Login could not be saved; try again.",
            },
        }
        assert await asyncio.wait_for(
            asyncio.to_thread(provider.credential_store.read), 1
        ) is None

        def acquire_locks() -> None:
            with provider.credential_store.store.credential_guard(
                provider.credential_store.account, timeout=0.5
            ):
                pass
            with AccountStore(home).provider_login_lock("anthropic"):
                pass

        await asyncio.wait_for(asyncio.to_thread(acquire_locks), 1)
    finally:
        await _close(server, writer)


async def test_cancelled_serve_login_never_publishes_and_releases_locks(
    tmp_path, monkeypatch
):
    home = tmp_path / "zeta"
    old = OAuthTokens("old-access", "old-refresh", 4_000_000_000)
    new = OAuthTokens("new-access", "new-refresh", 4_000_000_000)
    exchanged = iter((old, new))
    captured: list[dict[str, str]] = []
    save_entered = threading.Event()
    save_finished = threading.Event()
    lock_ready = threading.Event()
    release_lock = threading.Event()
    published: list[OAuthTokens] = []

    def build(state, _challenge, redirect):
        captured.append({"state": state, "redirect": redirect})
        return "https://authorize.invalid/"

    async def exchange(*_args):
        return next(exchanged)

    provider = replace(
        await asyncio.to_thread(providers.build_login_provider, "anthropic", home),
        callback_port=0,
        build_authorization_url=build,
        exchange_authorization_code=exchange,
    )
    real_save = provider.credential_store.save
    real_publish = provider.credential_store._publish

    def observed_save(*args, **kwargs):
        save_entered.set()
        try:
            return real_save(*args, **kwargs)
        finally:
            save_finished.set()

    def observed_publish(home_fd, value):
        published.append(value)
        return real_publish(home_fd, value)

    monkeypatch.setattr(provider.credential_store, "save", observed_save)
    monkeypatch.setattr(provider.credential_store, "_publish", observed_publish)
    monkeypatch.setattr(providers, "build_anthropic_authorization_url", build)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    monkeypatch.setattr(server_login, "build_login_provider", lambda *_args: provider)

    def hold_lifecycle_lock() -> None:
        with provider.credential_store.store.lifecycle_guard(
            provider.credential_store.account
        ):
            lock_ready.set()
            release_lock.wait()

    fd_before = len(os.listdir("/dev/fd"))
    holder = threading.Thread(target=hold_lifecycle_lock)
    holder.start()
    assert lock_ready.wait(timeout=5)
    manager = Logins(home)
    first_url = asyncio.get_running_loop().create_future()
    first = asyncio.create_task(manager._run("claude", first_url))
    try:
        await asyncio.wait_for(first_url, 3)
        assert await callback(
            captured[0]["redirect"],
            {"code": "first", "state": captured[0]["state"]},
        ) == 200
        assert await asyncio.to_thread(save_entered.wait, 5)
        first.cancel()
        assert not first.done()
        release_lock.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.to_thread(save_finished.wait, 5)

        second_url = asyncio.get_running_loop().create_future()
        second = asyncio.create_task(manager._run("claude", second_url))
        await asyncio.wait_for(second_url, 3)
        assert await callback(
            captured[1]["redirect"],
            {"code": "second", "state": captured[1]["state"]},
        ) == 200
        assert await asyncio.wait_for(second, 3) == {"state": "succeeded"}
    finally:
        release_lock.set()
        holder.join(timeout=5)

    assert not holder.is_alive()
    assert published == [new]
    assert await asyncio.to_thread(provider.credential_store.read) == new
    with AccountStore(home).provider_login_lock("anthropic"):
        pass
    assert len(os.listdir("/dev/fd")) == fd_before


async def test_login_port_conflict_has_specific_failure(tmp_path, monkeypatch):
    async def raise_port_conflict(*args, **kwargs):
        raise LoginPortInUseError("codex", 1455)

    monkeypatch.setattr(server_login, "run_login", raise_port_conflict)
    manager = Logins(tmp_path / "zeta")

    result = await manager._run("codex", asyncio.get_running_loop().create_future())

    assert result == {
        "state": "failed",
        "error": {
            "code": "login_port_in_use",
            "message": "Codex login cannot start because port 1455 is already in use. "
            "Close any other Codex login and retry.",
        },
    }


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
    from tests.support.fake_backend import FakeBackend
    from tests.test_server import _event
    from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType
    from zeta.server import model_selection

    server = ZetaServer(home=tmp_path / "zeta", port=0, provider="claude", model="claude-sonnet-4-6",
                        backend_factory=lambda provider, model, home: (FakeBackend([]), model))
    reader, writer = await _connect(server)
    try:
        await rpc(reader, writer, "hello", protocol_version=version)
        created = await rpc(reader, writer, "new_session")
        assert "result" in created, created
        await model_selection.apply(server.runtime, "gpt-5.4", "ask")
        await server._client._event(
            StreamEvent(StreamEventType.ERROR, error=ErrorInfo(
                "http_error" if status else "auth_error", "Not a string classification hint", status, provider_error)),
            session_id=server.runtime.session_id, background=background,
        )
        event = await _event(reader, "error")
        assert event["data"].get("login_provider") == expected
        if expected:
            assert server.runtime.metadata.provider == (
                "codex" if status is None else "claude"
            )
    finally:
        await _close(server, writer)


async def test_cancel_during_provider_lock_release_reports_success(
    tmp_path, monkeypatch
):
    expected = OAuthTokens("access", "refresh", 4_000_000_000)
    captured = {}
    release_entered = threading.Event()
    release_exit = threading.Event()

    class LoginLock:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            release_entered.set()
            assert release_exit.wait(3)

    def provider_login_lock(_self, _provider):
        return LoginLock()

    def build(state, _challenge, redirect):
        captured.update(state=state, redirect=redirect)
        return "https://authorize.invalid/"

    async def exchange(*_args):
        return expected

    monkeypatch.setattr(AccountStore, "provider_login_lock", provider_login_lock)
    monkeypatch.setattr(providers, "build_anthropic_authorization_url", build)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    server, reader, writer, _ = await connect(tmp_path)
    try:
        assert (await rpc(reader, writer, "login_start", provider="claude"))["result"]["state"] == "pending"
        await callback(captured["redirect"], {"code": "synthetic", "state": captured["state"]})
        assert await asyncio.to_thread(release_entered.wait, 3)

        task = server._client.logins.tasks["claude"]
        task.cancel()
        task.cancel()
        release_exit.set()

        assert (await rpc(reader, writer, "login_cancel", provider="claude"))["result"] == {"state": "succeeded"}
        assert task.cancelling() == 0
        saved = await asyncio.to_thread(credential_store, "claude", home=server.home)
        assert await asyncio.to_thread(saved.read) == expected
    finally:
        release_exit.set()
        await _close(server, writer)


async def test_cancel_during_success_cleanup_reports_published_login(tmp_path, monkeypatch):
    import threading

    from zeta.core import login_flow

    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    create = login_flow._create_redirect_server
    captured = {}

    def create_server(handler, *, port):
        server = create(handler, port=port)
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
        # Publication already committed, so cancellation cannot report that
        # credentials were left unchanged.
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert (await rpc(reader, writer, "login_cancel", provider="claude"))["result"] == {"state": "succeeded"}
        listener = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", urlsplit(captured["redirect"]).port)
        listener.close()
        await listener.wait_closed()
    finally:
        release.set()
        await _close(server, writer)
