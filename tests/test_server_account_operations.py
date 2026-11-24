"""Forced orderings across negotiated controllers using one account store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from urllib.parse import urlencode

import pytest

from tests.support.credentials import named_account_credentials
from tests.support.owned import await_or_fail
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from tests.test_server_login import callback
from tests.test_server_replay_table import _backend_factory
from zeta.accounts import AccountId, AccountStore, AccountUsage
from zeta.accounts.session import SessionAccountContext
from zeta.core import login_flow
from zeta.oauth import OAuthTokens
from zeta.providers import login as providers
from zeta.server import accounts
from zeta.server import login as server_login
from zeta.server.runtime import ServerRuntime
from zeta.server.server import _Client


@asynccontextmanager
async def controllers(home, transport, monkeypatch, count=2, *, multi=True):
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(home))
    for provider, name in (("codex", "first"), ("codex", "second"), ("codex", "third"), ("anthropic", "target"), ("anthropic", "other")):
        await asyncio.to_thread(
            named_account_credentials(home, provider, name).save,
            OAuthTokens(*("opaque-fixture",) * 2, 4_000_000_000),
        )
    await asyncio.to_thread(AccountStore(home).set_default, AccountId("codex", "first"))
    def backend(runtime, provider, model, home, **kwargs):
        fake, model = _backend_factory(provider, model, home)
        fake.model = model
        return fake, model

    monkeypatch.setattr(ServerRuntime, "_build_backend", backend)
    server = replay_server(home, transport, _backend_factory)
    clients, sessions = [], []
    await await_or_fail(server.start(), what="start serve")
    try:
        for _ in range(count):
            client = await await_or_fail(connect_replay(server, transport), what="connect controller")
            clients.append(client)
            extra = ("multi_controller", "account_list_idle_v1") if multi else ()
            hello = await negotiate(client, replay=False, extra=extra)
            assert ("multi_controller" in hello["result"]["capabilities"]["features"]) == multi
            created = await client.request(2, "new_session", {})
            sessions.append(created["result"]["session"]["session_id"])
        yield server, clients, sessions
    finally:
        await await_or_fail(
            asyncio.gather(*(client.close() for client in clients)), what="close controllers"
        )
        await await_or_fail(server.close(), what="close serve")


async def _empty_usage():
    return {}


def record(server, session):
    return next(record for record in server.controllers.records() if record.runtime.opened is not None and record.runtime.session_id == session)


def selected(server, session):
    return record(server, session).runtime.loop.account_context.snapshot()["codex"]


async def start_login(client, monkeypatch):
    captured = {}

    def authorization_url(state, challenge, redirect):
        captured.update(state=state, redirect=redirect)
        return "https://authorize.invalid/?" + urlencode({"state": state})

    async def exchange(*args):
        return OAuthTokens(*("opaque-fixture",) * 2, 4_000_000_000)

    build_provider = server_login.build_login_provider
    monkeypatch.setattr(
        server_login, "build_login_provider",
        lambda *args: replace(build_provider(*args), callback_port=0),
    )
    monkeypatch.setattr(providers, "build_anthropic_authorization_url", authorization_url)
    monkeypatch.setattr(providers, "exchange_anthropic_authorization_code", exchange)
    result = await client.request(5, "login_start", {"provider": "claude", "account": "anthropic:target"})
    assert result.get("result", {}).get("state") == "pending"
    status = await await_or_fail(
        callback(captured["redirect"], {"code": "fixture", "state": captured["state"]}),
        what="finish synthetic browser callback",
    )
    assert status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("first,second", (
    ("switch", "logout"), ("switch", "refresh"), ("publication", "logout"), ("switch", "switch"),
))
async def test_account_intervals_do_not_overlap(tmp_path, monkeypatch, transport, first, second):
    async with controllers(tmp_path, transport, monkeypatch) as (server, (a, b), (sid_a, sid_b)):
        entered, release, contender = asyncio.Event(), asyncio.Event(), asyncio.Event()
        order = []
        original_commit = SessionAccountContext._commit_and_apply
        original_publish = login_flow._publish_credentials
        original_switch = accounts.switch
        original_logout = accounts.logout
        refresh_body = "refresh_accounts" if hasattr(accounts, "refresh_accounts") else "refresh"
        original_refresh = getattr(accounts, refresh_body)
        original_dispatch = _Client._dispatch
        context_a = record(server, sid_a).runtime.loop.account_context
        runtime_b = record(server, sid_b).runtime
        backend_a = record(server, sid_a).runtime.loop.backend
        backend_b = runtime_b.loop.backend

        async def first_interval():
            entered.set()
            await await_or_fail(release.wait(), what="release first account interval")
            order.append("first released")

        async def commit(context, *args):
            if context is context_a:
                await first_interval()
            return await original_commit(context, *args)

        async def publish(*args):
            await first_interval()
            return await original_publish(*args)

        async def switch(runtime, *args):
            if runtime is runtime_b:
                order.append("second entered")
            return await original_switch(runtime, *args)

        async def logout(*args):
            order.append("second entered")
            return await original_logout(*args)

        async def refresh(*args):
            order.append("second entered")
            return await original_refresh(*args)

        async def no_usage_network(self):
            return {}

        async def dispatch(client, request_id, method, params):
            if request_id == 6:
                # Signal the test after dispatch has run to its first suspension.
                asyncio.get_running_loop().call_soon(contender.set)
            return await original_dispatch(client, request_id, method, params)

        monkeypatch.setattr(SessionAccountContext, "_commit_and_apply", commit)
        monkeypatch.setattr(login_flow, "_publish_credentials", publish)
        monkeypatch.setattr(accounts, "switch", switch)
        monkeypatch.setattr(accounts, "logout", logout)
        monkeypatch.setattr(accounts, refresh_body, refresh)
        monkeypatch.setattr(AccountUsage, "refresh_stale", no_usage_network)
        monkeypatch.setattr(_Client, "_dispatch", dispatch)
        target = "anthropic:target" if first == "publication" else "codex:second"
        try:
            if first == "publication":
                await start_login(a, monkeypatch)
            else:
                await a.send(5, "account_switch", {"session_id": sid_a, "account": target})
            await await_or_fail(entered.wait(), what="first account interval entered")
            params = {"session_id": sid_b}
            if second != "refresh":
                params["account"] = "codex:third" if second == "switch" else target
            await b.send(6, f"account_{second}", params)
            await await_or_fail(contender.wait(), what="dispatch competing account request")
        finally:
            release.set()

        result = await b.response(6)
        assert order == ["first released", "second entered"], "competing account operation entered before the first released"
        if first == "publication":
            login_task = next(c.logins.tasks["claude"] for c in server.connections.clients() if "claude" in c.logins.tasks)
            await await_or_fail(asyncio.shield(login_task), what="settle login")
            assert (await a.request(7, "login_status", {"provider": "claude"}))["result"] == {"state": "succeeded"}
        else:
            assert "result" in await a.response(5)

        assert selected(server, sid_a) == AccountId("codex", "first" if first == "publication" else "second")
        assert selected(server, sid_b) == AccountId("codex", "third" if second == "switch" else "first")
        assert (record(server, sid_a).runtime.loop.backend is not backend_a) == (first == "switch")
        assert (runtime_b.loop.backend is not backend_b) == (second == "switch")
        if (first, second) == ("switch", "logout"):
            assert result["error"]["code"] == -32005
            assert await asyncio.to_thread(AccountStore(tmp_path).contains, AccountId("codex", "second"))
        else:
            assert "result" in result
        if first == "publication":
            assert not await asyncio.to_thread(AccountStore(tmp_path).contains, AccountId("anthropic", "target"))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("account_logout", "account_refresh"))
async def test_unrelated_busy_record_does_not_block_account_operation(tmp_path, monkeypatch, method):
    async def no_usage_network(self):
        return {}

    monkeypatch.setattr(AccountUsage, "refresh_stale", no_usage_network)
    async with controllers(tmp_path, "unix", monkeypatch, count=3) as (server, (a, _, _), (sid_a, sid_b, sid_c)):
        affected = record(server, sid_a).control
        unrelated = record(server, sid_b).control
        transition = record(server, sid_c).control
        params = {"session_id": sid_a, "account": "anthropic:target"}
        if method == "account_logout":
            selected_target = await a.request(3, "account_switch", params)
            assert "result" in selected_target
            monkeypatch.setattr(affected, "_turn_task", asyncio.current_task())
            try:
                response = await a.request(4, method, params)
                assert response["error"]["code"] == -32004
            finally:
                monkeypatch.setattr(affected, "_turn_task", None)
            params["account"] = "anthropic:other"
        # The current task remains active across both wire replies.
        monkeypatch.setattr(unrelated, "_turn_task", asyncio.current_task())
        try:
            if method == "account_logout":
                with transition.reserve_session_swap():
                    response = await a.request(5, method, params)
                assert response["error"]["code"] == -32004
            response = await a.request(6, method, params)
            assert "result" in response, "unrelated running record blocked account operation"
            assert unrelated.state_change_busy()
        finally:
            monkeypatch.setattr(unrelated, "_turn_task", None)


@pytest.mark.asyncio
async def test_legacy_cancel_holds_owner_until_switch_worker_settles(tmp_path, monkeypatch):
    async with controllers(tmp_path, "unix", monkeypatch, count=1, multi=False) as (server, (peer,), (sid,)):
        entered, cancel_seen = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        order = []
        original_worker = SessionAccountContext._locked_apply_worker
        original_write = _Client._write_physical
        responses = []

        async def capture(client, payload):
            if json.loads(payload).get("id") in {3, 4, 8, 9}:
                responses.append(payload)
            return await original_write(client, payload)

        monkeypatch.setattr(_Client, "_write_physical", capture)
        params = {"session_id": sid, "account": "codex:first"}
        assert "result" in await peer.request(3, "account_switch", params)
        assert (await peer.request(4, "account_switch", {"session_id": sid}))["error"]["code"] == -32602
        assert (await peer.request(8, "account_logout", params))["error"]["code"] == -32005
        monkeypatch.setattr(AccountUsage, "refresh_stale", lambda self: _empty_usage())
        assert "result" in await peer.request(9, "account_refresh", {"session_id": sid})
        print("legacy account frame sha256:", hashlib.sha256(b"".join(responses)).hexdigest())

        def worker(context, event_loop, session_id, cancelled, operation, pins):
            loop.call_soon_threadsafe(entered.set)
            assert cancelled.wait(10), "legacy cancellation did not reach the worker"
            loop.call_soon_threadsafe(cancel_seen.set)
            assert release.wait(10), "legacy switch worker was not released"
            try:
                return original_worker(context, event_loop, session_id, cancelled, operation, pins)
            finally:
                order.append("switch worker settled")

        refresh_body = "refresh_accounts" if hasattr(accounts, "refresh_accounts") else "refresh"
        original_refresh = getattr(accounts, refresh_body)

        async def refresh_accounts(*args):
            order.append("refresh started")
            return await original_refresh(*args)

        async def no_usage_network(self):
            return {}

        monkeypatch.setattr(accounts, refresh_body, refresh_accounts)
        monkeypatch.setattr(AccountUsage, "refresh_stale", no_usage_network)
        monkeypatch.setattr(SessionAccountContext, "_locked_apply_worker", worker)
        await peer.send(5, "account_switch", {"session_id": sid, "account": "codex:second"})
        refresh = None
        try:
            await await_or_fail(entered.wait(), what="legacy switch worker entered")
            client, = server.connections.clients()
            handler = client._run_task
            assert handler is not None
            handler.cancel()
            await await_or_fail(cancel_seen.wait(), what="legacy worker observed cancellation")
            # Legacy sockets dispatch one request at a time. Exercise a queued
            # refresh at the dispatcher while that socket's switch is settling.
            dispatched = asyncio.Event()

            async def contender():
                asyncio.get_running_loop().call_soon(dispatched.set)
                return await client._dispatch(6, "account_refresh", {"session_id": sid})

            refresh = asyncio.create_task(contender())
            await await_or_fail(dispatched.wait(), what="legacy refresh dispatched")
            assert not handler.done()
        finally:
            release.set()
            if refresh is not None:
                await await_or_fail(refresh, what="legacy refresh completed")
        await await_or_fail(asyncio.gather(handler, return_exceptions=True), what="legacy switch settled")
        assert record(server, sid).runtime.loop.account_context.account_state_unknown
        assert order == ["switch worker settled", "refresh started"], "legacy refresh overlapped cancelled switch"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("switch", "logout", "refresh"))
@pytest.mark.parametrize("committed", (False, True), ids=("before-commit", "after-commit"))
async def test_cancel_account_operation_at_commit_boundary(tmp_path, monkeypatch, kind, committed):
    async with controllers(tmp_path, "unix", monkeypatch) as (server, (a, b), (sid_a, sid_b)):
        entered, dispatched, cancellation_checkpoint = (asyncio.Event() for _ in range(3))
        release = threading.Event()
        loop = asyncio.get_running_loop()
        mutations = []
        original_prepare = SessionAccountContext._prepare
        original_apply = SessionAccountContext._apply
        original_logout = AccountStore.logout
        original_dispatch = _Client._dispatch
        context_a = record(server, sid_a).runtime.loop.account_context
        blocker = not committed and kind != "switch"

        def gate():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "account worker was not released"

        def prepare(context, *args):
            candidate = original_prepare(context, *args)
            if not committed:
                gate()
            return candidate

        def apply(context, *args):
            if context is context_a and not blocker:
                mutations.append("switch")
            return original_apply(context, *args)

        def logout(store, *args, **kwargs):
            gate()
            mutations.append("logout")
            return original_logout(store, *args, **kwargs)

        async def refresh(self):
            await asyncio.to_thread(gate)
            mutations.append("refresh")
            return {}

        original_commit = SessionAccountContext._commit_and_apply

        async def commit(context, *args):
            if committed:
                await asyncio.to_thread(gate)
            return await original_commit(context, *args)

        async def dispatch(client, request_id, method, params):
            if request_id == 6:
                loop.call_soon(dispatched.set)
            return await original_dispatch(client, request_id, method, params)

        monkeypatch.setattr(SessionAccountContext, "_prepare", prepare)
        monkeypatch.setattr(SessionAccountContext, "_commit_and_apply", commit)
        monkeypatch.setattr(SessionAccountContext, "_apply", apply)
        monkeypatch.setattr(AccountStore, "logout", logout)
        monkeypatch.setattr(AccountUsage, "refresh_stale", refresh)
        monkeypatch.setattr(_Client, "_dispatch", dispatch)
        def client_for(session):
            return next(
                c for c in server.connections.clients()
                if server.controllers.controller_for(c) is record(server, session)
            )
        target = "codex:second" if kind == "switch" else "anthropic:target"
        try:
            if blocker:
                await a.send(5, "account_switch", {"session_id": sid_a, "account": "codex:second"})
                await await_or_fail(entered.wait(), what="blocking switch entered")
                await b.send(6, f"account_{kind}", {"session_id": sid_b, "account": target})
                await await_or_fail(dispatched.wait(), what="queued mutation dispatched")
                handler = client_for(sid_b)._run_task
            else:
                await a.send(5, f"account_{kind}", {"session_id": sid_a, "account": target})
                await await_or_fail(entered.wait(), what="mutation reached cancellation gate")
                handler = client_for(sid_a)._run_task
            assert handler is not None
            handler.cancel()
            loop.call_soon(cancellation_checkpoint.set)
            await await_or_fail(cancellation_checkpoint.wait(), what="deliver request cancellation")
            if not blocker:
                assert not handler.done(), "account worker was abandoned before it settled"
            if blocker:
                await await_or_fail(asyncio.gather(handler, return_exceptions=True), what="cancel queued mutation")
        finally:
            release.set()
        await await_or_fail(asyncio.gather(handler, return_exceptions=True), what="cancelled request settled")
        survivor = a if blocker else b
        response = await survivor.request(7, "account_switch", {"session_id": sid_a if blocker else sid_b})
        assert response["error"]["code"] == -32602
        assert mutations == ([kind] if committed else []), "cancelled mutation did not respect commit boundary"
        if kind == "switch":
            expected = AccountId("codex", "second" if committed else "first")
            assert context_a.snapshot()["codex"] == expected
            assert not context_a.account_state_unknown
        if kind == "logout":
            exists = await asyncio.to_thread(AccountStore(tmp_path).contains, AccountId("anthropic", "target"))
            assert exists != committed
