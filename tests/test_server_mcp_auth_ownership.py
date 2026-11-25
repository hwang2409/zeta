"""Connection ownership of headless MCP OAuth over real serve sockets."""

from __future__ import annotations

import asyncio
from itertools import count

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from tests.test_server_replay_table import _backend_factory
from zeta.mcp import oauth
from zeta.mcp.oauth_store import MCPCredentialStore, MCPOAuthToken


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", (
    "foreign-status", "foreign-cancel", "same-name", "different-name", "own-cancel",
    "disconnect", "disconnect-before-launch", "publish", "foreign-logout", "foreign-revision-logout", "shutdown",
))
@pytest.mark.asyncio
async def test_mcp_auth_ownership(tmp_path, transport, row, monkeypatch):
    names = ("github", "other")
    discovery = asyncio.Event()
    complete = {name: asyncio.Event() for name in names}
    cancelled = {name: asyncio.Event() for name in names}
    settled = {name: asyncio.Event() for name in names}
    token = MCPOAuthToken(
        "fake-access", None, None, "Bearer", None, "https://auth.invalid",
        "https://auth.invalid/token", "https://auth.invalid/authorize", "test", None,
        "http://127.0.0.1/callback", "https://mcp.invalid",
    )

    async def authorize(*, server_name, launch_ready, **kwargs):
        task = asyncio.current_task()
        assert task is not None
        task.add_done_callback(lambda _: settled[server_name].set())
        try:
            if row == "disconnect-before-launch" and server_name == "github":
                discovery.set()
            else:
                launch_ready(f"http://127.0.0.1/launch/{server_name}")
            await await_or_fail(complete[server_name].wait(), what=f"finish OAuth for {server_name}")
        except asyncio.CancelledError:
            cancelled[server_name].set()
            raise
        return token

    # test-contract: seam replace the OAuth browser/provider exchange; keep real credential publication.
    monkeypatch.setattr(oauth, "authorize", authorize)
    server = replay_server(tmp_path, transport, _backend_factory)
    clients = []
    ids = count(2)

    async def rpc(client, method, params):
        return await await_or_fail(client.request(next(ids), method, params), what=method)

    async def auth(client, action, name="github", **params):
        reply = await rpc(client, f"mcp_auth_{action}", {"name": name, **params})
        return reply["result"]["data"] if "result" in reply else reply["error"]

    async def connect():
        client = await await_or_fail(connect_replay(server, transport), what="connect OAuth controller")
        clients.append(client)
        hello = await await_or_fail(negotiate(
            client, replay=False, extra=("multi_controller", "mcp_management_v1"),
        ), what="negotiate OAuth controller")
        assert "multi_controller" in hello["result"]["capabilities"]["features"]
        assert "result" in await rpc(client, "new_session", {})
        return client

    try:
        a, b = await connect(), await connect()
        for name in names:
            listed = await rpc(a, "mcp_server_list", {})
            added = await rpc(a, "mcp_server_add", {"revision": listed["result"]["revision"], "mutation": {
                "name": name, "scope": "user", "url": "https://mcp.invalid", "oauth": True,
            }})
            assert "result" in added, added
        if row == "disconnect-before-launch":
            await await_or_fail(a.send(next(ids), "mcp_auth_start", {"name": "github"}), what="start OAuth discovery")
            await await_or_fail(discovery.wait(), what="OAuth discovery entered")
            await await_or_fail(a.close(), what="disconnect before launch")
            await await_or_fail(cancelled["github"].wait(), what="disconnect cancels OAuth discovery")
            return
        assert await auth(a, "start") == {
            "state": "pending", "authorization_url": "http://127.0.0.1/launch/github",
        }
        pending = {"state": "pending"}
        idle = {"state": "idle"}
        if row == "foreign-status":
            assert await auth(b, "status") == idle
        elif row == "foreign-cancel":
            assert await auth(b, "cancel") == idle
            assert await auth(a, "status") == pending
            assert not cancelled["github"].is_set()
        elif row == "same-name":
            refused = await auth(b, "start")
            assert refused == {"code": -32004, "message": "another process is writing this resource",
                               "data": {"code": "operation_busy"}}
        elif row == "different-name":
            assert await auth(b, "start", "other") == {
                "state": "pending", "authorization_url": "http://127.0.0.1/launch/other",
            }
        elif row == "own-cancel":
            assert await auth(a, "cancel") == {"state": "cancelled"}
            assert cancelled["github"].is_set()
        elif row in ("disconnect", "shutdown"):
            assert (await auth(b, "start", "other"))["state"] == "pending"
            if row == "disconnect":
                await await_or_fail(a.close(), what="disconnect A")
                await await_or_fail(cancelled["github"].wait(), what="disconnect cancels A's OAuth")
                assert await auth(b, "status", "other") == pending
                assert not cancelled["other"].is_set()
            else:
                await await_or_fail(server.close(), what="shut down with two OAuth attempts")
                assert all(event.is_set() for event in cancelled.values())
        elif row == "publish":
            complete["github"].set()
            await await_or_fail(settled["github"].wait(), what="settle credential publication")
            assert await auth(a, "status") == {"state": "succeeded"}
            store = MCPCredentialStore(home=tmp_path)
            assert (await await_or_fail(store.load("github"), what="read home credential")).token == token
            assert await auth(b, "logout", scope="user") == {"state": "unauthorized"}
            assert (await await_or_fail(store.load("github"), what="read shared logout")).token is None
        else:
            params = {"scope": "user"}
            if row == "foreign-revision-logout":
                listed = await rpc(b, "mcp_server_list", {})
                params["revision"] = listed["result"]["revision"]
            assert await auth(b, "logout", **params) == {"state": "unauthorized"}
            assert await auth(a, "status") == pending
            assert not cancelled["github"].is_set()
    finally:
        for event in complete.values():
            event.set()
        for client in clients:
            await await_or_fail(client.close(), what="close OAuth socket")
        await await_or_fail(server.close(), what="close OAuth server")
