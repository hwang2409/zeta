"""MCP management permissions and on-disk compare-and-swap through serve."""

from __future__ import annotations

import asyncio
import fcntl
import json

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeContract


async def _connect(contract, role="controller"):
    client = await contract.connect(role)
    hello = await client.request("hello", "hello", {
        "protocol_version": "1.2", "features": ["serve_observers", "mcp_management_v1"], "role": role,
    })
    assert "mcp_management_v1" in hello["result"]["capabilities"]["features"]
    return client


@pytest.mark.asyncio
async def test_observer_can_list_but_cannot_mutate(tmp_path):
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        client = await _connect(contract, "observer")
        listed = await client.request("list", "mcp_server_list")
        assert listed["result"]["data"] == {"items": []}
        for action in ("add", "remove", "set_enabled", "set_trusted"):
            method = f"mcp_server_{action}"
            reply = await client.request(action, method, {"mutation": "intentionally malformed"})
            assert reply["error"]["code"] == -32008
            assert reply["error"]["data"] == {
                "code": "observer_read_only", "reason": "observer_read_only", "method": method,
            }
        assert not (tmp_path / "home" / "mcp.json").exists()
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_two_serve_processes_cannot_both_write_the_same_revision(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    config = shared / "user-mcp.json"
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(config))
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
        (root / ".zeta").symlink_to(shared, target_is_directory=True)
    first = await ServeContract.start(roots[0], "websocket")
    try:
        second = await ServeContract.start(roots[1], "websocket")
        try:
            clients = [await _connect(first), await _connect(second)]
            revisions = [(await client.request("list", "mcp_server_list"))["result"]["revision"] for client in clients]
            assert revisions[0] == revisions[1]
            replies = await await_or_fail(asyncio.gather(*(
                client.request("add", "mcp_server_add", {
                    "revision": revision, "mutation": {"scope": "user", "name": name, "command": "echo"},
                })
                for client, revision, name in zip(clients, revisions, ("a", "b"), strict=True)
            )), what="competing MCP writes")
            winner, = [reply["result"] for reply in replies if "result" in reply]
            stale, = [reply["error"] for reply in replies if "error" in reply]
            assert stale["code"] == -32009
            assert stale["data"] == {"code": "stale_mcp_revision", "reason": "stale_mcp_revision"}
            name, = json.loads(config.read_text())["servers"]
            assert winner["data"]["items"][0]["name"] == name
            assert winner["revision"] != revisions[0]

            with (shared / ".user-mcp.json.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = await clients[0].request("busy", "mcp_server_remove", {
                    "revision": winner["revision"], "mutation": {"name": name, "scope": "user"},
                })
            assert blocked["error"]["code"] == -32004
            assert blocked["error"]["data"]["code"] == "operation_busy"
            assert name in json.loads(config.read_text())["servers"]
        finally:
            await second.close()
    finally:
        await first.close()
