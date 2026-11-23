"""Slice 6a through the authenticated serve WebSocket."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer, protocol, read_catalog
from zeta.storage_format import WRITER_EPOCH

FEATURE = "state_read_v1"
METHOD = "state_catalog"
PROJECT_METHODS = ["state_projects", "state_memory", "state_memory_detail", "state_inbox",
                   "state_inbox_detail", "state_attention"]
SESSION_METHODS = {"state_sessions", "state_session", "state_agents", "state_background_tasks"}
HOME_METHODS = {"state_settings", "state_accounts", "state_storage", "state_retention"}
KINDS = {"profiles": "agents", "skills": "skills", "commands": "commands"}
MARKER = "PRIVATE-DEFINITION-BODY"
INVALID = {"code": -32602, "message": "invalid params", "data": {"code": "invalid_params"}}


def _stat(home):
    return {path.relative_to(home): path.stat().st_mtime_ns for path in (home, *home.rglob("*"))}


@asynccontextmanager
async def _connect(server, *, observer=False, negotiated=True):
    ws = server._websocket
    assert ws is not None
    token = ws.observer_token if observer else ws.controller_token
    async with connect(
        f"ws://127.0.0.1:{ws.port}/", subprotocols=["zeta.v1", f"zeta.bearer.{token}"],
    ) as websocket:
        client = ServeClient(websocket=websocket)
        features = ([FEATURE] if negotiated else []) + (["serve_observers", "ping"] if observer else [])
        response = await client.request("hello", "hello", {
            "protocol_version": "1.2", "features": features,
            "role": "observer" if observer else "controller",
        })
        assert MARKER not in json.dumps(response)
        assert set(response["result"]["capabilities"].get("features", [])) == set(features)
        assert {method for method in response["result"]["capabilities"]["requests"] if method.startswith("state_")} == (
            {METHOD, *SESSION_METHODS, *HOME_METHODS, *(m for m in PROJECT_METHODS if not observer or not m.endswith("_detail"))}
            if negotiated else set()
        )
        yield client


async def _call(client, **params):
    raw = await client.request_raw("page", METHOD, params)
    assert MARKER not in raw
    return json.loads(raw)


@pytest.mark.parametrize("case", (
    "profiles", "skills", "commands", "tie", "frame", "shorter_cursor", "cap", "invalid", "unnegotiated",
    "missing", "future", "malformed", "responsive", "retry", "stale",
))
async def test_state_catalog_websocket(tmp_path: Path, monkeypatch, case):
    home = tmp_path / "home"
    home.mkdir()
    kind = case if case in KINDS else "profiles"
    folder = home / KINDS[kind]
    folder.mkdir()
    def seed(name, description, index):
        front = "" if kind == "commands" else f"name: '{name}'\n"
        text = f"---\n{front}description: '{description}'\n---\n{MARKER}\n"
        if kind == "skills":
            path = folder / name / "SKILL.md"
            path.parent.mkdir()
        else:
            path = folder / f"{index}.md"
        path.write_text(text)

    if case == "shorter_cursor":
        for i, (name, description) in enumerate([
            ("!a", "meta"), ("!b", "meta"), ("!c" * 750, "meta"), ("!d", "meta"), ("!e", "x" * 1400),
        ]):
            seed(name, description, i)
    elif case == "cap":
        for i in reversed(range(1001)):
            seed(f"!row-{i:04}", "row", i)
    else:
        for i in range(4):
            seed(f"!s6a-{i}", f"{'x' * 480}{i}", f"!s6a-{i}")
    manifest = home / "storage-format.json"
    if case != "missing":
        manifest.write_text("{}" if case == "malformed" else json.dumps({
            "writer_epoch": WRITER_EPOCH + (case == "future"),
        }))
    server = ZetaServer(
        home=tmp_path / "serve-home", socket_path=short_socket_path(home), ws_port=0,
        ws_discovery=tmp_path / "websocket.json", provider="codex",
    )
    await await_or_fail(server.start(), what="serve start")
    # Startup and the test fixtures use writer leases. Give the actual request a pristine catalog home.
    monkeypatch.setattr(server, "home", home)
    try:
        async with _connect(server, observer=case == "skills", negotiated=case != "unnegotiated") as client:
            before = _stat(home)
            if case in {"missing", "future", "malformed"}:
                reply = await _call(client, kind=kind)
                assert _stat(home) == before, "catalog read created or touched home state"
                error = reply["error"]
                if case == "missing":
                    assert error == {"code": -32005, "message": "state unavailable", "data": {
                        "code": "state_unavailable", "retryable": True,
                    }}
                else:
                    assert error == {"code": -32000, "message": "stored state is invalid", "data": {
                        "code": "corrupt_state", "resource": "storage-format.json",
                    }}
            elif case == "unnegotiated":
                assert (await _call(client, kind=kind))["error"]["code"] == -32601
            elif case == "invalid":
                first = (await _call(client, kind=kind, limit=1))["result"]
                for params in ({}, {"kind": []}, {"kind": "unknown"}, {"kind": kind, "extra": 0},
                               {"kind": kind, "cursor": "bad"}, {"kind": kind, "cursor": []},
                               {"kind": "skills", "cursor": first["next_cursor"]},
                               {"kind": kind, "cursor": first["next_cursor"] + "!!!!"},
                               *({"kind": kind, "limit": value} for value in (True, None, 0, 201, 1.5))):
                    assert (await _call(client, **params))["error"] == INVALID
            elif case == "responsive":
                loop = asyncio.get_running_loop()
                entered = asyncio.Event()
                released = threading.Event()
                finished = threading.Event()
                original = read_catalog.discover_session_agents

                def parked(**kwargs):
                    loop.call_soon_threadsafe(entered.set)
                    released.wait(5)
                    finished.set()
                    return original(**kwargs)

                monkeypatch.setattr(read_catalog, "discover_session_agents", parked)
                async with _connect(server, observer=True) as other:
                    request = asyncio.create_task(_call(client, kind=kind))
                    try:
                        await await_or_fail(entered.wait(), what="owner entered")
                        assert "result" in await other.request("ping", "ping")
                        assert not finished.is_set(), "serve could not answer ping while the owner was blocked"
                    finally:
                        released.set()
                        assert "result" in await await_or_fail(request, what="catalog read")
            elif case in {"retry", "stale"}:
                original = read_catalog.discover_session_agents
                calls = 0

                def changing(**kwargs):
                    nonlocal calls
                    calls += 1
                    if case == "stale" or calls == 1:
                        (folder / f"!added-{calls}.md").write_text(
                            f"---\nname: '!added-{calls}'\ndescription: added\n---\n{MARKER}\n"
                        )
                    return original(**kwargs)

                monkeypatch.setattr(read_catalog, "discover_session_agents", changing)
                reply = await _call(client, kind=kind)
                assert calls == 2
                if case == "retry":
                    assert reply["result"]["items"][0]["name"] == "!added-1"
                else:
                    assert reply["error"] == {"code": -32009, "message": "snapshot changed; retry", "data": {
                        "code": "stale_state_snapshot", "retryable": True,
                    }}
            elif case == "shorter_cursor":
                three = (await _call(client, kind=kind, limit=3))["result"]
                four = (await _call(client, kind=kind, limit=4))["result"]
                codec = protocol.FrameCodec()
                budget = codec.response_size("page", four)
                assert codec.response_size("page", three) > budget
                monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", budget)
                first = (await _call(client, kind=kind))["result"]
                assert first == four
                last = (await _call(client, kind=kind, cursor=first["next_cursor"]))["result"]
                assert last["items"][0] == {"name": "!e", "description": "x" * 1400, "source": "home"}
            elif case == "cap":
                rows = [
                    {"name": f"!row-{i:04}", "description": "row", "source": "home"} for i in range(1000)
                ]
                page = (await _call(client, kind=kind))["result"]
                assert page["items"] == rows[:50]
                seen = page["items"]
                while page["next_cursor"] is not None:
                    page = (await _call(client, kind=kind, limit=200, cursor=page["next_cursor"]))["result"]
                    seen += page["items"]
                assert seen == rows
            else:
                if case == "tie":
                    original = read_catalog.discover_session_agents

                    def tied(**kwargs):
                        catalog = original(**kwargs)
                        home_entry = catalog.find("!s6a-0")
                        built_in = replace(home_entry, source="packaged", selection_guidance="tie")
                        return replace(catalog, agents=(*catalog.agents, built_in))

                    monkeypatch.setattr(read_catalog, "discover_session_agents", tied)
                expected = (await _call(client, kind=kind, limit=200))["result"]
                assert set(expected) == {"kind", "items", "next_cursor"}
                assert expected["kind"] == kind and expected["next_cursor"] is None
                items = expected["items"]
                assert [row["name"] for row in items if row["source"] == "home"] == [f"!s6a-{i}" for i in range(4)]
                assert items == sorted(items, key=lambda row: (row["name"], row["source"]))
                assert all(set(row) == {"name", "description", "source"} for row in items)
                assert {row["source"] for row in items} == ({"home"} if kind == "skills" else {"built_in", "home"})
                limit = 1
                if case == "frame":
                    codec = protocol.FrameCodec()
                    two = (await _call(client, kind=kind, limit=2))["result"]
                    three = (await _call(client, kind=kind, limit=3))["result"]
                    budget = codec.response_size("page", two)
                    assert codec.response_size("page", three) > budget
                    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", budget)
                    limit = 200
                first = page = (await _call(client, kind=kind, limit=limit))["result"]
                seen = list(page["items"])
                while page["next_cursor"] is not None:
                    page = (await _call(client, kind=kind, limit=limit, cursor=page["next_cursor"]))["result"]
                    assert page["items"] and len(seen) < len(items)
                    seen.extend(page["items"])
                assert seen == items
                if case == "tie":
                    assert [(row["name"], row["source"]) for row in seen[:2]] == [
                        ("!s6a-0", "built_in"), ("!s6a-0", "home"),
                    ]
                if case == "frame":
                    assert first == two
                    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 512)
                    assert (await _call(client, kind=kind))["error"]["code"] == -32007
                assert _stat(home) == before, "catalog read created or touched home state"
    finally:
        await await_or_fail(server.close(), what="serve close")
