"""Home state and full state negotiation through the serve websocket."""

import asyncio
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from time import time

import pytest
from websockets.asyncio.client import connect

from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from tests.test_state_catalog import (
    HOME_METHODS,
    INVALID,
    PROJECT_METHODS,
    SESSION_METHODS,
    _connect,
)
from tests.test_state_projects import seed, state_stat
from zeta.mcp.config import home_config_path
from zeta.server import ZetaServer, protocol, read_home_state
from zeta.server.handlers.registry import HANDLERS
from zeta.storage_format import WRITER_EPOCH

SECRETS = tuple(f"fake-S6D-{key}-never-return" for key in (
    "settings", "env", "header", "arg", "access", "refresh", "email", "unknown-usage", "url",
))


def _seed_secrets(home):
    settings, env, header, arg, access, refresh, email, unknown, url = SECRETS
    (home / "settings.toml").write_text(
        f'model = "gpt-5.6-luna"\nprovider = "codex"\nyolo = false\n'
        f'api_key = "{settings}"\nollama_base_url = "http://{settings}.invalid"\n'
        '[agents]\nmax_depth = 2\n[retention]\ndays = 42\n'
    )
    home_config_path(home).write_text(json.dumps({"servers": {
        "alpha": {"transport": "stdio", "command": "not-run", "args": [arg], "env": {"TOKEN": env}},
        "beta": {"transport": "streamable-http", "url": f"https://user:{url}@example.invalid/",
                 "headers": {"Authorization": header}},
    }}))
    catalog = {}
    for name in ("alpha", "beta"):
        catalog[f"codex:{name}"] = {"provider": "codex", "name": name}
        path = home / "accounts" / "codex" / name
        path.mkdir(parents=True)
        (path / "oauth.json").write_text(json.dumps({
            "version": 1, "access_token": access, "refresh_token": refresh,
            "expires_at": time() + (3600 if name == "alpha" else -3600), "email": f"{email}@example.invalid",
        }))
        (path / "usage.json").write_text(json.dumps({"version": 1, "outcome": unknown}))
    (home / "accounts.json").write_text(json.dumps({
        "version": 1, "accounts": catalog, "defaults": {"codex": "codex:alpha"},
    }))


def _server(home, monkeypatch, **options):
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(home))
    return ZetaServer(home=home, socket_path=short_socket_path(home), ws_port=0,
                      ws_discovery=home.parent / "ws.json", provider="codex", **options)


@pytest.mark.asyncio
async def test_state_home_websocket_secrets(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    project, ids = await asyncio.to_thread(seed, home)
    sid = next((home / "sessions").glob("*/meta.json")).parent.name
    _seed_secrets(home)
    server = _server(home, monkeypatch)
    await await_or_fail(server.start(), what="serve start")
    try:
        before = state_stat(home)
        rows = [
            ("state_projects", {}),
            ("state_memory", {"project_id": project}),
            ("state_memory_detail", {"project_id": project, "memory_id": ids["state_memory"][0]}),
            ("state_inbox", {"project_id": project}),
            ("state_inbox_detail", {"project_id": project, "message_id": ids["state_inbox"][0]}),
            ("state_attention", {}), ("state_sessions", {}),
            ("state_session", {"session_id": sid, "collection": "agents"}),
            ("state_agents", {}), ("state_background_tasks", {}),
            ("state_catalog", {"kind": "profiles"}),
            ("state_settings", {"collection": "mcp"}), ("state_settings", {"collection": "redacted_keys"}),
            ("state_accounts", {"collection": "accounts"}), ("state_accounts", {"collection": "warnings"}),
            ("state_storage", {}), ("state_retention", {}),
        ]
        assert {method for method, _ in rows} == {"state_catalog", *PROJECT_METHODS, *SESSION_METHODS, *HOME_METHODS}
        for observer in (False, True):
            async with _connect(server, observer=observer) as client:
                async def call(method, params):
                    raw = await client.request_raw("page", method, params)
                    assert not any(secret in raw for secret in SECRETS), f"seeded secret reached {method}"
                    return json.loads(raw)

                results = {}
                for method, params in rows:
                    reply = await call(method, params)
                    if observer and method.endswith("_detail"):
                        assert reply["error"]["code"] == -32008
                    else:
                        results[method, params.get("collection")] = reply["result"]
                assert results["state_settings", "mcp"] == {
                    "model": "gpt-5.6-luna", "provider": "codex", "approval_mode": "ask", "max_depth": 2,
                    "otel": {"enabled": False, "endpoint": None}, "collection": "mcp", "next_cursor": None,
                    "items": [{"name": "alpha", "transport": "stdio", "configured": True},
                              {"name": "beta", "transport": "streamable-http", "configured": True}],
                }
                assert results["state_settings", "redacted_keys"]["items"] == ["api_key"]
                assert results["state_accounts", "accounts"]["items"] == [
                    {"provider": "codex", "account_id": f"codex:{name}", "status": status,
                     "selected": selected, "refreshable": True, "last_checked_ns": None}
                    for name, status, selected in (("alpha", "selected", True), ("beta", "error", False))
                ]
                store = results["state_storage", None]
                assert store.keys() == {"writer_epoch", "sessions", "projects", "accounts", "telemetry", "warnings", "next_cursor"}
                assert store["writer_epoch"] == WRITER_EPOCH
                assert [store[key]["count"] for key in ("sessions", "projects", "accounts")] == [1, 3, 2]
                assert store["accounts"]["bytes"] > 0
                assert results["state_retention", None] == {
                    "session_days": 42, "telemetry_raw_days": 14, "telemetry_rollup_days": 90,
                    "telemetry_max_bytes": 1073741824, "last_telemetry_prune_ns": None,
                }
                for method, collection in (("state_settings", "mcp"), ("state_accounts", "accounts"), ("state_accounts", "warnings")):
                    params = {"collection": collection, "limit": 1}
                    first = (await call(method, params))["result"]
                    assert len(first["items"]) == 1 and first["next_cursor"]
                    second = (await call(method, {**params, "cursor": first["next_cursor"]}))["result"]
                    assert first["items"] + second["items"] == results[method, collection]["items"][:2]
                    assert (await call("state_storage", {"cursor": first["next_cursor"]}))["error"] == INVALID
                for method in HOME_METHODS:
                    assert (await call(method, {"limit": 201, "extra": SECRETS[0]}))["error"] == INVALID
        assert not any(secret in caplog.text for secret in SECRETS), "seeded secret reached serve logs"
        assert state_stat(home) == before
    finally:
        await await_or_fail(server.close(), what="serve close")


@pytest.mark.asyncio
@pytest.mark.parametrize("fill_other_areas", [True, False], ids=["all-areas-capped", "only-sessions-capped"])
async def test_state_storage_operation_cap(tmp_path, monkeypatch, fill_other_areas):
    home = tmp_path / "home"
    server = _server(home, monkeypatch)
    await await_or_fail(server.start(), what="serve start")
    try:
        entry_limit = 8
        monkeypatch.setattr(read_home_state, "STORAGE_ENTRY_LIMIT", entry_limit)
        sessions = home / "sessions"
        project = home / "projects" / "example"
        account = home / "accounts" / "codex" / "example"
        for directory in (sessions, project, account):
            directory.mkdir(parents=True, exist_ok=True)
            files = entry_limit + 1 if directory == sessions or fill_other_areas else 1
            for i in range(files):
                (directory / f"data-{i}").write_bytes(b"x")
        async with _connect(server, observer=True) as client:
            stat_count = 0
            scandir = os.scandir

            class Entry:
                def __init__(self, entry):
                    self.entry = entry
                    self.name = entry.name

                def stat(self, **kwargs):
                    nonlocal stat_count
                    stat_count += 1
                    return self.entry.stat(**kwargs)

            @contextmanager
            def counted(fd):
                with scandir(fd) as source:
                    yield map(Entry, source) if isinstance(fd, int) else source

            with monkeypatch.context() as patch:
                # test-contract: count metadata operations; the request still uses real files.
                patch.setattr(os, "scandir", counted)
                first = (await client.request("cap", "state_storage", {"limit": 1}))["result"]
            assert first["projects"]["count"] > 0
            assert first["sessions"]["bytes"] == entry_limit
            assert first["sessions"]["complete"] is False
            if fill_other_areas:
                assert stat_count == 3 * entry_limit
                assert first["projects"]["complete"] is False
                assert first["accounts"]["complete"] is False
                assert first["warnings"] == ["accounts: partial storage scan"]
                second = (await client.request(
                    "cap-next", "state_storage", {"limit": 1, "cursor": first["next_cursor"]},
                ))["result"]
                assert second["warnings"] == ["projects: partial storage scan"]
                last = (await client.request("cap-last", "state_storage", {"cursor": second["next_cursor"]}))["result"]
                assert last["warnings"] == ["sessions: partial storage scan"]
                assert last["next_cursor"] is None
            else:
                assert first["projects"] == {"count": 1, "bytes": 1, "complete": True}
                assert first["accounts"] == {"count": 1, "bytes": 1, "complete": True}
                assert first["warnings"] == ["sessions: partial storage scan"]
                assert first["next_cursor"] is None
    finally:
        await await_or_fail(server.close(), what="serve close")


@pytest.mark.asyncio
@pytest.mark.parametrize("multi", (False, True), ids=("dormant", "multi"))
async def test_state_port_gate(tmp_path, monkeypatch, multi):
    features = (*protocol.FEATURES, "multi_controller") if multi else protocol.FEATURES
    server = _server(
        tmp_path / "home", monkeypatch,
        _supported_features_for_test=frozenset(features) if multi else None,
    )
    await await_or_fail(server.start(), what="serve start")
    try:
        async with _connect(server, negotiated=False) as client:
            assert (await client.request("missing", "state_storage"))["error"]["code"] == -32601
        document = (Path(__file__).parents[1] / "docs" / "serve-protocol.md").read_text()
        section = document.split("## method table", 1)[1].split("## common JSON-RPC shapes", 1)[0]
        docs_rows = re.findall(r"^\| `([^`]+)` \| `[^`]+` \| `([^`]+)` \| `[^`]+` \|$", section, re.MULTILINE)

        async def negotiate():
            ws = server._websocket
            assert ws is not None
            async with connect(f"ws://127.0.0.1:{ws.port}/", subprotocols=["zeta.v1", f"zeta.bearer.{ws.controller_token}"]) as socket:
                return (await ServeClient(websocket=socket).request("hello", "hello", {
                    "protocol_version": "1.2", "features": list(features),
                }))["result"]["capabilities"]

        caps = await negotiate()
        requests = set(caps["requests"])
        assert "state_read_v1" in caps["features"]
        assert ("multi_controller" in caps["features"]) is multi
        expected = {"state_catalog", *PROJECT_METHODS, *SESSION_METHODS, *HOME_METHODS}
        assert {method.wire for method in HANDLERS if method.feature == "state_read_v1"} == expected
        assert {wire for wire, feature in docs_rows if feature == "state_read_v1"} == expected
        assert expected <= requests
        for feature in caps["features"]:
            for wire, name in docs_rows:
                if name == feature:
                    assert protocol.MethodDescriptor.for_wire(wire) in HANDLERS
                    assert wire in requests
        # test-contract: simulate a partial port without changing the hello path.
        monkeypatch.delitem(HANDLERS, protocol.MethodDescriptor.STATE_RETENTION)
        partial = await negotiate()
        assert "state_read_v1" not in partial["features"]
        assert not expected.intersection(partial["requests"])
    finally:
        await await_or_fail(server.close(), what="serve close")
