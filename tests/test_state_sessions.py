"""Metadata session reads through the negotiated websocket."""

import asyncio
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from zeta.core import agent_progress
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.server import ZetaServer, protocol
from zeta.tools._shared.process import background_tasks_for_session

MARKER = "PRIVATE-CONTENT-STATE-S6C"
METHODS = {"state_sessions", "state_session", "state_agents", "state_background_tasks"}
START = "2026-11-08T00:00:00+00:00"


async def seed(home, cwd):
    manager = SessionManager(home)
    with ExitStack() as cleanup:
        opened = await asyncio.to_thread(manager.create, provider="codex", model="offline", cwd=cwd, auto_project=False,
                                name="synthetic", system_prompt=MARKER)
        root = opened.store
        cleanup.callback(root.close)
        sid = opened.metadata.session_id
        direct_run = f"{sid}:1"
        runs = [f"{direct_run}:{i}" for i in range(1, 5)]
        paths = []
        direct = ConversationStore(root.session_dir / "agents", session_id="1")
        cleanup.callback(direct.close)
        for i, run_id in enumerate([direct_run, *runs]):
            child = direct if i == 0 else ConversationStore(direct.session_dir / "agents", session_id=str(i))
            if i:
                cleanup.callback(child.close)
            child.start_agent_lifecycle(handle=run_id, started_at=START, depth=2 if i else 1,
                                        agent_type="general", description=MARKER)
            paths.append(child.agent_lifecycle_path)
            cleanup.callback(agent_progress.remove_live_agent_lifecycle, paths[-1])
        registry = background_tasks_for_session(root, command_niceness=10)
        try:
            tasks = [(await registry.start(f"printf {MARKER}", cwd, notify_on_exit=False))[0]
                     for _ in range(5)]
        finally:
            await await_or_fail(registry.close(), what="synthetic task registry close")
        return sid, direct_run, runs, tasks, paths


@pytest.mark.parametrize("case", ("operations", "agents", "background_tasks", "state_agents", "state_background_tasks"))
async def test_session_state_websocket(tmp_path, monkeypatch, case):
    home = tmp_path / "home"
    session_count = 8
    if case == "operations":
        def seed_older_sessions():
            manager = SessionManager(home)
            for i in range(session_count - 1):
                manager.create(provider="codex", model="offline", cwd=tmp_path,
                               name=f"older {i}", auto_project=False).store.close()

        await asyncio.to_thread(seed_older_sessions)
    sid, direct_run, runs, tasks, paths = await seed(home, tmp_path)
    server = ZetaServer(home=home, socket_path=short_socket_path(home), ws_port=0,
                        ws_discovery=tmp_path / "websocket.json", provider="codex")
    await await_or_fail(server.start(), what="serve start")
    try:
        ws = server._websocket
        assert ws is not None
        observer = case in {"operations", "agents"}
        role = "observer" if observer else "controller"
        token = ws.observer_token if observer else ws.controller_token
        async with connect(f"ws://127.0.0.1:{ws.port}/", subprotocols=["zeta.v1", f"zeta.bearer.{token}"]) as socket:
            client = ServeClient(websocket=socket)
            features = ["state_read_v1", "serve_observers"] if observer else ["state_read_v1"]
            hello = await client.request("hello", "hello", {"protocol_version": "1.2", "features": features, "role": role})
            capabilities = hello["result"]["capabilities"]
            assert "state_read_v1" in capabilities["features"]
            assert METHODS <= set(capabilities["requests"])

            async def call(method, params):
                raw = await client.request_raw("page", method, params)
                assert MARKER not in raw
                return json.loads(raw)

            if case == "operations":
                opens = Counter()
                sessions = {path.name for path in (home / "sessions").iterdir()}
                stat_calls = Counter()
                page_limit = 1

                def count(calls, original):
                    def wrapped(name, *args, **kwargs):
                        if not isinstance(name, int):
                            basename = Path(os.fspath(name)).name
                            calls["session directory" if basename in sessions else basename] += 1
                        return original(name, *args, **kwargs)
                    return wrapped

                with monkeypatch.context() as patch:
                    patch.setattr(os, "open", count(opens, os.open))
                    patch.setattr(os, "stat", count(stat_calls, os.stat))
                    result = (await call("state_sessions", {"limit": page_limit}))["result"]
                assert opens["conversation.jsonl"] == 0, "session list opened child transcripts"
                assert opens["agent_lifecycle.json"] == 0, "session list opened child lifecycles"
                assert opens["meta.json"] == session_count, "session list must open one metadata file per session"
                assert opens["session directory"] + opens["meta.json"] <= 2 * session_count + page_limit
                assert opens["agents"] == opens["background_tasks.json"] == stat_calls["conversation.jsonl"] == page_limit
                assert result["next_cursor"]
                session, = result["items"]
                assert session["session_id"] == sid
                assert session["child_count"] == 1
                assert session["background_count"] == 5
                assert session["title"] == "synthetic"
                return

            detail = not case.startswith("state_")
            collection = case.removeprefix("state_")
            method = "state_session" if detail else case
            params = {"session_id": sid, **({"collection": collection} if detail else {})}
            reply = (await call(method, params))["result"]
            if detail:
                assert reply["session"]["session_id"] == sid
                assert reply["collection"] == collection
            first = (await call(method, {**params, "limit": 2}))["result"]
            cursor = first["next_cursor"]
            assert cursor
            assert first["items"] == reply["items"][:2]
            assert (await call(method, {**params, "cursor": cursor}))["result"]["items"] == reply["items"][2:]
            mismatch = {**params, "cursor": cursor, "session_id": "another"}
            assert (await call(method, mismatch))["error"]["data"]["code"] == "invalid_params"
            if detail:
                other = "background_tasks" if collection == "agents" else "agents"
                assert (await call(method, {**params, "cursor": cursor, "collection": other}))["error"]["code"] == -32602
                codec = protocol.FrameCodec()
                budget = codec.response_size("page", first)
                third = (await call(method, {**params, "limit": 3}))["result"]
                assert codec.response_size("page", third) > budget
                with monkeypatch.context() as patch:
                    patch.setattr(protocol, "MAX_FRAME_BYTES", budget)
                    assert (await call(method, params))["result"] == first
            if collection == "agents":
                assert [row["run_id"] for row in reply["items"]] == sorted([direct_run, *runs], reverse=True)
                nested = reply["items"][0]
                assert nested == {"run_id": runs[-1], "parent_run_id": direct_run, "session_id": sid,
                                  "profile": "general", "status": "running", "background": False,
                                  "started_ns": 1794096000000000000, "ended_ns": None}
                agent_progress.publish_live_agent_lifecycle(paths[-1], {
                    "handle": runs[-1], "state": "running", "started_at": START, "finished_at": None,
                    "depth": 2, "agent_type": "general", "current_step": MARKER,
                })
                try:
                    assert (await call(method, params))["result"]["items"][0] == nested
                finally:
                    agent_progress.remove_live_agent_lifecycle(paths[-1])
            else:
                assert [row["task_id"] for row in reply["items"]] == sorted(tasks, reverse=True)
                assert all(row["session_id"] == sid and row["summary"] is None for row in reply["items"])
    finally:
        await await_or_fail(server.close(), what="serve close")
