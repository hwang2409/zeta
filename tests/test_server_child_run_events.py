from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.core.agent_progress import post_child_progress
from zeta.core.store import ConversationStore

pytest_plugins = ("tests.support.socket_fixtures",)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    "omitted",
    ("session_replay_seq", "child_run_history", "child_runs"),
)
async def test_child_run_events_requires_every_prerequisite_over_both_transports(
    tmp_path: Path, transport: Transport, omitted: str
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        features = {
            "child_run_events",
            "session_replay_seq",
            "child_run_history",
            "child_runs",
        }
        features.remove(omitted)
        hello = await client.request(
            1,
            "hello",
            {"protocol_version": "1.2", "features": sorted(features)},
        )
        capabilities = hello["result"]["capabilities"]
        assert "child_run_events" not in capabilities["features"]
        assert "child_run_events" not in capabilities["requests"]

        reply = await client.request(
            2,
            "child_run_events",
            {"session_id": "missing", "run_id": "missing:1"},
        )
        assert reply["error"]["code"] == -32601
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_child_run_events_rejects_malformed_params_over_both_transports(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        await negotiate(client, extra=("child_run_history", "child_run_events", "child_runs"), protocol_version="1.2")
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        reply = await client.request(3, "child_run_events", {"session_id": session_id})
        assert reply["error"]["code"] == -32602
        assert "session_id and run_id" in reply["error"]["message"]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_child_run_events_returns_real_join_over_both_transports(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        hello = await negotiate(
            client,
            extra=("child_run_history", "child_run_events", "child_runs"),
            protocol_version="1.2",
        )
        assert "child_run_events" in hello["result"]["capabilities"]["features"]
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        child = ConversationStore(
            server.runtime.opened.store.session_dir / "agents",
            session_id="1",
            _storage_lease=server.runtime.opened.store._storage_lease,
        )
        child.start_agent_lifecycle(
            handle=f"{session_id}:1", started_at="2026-01-01T00:00:00+00:00",
            depth=1, agent_type="worker", description="child",
        )
        child.close()
        reply = await client.request(
            3, "child_run_events", {"session_id": session_id, "run_id": f"{session_id}:1"}
        )
        assert "error" not in reply, reply
        result: dict[str, Any] = reply["result"]
        assert result["replay"]["history_seq"] == 0
        elapsed = result["child"]["elapsed_seconds"]
        # test-contract: latency the public row includes the product-reported elapsed value
        assert result["child"] == {
            "run_id": f"{session_id}:1",
            "session_id": session_id,
            "parent_run_id": None,
            "description": "child",
            "agent_type": "worker",
            "status": "running",
            "accepts_follow_ups": False,
            "turns_used": 0,
            "tool_count": 0,
            "current_step": "",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
            "elapsed_seconds": elapsed,
        }
        assert isinstance(elapsed, float)
        assert isinstance(result["revision"], int)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_late_unknown_child_join_activates_no_stream(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    child = None
    try:
        await negotiate(
            client,
            extra=("child_run_history", "child_run_events", "child_runs"),
            protocol_version="1.2",
        )
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        run_id = f"{session_id}:1"
        failed = await client.request(
            3, "child_run_events", {"session_id": session_id, "run_id": run_id}
        )
        assert failed["error"]["code"] == -32012

        child = ConversationStore(
            server.runtime.opened.store.session_dir / "agents",
            session_id="1",
            _storage_lease=server.runtime.opened.store._storage_lease,
        )
        child.start_agent_lifecycle(
            handle=run_id,
            started_at="2026-01-01T00:00:00+00:00",
            depth=1,
            agent_type="worker",
            description="late child",
        )
        lifecycle = child.agent_lifecycle()
        assert lifecycle is not None
        post_child_progress(child, lifecycle)
        applied = asyncio.Event()
        asyncio.get_running_loop().call_soon(applied.set)
        await await_or_fail(applied.wait(), what="late child progress applied")
        await client.request(4, "status", {})
        assert not any(
            frame.get("params", {}).get("event") == "child_progress"
            for frame in client.buffered
        ), "the failed child join activated a stream"
    finally:
        if child is not None:
            child.close()
        await client.close()
        await server.close()
