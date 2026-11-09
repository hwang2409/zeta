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
from zeta.core.agent_progress import post_child_progress, publish_child_started
from zeta.core.child_run_row import project_child_run_row
from zeta.core.store import ConversationStore
from zeta.server.protocol import (
    CHILD_RUN_EVENTS_PREREQUISITES,
    FEATURES,
    EventDescriptor,
    MethodDescriptor,
)
from zeta.server.session_replay.rows import CARRIED, LIVE, ROUTES

pytest_plugins = ("tests.support.socket_fixtures",)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


def test_child_run_row_has_the_z5_shape() -> None:
    row = project_child_run_row(
        {"handle": "top:1", "description": "task", "agent_type": "worker", "state": "running", "current_step": "work", "started_at": "now"},
        accepts_follow_ups=True,
    )
    assert list(row) == [
        "run_id", "session_id", "parent_run_id", "description", "agent_type", "status",
        "accepts_follow_ups", "turns_used", "tool_count", "current_step", "started_at",
        "elapsed_seconds",
    ]
    assert row["run_id"] == "top:1"
    assert row["session_id"] == "top"
    assert row["parent_run_id"] is None


def test_child_run_events_prerequisites_are_derived_from_method_features() -> None:
    assert CHILD_RUN_EVENTS_PREREQUISITES == frozenset(
        descriptor.feature
        for descriptor in MethodDescriptor
        if descriptor.wire in {"session_history_and_subscribe", "child_run_history"}
        and descriptor.feature is not None
    )
    assert CHILD_RUN_EVENTS_PREREQUISITES <= set(FEATURES)


def test_child_lifecycle_routes_are_explicit() -> None:
    assert ROUTES[(EventDescriptor.CHILD_STARTED, "")] is CARRIED
    assert ROUTES[(EventDescriptor.CHILD_PROGRESS, "")] is LIVE
    assert ROUTES[(EventDescriptor.CHILD_ENDED, "")] is CARRIED
    assert ROUTES[(EventDescriptor.APPROVAL_REQUEST, "child")] is LIVE
    assert ROUTES[(EventDescriptor.APPROVAL_END, "child")] is LIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_child_run_events_rejects_malformed_params_over_both_transports(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        await negotiate(client, extra=("child_run_history", "child_run_events"), protocol_version="1.2")
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
async def test_top_join_receives_child_started_over_both_transports(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    try:
        await negotiate(client, extra=("child_run_history", "child_run_events"), protocol_version="1.2")
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        joined = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
        stream_id = joined["result"]["replay"]["stream_id"]
        publish_child_started(
            f"{session_id}:1", 1,
            {"run_id": f"{session_id}:1", "task": "task", "name": "worker", "started_at": "now"},
            accepts_follow_ups=True,
        )
        event = await client.event("child_started")
        assert event["stream_id"] == stream_id
        assert event["seq"] == 1
        assert event["child"]["run_id"] == f"{session_id}:1"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_child_join_response_precedes_progress_posted_during_write(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    entered, release = asyncio.Event(), asyncio.Event()
    assert server.connections is not None
    original_read = server.connections.read_child_run_events

    async def gated(connection: Any, params: dict[str, Any]) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="release child join snapshot")
        return await original_read(connection, params)

    monkeypatch.setattr(server.connections, "read_child_run_events", gated)
    try:
        await negotiate(client, extra=("child_run_history", "child_run_events"), protocol_version="1.2")
        created = await client.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        child = ConversationStore(
            server.runtime.opened.store.session_dir / "agents", session_id="1",
            _storage_lease=server.runtime.opened.store._storage_lease,
        )
        child.start_agent_lifecycle(
            handle=f"{session_id}:1", started_at="now", depth=1,
            agent_type="worker", description="child",
        )
        nested = ConversationStore(
            child.session_dir / "agents", session_id="1",
            _storage_lease=server.runtime.opened.store._storage_lease,
        )
        nested.start_agent_lifecycle(
            handle=f"{session_id}:1:1", started_at="now", depth=2,
            agent_type="worker", description="nested",
        )
        await client.send(3, "child_run_events", {"session_id": session_id, "run_id": f"{session_id}:1"})
        await await_or_fail(entered.wait(), what="child join snapshot entered")
        post_child_progress(nested, {"handle": f"{session_id}:1:1", "state": "running", "current_step": "first"})
        release.set()
        response = await client.response(3)
        post_child_progress(nested, {"handle": f"{session_id}:1:1", "state": "running", "current_step": "second"})
        event = await await_or_fail(client.event("child_progress"), what="post-admission child progress")
        assert response["result"]["replay"]["history_seq"] == 0
        assert event["seq"] == 1
        assert event["child"]["run_id"] == f"{session_id}:1:1"
        assert event["revision"] > 0
        assert event["child"]["current_step"] == "second"
    finally:
        release.set()
        if "nested" in locals():
            nested.close()
        if "child" in locals():
            child.close()
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
            extra=("child_run_history", "child_run_events"),
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
        assert result["child"]["run_id"] == f"{session_id}:1"
        assert isinstance(result["revision"], int)
    finally:
        await client.close()
        await server.close()
