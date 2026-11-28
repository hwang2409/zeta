from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_replay_table import _backend_factory
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.server.connections import EVENT_FEATURES
from zeta.server.protocol import EventDescriptor


@pytest.mark.parametrize(("transport", "case"), [
    ("unix", "source"), ("websocket", "source"),
    ("unix", "pause"), ("websocket", "pause"),
    ("unix", "owner-and-subscriber"), ("unix", "global"),
    ("unix", "parked"), ("unix", "replay-terminal"), ("unix", "legacy-precommit"),
])
@pytest.mark.asyncio
async def test_event_routing_table(tmp_path: Path, transport: Transport, case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(EVENT_FEATURES) == set(EventDescriptor) and len(EVENT_FEATURES) == 47
    assert Counter(route.destination for route in EVENT_FEATURES.values()) == {
        "session": 44, "connection": 2, "global": 1,
    }
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(2)]
    try:
        first, second = clients
        if case == "legacy-precommit":
            control = server.controllers.attached().control
            ready = control.controller_ready

            async def emit_before_commit():
                assert await control._notify(EventDescriptor.ASSISTANT_DELTA, delta="precommit")
                await ready()

            monkeypatch.setattr(control, "controller_ready", emit_before_commit)
            assert "result" in await negotiate(first, replay=False, protocol_version="1.0")
            await first.request(4, "status", {})
            assert first.wire_received[0]["id"] == 1
            assert first.wire_received[1] == {
                "jsonrpc": "2.0", "method": "event",
                "params": {"event": "assistant_delta", "delta": "precommit"},
            }
            return
        if case in {"parked", "replay-terminal"}:
            for client in clients:
                assert "result" in await negotiate(client, extra=("multi_controller", "session_park", "resume_attach"))
            created = await second.request(2, "new_session", {})
            session_id = created["result"]["session"]["session_id"]
            if case == "parked":
                connection = server.connections.clients()[1]
                record = server.controllers.controller_for(connection)
                monkeypatch.setattr(record.control, "holds_work", lambda: True)
                await second.request(3, "new_session", {})
                name = "parked_session_changed"
            else:
                for request_id in (2, 3):
                    joined = await second.request(request_id + 1, "session_history_and_subscribe", {"session_id": session_id})
                    assert "result" in joined, joined
                name = "replay_stream_ended"
            expected = [[], [session_id]]
        elif case == "global":
            observer, ineligible = [await connect_replay(server, transport) for _ in range(2)]
            clients.extend((observer, ineligible))
            for client in (first, second):
                assert "result" in await negotiate(client, extra=("multi_controller", "zeta_updated"))
                assert "result" in await client.request(2, "new_session", {})
            assert "result" in await negotiate(observer, observer=True, extra=("zeta_updated",))
            assert "result" in await negotiate(ineligible, observer=True)
            await server.publish_update("old", "new")
            name = "zeta_updated"
            expected = [[None], [None], [None], []]
        else:
            for client in clients:
                assert "result" in await negotiate(client, extra=("multi_controller", "live_config_changed"))
                assert "result" in await client.request(2, "new_session", {})
            pairs = [(conn, server.controllers.controller_for(conn))
                     for conn in server.connections.clients()]
            (_, first_record), (_, second_record) = pairs
            if case == "pause":
                server.connections.release(pairs[0][0])
            target, record = (first, first_record) if case == "owner-and-subscriber" else (second, second_record)
            session_id = record.runtime.session_id
            if case == "owner-and-subscriber":
                joined = await target.request(3, "session_history_and_subscribe", {"session_id": session_id})
                assert "result" in joined, joined
            await record.control._event(StreamEvent(StreamEventType.CONFIG_CHANGED, data={}), session_id=session_id)
            name = "config_changed"
            expected = [[session_id], []] if target is first else [[], [session_id]]
        # A response drains prior output on that connection. No timeout proves absence.
        for index, client in enumerate(clients):
            if case == "pause" and index == 0:
                continue
            await client.request(99, "status", {})
            received = [params["session_id"] for frame in client.wire_received
                        if (params := frame.get("params", {})).get("event") == name]
            assert received == expected[index], f"{case}: physical client {index} must get exactly its selected copies"
    finally:
        for client in clients:
            await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_finalization_cannot_overtake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = replay_server(tmp_path, "unix", _backend_factory)
    client = await connect_replay(server, "unix")
    release = asyncio.Event()
    task = None
    try:
        await negotiate(client, extra=("live_config_changed", "served_turn_end"))
        assert "result" in await client.request(2, "new_session", {})
        control = server.controllers.attached().control
        paused = asyncio.Event()
        order = []
        notify, finalize = control._notify, control.deliveries.finalize_or_fail

        async def stream(*args, **kwargs):
            yield StreamEvent(StreamEventType.CONFIG_CHANGED, data={})
            yield StreamEvent(StreamEventType.TURN_END, data={})
            yield StreamEvent(StreamEventType.AGENT_END, data={})

        async def hold(event, *args, **kwargs):
            if event == "config_changed":
                paused.set()
                await await_or_fail(release.wait(), what="release config publication")
            return await notify(event, *args, **kwargs)

        async def watch_finalize(*args, **kwargs):
            order.append("finalization")
            return await finalize(*args, **kwargs)

        monkeypatch.setattr(control.runtime.loop, "run_turn", stream)
        # test-contract: seam holds the ordinary event before output admission
        monkeypatch.setattr(control, "_notify", hold)
        monkeypatch.setattr(control.deliveries, "finalize_or_fail", watch_finalize)
        task = asyncio.create_task(control._run_turn("test", turn_id="0" * 32))
        await await_or_fail(paused.wait(), what="last ordinary event enters route")
        await client.request(3, "status", {})
        assert order == [], "turn finalization overtook the paused ordinary event"
        release.set()
        await await_or_fail(task, what="turn finishes after config publication")
        await client.request(4, "status", {})
        assert order == ["finalization"]
        assert [params["event"] for frame in client.wire_received
                if (params := frame.get("params", {})).get("event")] == [
            "config_changed", "turn_end", "agent_end",
        ]
    finally:
        release.set()
        if task is not None:
            await await_or_fail(task, what="turn settled during teardown")
        await client.close()
        await server.close()
