"""real serve-socket coverage for `turn_reconnect`: a turn outlives its controller."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import frames_until_event
from tests.test_server import _connect, _event, _request
from tests.test_server_mcp_bindings import (
    LocalMCP,
    _wait_for_controller_release,
    allow_local_http,
)
from zeta.protocol.types import (
    CompletionBackend,
    ContentBlock,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

RECONNECT = ["approval_reconnect", "turn_reconnect", "served_turn_end", "delivery_id"]


class HeldBackend(CompletionBackend):
    """One scripted reply per provider request; a held request stops mid-stream."""

    def __init__(self, replies: Sequence[tuple[list[ContentBlock], bool]]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.streaming = [asyncio.Event() for _ in self.replies]
        self.release = [asyncio.Event() for _ in self.replies]

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        index = self.calls
        self.calls += 1
        blocks, held = self.replies[index]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        self.streaming[index].set()
        if held:
            await await_or_fail(self.release[index].wait(), what=f"release {index}")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _server(tmp_path: Path, backend: CompletionBackend) -> ZetaServer:
    return ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )


async def _hello(reader, writer, request_id: int, features: list[str]) -> list[str]:
    hello = await _request(reader, writer, request_id, "hello", {
        "protocol_version": "1.2", "features": features
    })
    return hello[-1]["result"]["capabilities"]["features"]


async def _reconnect(server: ZetaServer, request_id: int, features: list[str]):
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", server.port), 10
    )
    await _hello(reader, writer, request_id, features)
    return reader, writer


async def _drop(writer: asyncio.StreamWriter, server: ZetaServer) -> None:
    writer.close()
    await writer.wait_closed()
    await _wait_for_controller_release(server)


async def _close(server: ZetaServer, *writers) -> None:
    for writer in writers:
        if writer is not None and not writer.is_closing():
            writer.close()
            await writer.wait_closed()
    await server.close()


def _replies(frames: list[dict[str, Any]]) -> list[str]:
    return [
        block["text"]
        for frame in frames
        if frame.get("params", {}).get("event") == "assistant_message"
        for block in frame["params"]["message"]["content"]
        if block.get("type") == "text"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("features", "kept"),
    [(RECONNECT, True), (["approval_reconnect", "served_turn_end"], False)],
    ids=["turn_reconnect", "approval_reconnect_only"],
)
async def test_reload_mid_stream_keeps_turn_for_reattached_client(
    tmp_path: Path, features: list[str], kept: bool
) -> None:
    backend = HeldBackend([([TextContent("the reply")], True)])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    writer2 = None
    try:
        negotiated = await _hello(reader, writer, 1, features)
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "hi"})
        await await_or_fail(backend.streaming[0].wait(), what="mid-stream")

        await _drop(writer, server)
        reader2, writer2 = await _reconnect(server, 4, features)
        status = (await _request(reader2, writer2, 5, "status", {}))[-1]["result"]
        if not kept:
            assert status["state"] == "idle"
            assert status["last_turn_outcome"]["reason"] == "disconnect"
            assert "turn_reconnect" not in negotiated
            return
        assert status["state"] == "running", status
        assert "turn_reconnect" in negotiated
        backend.release[0].set()
        frames = await frames_until_event(reader2, "agent_end")
        assert _replies(frames) == ["the reply"]
        status = (await _request(reader2, writer2, 6, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "completed"
    finally:
        for release in backend.release:
            release.set()
        await _close(server, writer2, writer)


@pytest.mark.asyncio
async def test_turn_reconnect_requires_approval_reconnect(tmp_path: Path) -> None:
    server = _server(tmp_path, HeldBackend([]))
    reader, writer = await _connect(server)
    try:
        assert "turn_reconnect" not in await _hello(reader, writer, 1, ["turn_reconnect"])
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_nobody_returns_within_grace_cancels_turn_with_disconnect(
    tmp_path: Path,
) -> None:
    backend = HeldBackend([([TextContent("never seen")], True)])
    server = _server(tmp_path, backend)
    deadline = asyncio.Event()

    async def injected_deadline() -> None:
        await await_or_fail(deadline.wait(), what="injected grace deadline")

    server.controllers.attached().control._grace_deadline = injected_deadline
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _hello(reader, writer, 1, RECONNECT)
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "hi"})
        await await_or_fail(backend.streaming[0].wait(), what="mid-stream")
        steer = await _request(reader, writer, 4, "steer", {
            "text": "later", "delivery_id": "steer-1"
        })
        assert steer[-1]["result"] == {"accepted": True}

        await _drop(writer, server)
        assert server.controllers.attached().runtime.state.status == "running"
        grace = server.controllers.attached().control._grace_task
        assert grace is not None and not grace.done()
        deadline.set()
        await await_or_fail(asyncio.shield(grace), what="grace expiry")

        reader2, writer2 = await _reconnect(server, 5, RECONNECT)
        status = (await _request(reader2, writer2, 6, "status", {}))[-1]["result"]
        assert status["state"] == "idle"
        assert status["last_turn_outcome"]["reason"] == "disconnect"
        dropped = server.controllers.attached().runtime.opened.store.client_delivery("steer-1")
        assert dropped.status == "dropped" and dropped.reason == "disconnect"
        assert backend.calls == 1
    finally:
        deadline.set()
        for release in backend.release:
            release.set()
        await _close(server, writer2, writer)


async def _late_steer_at_turn_end(server: ZetaServer, reader, writer, backend):
    """Queue a steer the finished turn did not consume; hold its end-of-turn step.

    Returns the event that lets the end-of-turn decision run.
    """
    store = server.controllers.attached().runtime.opened.store
    committed = asyncio.Event()
    finalizing = asyncio.Event()
    publish = asyncio.Event()
    decide = asyncio.Event()
    at_decision = asyncio.Event()
    append = store.append_client_delivery_async
    finalize = server.controllers.attached().control.deliveries.finalize_turn
    continue_after_turn = server.controllers.attached().control.deliveries.continue_after_turn

    async def pause_after_commit(*args: Any, **kwargs: Any) -> None:
        await append(*args, **kwargs)
        if args[0] == "late-steer":
            committed.set()
            await await_or_fail(publish.wait(), what="steer publication")

    async def track_finalize(*args: Any, **kwargs: Any) -> None:
        finalizing.set()
        await finalize(*args, **kwargs)

    async def hold_decision(*args: Any, **kwargs: Any) -> None:
        at_decision.set()
        await await_or_fail(decide.wait(), what="end-of-turn decision")
        await continue_after_turn(*args, **kwargs)

    store.append_client_delivery_async = pause_after_commit
    server.controllers.attached().control.deliveries.finalize_turn = track_finalize
    server.controllers.attached().control.deliveries.continue_after_turn = hold_decision
    steer = asyncio.create_task(_request(reader, writer, 10, "steer", {
        "text": "late steer", "delivery_id": "late-steer"
    }))
    await await_or_fail(committed.wait(), what="durable steer commit")
    backend.release[0].set()
    await await_or_fail(finalizing.wait(), what="turn finalization")
    publish.set()
    assert (await await_or_fail(steer, what="steer response"))[-1]["result"] == {
        "accepted": True
    }
    await await_or_fail(at_decision.wait(), what="end-of-turn step")
    return decide


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("features", "kept"),
    [(["delivery_id"], False), (RECONNECT, True)],
    ids=["plain", "turn_reconnect"],
)
async def test_disconnect_before_end_of_turn_decision_follows_grace_rule(
    tmp_path: Path, features: list[str], kept: bool
) -> None:
    backend = HeldBackend([
        ([TextContent("first")], True), ([TextContent("steered")], False)
    ])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    decide = None
    try:
        await _hello(reader, writer, 1, features)
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "first"})
        await await_or_fail(backend.streaming[0].wait(), what="first request")
        decide = await _late_steer_at_turn_end(server, reader, writer, backend)

        await _drop(writer, server)
        decide.set()
        handoff = server.controllers.attached().control._turn_handoff_task
        await await_or_fail(asyncio.shield(handoff), what="end-of-turn step")
        delivery = server.controllers.attached().runtime.opened.store.client_delivery("late-steer")
        if not kept:
            assert delivery.status == "dropped" and delivery.reason == "disconnect"
            assert backend.calls == 1
            return
        await await_or_fail(backend.streaming[1].wait(), what="steered turn")
        steered = server.controllers.attached().control._turn_task
        assert steered is not None
        await await_or_fail(asyncio.shield(steered), what="steered turn end")
        assert server.controllers.attached().runtime.opened.store.last_turn_end().data["reason"] == "completed"
    finally:
        if decide is not None:
            decide.set()
        for release in backend.release:
            release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_close_finishes_end_of_turn_step_before_runtime_closes(
    tmp_path: Path,
) -> None:
    backend = HeldBackend([([TextContent("first")], True)])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    order: list[str] = []
    decide = None
    try:
        await _hello(reader, writer, 1, ["delivery_id"])
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "first"})
        await await_or_fail(backend.streaming[0].wait(), what="first request")
        decide = await _late_steer_at_turn_end(server, reader, writer, backend)

        store = server.controllers.attached().runtime.opened.store
        drop = store.drop_client_steering_async
        runtime_close = server.controllers.attached().runtime.close
        finish_handoff = getattr(server.controllers.attached().control, "_finish_turn_handoff", None)

        async def record_drop(*args: Any, **kwargs: Any):
            order.append("drop steering")
            return await drop(*args, **kwargs)

        async def record_runtime_close(*args: Any, **kwargs: Any):
            order.append("runtime close")
            # without the fix the end-of-turn step resumes only here.
            decide.set()
            return await runtime_close(*args, **kwargs)

        async def release_on_wait() -> None:
            decide.set()
            await finish_handoff()

        store.drop_client_steering_async = record_drop
        server.controllers.attached().runtime.close = record_runtime_close
        if finish_handoff is not None:
            server.controllers.attached().control._finish_turn_handoff = release_on_wait
        await await_or_fail(server.close(), what="server close")
        assert order == ["drop steering", "runtime close"], order
    finally:
        if decide is not None:
            decide.set()
        for release in backend.release:
            release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_attach_takes_over_turn_in_grace_and_rebinds_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = HeldBackend([
        ([TextContent("mounted")], False),
        ([TextContent("kept reply")], True),
        ([ToolUseContent(ToolCall("page-1", "page__echo", {}))], False),
        ([TextContent("used page")], False),
    ])
    server = _server(tmp_path, backend)
    features = [
        *RECONNECT, "new_session_settings", "session_mcp_bindings", "resume_attach"
    ]
    old = {"server": "page", "headers": {"Authorization": "Bearer old-page"}}
    new = {"server": "page", "headers": {"Authorization": "Bearer new-page"}}
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _hello(reader, writer, 1, features)
        session_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": "allow", "mcp_bindings": [old],
        }))[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "agent_end")
        await _request(reader, writer, 4, "send", {"text": "reply"})
        await await_or_fail(backend.streaming[1].wait(), what="mid-stream")

        await _drop(writer, server)
        # achilles reload: a bare hello, then resume of the session with bindings.
        reader2, writer2 = await _reconnect(server, 5, features)
        attached = await _request(reader2, writer2, 6, "resume", {
            "session_id": session_id, "mcp_bindings": [new]
        })
        assert attached[-1].get("result", {}).get("attached") == "live_turn", attached
        backend.release[1].set()
        frames = await frames_until_event(reader2, "agent_end")
        assert _replies(frames) == ["kept reply"]
        status = (await _request(reader2, writer2, 7, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "completed"

        await _request(reader2, writer2, 8, "send", {"text": "use page"})
        await _event(reader2, "agent_end")
        calls = [
            header
            for method, header in zip(mcp.methods, mcp.headers, strict=False)
            if method == "tools/call"
        ]
        assert calls == ["Bearer new-page"]
    finally:
        for release in backend.release:
            release.set()
        await _close(server, writer2, writer)
        await mcp.close()


@pytest.mark.asyncio
async def test_turn_ending_inside_grace_ends_it_so_later_wake_survives_deadline(
    tmp_path: Path,
) -> None:
    backend = HeldBackend([
        ([TextContent("first")], True), ([TextContent("wake reply")], True)
    ])
    server = _server(tmp_path, backend)
    deadline = asyncio.Event()

    async def injected_deadline() -> None:
        await await_or_fail(deadline.wait(), what="injected grace deadline")

    server.controllers.attached().control._grace_deadline = injected_deadline
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, 1, RECONNECT)
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "first"})
        await await_or_fail(backend.streaming[0].wait(), what="first request")
        first = server.controllers.attached().control._turn_task
        assert first is not None

        await _drop(writer, server)
        grace = server.controllers.attached().control._grace_task
        assert grace is not None
        backend.release[0].set()
        await await_or_fail(asyncio.shield(first), what="first turn end")

        # a background completion wakes the session after the client's turn ended.
        store = server.controllers.attached().runtime.opened.store
        store.append_agent_notification(
            "child",
            child_session_path="/tmp/child",
            description="child",
            status="completed",
            text="child done",
        )
        sink = server.controllers.attached().runtime._background_wake_sink
        assert sink is not None
        sink(server.controllers.attached().runtime.session_id)
        await await_or_fail(backend.streaming[1].wait(), what="wake turn")
        wake = server.controllers.attached().control._turn_task
        assert wake is not None

        # the old grace deadline passes while the wake turn runs.
        deadline.set()
        await await_or_fail(asyncio.shield(grace), what="grace end")
        backend.release[1].set()
        await await_or_fail(
            asyncio.gather(wake, return_exceptions=True), what="wake turn end"
        )
        assert store.last_turn_end().data["reason"] == "completed"
        assert backend.calls == 2
    finally:
        deadline.set()
        for release in backend.release:
            release.set()
        await _close(server, writer)
