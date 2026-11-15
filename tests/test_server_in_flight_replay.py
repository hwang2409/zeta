"""real serve-socket coverage for `in_flight_replay`: a client that reattaches
mid-reply and joins with `session_history_and_subscribe` sees the whole
in-flight reply once, then the live deltas, with no gap and no duplicate."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import frames_until_event
from tests.test_server import _connect, _request
from tests.test_server_turn_reconnect import RECONNECT, _close, _drop, _reconnect
from zeta.protocol.types import (
    CompletionBackend,
    ContentBlock,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

FEATURES = [
    *RECONNECT,
    "new_session_settings",
    "resume_attach",
    "session_replay_seq",
    "in_flight_replay",
]
HOLD = object()


class GatedBackend(CompletionBackend):
    """Stream scripted blocks; each HOLD waits for the test to open its gate."""

    def __init__(self, replies: Sequence[tuple[list[object], list[ContentBlock]]]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.held: list[asyncio.Event] = []
        self.gates: list[asyncio.Event] = []
        for steps, _ in self.replies:
            for step in steps:
                if step is HOLD:
                    self.held.append(asyncio.Event())
                    self.gates.append(asyncio.Event())

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        gate = sum(
            step is HOLD for steps, _ in self.replies[: self.calls] for step in steps
        )
        steps, final = self.replies[self.calls]
        self.calls += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for step in steps:
            if step is HOLD:
                self.held[gate].set()
                await await_or_fail(self.gates[gate].wait(), what=f"gate {gate}")
                gate += 1
                continue
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=step)
        yield StreamEvent(
            StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, final)
        )

    def open_all(self) -> None:
        for gate in self.gates:
            gate.set()


def _server(tmp_path: Path, backend: CompletionBackend) -> ZetaServer:
    return ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )


def _published(server: ZetaServer, event: str, match: str) -> asyncio.Event:
    """An event set once serve publishes a frame whose text contains ``match``."""
    seen = asyncio.Event()
    connections = server.connections
    original = connections.publish_controller

    async def publish(name, session_id, **fields):
        written = await original(name, session_id, **fields)
        text = str(fields.get("delta", fields.get("output", fields.get("message", ""))))
        if name == event and match in text:
            seen.set()
        return written

    connections.publish_controller = publish
    return seen


class ClientView:
    """What a replay client renders: committed replies plus one streaming reply."""

    def __init__(self, history: list[dict[str, Any]]) -> None:
        self.committed = [
            "".join(b.get("text", "") for b in row["content"] if b.get("type") == "text")
            for row in history
            if row.get("role") == "assistant"
            and any(b.get("type") == "text" for b in row.get("content", ()))
        ]
        self.text = ""
        self.thinking = ""
        self.output: dict[str, str] = {}
        self.seqs: list[int] = []

    def apply(self, stream_id: str, frames: list[dict[str, Any]]) -> None:
        for frame in frames:
            params = frame.get("params", {})
            if frame.get("method") != "event" or params.get("stream_id") != stream_id:
                continue
            self.seqs.append(params["seq"])
            name = params["event"]
            if name == "assistant_delta" and params["kind"] == "thinking":
                self.thinking += params["delta"]
            elif name == "assistant_delta":
                self.text += params["delta"]
            elif name == "tool_output":
                call = params["tool_call"]["id"]
                self.output[call] = self.output.get(call, "") + params["output"]
            elif name == "assistant_message":
                text = "".join(
                    b["text"] for b in params["message"]["content"] if b["type"] == "text"
                )
                thinking = "".join(
                    b["text"] for b in params["message"]["content"] if b["type"] == "thinking"
                )
                # the streamed reply must equal its commit: no gap, no duplicate
                assert (self.thinking, self.text) == (thinking, text)
                self.committed.append(text)
                self.text = ""
                self.thinking = ""


async def _start(server: ZetaServer, reader, writer, approval: str = "ask") -> str:
    await _request(reader, writer, 1, "hello", {
        "protocol_version": "1.2", "features": FEATURES
    })
    created = await _request(reader, writer, 2, "new_session", {
        "provider": "codex", "model": "gpt-5.6-sol", "approval_mode": approval,
    })
    session_id = created[-1]["result"]["session"]["session_id"]
    await _request(reader, writer, 3, "send", {"text": "go"})
    return session_id


async def _reattach(server: ZetaServer, session_id: str, writer):
    """achilles reload: drop, hello, resume (attach), then the atomic join."""
    await _drop(writer, server)
    reader2, writer2 = await _reconnect(server, 4, FEATURES)
    attached = await _request(reader2, writer2, 5, "resume", {"session_id": session_id})
    assert attached[-1]["result"].get("attached") == "live_turn", attached
    return reader2, writer2


async def _join(reader, writer, session_id: str) -> tuple[ClientView, str]:
    frames = await _request(reader, writer, 6, "session_history_and_subscribe", {
        "session_id": session_id
    })
    result = frames[-1]["result"]
    view = ClientView(result["messages"])
    stream_id = result["replay"]["stream_id"]
    return view, stream_id


@pytest.mark.asyncio
async def test_reattach_mid_reply_shows_whole_reply_then_live_deltas(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([(
        [ThinkingContent("plan "), TextContent("line 0 "), TextContent("line 1 "), HOLD,
         TextContent("line 2")],
        [ThinkingContent("plan "), TextContent("line 0 line 1 line 2")],
    )])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    writer2 = None
    try:
        session_id = await _start(server, reader, writer)
        await await_or_fail(backend.held[0].wait(), what="mid-reply")
        reader2, writer2 = await _reattach(server, session_id, writer)

        view, stream_id = await _join(reader2, writer2, session_id)
        backend.open_all()
        view.apply(stream_id, await frames_until_event(reader2, "agent_end"))

        assert view.committed == ["line 0 line 1 line 2"]
        assert view.seqs == list(range(1, len(view.seqs) + 1))
    finally:
        backend.open_all()
        await _close(server, writer2, writer)


@pytest.mark.asyncio
async def test_forced_order_reply_ends_and_next_streams_during_join_history_read(
    tmp_path: Path,
) -> None:
    """Forced bad order: while the join reads history off the loop, the reply
    streams on, commits, a tool runs, and the next reply starts. Each delta
    reaches the client once and the committed reply is not shown twice."""
    target = tmp_path / "note.txt"
    target.write_text("note\n")
    call = ToolUseContent(ToolCall("read-1", "read", {"path": str(target)}))
    backend = GatedBackend([
        ([TextContent("first "), HOLD, TextContent("reply")],
         [TextContent("first reply"), call]),
        ([TextContent("second "), HOLD, TextContent("reply")],
         [TextContent("second reply")]),
    ])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    writer2 = None
    try:
        session_id = await _start(server, reader, writer, approval="allow")
        await await_or_fail(backend.held[0].wait(), what="first reply mid-stream")
        reader2, writer2 = await _reattach(server, session_id, writer)

        connections = server.connections
        original_read = connections.read_replay_history
        reading = asyncio.Event()
        proceed = asyncio.Event()

        async def held_read(*args, **kwargs):
            reading.set()
            await await_or_fail(proceed.wait(), what="held history read")
            return await original_read(*args, **kwargs)

        connections.read_replay_history = held_read
        second_streaming = _published(server, "assistant_delta", "second")
        join = asyncio.create_task(_join(reader2, writer2, session_id))
        try:
            await await_or_fail(reading.wait(), what="join reading history")
            backend.gates[0].set()
            await await_or_fail(backend.held[1].wait(), what="second reply mid-stream")
            await await_or_fail(second_streaming.wait(), what="second delta published")
            proceed.set()
            view, stream_id = await await_or_fail(join, what="join")
        finally:
            proceed.set()
            if not join.done():
                join.cancel()
        backend.gates[1].set()
        view.apply(stream_id, await frames_until_event(reader2, "agent_end"))

        assert view.committed == ["first reply", "second reply"]
        assert view.text == ""
        assert view.seqs == list(range(1, len(view.seqs) + 1))
    finally:
        backend.open_all()
        await _close(server, writer2, writer)


@pytest.mark.asyncio
async def test_reattach_while_tool_streams_shows_output_so_far(tmp_path: Path) -> None:
    go = tmp_path / "go"
    # the tool waits for the test's file, so out-2 comes after the join
    command = f"echo out-1; while [ ! -e {go} ]; do sleep 0.05; done; echo out-2"
    call = ToolUseContent(ToolCall("bash-1", "bash", {"command": command}))
    backend = GatedBackend([
        ([TextContent("running")], [TextContent("running"), call]),
        ([TextContent("ran")], [TextContent("ran")]),
    ])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    out_1 = _published(server, "tool_output", "out-1")
    writer2 = None
    try:
        session_id = await _start(server, reader, writer, approval="allow")
        await await_or_fail(out_1.wait(), what="first tool output")
        reader2, writer2 = await _reattach(server, session_id, writer)

        view, stream_id = await _join(reader2, writer2, session_id)
        go.touch()
        view.apply(stream_id, await frames_until_event(reader2, "agent_end"))

        assert view.output.get("bash-1") == "out-1\nout-2\n", view.output
        assert view.committed == ["running", "ran"]
    finally:
        backend.open_all()
        go.touch()
        await _close(server, writer2, writer)


@pytest.mark.asyncio
async def test_join_without_feature_is_unchanged(tmp_path: Path) -> None:
    backend = GatedBackend([(
        [TextContent("line 0 "), HOLD, TextContent("line 1")],
        [TextContent("line 0 line 1")],
    )])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    writer2 = None
    try:
        session_id = await _start(server, reader, writer)
        await await_or_fail(backend.held[0].wait(), what="mid-reply")
        await _drop(writer, server)
        features = [f for f in FEATURES if f != "in_flight_replay"]
        reader2, writer2 = await _reconnect(server, 4, features)
        await _request(reader2, writer2, 5, "resume", {"session_id": session_id})
        frames = await _request(reader2, writer2, 6, "session_history_and_subscribe", {
            "session_id": session_id
        })
        stream_id = frames[-1]["result"]["replay"]["stream_id"]
        backend.open_all()
        frames = await frames_until_event(reader2, "agent_end")
        deltas = [
            f["params"]["delta"] for f in frames
            if f.get("params", {}).get("event") == "assistant_delta"
            and f["params"].get("stream_id") == stream_id
        ]
        assert deltas == ["line 1"]
    finally:
        backend.open_all()
        await _close(server, writer2, writer)


@pytest.mark.asyncio
async def test_join_after_aborted_turn_replays_nothing(tmp_path: Path) -> None:
    go = tmp_path / "go"
    command = f"echo out-1; while [ ! -e {go} ]; do sleep 0.05; done"
    call = ToolUseContent(ToolCall("bash-1", "bash", {"command": command}))
    backend = GatedBackend([([TextContent("running")], [TextContent("running"), call])])
    server = _server(tmp_path, backend)
    reader, writer = await _connect(server)
    out_1 = _published(server, "tool_output", "out-1")
    try:
        session_id = await _start(server, reader, writer, approval="allow")
        await await_or_fail(out_1.wait(), what="tool output")
        # the server answers abort after the canceled turn stopped
        aborted = await _request(reader, writer, 4, "abort", {})
        assert aborted[-1]["result"] == {"aborted": True}

        frames = await _request(reader, writer, 6, "session_history_and_subscribe", {
            "session_id": session_id
        })
        # frames the join sends after its response arrive before the next response
        frames += await _request(reader, writer, 7, "status", {})
        assert not [
            f for f in frames
            if f.get("params", {}).get("event") in {"assistant_delta", "tool_output"}
        ]
    finally:
        go.touch()
        await _close(server, writer)
