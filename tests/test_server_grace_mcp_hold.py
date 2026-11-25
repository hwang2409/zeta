"""real serve-socket coverage: a bound MCP call in a turn's reconnect grace waits
for the client's return, bounded by the grace deadline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import frames_until_event
from tests.test_mcp_oauth import _monkey_home, _write_token
from tests.test_server import _connect, _event, _request
from tests.test_server_mcp_bindings import LocalMCP, allow_local_http
from tests.test_server_turn_reconnect import (
    RECONNECT,
    HeldBackend,
    _close,
    _drop,
    _hello,
    _reconnect,
    _replies,
    _server,
)
from zeta.core.abort import AbortSignal
from zeta.mcp import MCPServerConfig, StreamableHTTPMCPClient
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer

FEATURES = [
    *RECONNECT, "new_session_settings", "session_mcp_bindings", "resume_attach"
]
OLD = {"server": "page", "headers": {"Authorization": "Bearer old-page"}}
NEW = {"server": "page", "headers": {"Authorization": "Bearer new-page"}}
DETACHED = "client that provides it disconnected"
MAY_HAVE_RUN = "may or may not have taken effect"


class Grace:
    """One page MCP server, a held backend, and an injected grace deadline."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        allow_local_http(monkeypatch)
        self.tmp_path = tmp_path
        self.mcp = LocalMCP()
        self.deadline = asyncio.Event()
        self.deadlines_started = 0
        self.handled = asyncio.Event()
        self.parked: list[bool] = []
        self.writers: list[asyncio.StreamWriter] = []
        self.backend: HeldBackend | None = None
        self.server: ZetaServer | None = None

    async def start(self, replies) -> tuple[ZetaServer, HeldBackend]:
        port = await self.mcp.start()
        (self.tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
            "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
        }}}))
        self.backend = HeldBackend(replies)
        self.server = _server(self.tmp_path, self.backend)

        async def injected_deadline() -> None:
            self.deadlines_started += 1
            await await_or_fail(self.deadline.wait(), what="injected grace deadline")

        self.server.controllers.attached().control._grace_deadline = injected_deadline
        return self.server, self.backend

    async def open_session(self, approval_mode: str = "allow"):
        assert self.server is not None
        reader, writer = await _connect(self.server)
        self.writers.append(writer)
        await _hello(reader, writer, 1, FEATURES)
        session_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": approval_mode, "mcp_bindings": [OLD],
        }))[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "agent_end")
        self._track_page_calls()
        return reader, writer, session_id

    def _track_page_calls(self) -> None:
        assert self.server is not None and self.server.controllers.attached().runtime.loop is not None
        mount = self.server.controllers.attached().runtime.loop._mcp_mount
        assert mount is not None
        actor = mount._actors["page"]
        handle = actor._handle_call

        def track(request: Any) -> None:
            handle(request)
            # parked: kept for the client's return instead of answered at once.
            self.parked.append(
                request.identifier in actor._requests and request.task is None
            )
            self.handled.set()

        actor._handle_call = track

    async def call_waits(self) -> None:
        await await_or_fail(self.handled.wait(), what="bound call reached the actor")
        assert self.parked == [True], "the bound call failed at once instead of waiting"
        assert self.page_calls() == []

    async def reconnect(self, request_id: int):
        assert self.server is not None
        reader, writer = await _reconnect(self.server, request_id, FEATURES)
        self.writers.append(writer)
        return reader, writer

    def page_calls(self) -> list[str | None]:
        return [
            header
            for method, header in zip(self.mcp.methods, self.mcp.headers, strict=False)
            if method == "tools/call"
        ]

    async def close(self) -> None:
        self.deadline.set()
        if self.backend is not None:
            for release in self.backend.release:
                release.set()
        if self.server is not None:
            await _close(self.server, *reversed(self.writers))
        await self.mcp.close()


TOOL_TURN = [
    ([TextContent("mounted")], False),
    ([ToolUseContent(ToolCall("page-1", "page__echo", {}))], True),
    ([TextContent("after page")], False),
]


@pytest.mark.asyncio
async def test_bound_call_in_grace_waits_and_runs_once_on_new_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    try:
        _, backend = await grace.start(TOOL_TURN)
        reader, writer, session_id = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(backend.streaming[1].wait(), what="tool call streaming")

        await _drop(writer, grace.server)
        backend.release[1].set()
        await grace.call_waits()

        reader2, writer2 = await grace.reconnect(5)
        attached = await _request(reader2, writer2, 6, "resume", {
            "session_id": session_id, "mcp_bindings": [NEW]
        })
        assert attached[-1].get("result", {}).get("attached") == "live_turn", attached
        tool_end = await _event(reader2, "tool_end")
        frames = await frames_until_event(reader2, "agent_end")

        assert tool_end["tool_result"]["is_error"] is False, tool_end
        assert grace.page_calls() == ["Bearer new-page"]
        assert _replies(frames) == ["after page"]
        status = (await _request(reader2, writer2, 7, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "completed"
    finally:
        await grace.close()


@pytest.mark.asyncio
async def test_bound_call_without_reattach_gets_detached_text_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    try:
        _, backend = await grace.start([
            *TOOL_TURN[:2], ([TextContent("finished without the page")], False)
        ])
        reader, writer, _ = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(backend.streaming[1].wait(), what="tool call streaming")

        await _drop(writer, grace.server)
        # a bare hello reclaims the turn but re-sends no binding.
        reader2, writer2 = await grace.reconnect(5)
        backend.release[1].set()
        await grace.call_waits()
        grace.deadline.set()
        tool_end = await _event(reader2, "tool_end")
        frames = await frames_until_event(reader2, "agent_end")

        result = tool_end["tool_result"]
        assert result["is_error"] is True
        assert DETACHED in result["content"], result
        # it never reached the MCP server, so it does not claim it may have run.
        assert MAY_HAVE_RUN not in result["content"], result
        assert grace.page_calls() == []
        assert _replies(frames) == ["finished without the page"]
        status = (await _request(reader2, writer2, 6, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "completed"
    finally:
        await grace.close()


@pytest.mark.asyncio
async def test_abort_during_wait_releases_call_and_it_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    try:
        server, backend = await grace.start(TOOL_TURN)
        reader, writer, session_id = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(backend.streaming[1].wait(), what="tool call streaming")

        await _drop(writer, server)
        reader2, writer2 = await grace.reconnect(5)
        backend.release[1].set()
        await grace.call_waits()
        turn = server.controllers.attached().control._turn_task
        assert turn is not None
        aborted = await _request(reader2, writer2, 6, "abort", {})
        assert aborted[-1]["result"] == {"aborted": True}
        await await_or_fail(asyncio.wait({turn}), what="aborted turn end")
        actor = server.controllers.attached().runtime.loop._mcp_mount._actors["page"]
        assert actor._held_calls() == []

        # a binding that returns after the abort must not run the dropped call.
        await _request(reader2, writer2, 7, "resume", {
            "session_id": session_id, "mcp_bindings": [NEW]
        })
        status = (await _request(reader2, writer2, 8, "status", {}))[-1]["result"]
        assert status["state"] == "idle", status
        assert status["last_turn_outcome"]["reason"] != "completed", status
        assert grace.page_calls() == []
    finally:
        await grace.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["reattach", "deadline"])
async def test_approval_grace_deadline_also_bounds_the_approved_bound_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    try:
        server, _ = await grace.start([
            ([TextContent("mounted")], False),
            ([ToolUseContent(ToolCall("page-1", "page__echo", {}))], False),
            ([TextContent("after page")], False),
        ])
        reader, writer, session_id = await grace.open_session(approval_mode="ask")
        await _request(reader, writer, 4, "send", {"text": "use page"})
        approval = await _event(reader, "approval_request")

        await _drop(writer, server)
        reader2, writer2 = await grace.reconnect(5)
        assert server.controllers.attached().control._grace_task is None
        approved = await _request(reader2, writer2, 6, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in approved[-1], approved
        await grace.call_waits()
        if ending == "reattach":
            await _request(reader2, writer2, 7, "resume", {
                "session_id": session_id, "mcp_bindings": [NEW]
            })
        else:
            grace.deadline.set()
        tool_end = await _event(reader2, "tool_end")
        await frames_until_event(reader2, "agent_end")

        # the approval's grace started the one deadline that bounded the call.
        assert grace.deadlines_started == 1
        if ending == "reattach":
            assert tool_end["tool_result"]["is_error"] is False, tool_end
            assert grace.page_calls() == ["Bearer new-page"]
        else:
            assert DETACHED in tool_end["tool_result"]["content"], tool_end
            assert grace.page_calls() == []
    finally:
        await grace.close()


IN_FLIGHT_TURN = [
    ([TextContent("mounted")], False),
    ([ToolUseContent(ToolCall("page-1", "page__echo", {}))], False),
    ([TextContent("after page")], False),
]


async def _page_results(reader, writer, request_id: int, session_id: str) -> list:
    rows = (await _request(reader, writer, request_id, "session_history", {
        "session_id": session_id
    }))[-1]["result"]["messages"]
    return [
        row["tool_result"]
        for row in rows
        if (row.get("tool_result") or {}).get("tool_call_id") == "page-1"
    ]


@pytest.mark.asyncio
async def test_call_in_flight_at_disconnect_returns_its_real_result_after_reattach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    gate = asyncio.Event()
    grace.mcp.call_gate = gate
    try:
        server, _ = await grace.start(IN_FLIGHT_TURN)
        reader, writer, session_id = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(grace.mcp.tool_called.wait(), what="call in flight")
        turn = server.controllers.attached().control._turn_task
        assert turn is not None

        await _drop(writer, server)
        reader2, writer2 = await grace.reconnect(5)
        await _request(reader2, writer2, 6, "resume", {
            "session_id": session_id, "mcp_bindings": [NEW]
        })
        # the old client sends nothing new with the departed page's header.
        actor = server.controllers.attached().runtime.loop._mcp_mount._actors["page"]
        (old_client,) = actor._draining
        listener = old_client._notification_task
        assert listener is not None
        await await_or_fail(asyncio.wait({listener}), what="old notification stream stop")
        # the server answers the call that was sent before the disconnect.
        gate.set()
        await await_or_fail(asyncio.wait({turn}), what="turn end")

        results = await _page_results(reader2, writer2, 7, session_id)
        assert [row["content"] for row in results] == ["ok"], (
            "the model did not get the in-flight call's real result", results
        )
        assert grace.page_calls() == ["Bearer old-page"], "the call ran more than once"
        status = (await _request(reader2, writer2, 8, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "completed"
        assert actor._draining == set(), "the old client was never closed"
    finally:
        gate.set()
        await grace.close()


@pytest.mark.asyncio
async def test_call_in_flight_without_reattach_ends_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    gate = asyncio.Event()
    grace.mcp.call_gate = gate
    try:
        server, _ = await grace.start(IN_FLIGHT_TURN)
        reader, writer, session_id = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(grace.mcp.tool_called.wait(), what="call in flight")
        turn = server.controllers.attached().control._turn_task
        assert turn is not None

        await _drop(writer, server)
        grace.deadline.set()
        await await_or_fail(asyncio.wait({turn}), what="turn end at the deadline")
        reader2, writer2 = await grace.reconnect(5)
        results = await _page_results(reader2, writer2, 6, session_id)
        # the grace ended the turn, so the call ends with it, as on a disconnect.
        assert len(results) == 1 and results[0]["is_error"] is True, results
        status = (await _request(reader2, writer2, 7, "status", {}))[-1]["result"]
        assert status["last_turn_outcome"]["reason"] == "disconnect", status
        actor = server.controllers.attached().runtime.loop._mcp_mount._actors["page"]
        assert actor._draining == set(), "the old client was never closed"
        assert grace.page_calls() == ["Bearer old-page"]
    finally:
        gate.set()
        await grace.close()


@pytest.mark.asyncio
async def test_in_flight_call_answered_404_sends_nothing_with_old_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    gate = asyncio.Event()
    grace.mcp.call_gate = gate
    # session gone: an open client would re-initialize and resend the call.
    grace.mcp.call_error = b"404 Not Found"
    try:
        server, _ = await grace.start(IN_FLIGHT_TURN)
        reader, writer, session_id = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(grace.mcp.tool_called.wait(), what="call in flight")
        turn = server.controllers.attached().control._turn_task
        assert turn is not None
        sent_before = len(grace.mcp.headers)

        await _drop(writer, server)
        reader2, writer2 = await grace.reconnect(5)
        await _request(reader2, writer2, 6, "resume", {
            "session_id": session_id, "mcp_bindings": [NEW]
        })
        gate.set()
        await await_or_fail(asyncio.wait({turn}), what="turn end")

        after = grace.mcp.headers[sent_before:]
        assert "Bearer old-page" not in after, (
            "a request after the disconnect carried the gone client's header", after
        )
        (result,) = await _page_results(reader2, writer2, 7, session_id)
        assert DETACHED in result["content"] and MAY_HAVE_RUN in result["content"], result
    finally:
        gate.set()
        await grace.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("seal_at", ["answer_401", "credential_load"])
async def test_sealed_oauth_call_sends_nothing_new_with_old_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seal_at: str
) -> None:
    home = _monkey_home(monkeypatch, tmp_path)
    _write_token("page", home, access_token="stale-token")
    in_flight = asyncio.Event()
    answer = asyncio.Event()
    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if str(request.url).endswith("/token"):
            return httpx.Response(200, request=request, json={
                "access_token": "fresh-token", "token_type": "Bearer"
            })
        if request.method != "POST":
            return httpx.Response(405, request=request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "tools/call":
            in_flight.set()
            await await_or_fail(answer.wait(), what="test releases the call")
            return httpx.Response(401, text="expired", request=request)
        return httpx.Response(200, request=request, json={
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
        })

    config = MCPServerConfig(
        "page", "streamable-http", url="https://mcp.test/rpc",
        auth_type="oauth", headers={"X-Page": "old-page"},
    )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHTTPMCPClient(config, client=transport, home=str(home))
    call: asyncio.Task[Any] | None = None
    try:
        await client.connect()
        if seal_at == "credential_load":
            store = client._credential_store
            load = store.load

            async def gated_load(name: str):
                snapshot = await load(name)
                in_flight.set()
                await await_or_fail(answer.wait(), what="test releases the load")
                return snapshot

            monkeypatch.setattr(store, "load", gated_load)
        call = asyncio.create_task(client.call_tool("echo", {}, AbortSignal()))
        await await_or_fail(in_flight.wait(), what="call in flight")
        sent_before = len(sent)
        client.seal()
        answer.set()
        result = await await_or_fail(call, what="sealed call ends")

        assert sent[sent_before:] == [], (
            "the sealed client sent a new request",
            [(r.method, str(r.url), r.headers.get("x-page")) for r in sent[sent_before:]],
        )
        assert result["isError"] is True
        assert DETACHED in result["content"][0]["text"], result
    finally:
        answer.set()
        if call is not None:
            await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), 10)
        await client.close()
        await transport.aclose()


@pytest.mark.asyncio
async def test_server_close_during_wait_does_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grace = Grace(tmp_path, monkeypatch)
    try:
        server, backend = await grace.start(TOOL_TURN)
        reader, writer, _ = await grace.open_session()
        await _request(reader, writer, 4, "send", {"text": "use page"})
        await await_or_fail(backend.streaming[1].wait(), what="tool call streaming")

        await _drop(writer, server)
        backend.release[1].set()
        await grace.call_waits()
        turn = server.controllers.attached().control._turn_task
        assert turn is not None
        # the deadline never fires: only close may end the wait.
        await await_or_fail(server.close(), what="server close")
        assert turn.done()
        assert grace.page_calls() == []
    finally:
        await grace.close()
