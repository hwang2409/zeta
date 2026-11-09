"""Real-transport coverage for negotiated session-grant control."""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console
from websockets.asyncio.client import connect

from tests.support import server_client
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.server_client import _buffer, approve_and_collect_grant, read
from tests.test_server import (
    TIMEOUT,
    _connect,
    _event,
    _frames_until_event,
    _read,
    _request,
    _socket_path,
)
from tests.test_server_observers import _observer
from tests.test_server_websocket import (
    _connect as _connect_websocket,
)
from tests.test_server_websocket import (
    _read_json,
    _send_json,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageRole, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer, approval_lifecycle, approval_settlement
from zeta.server.approval_lifecycle import ExternalEnd
from zeta.server.protocol import EventDescriptor
from zeta.tui.app import TUIApp

FEATURES = ["session_grants"]
PRECEDENCE = ["deny", "ask", "allow", "default"]


async def _approve_after_external_end(
    server: ZetaServer,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str,
    request_id: int,
    approval_request_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approval_module = importlib.import_module("zeta.server.approval_lifecycle")
    committed = asyncio.Event()
    release = asyncio.Event()
    external_entered = asyncio.Event()
    original_persist = approval_settlement.persist_durably
    original_finish = server.control._approvals._finish_record

    async def gated_finish(record: object, resolution: dict[str, str]) -> None:
        external_entered.set()
        await original_finish(record, resolution)  # type: ignore[arg-type]

    async def gated_persist(*args: object, **kwargs: object) -> object:
        result = await original_persist(*args, **kwargs)
        committed.set()
        await await_or_fail(release.wait(), what="release durable approval")
        return result

    monkeypatch.setattr(approval_settlement, "persist_durably", gated_persist)
    # test-contract: seam force the ExternalEnd ownership interleaving
    monkeypatch.setattr(server.control._approvals, "_finish_record", gated_finish)
    approve = asyncio.create_task(
        _request(
            reader,
            writer,
            request_id,
            "approve",
            {"request_id": approval_request_id, "scope": "always_tool"},
        )
    )
    try:
        await await_or_fail(committed.wait(), what="durable approval commit")
        external_end = asyncio.create_task(
            server.control._approvals.transition(
                approval_module.ExternalEnd(approval_request_id, session_id, {})
            )
        )
        await await_or_fail(external_entered.wait(), what="external approval finish")
        await await_or_fail(external_end, what="external approval end")
        release.set()
        approve_frames = await await_or_fail(approve, what="approve response")
        response = approve_frames[-1]["result"]
        assert response["request_id"] == approval_request_id
        assert response["scope"] == "always_tool"
        buffered = list(_buffer(reader))
        grant_events = [
            frame
            for frame in buffered
            if frame.get("params", {}).get("event") == "session_grants_changed"
        ]
        assert len(grant_events) == 1, buffered
        assert grant_events[0]["params"]["data"]["change"] == "grant"
        assert grant_events[0]["params"]["data"]["changed"] == ["bash"]
        return approve_frames, buffered
    finally:
        release.set()


class _DelegatedBackend:
    def __init__(self) -> None:
        self.child_one_ready = asyncio.Event()
        self.release_child_one = asyncio.Event()
        self.child_two_ready = asyncio.Event()
        self.release_child_two = asyncio.Event()

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any):
        del tool_schemas
        last_user_index, last_user = next(
            (
                (index, block.text)
                for index, message in reversed(list(enumerate(messages)))
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            (-1, ""),
        )
        has_tool_result = any(
            message.role is MessageRole.TOOL_RESULT
            for message in messages[last_user_index + 1 :]
        )
        if last_user == "grant" and not has_tool_result:
            turn = ScriptedTurn(
                tool_calls=[
                    ToolCall("parent-bash", "bash", {"command": "printf parent"})
                ]
            )
        elif last_user in {"first delegate", "second delegate"} and not has_tool_result:
            number = "1" if last_user == "first delegate" else "2"
            turn = ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        f"agent-{number}",
                        "agent",
                        {
                            "prompt": f"child {number}",
                            "description": f"child {number}",
                            "background": True,
                        },
                    )
                ]
            )
        elif last_user in {"child 1", "child 2"} and not has_tool_result:
            number = last_user[-1]
            if number == "1":
                self.child_one_ready.set()
                # test-contract: seam the owning journey supplies the one deadline.
                await self.release_child_one.wait()
            elif number == "2":
                self.child_two_ready.set()
                # test-contract: seam the owning journey supplies the one deadline.
                await self.release_child_two.wait()
            turn = ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        f"child-bash-{number}",
                        "bash",
                        {"command": f"printf child-{number}"},
                    )
                ]
            )
        else:
            turn = ScriptedTurn([TextContent("done")])
        backend = FakeBackend([turn])
        async for event in FakeBackend.complete.__wrapped__(backend, messages, []):
            yield event


class _GatedBackend(FakeBackend):
    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.release_next_call = asyncio.Event()

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any):
        if len(self.calls) == 1:
            await asyncio.wait_for(self.release_next_call.wait(), 10)
        async for event in FakeBackend.complete.__wrapped__(
            self, messages, tool_schemas
        ):
            yield event


@pytest.mark.asyncio
async def test_approve_collector_reports_structured_error_frame() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "error": {
                        "code": -32000,
                        "message": "approval request is no longer pending",
                    },
                }
            )
            + "\n"
        ).encode()
    )

    class _Writer:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    with pytest.raises(AssertionError) as failure:
        await approve_and_collect_grant(
            reader, _Writer(), 4, "call-1", "always_tool", ["bash"]
        )
    message = str(failure.value)
    assert "approve response" in message
    assert "approval request is no longer pending" in message
    assert "all observed frames" in message


@pytest.mark.asyncio
async def test_approve_collector_reports_response_without_grant_event() -> None:
    reader = asyncio.StreamReader()
    response = {
        "jsonrpc": "2.0",
        "id": 4,
        "result": {
            "accepted": True,
            "decision": "approve",
            "request_id": "call-1",
            "scope": "always_tool",
        },
    }
    turn_end = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"event": "turn_end", "data": {"turn_id": "turn-1"}},
    }
    reader.feed_data((json.dumps(response) + "\n").encode())
    reader.feed_data((json.dumps(turn_end) + "\n").encode())
    reader.feed_eof()

    class _Writer:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    with pytest.raises(AssertionError) as failure:
        await approve_and_collect_grant(
            reader, _Writer(), 4, "call-1", "always_tool", ["bash"]
        )
    message = str(failure.value)
    assert "approve response without session_grants_changed" in message
    assert repr(response) in message
    assert repr(turn_end) in message
    assert "all observed frames" in message


@pytest.mark.asyncio
async def test_approve_collector_restores_unrelated_frames() -> None:
    reader = asyncio.StreamReader()
    turn_end = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"event": "turn_end", "data": {"turn_id": "turn-1"}},
    }
    response = {
        "jsonrpc": "2.0",
        "id": 4,
        "result": {
            "accepted": True,
            "decision": "approve",
            "request_id": "call-1",
            "scope": "always_tool",
        },
    }
    grant = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "event": "session_grants_changed",
            "data": {"change": "grant", "changed": ["bash"]},
        },
    }
    reader.feed_data(
        (
            "\n".join(json.dumps(frame) for frame in (turn_end, response, grant)) + "\n"
        ).encode()
    )

    class _Writer:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            return None

    await approve_and_collect_grant(
        reader, _Writer(), 4, "call-1", "always_tool", ["bash"]
    )
    assert await asyncio.wait_for(read(reader), 10) == turn_end


async def _wait_idle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, request_id: int
) -> int:
    while True:
        result = (await _request(reader, writer, request_id, "status"))[-1]["result"]
        request_id += 1
        if result["state"] == "idle":
            return request_id


async def _hello_and_create(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> tuple[dict[str, Any], str]:
    hello = (
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.2", "features": FEATURES},
        )
    )[-1]["result"]
    assert "session_grants" in hello["capabilities"]["features"]
    created = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[
        -1
    ]["result"]
    reset = await _event(reader, "session_grants_changed")
    assert reset["data"]["change"] == "reset"
    return hello, created["session"]["session_id"]


@pytest.mark.parametrize("transport", ["unix", "tcp"])
@pytest.mark.asyncio
async def test_session_grant_list_revoke_and_next_call_asks_over_stream_transport(
    tmp_path: Path,
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ToolCall("call-1", "bash", {"command": "printf first"})
    second = ToolCall("call-2", "bash", {"command": "printf second"})
    backend = _GatedBackend(
        [
            ScriptedTurn(tool_calls=[first]),
            ScriptedTurn(tool_calls=[second]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path) if transport == "unix" else None,
        port=0 if transport == "tcp" else None,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        hello, session_id = await _hello_and_create(reader, writer)
        assert "session_grants" in hello["capabilities"]["features"]
        assert {"session_grants", "revoke_session_grants"} <= set(
            hello["capabilities"]["requests"]
        )

        await _request(reader, writer, 3, "send", {"text": "first"})
        assert (await _event(reader, "approval_request"))["request_id"] == "call-1"
        approve_frames, buffered = await _approve_after_external_end(
            server,
            reader,
            writer,
            monkeypatch,
            session_id=session_id,
            request_id=4,
            approval_request_id="call-1",
        )
        assert approve_frames[-1]["result"]["accepted"] is True
        grant_events = [
            frame
            for frame in buffered
            if frame.get("params", {}).get("event") == "session_grants_changed"
        ]
        assert len(grant_events) == 1, buffered
        assert grant_events[0]["params"]["data"]["changed"] == ["bash"]
        await _event(reader, "session_grants_changed")

        snapshot = (await _request(reader, writer, 5, "session_grants"))[-1]["result"]
        assert snapshot == {
            "session_id": session_id,
            "default": "ask",
            "precedence": PRECEDENCE,
            "rules": {"deny": [], "ask": [], "allow": ["bash"]},
            "session_grants": ["bash"],
        }

        if transport == "unix":
            output = StringIO()
            assert server.runtime.loop is not None
            app = TUIApp(
                server.runtime.loop,
                provider="codex",
                model="offline",
                zeta_home=tmp_path,
                approval_policy=server.runtime.policy,
                console=Console(file=output, width=160, force_terminal=False),
            )
            await app._handle_prompt_value("/permissions")
            assert "Rule precedence: deny > ask > allow > default" in output.getvalue()
            assert "Session grants (/always):" in output.getvalue()
            assert "bash" in output.getvalue()

        revoked = (
            await _request(
                reader, writer, 6, "revoke_session_grants", {"grant": "bash"}
            )
        )[-1]["result"]
        revoke_event = await _event(reader, "session_grants_changed")
        assert revoked["removed"] == ["bash"]
        assert revoked["grants"]["session_grants"] == []
        assert revoke_event["data"] == {
            "change": "revoke",
            "changed": ["bash"],
            "grants": revoked["grants"],
        }

        if transport == "unix":
            await app._handle_prompt_value("/permissions")
            assert "Session grants (/always):" in output.getvalue()
            assert "none" in output.getvalue()

        backend.release_next_call.set()
        assert (await _event(reader, "approval_request"))["request_id"] == "call-2"
        await _request(reader, writer, 7, "deny", {"request_id": "call-2"})
        await _frames_until_event(reader, "turn_end")
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_always_tool_grant_survives_external_end_during_approve_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ToolCall("call-1", "bash", {"command": "printf first"})
    second = ToolCall("call-2", "bash", {"command": "printf second"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[first]),
            ScriptedTurn(tool_calls=[second]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    committed = asyncio.Event()
    release = asyncio.Event()
    external_entered = asyncio.Event()
    original_persist = approval_settlement.persist_durably
    original_finish = server.control._approvals._finish_record

    async def gated_finish(record: object, resolution: dict[str, str]) -> None:
        external_entered.set()
        await original_finish(record, resolution)  # type: ignore[arg-type]

    async def gated_persist(*args: object, **kwargs: object) -> object:
        result = await original_persist(*args, **kwargs)
        committed.set()
        await asyncio.wait_for(release.wait(), 10)
        return result

    monkeypatch.setattr(approval_settlement, "persist_durably", gated_persist)
    # test-contract: seam force the ExternalEnd ownership interleaving
    monkeypatch.setattr(server.control._approvals, "_finish_record", gated_finish)
    try:
        _, session_id = await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "first"})
        assert (await _event(reader, "approval_request"))["request_id"] == "call-1"

        approve = asyncio.create_task(
            _request(
                reader,
                writer,
                4,
                "approve",
                {"request_id": "call-1", "scope": "always_tool"},
            )
        )
        await await_or_fail(committed.wait(), what="durable approval commit")
        external_end = asyncio.create_task(
            server.control._approvals.transition(ExternalEnd("call-1", session_id, {}))
        )
        await await_or_fail(external_entered.wait(), what="external approval finish")
        await await_or_fail(external_end, what="external approval end")
        release.set()
        approve_frames = await await_or_fail(approve, what="approve response")
        assert approve_frames[-1]["result"]["decision"] == "approve"
        buffered = list(_buffer(reader))
        grant_events = [
            frame
            for frame in buffered
            if frame.get("params", {}).get("event") == "session_grants_changed"
        ]
        assert len(grant_events) == 1, buffered
        assert grant_events[0]["params"]["data"]["changed"] == ["bash"]

        await _frames_until_event(reader, "turn_end")
        snapshot = (await _request(reader, writer, 5, "session_grants"))[-1]["result"]
        assert snapshot["session_id"] == session_id
        assert snapshot["session_grants"] == ["bash"]

        await _request(reader, writer, 6, "send", {"text": "second"})
        frames = await _frames_until_event(reader, "turn_end")
        assert not any(
            frame.get("params", {}).get("event") == "approval_request"
            for frame in frames
        ), frames
    finally:
        release.set()
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_session_grant_flow_over_authenticated_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "bash", {"command": "printf websocket"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        port=0,
        ws_port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    await server.start()
    assert server._websocket is not None
    websocket = await connect(
        f"ws://127.0.0.1:{server._websocket.port}/",
        subprotocols=["zeta.v1", f"zeta.bearer.{server._websocket.token}"],
    )
    assert websocket.subprotocol == "zeta.v1"

    async def request(
        request_id: int, method: str, params: dict[str, object]
    ) -> list[dict[str, Any]]:
        async def read_response() -> list[dict[str, Any]]:
            await websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    },
                    separators=(",", ":"),
                )
            )
            frames = []
            while True:
                payload = await websocket.recv()
                assert isinstance(payload, str), "serve sent a binary WebSocket frame"
                frame = json.loads(payload)
                frames.append(frame)
                if frame.get("id") == request_id:
                    return frames

        return await await_or_fail(
            read_response(), what=f"{method} WebSocket response"
        )

    async def event(name: str) -> dict[str, Any]:
        async def read_event() -> dict[str, Any]:
            while True:
                payload = await websocket.recv()
                assert isinstance(payload, str), "serve sent a binary WebSocket frame"
                frame = json.loads(payload)
                if frame.get("params", {}).get("event") == name:
                    return frame

        return await await_or_fail(read_event(), what=f"{name} WebSocket event")

    setup_started = asyncio.Event()
    release_setup = asyncio.Event()
    original_run_setup = server.runtime._run_setup

    async def gated_run_setup(*args: Any, **kwargs: Any):
        setup_started.set()
        await await_or_fail(
            release_setup.wait(), what="test to release new_session setup"
        )
        return await original_run_setup(*args, **kwargs)

    # test-contract: seam no public setup gate exposes request/setup ordering
    monkeypatch.setattr(server.runtime, "_run_setup", gated_run_setup)
    create_task: asyncio.Task[list[dict[str, Any]]] | None = None
    try:
        hello = (
            await request(
                1,
                "hello",
                {
                    "protocol_version": "1.2",
                    "features": FEATURES,
                },
            )
        )[-1]["result"]
        assert "session_grants" in hello["capabilities"]["features"]
        create_task = asyncio.create_task(
            request(2, "new_session", {"provider": "codex"})
        )
        await await_or_fail(setup_started.wait(), what="new_session setup to start")
        assert not create_task.done(), "new_session responded before setup completed"
        release_setup.set()
        created = await await_or_fail(
            create_task, what="new_session WebSocket response after setup release"
        )
        assert any(
            frame.get("params", {}).get("data", {}).get("change") == "reset"
            for frame in created
        )
        sent = await request(3, "send", {"text": "run"})
        approval = next(
            (
                frame
                for frame in sent
                if frame.get("params", {}).get("event") == "approval_request"
            ),
            None,
        )
        if approval is None:
            approval = await event("approval_request")
        approved = await request(
            4,
            "approve",
            {
                "request_id": "call-1",
                "scope": "always_tool",
            },
        )
        assert any(
            frame.get("params", {}).get("data", {}).get("change") == "grant"
            for frame in approved
        )
        snapshot = (await request(5, "session_grants", {}))[-1]["result"]
        assert snapshot["session_grants"] == ["bash"]
        revoked = await request(6, "revoke_session_grants", {"grant": "bash"})
        assert any(
            frame.get("params", {}).get("data", {}).get("change") == "revoke"
            for frame in revoked
        )
        assert revoked[-1]["result"]["grants"]["session_grants"] == []
    finally:
        release_setup.set()
        if create_task is not None and not create_task.done():
            create_task.cancel()
            await await_or_fail(
                asyncio.gather(create_task, return_exceptions=True),
                what="canceled new_session request cleanup",
            )
        await websocket.close()
        await close_serve(server)


@pytest.mark.asyncio
async def test_revoking_grant_leaves_matching_pending_approval_unchanged(
    tmp_path: Path,
) -> None:
    first = ToolCall("call-1", "bash", {"command": "printf first"})
    pending = ToolCall("call-2", "bash", {"command": "printf pending"})
    next_call = ToolCall("call-3", "bash", {"command": "printf next"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[first]),
            ScriptedTurn([TextContent("first done")]),
            ScriptedTurn(tool_calls=[pending]),
            ScriptedTurn([TextContent("pending done")]),
            ScriptedTurn(tool_calls=[next_call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        ws_port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await asyncio.wait_for(_connect(server), TIMEOUT)
    observer_writer: asyncio.StreamWriter | None = None

    async def observer_request(
        request_id: int, method: str, params: dict[str, object]
    ) -> dict[str, Any]:
        assert observer_writer is not None
        await asyncio.wait_for(
            _send_json(
                observer_writer,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
            ),
            TIMEOUT,
        )
        while True:
            frame = await asyncio.wait_for(_read_json(observer_reader), TIMEOUT)
            if frame.get("id") == request_id:
                return frame

    try:
        await asyncio.wait_for(_hello_and_create(reader, writer), TIMEOUT)
        server.runtime.policy.always_ask = ("bash",)
        observer_reader, observer_writer, status, _headers = await asyncio.wait_for(
            _connect_websocket(
                server, token=None, protocol_token=server._websocket.token
            ),
            TIMEOUT,
        )
        assert status == 101
        hello = await observer_request(
            10,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "session_grants"],
            },
        )
        assert "session_grants" in hello["result"]["capabilities"]["features"]

        await asyncio.wait_for(
            _request(reader, writer, 3, "send", {"text": "create grant"}), TIMEOUT
        )
        first_approval = await asyncio.wait_for(
            _event(reader, "approval_request"), TIMEOUT
        )
        assert first_approval["request_id"] == "call-1"
        await approve_and_collect_grant(
            reader, writer, 4, "call-1", "always_tool", ["bash"]
        )
        await asyncio.wait_for(_frames_until_event(reader, "turn_end"), TIMEOUT)
        await asyncio.wait_for(_wait_idle(reader, writer, 5), TIMEOUT)

        await asyncio.wait_for(
            _request(reader, writer, 20, "send", {"text": "leave pending"}), TIMEOUT
        )
        second_approval = await asyncio.wait_for(
            _event(reader, "approval_request"), TIMEOUT
        )
        assert second_approval["request_id"] == "call-2"
        revoked = (
            await asyncio.wait_for(
                _request(
                    reader,
                    writer,
                    21,
                    "revoke_session_grants",
                    {"grant": "bash"},
                ),
                TIMEOUT,
            )
        )[-1]["result"]
        assert revoked["removed"] == ["bash"]
        status_result = (
            await asyncio.wait_for(_request(reader, writer, 22, "status"), TIMEOUT)
        )[-1]["result"]
        assert [item["request_id"] for item in status_result["pending_approvals"]] == [
            "call-2"
        ]
        assert server.runtime.opened.store.approval_resolution_data("call-2") is None

        observer_denied = await observer_request(
            11,
            "approve",
            {"request_id": "call-2", "scope": "once"},
        )
        assert observer_denied["error"]["data"]["code"] == "observer_read_only"
        await asyncio.wait_for(
            _request(
                reader,
                writer,
                23,
                "approve",
                {"request_id": "call-2", "scope": "once"},
            ),
            TIMEOUT,
        )
        snapshot = (
            await asyncio.wait_for(
                _request(reader, writer, 24, "session_grants"), TIMEOUT
            )
        )[-1]["result"]
        assert snapshot["session_grants"] == []

        await asyncio.wait_for(_frames_until_event(reader, "turn_end"), TIMEOUT)
        await asyncio.wait_for(_wait_idle(reader, writer, 25), TIMEOUT)
        await asyncio.wait_for(
            _request(reader, writer, 30, "send", {"text": "ask again"}), TIMEOUT
        )
        approval = await asyncio.wait_for(_event(reader, "approval_request"), TIMEOUT)
        assert approval["request_id"] == "call-3"
        await asyncio.wait_for(
            _request(reader, writer, 31, "deny", {"request_id": "call-3"}), TIMEOUT
        )
        await asyncio.wait_for(_frames_until_event(reader, "turn_end"), TIMEOUT)
    finally:
        await asyncio.wait_for(
            close_serve(
                server,
                writer,
                *([observer_writer] if observer_writer is not None else []),
            ),
            TIMEOUT,
        )


@pytest.mark.asyncio
async def test_slow_observer_does_not_delay_grant_or_revoke_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "bash", {"command": "printf observer"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    observer_writer: asyncio.StreamWriter | None = None
    replacement_writer: asyncio.StreamWriter | None = None
    release_observer = asyncio.Event()
    release_controller_read = asyncio.Event()
    try:
        await _hello_and_create(reader, writer)
        observer_reader, observer_writer = await _observer(
            server, 10, features=["serve_observers", "session_grants"]
        )
        await _request(reader, writer, 3, "send", {"text": "run"})
        await _event(reader, "approval_request")
        assert server.connections is not None
        observer_connection = server.connections._observers[0].connection
        controller_connection = server.connections._controller.connection
        original_observer_write = observer_connection._write_physical
        original_controller_write = controller_connection._write_physical
        original_client_read = server_client.read
        observer_blocked = asyncio.Event()
        approve_response_written = asyncio.Event()
        revoke_response_written = asyncio.Event()

        async def block_observer(payload: bytes) -> bool:
            if b"session_grants_changed" in payload:
                observer_blocked.set()
                await asyncio.wait_for(release_observer.wait(), 10)
            return await original_observer_write(payload)

        async def observe_controller_write(payload: bytes) -> bool:
            written = await original_controller_write(payload)
            if b'"id":4' in payload:
                approve_response_written.set()
            if b'"id":5' in payload:
                revoke_response_written.set()
            return written

        async def block_controller_read(
            stream: asyncio.StreamReader,
        ) -> dict[str, Any]:
            await asyncio.wait_for(release_controller_read.wait(), 10)
            return await original_client_read(stream)

        # test-contract: seam gate the real observer and controller transports
        monkeypatch.setattr(observer_connection, "_write_physical", block_observer)
        # test-contract: seam observe the real controller transport write
        monkeypatch.setattr(
            controller_connection, "_write_physical", observe_controller_write
        )
        monkeypatch.setattr(server_client, "read", block_controller_read)
        approve = asyncio.create_task(
            approve_and_collect_grant(
                reader, writer, 4, "call-1", "always_tool", ["bash"]
            )
        )
        await asyncio.wait_for(observer_blocked.wait(), 10)
        await asyncio.wait_for(approve_response_written.wait(), 10)
        # test-contract: seam real controller wire write precedes observer release
        assert not release_observer.is_set(), (
            "controller response required observer write completion"
        )
        release_controller_read.set()
        controller_event = await await_or_fail(
            approve, what="controller grant response after wire write"
        )
        release_observer.set()
        observer_event = await _event(observer_reader, "session_grants_changed")
        assert observer_event["data"] == controller_event["data"]
        await _frames_until_event(reader, "turn_end")

        observer_blocked.clear()
        release_observer = asyncio.Event()
        revoke = asyncio.create_task(
            _request(
                reader,
                writer,
                5,
                "revoke_session_grants",
                {"grant": "bash"},
            )
        )
        await asyncio.wait_for(observer_blocked.wait(), 10)
        await asyncio.wait_for(revoke_response_written.wait(), 10)
        # test-contract: seam real controller wire write precedes observer release
        assert not release_observer.is_set(), (
            "revoke response required observer write completion"
        )
        revoke_frames = await await_or_fail(
            revoke, what="controller revoke response after wire write"
        )
        assert revoke_frames[-1]["result"]["removed"] == ["bash"]
        release_observer.set()
        revoke_event = await _event(observer_reader, "session_grants_changed")
        assert revoke_event["data"]["grants"]["session_grants"] == []
        assert (await _request(observer_reader, observer_writer, 11, "session_grants"))[
            -1
        ]["result"] == revoke_event["data"]["grants"]

        denied = (
            await _request(
                observer_reader,
                observer_writer,
                12,
                "revoke_session_grants",
                {"grant": "bash"},
            )
        )[-1]["error"]
        assert denied["code"] == -32008
        assert denied["data"] == {
            "code": "observer_read_only",
            "method": "revoke_session_grants",
        }

        observer_writer.close()
        await observer_writer.wait_closed()
        observer_writer = None
        replacement_reader, replacement_writer = await _observer(
            server, 20, features=["serve_observers", "session_grants"]
        )
        snapshot = (
            await _request(replacement_reader, replacement_writer, 21, "session_grants")
        )[-1]["result"]
        assert snapshot["session_grants"] == []
    finally:
        release_observer.set()
        release_controller_read.set()
        await close_serve(
            server,
            writer,
            *[
                candidate
                for candidate in (observer_writer, replacement_writer)
                if candidate is not None
            ],
        )


@pytest.mark.asyncio
async def test_observer_write_failure_does_not_block_approved_tool_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "bash", {"command": "printf approved"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    observer_writer: asyncio.StreamWriter | None = None
    try:
        await _hello_and_create(reader, writer)
        _observer_reader, observer_writer = await _observer(
            server, 10, features=["serve_observers", "session_grants"]
        )
        await _request(reader, writer, 3, "send", {"text": "run"})
        await _event(reader, "approval_request")
        assert server.connections is not None
        observer_connection = server.connections._observers[0].connection
        write_failed = asyncio.Event()

        async def fail_grant_event(payload: bytes) -> bool:
            if b"session_grants_changed" in payload:
                write_failed.set()
                raise ConnectionError("simulated observer write failure")
            return True

        # test-contract: seam fail the real observer transport write
        monkeypatch.setattr(observer_connection, "_write_physical", fail_grant_event)
        approve_response = (
            await _request(
                reader,
                writer,
                4,
                "approve",
                {"request_id": "call-1", "scope": "always_tool"},
            )
        )[-1]
        assert "result" in approve_response, approve_response
        assert approve_response["result"] == {
            "accepted": True,
            "decision": "approve",
            "request_id": "call-1",
            "scope": "always_tool",
        }
        await asyncio.wait_for(write_failed.wait(), 10)
        await _frames_until_event(reader, "turn_end")
        snapshot = (await _request(reader, writer, 5, "session_grants"))[-1]["result"]
        assert snapshot["session_grants"] == ["bash"]
    finally:
        await close_serve(
            server,
            writer,
            *([observer_writer] if observer_writer is not None else []),
        )


@pytest.mark.asyncio
async def test_observer_write_failure_does_not_turn_revoke_into_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "bash", {"command": "printf grant"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    observer_writer: asyncio.StreamWriter | None = None
    try:
        await _hello_and_create(reader, writer)
        _observer_reader, observer_writer = await _observer(
            server, 10, features=["serve_observers", "session_grants"]
        )
        await _request(reader, writer, 3, "send", {"text": "run"})
        await _event(reader, "approval_request")
        await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": "call-1", "scope": "always_tool"},
        )
        await _frames_until_event(reader, "turn_end")

        assert server.connections is not None
        observer_connection = server.connections._observers[0].connection
        write_failed = asyncio.Event()

        async def fail_grant_event(payload: bytes) -> bool:
            if b"session_grants_changed" in payload:
                write_failed.set()
                raise ConnectionError("simulated observer write failure")
            return True

        # test-contract: seam fail the real observer transport write
        monkeypatch.setattr(observer_connection, "_write_physical", fail_grant_event)
        revoke_response = (
            await _request(
                reader,
                writer,
                5,
                "revoke_session_grants",
                {"grant": "bash"},
            )
        )[-1]
        assert "result" in revoke_response, revoke_response
        revoked = revoke_response["result"]
        assert revoked["removed"] == ["bash"]
        assert revoked["grants"]["session_grants"] == []
        await asyncio.wait_for(write_failed.wait(), 10)
        assert (await _request(reader, writer, 6, "session_grants"))[-1]["result"] == (
            revoked["grants"]
        )
    finally:
        await close_serve(
            server,
            writer,
            *([observer_writer] if observer_writer is not None else []),
        )


@pytest.mark.asyncio
async def test_session_grant_exactness_noops_feature_gate_and_fifo_sentinels(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        '[approval]\nallow = ["bash"]\n', encoding="utf-8"
    )
    call = ToolCall("call-1", "bash", {"command": "printf exact"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=home,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, backend_home: (
            backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    try:
        legacy = (
            await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        )[-1]["result"]
        assert "session_grants" not in legacy["capabilities"]["requests"]
        unavailable = (await _request(reader, writer, 2, "session_grants"))[-1]
        assert unavailable["error"] == {
            "code": -32601,
            "message": "session grants require the negotiated session_grants feature",
        }
    finally:
        await close_serve(server, writer)

    server = ZetaServer(
        home=home,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, backend_home: (
            backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    try:
        await _hello_and_create(reader, writer)
        assert server.runtime.policy is not None
        rule = server.runtime.policy.remember_allow(
            server.runtime.policy.request_from_store(
                "manual", ToolCall("manual", "bash", {"command": "printf exact"})
            )
        )
        assert str(rule) == "bash"
        snapshot = (await _request(reader, writer, 3, "session_grants"))[-1]["result"]
        assert snapshot["rules"]["allow"] == ["bash"]
        assert snapshot["session_grants"] == ["bash"]

        missing = (
            await _request(
                reader, writer, 4, "revoke_session_grants", {"grant": "bash(*)"}
            )
        )[-1]["error"]
        assert missing["data"] == {
            "code": "session_grant_not_found",
            "grant": "bash(*)",
        }
        # The read response is the positive FIFO sentinel: no change event can
        # precede it on this connection.
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "session_grants",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        sentinel = await _read(reader)
        assert sentinel["id"] == 5
        assert sentinel["result"]["session_grants"] == ["bash"]

        revoked = (
            await _request(
                reader, writer, 6, "revoke_session_grants", {"grant": "bash"}
            )
        )[-1]["result"]
        await _event(reader, "session_grants_changed")
        assert revoked["grants"]["rules"]["allow"] == ["bash"]
        assert revoked["grants"]["session_grants"] == []

        cleared = (
            await _request(reader, writer, 7, "revoke_session_grants", {"clear": True})
        )[-1]["result"]
        assert cleared["removed"] == []
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 8,
                        "method": "session_grants",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        sentinel = await _read(reader)
        assert sentinel["id"] == 8
        assert sentinel["result"]["session_grants"] == []
    finally:
        await close_serve(server, writer)


async def _collect_second_parent_terminal(
    reader: asyncio.StreamReader,
    release_child: asyncio.Event,
) -> tuple[list[dict[str, Any]], str]:
    frames: list[dict[str, Any]] = []
    while True:
        buffered = _buffer(reader)
        if buffered:
            frame = buffered.popleft()
        else:
            frame = await read(reader)
        frames.append(frame)
        if frame.get("params", {}).get("event") == "agent_end":
            release_child.set()
            return frames, "parent agent end"


async def _read_until_child_bash_two(
    reader: asyncio.StreamReader, frames: list[dict[str, Any]]
) -> str:
    while True:
        buffered = _buffer(reader)
        if buffered:
            frame = buffered.popleft()
        else:
            frame = await read(reader)
        frames.append(frame)
        params = frame.get("params", {})
        if params.get("tool_call", {}).get("id") != "child-bash-2":
            continue
        event = params.get("event")
        if event == "approval_request":
            return "child-bash-2 approval_request"
        if event in {"tool_start", "tool_end"}:
            return f"child-bash-2 {event} before any approval_request"


async def _first_child_bash_two_fact(
    reader: asyncio.StreamReader,
    frames: list[dict[str, Any]],
    publication_attempted: asyncio.Event,
) -> str:
    wire = asyncio.create_task(_read_until_child_bash_two(reader, frames))
    # test-contract: seam this coroutine is the await_or_fail argument.
    hook = asyncio.create_task(publication_attempted.wait())
    try:
        # test-contract: seam this coroutine is the await_or_fail argument.
        await asyncio.wait({wire, hook}, return_when=asyncio.FIRST_COMPLETED)
        if wire.done():
            return wire.result()
        return "publication hook"
    finally:
        for task in (wire, hook):
            task.cancel()
        # test-contract: seam this coroutine is the await_or_fail argument.
        await asyncio.gather(wire, hook, return_exceptions=True)


def test_approval_task_creation_has_one_owner() -> None:
    root = Path(__file__).parents[1]
    violations: list[tuple[str, int]] = []
    task_names = {"Task", "create_task", "ensure_future"}
    files = [
        root / "src/zeta/server/approval_lifecycle.py",
        *root.glob("src/zeta/server/approval_settlement/**/*.py"),
    ]
    for path in files:
        relative = str(path.relative_to(root))
        tree = ast.parse(path.read_text())
        imported_task_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "asyncio":
                for alias in node.names:
                    if alias.name in task_names:
                        imported_task_names.add(alias.asname or alias.name)

        class TaskCallVisitor(ast.NodeVisitor):
            def __init__(self, relative: str, imported_task_names: set[str]) -> None:
                self.classes: list[str] = []
                self.functions: list[str] = []
                self.relative = relative
                self.imported_task_names = imported_task_names

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.classes.append(node.name)
                self.generic_visit(node)
                self.classes.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                callee = node.func
                name = (
                    callee.id
                    if isinstance(callee, ast.Name)
                    else callee.attr
                    if isinstance(callee, ast.Attribute)
                    else None
                )
                is_task_call = name in task_names or name in self.imported_task_names
                allowed = (
                    self.relative.endswith("/approval_settlement/owner.py")
                    and self.classes == ["TaskOwner"]
                    and self.functions == ["start"]
                )
                if is_task_call and not allowed:
                    violations.append((self.relative, node.lineno))
                self.generic_visit(node)

        TaskCallVisitor(relative, imported_task_names).visit(tree)
    assert not violations, f"unowned approval task creation: {violations!r}"


@pytest.mark.asyncio
async def test_terminal_watchdog_failure_is_retrieved(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = approval_lifecycle.ApprovalLifecycle(
        request_turn_failure=lambda: None,
        policy=lambda: None,
        session_id=lambda: "session-1",
        parent_store=lambda: None,
        offload=cast(Any, None),
        is_closing=lambda: False,
    )
    request = approval_lifecycle.ApprovalRequest(
        "call-1", ToolCall("call-1", "bash", {"command": "true"}), child_instance_id="child-1"
    )
    lifecycle.register_delegated(request, cast(Any, None), "child-1")
    key = ("child-1", "call-1")
    assert lifecycle.release_or_hold_terminal(key, "session-1", {})

    async def fail_wait(*args: Any, **kwargs: Any) -> None:
        args[0].close()
        raise TimeoutError

    def fail_release(record: Any) -> None:
        raise RuntimeError("terminal release failed")

    monkeypatch.setattr(approval_lifecycle.asyncio, "wait_for", fail_wait)
    # test-contract: seam force the real terminal watchdog release to fail
    monkeypatch.setattr(lifecycle, "_release_terminal", fail_release)
    await lifecycle.wait_for_approval_settlement(0.1)
    assert any(
        record.message == "approval settlement task failed"
        and record.approval_key == key
        and record.kind == "terminal watchdog"
        for record in caplog.records
    )
    assert not lifecycle._settlement_tasks


@pytest.mark.asyncio
async def test_server_close_bounds_stuck_approval_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "bash", {"command": "true"})]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_persist = approval_settlement.persist_durably

    async def stuck_persist(*args: object, **kwargs: object) -> object:
        entered.set()
        await await_or_fail(release.wait(), what="stuck approval release")
        return await original_persist(*args, **kwargs)

    monkeypatch.setattr(approval_settlement, "persist_durably", stuck_persist)
    approve: asyncio.Task[object] | None = None
    try:
        await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "ask"})
        await _event(reader, "approval_request")
        approve = asyncio.create_task(
            _request(reader, writer, 4, "approve", {"request_id": "call-1"})
        )
        await await_or_fail(entered.wait(), what="stuck approval settlement")
        await server.close(approval_settlement_timeout=0.05)
        assert any(
            record.message == "approval settlement tasks stuck during close"
            and record.approval_keys == ["call-1"]
            for record in caplog.records
        )
    finally:
        release.set()
        if approve is not None:
            approve.cancel()
            await await_or_fail(
                asyncio.gather(approve, return_exceptions=True),
                what="approval request cleanup",
            )
        writer.close()
        await writer.wait_closed()
        await server.close()


@pytest.mark.asyncio
async def test_parent_grant_and_revoke_control_future_delegated_child_calls(
    tmp_path: Path,
) -> None:
    backend = _DelegatedBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "grant"})
        await _event(reader, "approval_request")
        await approve_and_collect_grant(
            reader, writer, 4, "parent-bash", "always_tool", ["bash"]
        )
        await _frames_until_event(reader, "turn_end")
        await _wait_idle(reader, writer, 20)

        delegated_send = await _request(
            reader, writer, 5, "send", {"text": "first delegate"}
        )
        assert delegated_send[-1].get("error") is None, delegated_send
        assert delegated_send[-1]["result"]["accepted"] is True
        while True:
            frame = await _read(reader)
            assert frame.get("params", {}).get("event") != "approval_request"
            params = frame.get("params", {})
            if (
                params.get("event") == "tool_end"
                and params.get("tool_call", {}).get("id") == "agent-1"
            ):
                break
        await _wait_idle(reader, writer, 30)

        publication_attempted = asyncio.Event()
        original_admit = server.control._approvals._admit

        def observe_publication(event, session_id, **fields):
            result = original_admit(event, session_id, **fields)
            if (
                event is EventDescriptor.APPROVAL_REQUEST
                and fields.get("delegated") is True
                and fields.get("tool_call", {}).get("id") == "child-bash-2"
            ):
                publication_attempted.set()
            return result

        server.control._approvals._admit = observe_publication
        await await_or_fail(
            backend.child_one_ready.wait(), what="child 1 post-revoke ordering gate"
        )

        async def second_delegate_journey() -> None:
            await _request(
                reader, writer, 6, "revoke_session_grants", {"grant": "bash"}
            )
            await _event(reader, "session_grants_changed")
            backend.release_child_one.set()
            child_one_approval = await _event(reader, "approval_request")
            assert child_one_approval["tool_call"]["id"] == "child-bash-1"
            assert child_one_approval["delegated"] is True
            child_one_deny = await _request(
                reader,
                writer,
                61,
                "deny",
                {"request_id": child_one_approval["request_id"]},
            )
            assert child_one_deny[-1].get("error") is None, child_one_deny
            await _frames_until_event(reader, "agent_end")
            second_send = await _request(
                reader, writer, 7, "send", {"text": "second delegate"}
            )
            # test-contract: seam this coroutine is the await_or_fail argument.
            await backend.child_two_ready.wait()
            snapshot = (await _request(reader, writer, 31, "session_grants"))[-1][
                "result"
            ]
            assert snapshot["session_grants"] == []
            frames, terminal = await _collect_second_parent_terminal(
                reader, backend.release_child_two
            )
            ended_on = await _first_child_bash_two_fact(
                reader, frames, publication_attempted
            )
            sentinel = await _request(reader, writer, 32, "session_grants")
            buffered = _buffer(reader)
            frames.extend(buffered)
            buffered.clear()
            frames.extend(sentinel)
            approval = next(
                (
                    frame["params"]
                    for frame in frames
                    if frame.get("params", {}).get("event") == "approval_request"
                    and frame.get("params", {}).get("tool_call", {}).get("id")
                    == "child-bash-2"
                ),
                None,
            )
            terminal = "delegated approval" if approval is not None else ended_on
            assert approval is not None, (
                "child-bash-2 did not request delegated approval; "
                f"wait ended on: {ended_on}; second send response={second_send!r}; "
                f"all buffered frames={frames!r}; "
                f"live parent session_grants={snapshot!r}"
            )
            assert terminal == "delegated approval"
            assert any(
                frame.get("params", {}).get("event") == "tool_end"
                and frame.get("params", {}).get("tool_call", {}).get("id")
                == "agent-2"
                for frame in frames
            )
            assert any(
                frame.get("params", {}).get("event") == "agent_end"
                for frame in frames
            )
            assert approval["tool_call"]["id"] == "child-bash-2"
            assert approval["delegated"] is True
            await _request(
                reader, writer, 8, "deny", {"request_id": approval["request_id"]}
            )
            await _frames_until_event(reader, "turn_end")

        await await_or_fail(
            second_delegate_journey(),
            timeout=300,
            what="second delegated approval journey",
        )
    finally:
        backend.release_child_one.set()
        backend.release_child_two.set()
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_duplicate_grant_emits_no_event_before_positive_fifo_sentinel(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    calls = [
        ToolCall("call-1", "bash", {"command": "printf one"}),
        ToolCall("call-2", "bash", {"command": "printf two"}),
    ]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[calls[0]]),
            ScriptedTurn([TextContent("first done")]),
        ]
    )
    server = ZetaServer(
        home=home,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "first call"})
        first = await _event(reader, "approval_request")
        await approve_and_collect_grant(
            reader, writer, 4, first["request_id"], "always_tool", ["bash"]
        )
        await _frames_until_event(reader, "turn_end")
        assert server.runtime.opened is not None
        server.runtime.opened.store.append_approval_request("call-2", calls[1])
        await _request(
            reader,
            writer,
            6,
            "approve",
            {"request_id": "call-2", "scope": "always_tool"},
        )
        # test-contract: transport inspect events before the raw response frame
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "session_grants",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        while True:
            sentinel = await _read(reader)
            assert sentinel.get("params", {}).get("event") != "session_grants_changed"
            if sentinel.get("id") == 7:
                break
        assert sentinel["result"]["session_grants"] == ["bash"]
        await _frames_until_event(reader, "turn_end")
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_replacement_controller_discards_cache_and_reads_current_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "bash", {"command": "printf handoff"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    replacement_writer: asyncio.StreamWriter | None = None
    try:
        _, session_id = await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "grant"})
        await _event(reader, "approval_request")
        approve_frames, buffered = await _approve_after_external_end(
            server,
            reader,
            writer,
            monkeypatch,
            session_id=session_id,
            request_id=4,
            approval_request_id="call-1",
        )
        assert approve_frames[-1]["result"]["accepted"] is True
        grant_events = [
            frame
            for frame in buffered
            if frame.get("params", {}).get("event") == "session_grants_changed"
        ]
        assert len(grant_events) == 1, buffered
        assert grant_events[0]["params"]["data"]["grants"]["session_grants"] == [
            "bash"
        ]
        await _frames_until_event(reader, "turn_end")

        assert server.connections is not None
        released = asyncio.Event()
        original_release = server.connections.release

        def release(connection: Any) -> None:
            original_release(connection)
            released.set()

        monkeypatch.setattr(server.connections, "release", release)
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)

        replacement_reader, replacement_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        hello = (
            await _request(
                replacement_reader,
                replacement_writer,
                10,
                "hello",
                {"protocol_version": "1.2", "features": FEATURES},
            )
        )[-1]["result"]
        assert "session_grants" in hello["capabilities"]["features"]
        # test-contract: transport read one snapshot response without event buffering
        replacement_writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 11,
                        "method": "session_grants",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        await replacement_writer.drain()
        snapshot_response = await _read(replacement_reader)
        assert snapshot_response["id"] == 11
        assert snapshot_response["result"]["session_grants"] == ["bash"]
    finally:
        await close_serve(
            server,
            *([replacement_writer] if replacement_writer is not None else []),
        )


@pytest.mark.asyncio
async def test_abort_canceled_settlement_keeps_controller_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall("abort-call", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    gate_enabled = False
    original_commit = ConversationStore._commit_storage_command

    async def gated_commit(store: ConversationStore, command: object) -> object:
        nonlocal gate_enabled
        result = await original_commit(store, command)
        if gate_enabled and type(command).__name__ == "_ResolveApprovalCommand":
            entered.set()
            await await_or_fail(release.wait(), what="abort settlement")
        return result

    # test-contract: seam gate the settlement's typed durable write owner
    monkeypatch.setattr(ConversationStore, "_commit_storage_command", gated_commit)
    reader, writer = await _connect(server)
    try:
        await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        gate_enabled = True
        abort = asyncio.create_task(_request(reader, writer, 4, "abort"))
        await await_or_fail(entered.wait(), what="abort settlement gate")
        record = server.control._approvals.state.records[approval["request_id"]]
        assert record.settlement is not None
        record.settlement.cancel()
        release.set()
        frames = await await_or_fail(abort, what="abort response")
        assert frames[-1]["result"] == {"aborted": True}
        approval_end = await _event(reader, "approval_end")
        assert approval_end["reason"] == "turn_canceled"
        status = (await _request(reader, writer, 5, "status"))[-1]
        assert status["result"]["state"] in {"idle", "running"}
    finally:
        release.set()
        await close_serve(server, writer)


@pytest.mark.parametrize("transport", ["unix", "websocket"])
@pytest.mark.parametrize(
    "case",
    [
        "normal_owner",
        "duplicate",
        "deny",
        "external_end",
        "cancel_settle",
        "cancel_finish",
        "cancel_publish",
        "cancel_history",
        "cancel_write",
        "cancel_owner_exit",
        "fail_approval_end_publish",
        "fail_grant_event_publish",
        "fail_response_write",
        "abort",
        "replacement",
        "restart_replay",
    ],
)
@pytest.mark.asyncio
async def test_session_grant_lifecycle_table_over_real_transports(
    tmp_path: Path, transport: str, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise grant ownership and every cancellation await over unix serve."""
    GrantPublication = getattr(approval_lifecycle, "GrantPublication", None)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "bash", {"command": "true"})]),
            ScriptedTurn(tool_calls=[ToolCall("call-2", "bash", {"command": "true"})]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path) if transport == "unix" else None,
        port=0 if transport == "websocket" else None,
        ws_port=0 if transport == "websocket" else None,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    if transport == "unix":
        reader, writer = await _connect(server)
    else:
        await server.start()
        reader, writer, status, _headers = await _connect_websocket(
            server, token=None, protocol_token=server._websocket.token
        )
        assert status == 101
    replacement_reader: asyncio.StreamReader | None = None
    replacement_writer: asyncio.StreamWriter | None = None
    observed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    async def read_websocket_json() -> dict[str, Any]:
        while True:
            try:
                return await _read_json(reader)
            except TimeoutError:
                continue
            except AssertionError as exc:
                raise AssertionError("server closed the websocket") from exc

    async def request(request_id: int, method: str, params: dict[str, object]) -> list[dict[str, Any]]:
        if transport == "unix":
            frames = await _request(reader, writer, request_id, method, params)
        else:
            await _send_json(writer, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            frames = []
            while True:
                frame = await read_websocket_json()
                frames.append(frame)
                if frame.get("id") == request_id:
                    break
        observed.extend(frames)
        if transport == "websocket":
            pending.extend(frames)
        return frames

    async def receive_response(request_id: int) -> list[dict[str, Any]]:
        frames = []
        while True:
            frame = await (
                _read(reader) if transport == "unix" else read_websocket_json()
            )
            frames.append(frame)
            if frame.get("id") == request_id:
                break
        observed.extend(frames)
        if transport == "websocket":
            pending.extend(frames)
        return frames

    async def event(name: str) -> dict[str, Any]:
        if transport == "websocket":
            for index, frame in enumerate(pending):
                if frame.get("params", {}).get("event") == name:
                    pending.pop(index)
                    return frame["params"]
        while True:
            frame = await (
                _event(reader, name)
                if transport == "unix"
                else read_websocket_json()
            )
            if transport == "websocket" and frame.get("params", {}).get("event") != name:
                pending.append(frame)
                continue
            observed.append(frame)
            return frame["params"] if transport == "websocket" else frame

    def buffered() -> list[dict[str, Any]]:
        return _buffer(reader) if transport == "unix" else pending

    async def frames_until(name: str) -> list[dict[str, Any]]:
        if transport == "unix":
            frames = await _frames_until_event(reader, name)
        else:
            frames = []
            while True:
                if pending:
                    frame = pending.pop(0)
                else:
                    frame = await read_websocket_json()
                frames.append(frame)
                if frame.get("params", {}).get("event") == name:
                    break
        observed.extend(frames)
        return frames

    async def hello_create() -> tuple[dict[str, Any], str]:
        hello = (await request(1, "hello", {"protocol_version": "1.2", "features": FEATURES}))[-1]["result"]
        assert "session_grants" in hello["capabilities"]["features"]
        created = (await request(2, "new_session", {"provider": "codex"}))[-1]["result"]
        reset = await event("session_grants_changed")
        assert reset["data"]["change"] == "reset"
        return hello, created["session"]["session_id"]
    try:
        _, session_id = await hello_create()
        await request( 3, "send", {"text": "grant"})
        assert (await event("approval_request"))["request_id"] == "call-1"
        if case in {"abort", "replacement"}:
            entered = asyncio.Event()
            release = asyncio.Event()
            original_commit = ConversationStore._commit_storage_command

            async def gated_replacement_commit(
                store: ConversationStore, command: object
            ) -> object:
                result = await original_commit(store, command)
                if type(command).__name__ == "_ResolveApprovalCommand":
                    # test-contract: seam gate the replacement's durable write owner
                    entered.set()
                    await await_or_fail(release.wait(), what="replacement settlement")
                return result

            # test-contract: seam gate the replacement settlement await
            monkeypatch.setattr(
                ConversationStore, "_commit_storage_command", gated_replacement_commit
            )
            abort = asyncio.create_task(request(4, "abort", {}))
            await await_or_fail(entered.wait(), what="replacement settlement gate")
            if case == "replacement":
                replacement_request = {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "new_session",
                    "params": {"provider": "codex"},
                }
                if transport == "unix":
                    writer.write((json.dumps(replacement_request) + "\n").encode())
                    await writer.drain()
                else:
                    await _send_json(writer, replacement_request)
            record = server.control._approvals.state.records["call-1"]
            assert record.settlement is not None
            record.settlement.cancel()
            release.set()
            abort_frames = await await_or_fail(abort, what="abort response")
            assert abort_frames[-1]["result"] == {"aborted": True}
            if case == "replacement":
                replacement_frames = await await_or_fail(
                    receive_response(5), what="new session response"
                )
                assert (
                    replacement_frames[-1]["result"]["session"]["session_id"]
                    != session_id
                )
            if case == "abort":
                await event("approval_end")
            approval_ends = [
                params
                for frame in observed
                if (params := frame.get("params", frame)).get("event")
                == "approval_end"
                and params.get("request_id") == "call-1"
            ]
            assert len(approval_ends) == 1
            assert approval_ends[0]["reason"] == "turn_canceled"
            status = (await request(6, "status", {}))[-1]
            assert status["result"]["state"] in {"idle", "running"}
            assert not server.control._approvals.state.records
            return
        cancel = case.startswith("cancel_")
        if case == "deny":
            response = (
                await request( 4, "deny", {"request_id": "call-1"})
            )[-1]
            assert response["result"]["decision"] == "deny"
            assert not any(
                frame.get("params", {}).get("event") == "session_grants_changed"
                for frame in buffered()
            )
        else:
            entered = asyncio.Event()
            release = asyncio.Event()
            settlement: asyncio.Task[object] | None = None
            gate_owner: asyncio.Task[object] | None = None
            all_frames: list[dict[str, Any]] = []
            failed_write = asyncio.Event()
            failed_payloads: list[bytes] = []
            if case.startswith("fail_"):
                assert server.connections is not None
                controller = server.connections._controller
                assert controller is not None
                original_write = controller.connection._write_physical

                async def fail_named_publish(payload: bytes) -> bool:
                    compact = payload.replace(b" ", b"")
                    should_fail = (
                        case == "fail_approval_end_publish"
                        and b'"event":"approval_end"' in compact
                        or case == "fail_grant_event_publish"
                        and b'"event":"session_grants_changed"' in compact
                        and b'"change":"grant"' in compact
                        or case == "fail_response_write"
                        and b'"id":4' in compact
                        and b'"result"' in compact
                    )
                    if should_fail and not failed_write.is_set():
                        failed_payloads.append(payload)
                        failed_write.set()
                        raise OSError(f"injected {case}")
                    return await original_write(payload)

                # test-contract: seam fail each controller publication write once
                monkeypatch.setattr(
                    controller.connection, "_write_physical", fail_named_publish
                )
            if case in {"cancel_settle", "external_end"}:
                original_commit = ConversationStore._commit_storage_command

                async def gated_commit(store: ConversationStore, command: object) -> object:
                    nonlocal gate_owner
                    is_resolution = type(command).__name__ == "_ResolveApprovalCommand"
                    if case == "cancel_settle" and is_resolution:
                        # test-contract: prove the settlement task owns the gated await
                        gate_owner = asyncio.current_task()
                        record = server.control._approvals.state.records["call-1"]
                        assert gate_owner is record.settlement
                        # test-contract: cancel before the real durable operation begins
                        entered.set()
                        await await_or_fail(release.wait(), what="durable settlement")
                    result = await original_commit(store, command)
                    if case == "external_end" and is_resolution:
                        # test-contract: prove the settlement task owns the gated await
                        gate_owner = asyncio.current_task()
                        record = server.control._approvals.state.records["call-1"]
                        assert gate_owner is record.settlement
                        # test-contract: gate after the real durable operation commits
                        entered.set()
                        await await_or_fail(release.wait(), what="durable settlement")
                    return result

                # test-contract: seam gate the typed durable write owner
                monkeypatch.setattr(
                    ConversationStore, "_commit_storage_command", gated_commit
                )
            elif case == "cancel_finish" and GrantPublication is not None:
                original_finish_grant = approval_lifecycle.finish_grant

                async def gated_finish_grant(
                    *args: object, **kwargs: object
                ) -> tuple[bool, bool, bool]:
                    nonlocal gate_owner
                    gate_owner = asyncio.current_task()
                    record = server.control._approvals.state.records["call-1"]
                    assert gate_owner is record.settlement
                    entered.set()
                    try:
                        await await_or_fail(release.wait(), what="grant finish")
                    except asyncio.CancelledError:
                        committed, published, _ = await original_finish_grant(
                            *args, **kwargs
                        )
                        return committed, published, True
                    return await original_finish_grant(*args, **kwargs)

                monkeypatch.setattr(
                    approval_lifecycle, "finish_grant", gated_finish_grant
                )
            elif case == "cancel_publish" and GrantPublication is not None:
                original_finish_grant = approval_lifecycle.finish_grant
                original_publish = GrantPublication.publish
                publish_entered = asyncio.Event()
                release_publish = asyncio.Event()

                async def gated_publish(
                    publication: GrantPublication, rule: str
                ) -> None:
                    publish_entered.set()
                    await await_or_fail(
                        release_publish.wait(), what="grant publication"
                    )
                    await original_publish(publication, rule)

                async def gated_publishing_finish(
                    *args: object, **kwargs: object
                ) -> tuple[bool, bool, bool]:
                    nonlocal gate_owner
                    operation = asyncio.create_task(
                        original_finish_grant(*args, **kwargs)
                    )
                    await await_or_fail(
                        publish_entered.wait(), what="grant publish entered"
                    )
                    gate_owner = asyncio.current_task()
                    record = server.control._approvals.state.records["call-1"]
                    assert gate_owner is record.settlement
                    entered.set()
                    try:
                        await await_or_fail(release.wait(), what="grant publish owner")
                    except asyncio.CancelledError:
                        release_publish.set()
                        committed, published, _ = await await_or_fail(
                            operation, what="canceled grant publication"
                        )
                        return committed, published, True
                    release_publish.set()
                    return await await_or_fail(
                        operation, what="completed grant publication"
                    )

                monkeypatch.setattr(GrantPublication, "publish", gated_publish)
                monkeypatch.setattr(
                    approval_lifecycle, "finish_grant", gated_publishing_finish
                )
            elif case == "cancel_history":
                original_history = ConversationStore.append_approval_history_async
                original_persist = approval_settlement.persist_durably
                original_finish = server.control._approvals._finish_record
                committed = asyncio.Event()
                release_persist = asyncio.Event()
                competing_entered = asyncio.Event()
                release_competing = asyncio.Event()

                async def gated_persist(*args: object, **kwargs: object) -> object:
                    result = await original_persist(*args, **kwargs)
                    committed.set()
                    await await_or_fail(
                        release_persist.wait(), what="approval settlement commit"
                    )
                    return result

                async def gated_finish(
                    record: Any, resolution: dict[str, str]
                ) -> None:
                    if record.target.key == "call-1":
                        owner = asyncio.current_task()
                        current = server.control._approvals.state.records.get("call-1")
                        if current is None:
                            return
                        if current.settlement is not owner:
                            competing_entered.set()
                            await await_or_fail(
                                release_competing.wait(),
                                what="competing approval publication",
                            )
                    await original_finish(record, resolution)

                async def gated_history(
                    store: ConversationStore, payload: dict[str, object]
                ) -> object:
                    nonlocal gate_owner
                    record = server.control._approvals.state.records["call-1"]
                    # test-contract: seam prove the settlement task owns this await
                    gate_owner = asyncio.current_task()
                    assert gate_owner is record.settlement
                    entered.set()
                    try:
                        await await_or_fail(release.wait(), what="approval history")
                    except asyncio.CancelledError:
                        await original_history(store, payload)
                        raise
                    return await original_history(store, payload)

                monkeypatch.setattr(
                    approval_settlement, "persist_durably", gated_persist
                )
                # test-contract: seam force a competing publication owner order
                monkeypatch.setattr(
                    server.control._approvals, "_finish_record", gated_finish
                )
                monkeypatch.setattr(
                    ConversationStore,
                    "append_approval_history_async",
                    gated_history,
                )
            elif case == "cancel_owner_exit":
                original_settle_record = (
                    approval_lifecycle.ApprovalLifecycle._settle_record
                )

                async def gated_owner_exit(
                    lifecycle: Any,
                    record: Any,
                    decision: str,
                    reason: str | None = None,
                ) -> dict[str, str]:
                    nonlocal settlement
                    nonlocal gate_owner
                    result = await original_settle_record(
                        lifecycle, record, decision, reason
                    )
                    settlement = asyncio.current_task()
                    gate_owner = settlement
                    # test-contract: seam capture the lifecycle owner task
                    assert settlement is not None
                    # test-contract: seam gate owner exit after shared admission
                    entered.set()
                    await await_or_fail(release.wait(), what="settlement owner exit")
                    return result

                # test-contract: seam gate the lifecycle owner exit
                monkeypatch.setattr(
                    approval_lifecycle.ApprovalLifecycle,
                    "_settle_record",
                    gated_owner_exit,
                )
            elif case == "cancel_write":
                assert server.connections is not None
                controller = server.connections._controller
                assert controller is not None
                original_write = controller.connection._write_physical

                async def gated_write(payload: bytes) -> bool:
                    nonlocal gate_owner
                    if b"session_grants_changed" in payload:
                        # test-contract: seam gate the real dead-peer write
                        gate_owner = asyncio.current_task()
                        entered.set()
                        await await_or_fail(release.wait(), what="grant physical write")
                    return await original_write(payload)

                # test-contract: seam gate the real physical-write await
                monkeypatch.setattr(
                    controller.connection, "_write_physical", gated_write
                )
            if case in {"cancel_finish", "cancel_publish"} and GrantPublication is None:
                original_grant_publish = server.control.publish_session_grant_change

                async def gated_grant_publish(change: Any) -> None:
                    nonlocal gate_owner
                    # test-contract: seam gate the named production await
                    gate_owner = asyncio.current_task()
                    record = server.control._approvals.state.records["call-1"]
                    assert gate_owner is record.settlement
                    entered.set()
                    await await_or_fail(release.wait(), what="grant publication")
                    await original_grant_publish(change)

                monkeypatch.setattr(
                    server.control, "publish_session_grant_change", gated_grant_publish
                )

            if case == "cancel_history":
                record = server.control._approvals.state.records["call-1"]
                server.control._approvals._change(
                    record,
                    target=replace(record.target, history_store=record.target.store),
                    history_published=True,
                )
            approve_params = {
                "request_id": "call-1",
                "scope": "always_tool",
            }
            approve: asyncio.Task[list[dict[str, Any]]] | None = None
            if case == "fail_response_write":
                approve_request = {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "approve",
                    "params": approve_params,
                }
                if transport == "unix":
                    writer.write((json.dumps(approve_request) + "\n").encode())
                    await writer.drain()
                else:
                    await _send_json(writer, approve_request)
                await await_or_fail(failed_write.wait(), what="approve response failure")
            else:
                approve = asyncio.create_task(request(4, "approve", approve_params))
            competing: asyncio.Task[object] | None = None
            if case == "cancel_history":
                await await_or_fail(committed.wait(), what="approval settlement commit")
                record = server.control._approvals.state.records["call-1"]
                resolution = record.target.store.approval_resolution_data("call-1")
                assert resolution is not None
                competing = asyncio.create_task(
                    server.control._approvals._finish_record(record, resolution)
                )
                await await_or_fail(
                    competing_entered.wait(), what="competing approval publication"
                )
                release_persist.set()
            if case == "external_end":
                await await_or_fail(entered.wait(), what="durable settlement")
                external_end = asyncio.create_task(
                    server.control._approvals.transition(
                        ExternalEnd("call-1", session_id, {})
                    )
                )
                release.set()
                await await_or_fail(external_end, what="external approval end")
            if cancel:
                await await_or_fail(entered.wait(), what=f"{case} gate")
                record = server.control._approvals.state.records.get("call-1")
                if record is not None:
                    settlement = record.settlement
                original_turn = server.control._turn_task
                assert original_turn is not None
                assert gate_owner is not None
                assert not gate_owner.done()
                if case == "cancel_write":
                    # test-contract: cancel_write aborts the real controller transport
                    assert (
                        gate_owner.get_coro().__qualname__
                        == "ConnectionRegistry._write_physical"
                    )
                    writer.transport.abort()
                    release.set()
                    await await_or_fail(
                        asyncio.gather(original_turn, return_exceptions=True),
                        what=f"{case} original turn task",
                    )
                    assert not server.control._approvals.state.records
                    if transport == "unix":
                        reader, writer = await asyncio.open_unix_connection(
                            str(server.socket_path)
                        )
                    else:
                        reader, writer, status, _headers = await _connect_websocket(
                            server, token=None, protocol_token=server._websocket.token
                        )
                        assert status == 101
                    pending.clear()
                    await request(
                        10,
                        "hello",
                        {
                            "protocol_version": "1.2",
                            "features": [*FEATURES, "approval_reconnect"],
                        },
                    )
                    snapshot_frames = await request(11, "session_grants", {})
                    assert snapshot_frames[-1]["result"]["session_grants"] == ["bash"]
                    # Force the real busy predicate to remain active while the
                    # replacement controller queries status and sends.
                    release_handoff = asyncio.Event()
                    handoff = asyncio.create_task(
                        await_or_fail(
                            release_handoff.wait(),
                            timeout=10,
                            what="replacement turn handoff gate",
                        )
                    )
                    server.control._turn_handoff_task = handoff
                    try:
                        busy_status = (await request(12, "status", {}))[-1]
                        assert busy_status["result"]["state"] == "running"
                        busy_send = await request(13, "send", {"text": "ask again"})
                        assert busy_send[-1]["error"]["code"] == -32004
                    finally:
                        release_handoff.set()
                        await await_or_fail(
                            handoff, timeout=10, what="replacement turn handoff"
                        )

                    idle_status = (await request(14, "status", {}))[-1]
                    assert idle_status["result"]["state"] == "idle"
                    send_frames = await request(15, "send", {"text": "ask again"})
                    assert send_frames[-1].get("result", {}).get("accepted") is True, (
                        "cancel_write replacement send was not accepted: "
                        f"{send_frames[-1]!r}"
                    )
                    next_turn = await frames_until("turn_end")
                    assert not any(
                        frame.get("params", {}).get("event") == "approval_request"
                        for frame in next_turn
                    ), "cancel_write replacement unexpectedly asked for approval"
                    changed = [
                        frame
                        for frame in observed
                        if frame.get("params", {}).get("event")
                        == "session_grants_changed"
                        and frame.get("params", {}).get("data", {}).get("change")
                        == "grant"
                    ]
                    assert len(changed) == 0, (
                        "cancel_write must publish no grant-change event to a dead "
                        f"peer; observed {changed!r}"
                    )
                    assert not server.control._approvals.state.records
                    return
                assert approve is not None
                if case in {"cancel_settle", "cancel_history"}:
                    assert gate_owner is settlement
                if settlement is not None:
                    assert not settlement.done()
                gate_owner.cancel()
                release.set()
                if case == "cancel_history":
                    release_competing.set()
                frames = await await_or_fail(approve, what=f"{case} approve response")
                all_frames.extend(frames)
                assert frames[-1]["result"]["decision"] == "approve"
                if competing is not None:
                    await await_or_fail(
                        competing, what="competing approval publication"
                    )
            elif case.startswith("fail_"):
                if approve is not None:
                    frames = await await_or_fail(approve, what=f"{case} response")
                    all_frames.extend(frames)
                await await_or_fail(failed_write.wait(), what=f"{case} write")
                assert len(failed_payloads) == 1
                status_frames = await await_or_fail(
                    request(10, "status", {}), what=f"{case} live controller status"
                )
                all_frames.extend(status_frames)
                assert status_frames[-1]["result"]["state"] in {
                    "idle",
                    "running",
                    "tool",
                }
                snapshot_frames = await request(11, "session_grants", {})
                all_frames.extend(snapshot_frames)
                assert snapshot_frames[-1]["result"]["session_grants"] == ["bash"]
                assert server.runtime.opened is not None
                resolution = server.runtime.opened.store.approval_resolution_data(
                    "call-1"
                )
                assert resolution is not None
                assert resolution["decision"] == "allow"
                wire_frames = (
                    [*all_frames, *buffered()]
                    if transport == "unix"
                    else all_frames
                )
                grant_events = [
                    frame
                    for frame in wire_frames
                    if frame.get("params", {}).get("event")
                    == "session_grants_changed"
                    and frame.get("params", {}).get("data", {}).get("change")
                    == "grant"
                ]
                approval_ends = [
                    frame
                    for frame in wire_frames
                    if frame.get("params", {}).get("event") == "approval_end"
                ]
                responses = [frame for frame in wire_frames if frame.get("id") == 4]
                assert len(grant_events) == (
                    0 if case == "fail_grant_event_publish" else 1
                )
                assert len(approval_ends) == (1 if case == "fail_response_write" else 0)
                assert len(responses) == (0 if case == "fail_response_write" else 1)
                if case == "fail_grant_event_publish":
                    assert responses[0]["error"]["message"] == (
                        "approval persistence failed: injected fail_grant_event_publish"
                    )
                elif case == "fail_approval_end_publish":
                    assert responses[0]["result"]["decision"] == "approve"
                assert not server.control._approvals.state.records
                return
            else:
                assert approve is not None
                frames = await await_or_fail(approve, what="approve response")
                all_frames.extend(frames)
                assert frames[-1]["result"]["decision"] == "approve"

            if cancel:
                if case == "cancel_history":
                    assert server.runtime.opened is not None
                    history = [
                        entry.data
                        for entry in server.runtime.opened.store.replay()
                        if entry.type == "approval_history"
                    ]
                    assert [row["event"] for row in history] == ["approval_end"]
                    assert history[0]["request_id"] == "call-1"
                published = [
                    frame
                    for frame in buffered()
                    if frame.get("params", {}).get("event") == "session_grants_changed"
                ]
                assert len(published) == 1, (
                    f"{case} must publish one grant-change event; observed {published!r}"
                )
                await await_or_fail(
                    asyncio.gather(original_turn, return_exceptions=True),
                    what=f"{case} original turn end",
                )
                status = (
                    await await_or_fail(
                        request(10, "status", {}),
                        what=f"{case} live controller status",
                    )
                )[-1]
                assert status["result"]["state"] == "idle", (
                    f"{case} live controller did not remain connected: {status!r}"
                )
                snapshot_frames = await request(11, "session_grants", {})
                all_frames.extend(snapshot_frames)
                snapshot = snapshot_frames[-1]
                assert snapshot["result"]["session_grants"] == ["bash"]
                send_frames = await request(12, "send", {"text": "ask again"}
                )
                all_frames.extend(send_frames)
                send_response = send_frames[-1]
                assert send_response.get("result", {}).get("accepted") is True
                next_turn = await await_or_fail(
                    frames_until("turn_end"),
                    what=f"{case} next turn end",
                )
                all_frames.extend(next_turn)
                assert not any(
                    frame.get("params", {}).get("event") == "approval_request"
                    for frame in next_turn
                )
                changed: list[dict[str, Any]] = []
                for frame in [*all_frames, *observed]:
                    if (
                        frame.get("params", {}).get("event") == "session_grants_changed"
                        and frame.get("params", {}).get("data", {}).get("change") == "grant"
                        and frame not in changed
                    ):
                        changed.append(frame)
                assert len(changed) == 1, (
                    f"{case} must publish exactly one grant-change event; observed {changed!r}"
                )
                assert not server.control._approvals.state.records, (
                    f"{case} left an approval record after grant settlement"
                )
            else:
                grant_events = [
                    frame
                    for frame in buffered()
                    if frame.get("params", {}).get("event") == "session_grants_changed"
                ]
                assert len(grant_events) == (0 if case == "deny" else 1)
                if case == "duplicate":
                    buffered().clear()
                    assert server.runtime.opened is not None
                    duplicate_call = ToolCall(
                        "call-2", "bash", {"command": "true"}
                    )
                    server.runtime.opened.store.append_approval_request(
                        "call-2", duplicate_call
                    )
                    before = (await request(5, "session_grants", {}))[-1]
                    assert before["result"]["session_grants"] == ["bash"]
                    duplicate_response = (
                        await request(6,
                            "approve",
                            {"request_id": "call-2", "scope": "always_tool"},
                        )
                    )[-1]
                    assert duplicate_response.get("result", {}).get("accepted") is True, (
                        f"duplicate approval was not accepted: {duplicate_response!r}"
                    )
                    sentinel = (await request(7, "session_grants", {}))[-1]
                    assert sentinel["result"]["session_grants"] == ["bash"]
                    duplicate_events = [
                        frame
                        for frame in buffered()
                        if frame.get("params", {}).get("event")
                        == "session_grants_changed"
                    ]
                    assert not duplicate_events, (
                        "duplicate grant changed the event stream: "
                        f"{duplicate_events!r}"
                    )
        if not cancel:
            restarted = case == "restart_replay"
            if restarted:
                await frames_until("turn_end")
                await close_serve(server, writer)
                writer = None  # type: ignore[assignment]
                replay_backend = FakeBackend(
                    [
                        ScriptedTurn(
                            tool_calls=[ToolCall("call-2", "bash", {"command": "true"})]
                        ),
                        ScriptedTurn([TextContent("done")]),
                    ]
                )
                server = ZetaServer(
                    home=tmp_path,
                    socket_path=_socket_path(tmp_path) if transport == "unix" else None,
                    port=0 if transport == "websocket" else None,
                    ws_port=0 if transport == "websocket" else None,
                    provider="codex",
                    backend_factory=lambda provider, model, home: (
                        replay_backend,
                        model or "offline",
                    ),
                )
                if transport == "unix":
                    reader, writer = await _connect(server)
                else:
                    await server.start()
                    reader, writer, status, _headers = await _connect_websocket(
                        server, token=None, protocol_token=server._websocket.token
                    )
                    assert status == 101
                pending.clear()
                await request(10,
                    "hello",
                    {"protocol_version": "1.2", "features": FEATURES},
                )
                await request(11,
                    "resume",
                    {"session_id": session_id},
                )
                reset = await event("session_grants_changed")
                assert reset["data"]["change"] == "reset"
            snapshot = (await request(6, "session_grants", {}))[-1][
                "result"
            ]
            expected_grants = [] if case in {"deny", "restart_replay"} else ["bash"]
            assert snapshot["session_grants"] == expected_grants
            if restarted:
                await request( 12, "send", {"text": "ask again"})
                assert (await event("approval_request"))[
                    "request_id"
                ] == "call-2"
                await request(13,
                    "deny",
                    {"request_id": "call-2"},
                )
                await frames_until("turn_end")
            else:
                await frames_until("turn_end")
            assert not server.control._approvals.state.records
    finally:
        release.set() if "release" in locals() else None
        if "release_competing" in locals():
            release_competing.set()
        if "release_publish" in locals():
            release_publish.set()
        if "competing" in locals() and competing is not None and not competing.done():
            await await_or_fail(competing, what="competing approval cleanup")
        await close_serve(
            server,
            writer,
            *([replacement_writer] if replacement_writer is not None else []),
        )
        assert not server.control._approvals._settlement_tasks
