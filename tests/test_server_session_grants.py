"""Real-transport coverage for negotiated session-grant control."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

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
from zeta.server import ZetaServer, approval_lifecycle
from zeta.server.approval_lifecycle import ExternalEnd
from zeta.tui.app import TUIApp

FEATURES = ["session_grants"]
PRECEDENCE = ["deny", "ask", "allow", "default"]


class _DelegatedBackend:
    def __init__(self) -> None:
        self.child_two_ready = asyncio.Event()
        self.release_child_two = asyncio.Event()

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
            if number == "2":
                self.child_two_ready.set()
                await await_or_fail(
                    self.release_child_two.wait(),
                    timeout=120,
                    what="release child-bash-2 after parent terminal",
                )
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
        async for event in FakeBackend([turn]).complete(messages, []):
            yield event


class _GatedBackend(FakeBackend):
    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.release_next_call = asyncio.Event()

    async def complete(self, messages: Any, tool_schemas: Any):
        if len(self.calls) == 1:
            await self.release_next_call.wait()
        async for event in super().complete(messages, tool_schemas):
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


async def _assert_completes_in_ready_turns(
    task: asyncio.Task[Any], *, turns: int = 20
) -> Any:
    loop = asyncio.get_running_loop()
    for _ in range(turns):
        if task.done():
            break
        turn = loop.create_future()
        loop.call_soon(turn.set_result, None)
        await turn
    assert task.done(), "controller response waited for observer output"
    return await task


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
        await approve_and_collect_grant(
            reader, writer, 4, "call-1", "always_tool", ["bash"]
        )

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
    original_persist = approval_lifecycle.persist_durably
    original_finish = server.control._approvals._finish_record

    async def gated_finish(record: object, resolution: dict[str, str]) -> None:
        external_entered.set()
        await original_finish(record, resolution)  # type: ignore[arg-type]

    async def gated_persist(*args: object, **kwargs: object) -> object:
        result = await original_persist(*args, **kwargs)
        committed.set()
        await asyncio.wait_for(release.wait(), 10)
        return result

    monkeypatch.setattr(approval_lifecycle, "persist_durably", gated_persist)
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
async def test_session_grant_flow_over_authenticated_websocket(tmp_path: Path) -> None:
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
    reader, writer, status, _headers = await _connect_websocket(
        server, token=None, protocol_token=server._websocket.token
    )
    assert status == 101

    async def request(
        request_id: int, method: str, params: dict[str, object]
    ) -> list[dict[str, Any]]:
        await _send_json(
            writer,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        frames = []
        while True:
            frame = await _read_json(reader)
            frames.append(frame)
            if frame.get("id") == request_id:
                return frames

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
        created = await request(2, "new_session", {"provider": "codex"})
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
        while approval is None:
            frame = await _read_json(reader)
            if frame.get("params", {}).get("event") == "approval_request":
                approval = frame
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
        await close_serve(server, writer)


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
    try:
        await _hello_and_create(reader, writer)
        observer_reader, observer_writer = await _observer(
            server, 10, features=["serve_observers", "session_grants"]
        )
        await _request(reader, writer, 3, "send", {"text": "run"})
        await _event(reader, "approval_request")
        assert server.connections is not None
        observer_connection = server.connections._observers[0].connection
        original_write = observer_connection._write_physical
        observer_blocked = asyncio.Event()
        release_observer = asyncio.Event()

        async def block_observer(payload: bytes) -> bool:
            if b"session_grants_changed" in payload:
                observer_blocked.set()
                await asyncio.wait_for(release_observer.wait(), 10)
            return await original_write(payload)

        # test-contract: seam gate the real observer transport write
        monkeypatch.setattr(observer_connection, "_write_physical", block_observer)
        approve = asyncio.create_task(
            approve_and_collect_grant(
                reader, writer, 4, "call-1", "always_tool", ["bash"]
            )
        )
        await asyncio.wait_for(observer_blocked.wait(), 10)
        controller_event = await asyncio.wait_for(
            _assert_completes_in_ready_turns(approve), 10
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
        revoke_frames = await asyncio.wait_for(
            _assert_completes_in_ready_turns(revoke), 10
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


async def _collect_second_delegated_approval(
    reader: asyncio.StreamReader,
    release_child: asyncio.Event,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    frames: list[dict[str, Any]] = []

    async def collect() -> tuple[dict[str, Any] | None, str]:
        while True:
            buffered = _buffer(reader)
            if buffered:
                frame = buffered.popleft()
            else:
                line = await reader.readline()
                if not line:
                    return None, "socket eof"
                frame = json.loads(line)
            frames.append(frame)
            params = frame.get("params", {})
            event = params.get("event")
            if event == "agent_end" and not release_child.is_set():
                release_child.set()
            if (
                event == "approval_request"
                and params.get("tool_call", {}).get("id") == "child-bash-2"
            ):
                return params, "delegated approval"

    approval, terminal = await await_or_fail(
        collect(), timeout=60, what="second delegated approval collection"
    )
    return approval, frames, terminal


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

        await _request(reader, writer, 6, "revoke_session_grants", {"grant": "bash"})
        await _event(reader, "session_grants_changed")
        second_send = await _request(
            reader, writer, 7, "send", {"text": "second delegate"}
        )
        await await_or_fail(
            backend.child_two_ready.wait(),
            timeout=60,
            what="child-bash-2 provider gate",
        )
        snapshot = (await _request(reader, writer, 31, "session_grants"))[-1]["result"]
        assert snapshot["session_grants"] == []
        approval, frames, terminal = await _collect_second_delegated_approval(
            reader, backend.release_child_two
        )
        assert approval is not None, (
            "child-bash-2 did not request delegated approval; "
            f"terminal={terminal}; second send response={second_send!r}; "
            f"all buffered frames={frames!r}; "
            f"live parent session_grants={snapshot!r}"
        )
        assert terminal == "delegated approval"
        assert any(
            frame.get("params", {}).get("event") == "tool_end"
            and frame.get("params", {}).get("tool_call", {}).get("id") == "agent-2"
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
    finally:
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
        await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "grant"})
        await _event(reader, "approval_request")
        await approve_and_collect_grant(
            reader, writer, 4, "call-1", "always_tool", ["bash"]
        )
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
        "restart_replay",
    ],
)
@pytest.mark.asyncio
async def test_session_grant_lifecycle_table_over_unix_protocol(
    tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
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
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    replacement_reader: asyncio.StreamReader | None = None
    replacement_writer: asyncio.StreamWriter | None = None
    try:
        _, session_id = await _hello_and_create(reader, writer)
        await _request(reader, writer, 3, "send", {"text": "grant"})
        assert (await _event(reader, "approval_request"))["request_id"] == "call-1"
        cancel = case.startswith("cancel_")
        if case == "deny":
            response = (
                await _request(reader, writer, 4, "deny", {"request_id": "call-1"})
            )[-1]
            assert response["result"]["decision"] == "deny"
            assert not any(
                frame.get("params", {}).get("event") == "session_grants_changed"
                for frame in _buffer(reader)
            )
        else:
            entered = asyncio.Event()
            release = asyncio.Event()
            settlement: asyncio.Task[object] | None = None
            if case in {"cancel_settle", "external_end"}:
                original_durable = ConversationStore._to_thread_durable

                async def gated_durable(*args: object, **kwargs: object) -> object:
                    result = await original_durable(*args, **kwargs)
                    # test-contract: seam gate the named production await
                    entered.set()
                    await await_or_fail(release.wait(), what="durable settlement")
                    return result

                # test-contract: seam gate the real durable await without a production seam
                monkeypatch.setattr(
                    ConversationStore, "_to_thread_durable", gated_durable
                )
            elif case == "cancel_finish" and GrantPublication is not None:
                original_finish = GrantPublication.finish

                async def gated_finish(
                    publication: GrantPublication,
                    request: Any,
                    committed: bool,
                    event_published: bool,
                ) -> tuple[bool, bool]:
                    # test-contract: seam gate the named production await
                    entered.set()
                    await await_or_fail(release.wait(), what="grant finish")
                    return await original_finish(
                        publication, request, committed, event_published
                    )

                monkeypatch.setattr(GrantPublication, "finish", gated_finish)
            elif case == "cancel_publish" and GrantPublication is not None:
                original_publish = GrantPublication.publish

                async def gated_publish(
                    publication: GrantPublication, rule: str
                ) -> None:
                    # test-contract: seam gate the named production await
                    entered.set()
                    await await_or_fail(release.wait(), what="grant publication")
                    await original_publish(publication, rule)

                monkeypatch.setattr(GrantPublication, "publish", gated_publish)
            elif case == "cancel_history":
                original_history = ConversationStore.append_approval_history_async

                async def gated_history(
                    store: ConversationStore, payload: dict[str, object]
                ) -> object:
                    result = await original_history(store, payload)
                    # test-contract: seam gate the named production await
                    entered.set()
                    await await_or_fail(release.wait(), what="approval history")
                    return result

                monkeypatch.setattr(
                    ConversationStore,
                    "append_approval_history_async",
                    gated_history,
                )
            elif case == "cancel_write":
                assert server.connections is not None
                controller = server.connections._controller
                assert controller is not None
                original_write = controller.connection._write_physical

                async def gated_write(payload: bytes) -> bool:
                    if b"session_grants_changed" in payload:
                        # test-contract: seam gate the named production await
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
                    # test-contract: seam gate the named production await
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
            approve = asyncio.create_task(
                _request(
                    reader,
                    writer,
                    4,
                    "approve",
                    {"request_id": "call-1", "scope": "always_tool"},
                )
            )
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
                cancel_target = settlement or approve
                if settlement is not None:
                    assert not settlement.done()
                cancel_target.cancel()
                release.set()
                with pytest.raises((asyncio.CancelledError, AssertionError)):
                    await await_or_fail(approve, what="canceled approve")
            else:
                frames = await await_or_fail(approve, what="approve response")
                assert frames[-1]["result"]["decision"] == "approve"

            if cancel:
                published = [
                    frame
                    for frame in _buffer(reader)
                    if frame.get("params", {}).get("event") == "session_grants_changed"
                ]
                assert len(published) == 1, (
                    f"{case} must publish one grant-change event before replacement; "
                    f"observed {published!r}"
                )
                assert not server.control._approvals.state.records, (
                    f"{case} left an approval record after grant settlement"
                )
                original_frames = list(_buffer(reader))
                _buffer(reader).clear()
                await await_or_fail(
                    asyncio.gather(original_turn, return_exceptions=True),
                    what=f"{case} original turn end",
                )
                (
                    replacement_reader,
                    replacement_writer,
                ) = await asyncio.open_unix_connection(str(server.socket_path))
                replacement_frames = await _request(
                    replacement_reader,
                    replacement_writer,
                    10,
                    "hello",
                    {"protocol_version": "1.2", "features": FEATURES},
                )
                snapshot_frames = await _request(
                    replacement_reader, replacement_writer, 11, "session_grants"
                )
                snapshot = snapshot_frames[-1]["result"]
                assert snapshot["session_grants"] == ["bash"]
                original_status = (
                    await await_or_fail(
                        _request(replacement_reader, replacement_writer, 12, "status"),
                        what=f"{case} original turn terminal status",
                    )
                )[-1]
                assert original_status["result"]["state"] == "idle", (
                    f"{case} replacement controller did not observe the original "
                    "turn reaching idle"
                )
                send_frames = await await_or_fail(
                    _request(
                        replacement_reader,
                        replacement_writer,
                        13,
                        "send",
                        {"text": "ask again"},
                    ),
                    what=f"{case} replacement send response",
                )
                send_response = send_frames[-1]
                assert send_response.get("result", {}).get("accepted") is True, (
                    f"{case} replacement send was not accepted: {send_response!r}"
                )
                next_turn = await await_or_fail(
                    _frames_until_event(replacement_reader, "turn_end"),
                    what=f"{case} replacement turn end",
                )
                assert not any(
                    frame.get("params", {}).get("event") == "approval_request"
                    for frame in next_turn
                ), f"{case} replacement unexpectedly asked for approval"
                replacement_frames.extend(snapshot_frames)
                replacement_frames.extend(_buffer(replacement_reader))
                changed = [
                    frame
                    for frame in [*original_frames, *replacement_frames]
                    if frame.get("params", {}).get("event") == "session_grants_changed"
                ]
                assert len(changed) == 1, (
                    f"{case} must publish exactly one grant-change event; "
                    f"observed {changed!r}"
                )
                assert not server.control._approvals.state.records, (
                    f"{case} left an approval record after replacement turn"
                )
            else:
                grant_events = [
                    frame
                    for frame in _buffer(reader)
                    if frame.get("params", {}).get("event") == "session_grants_changed"
                ]
                assert len(grant_events) == (0 if case == "deny" else 1)
                if case == "duplicate":
                    _buffer(reader).clear()
                    assert server.runtime.opened is not None
                    duplicate_call = ToolCall(
                        "call-2", "bash", {"command": "true"}
                    )
                    server.runtime.opened.store.append_approval_request(
                        "call-2", duplicate_call
                    )
                    before = (await _request(reader, writer, 5, "session_grants"))[-1]
                    assert before["result"]["session_grants"] == ["bash"]
                    duplicate_response = (
                        await _request(
                            reader,
                            writer,
                            6,
                            "approve",
                            {"request_id": "call-2", "scope": "always_tool"},
                        )
                    )[-1]
                    assert duplicate_response.get("result", {}).get("accepted") is True, (
                        f"duplicate approval was not accepted: {duplicate_response!r}"
                    )
                    sentinel = (await _request(reader, writer, 7, "session_grants"))[-1]
                    assert sentinel["result"]["session_grants"] == ["bash"]
                    duplicate_events = [
                        frame
                        for frame in _buffer(reader)
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
                await _frames_until_event(reader, "turn_end")
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
                    socket_path=_socket_path(tmp_path),
                    provider="codex",
                    backend_factory=lambda provider, model, home: (
                        replay_backend,
                        model or "offline",
                    ),
                )
                reader, writer = await _connect(server)
                await _request(
                    reader,
                    writer,
                    10,
                    "hello",
                    {"protocol_version": "1.2", "features": FEATURES},
                )
                await _request(
                    reader,
                    writer,
                    11,
                    "resume",
                    {"session_id": session_id},
                )
                reset = await _event(reader, "session_grants_changed")
                assert reset["data"]["change"] == "reset"
            snapshot = (await _request(reader, writer, 6, "session_grants"))[-1][
                "result"
            ]
            expected_grants = [] if case in {"deny", "restart_replay"} else ["bash"]
            assert snapshot["session_grants"] == expected_grants
            if restarted:
                await _request(reader, writer, 12, "send", {"text": "ask again"})
                assert (await _event(reader, "approval_request"))[
                    "request_id"
                ] == "call-2"
                await _request(
                    reader,
                    writer,
                    13,
                    "deny",
                    {"request_id": "call-2"},
                )
                await _frames_until_event(reader, "turn_end")
            else:
                await _frames_until_event(reader, "turn_end")
            assert not server.control._approvals.state.records
    finally:
        release.set() if "release" in locals() else None
        await close_serve(
            server,
            writer,
            *([replacement_writer] if replacement_writer is not None else []),
        )
