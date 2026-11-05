"""Protocol fixes found while building the gamma web client."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import (
    _close,
    _connect,
    _event,
    _frames_until_event,
    _read,
    _request,
    _socket_path,
)
from zeta.core.session import SessionMetadata
from zeta.protocol.types import TextContent, ToolCall
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.server import ZetaServer
from zeta.server.protocol import FEATURES as SERVER_FEATURES

FEATURES = [
    "session_cwd",
    "new_session_settings",
    "user_message_event",
    "list_sessions_paging",
    "ping",
    "assistant_reset",
]
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"


def _server(tmp_path: Path, backend: FakeBackend | None, cwd: Path) -> ZetaServer:
    backend = backend or FakeBackend([])
    return ZetaServer(
        home=tmp_path / "home",
        cwd=cwd,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )


async def _hello(reader, writer, features: list[str] | None = None) -> dict:
    params: dict[str, object] = {"protocol_version": "1.0", "client_version": "1.1"}
    if features is not None:
        params["features"] = features
    frames = await _request(reader, writer, "hello", "hello", params)
    return frames[-1]["result"]


async def _legacy_hello(reader, writer) -> dict:
    frames = await _request(
        reader, writer, "hello", "hello", {"protocol_version": "1.0"}
    )
    return frames[-1]["result"]


def _launch(tmp_path: Path) -> Path:
    launch = tmp_path / "launch"
    launch.mkdir(exist_ok=True)
    return launch


def _tool_stdout(end: dict) -> str:
    content = end["tool_result"]["content"]
    match = re.search(r"stdout:\n(.*?)\n", content)
    assert match is not None, content
    return match.group(1)


@pytest.mark.asyncio
async def test_controller_model_catalog_works_before_session_attachment(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        result = (await _request(reader, writer, 2, "model_catalog"))[-1]
        assert result["result"]["models"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_new_session_settings_apply_before_first_turn(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["new_session_settings"])
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"model": "gpt-5.4-mini", "approval_mode": "allow"},
        )
        assert created[-1]["result"]["session"]["model"] == "gpt-5.4-mini"
        await _request(reader, writer, 3, "send", {"text": "hello"})
        await _event(reader, "agent_end")
        assert server.runtime.metadata.model == "gpt-5.4-mini"
        assert server.runtime.policy.default.value == "allow"
        assert backend.calls
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_new_session_settings_invalid_does_not_create_session(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["new_session_settings"])
        result = (
            await _request(
                reader,
                writer,
                2,
                "new_session",
                {"model": "not-a-model", "approval_mode": "allow"},
            )
        )[-1]
        assert result["error"] == {
            "code": -32602,
            "message": "invalid model or approval mode",
        }
        assert server.runtime.opened is None
    finally:
        await _close(server, writer)


# Finding 1: resume uses the persisted session cwd.


async def _create_session_in(tmp_path: Path, cwd: Path) -> str:
    server = _server(tmp_path, None, cwd)
    reader, writer = await _connect(server)
    try:
        await _legacy_hello(reader, writer)
        created = await _request(reader, writer, 2, "new_session", {})
        return created[-1]["result"]["session"]["session_id"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_runs_tools_in_the_session_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    session_id = await _create_session_in(tmp_path, original)
    call = ToolCall("pwd-call", "bash", {"command": "pwd"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _legacy_hello(reader, writer)
        tool_end_written = asyncio.Event()
        controller = server.connections.controller_metadata().connection
        original_write = controller._write_physical

        async def gated_write(payload: bytes) -> bool:
            frame = json.loads(payload)
            if frame.get("id") == 4:
                await tool_end_written.wait()
            written = await original_write(payload)
            if (
                frame.get("method") == "event"
                and frame["params"].get("event") == "tool_end"
            ):
                tool_end_written.set()
            return written

        monkeypatch.setattr(controller, "_write_physical", gated_write)
        resumed = await _request(
            reader, writer, 2, "resume", {"session_id": session_id}
        )
        assert resumed[-1]["result"]["session"]["cwd"] == str(original.resolve())
        await _request(reader, writer, 3, "send", {"text": "where am I"})
        await _event(reader, "approval_request")
        await _request(reader, writer, 4, "approve", {"request_id": "pwd-call"})
        end = await _event(reader, "tool_end")
        assert _tool_stdout(end) == str(original.resolve())
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_rejects_a_session_whose_cwd_was_removed(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    session_id = await _create_session_in(tmp_path, original)
    shutil.rmtree(original)
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _legacy_hello(reader, writer)
        active = (await _request(reader, writer, 2, "new_session", {}))[-1]["result"]
        frames = await _request(reader, writer, 3, "resume", {"session_id": session_id})
        error = frames[-1]["error"]
        assert error["code"] == -32602
        assert "working directory no longer exists" in error["message"]
        assert str(original.resolve()) in error["message"]
        status = (await _request(reader, writer, 4, "status"))[-1]["result"]
        assert status["session"]["session_id"] == active["session"]["session_id"]
    finally:
        await _close(server, writer)


# Feature negotiation for the gamma extensions.


@pytest.mark.asyncio
async def test_hello_echoes_requested_supported_features_only(tmp_path: Path) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, [*FEATURES, "unknown_feature"])
        assert hello["protocol_version"] == "1.1"
        assert hello["capabilities"]["features"] == FEATURES
        assert "ping" in hello["capabilities"]["requests"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_clients_without_features_see_unchanged_capabilities(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer)
        assert "features" not in hello["capabilities"]
        assert "ping" not in hello["capabilities"]["requests"]
        frames = await _request(reader, writer, 2, "ping")
        assert frames[-1]["error"]["code"] == -32601
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_legacy_hello_ignores_features(tmp_path: Path) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        frames = await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.0", "features": FEATURES},
        )
        hello = frames[-1]["result"]
        assert hello["protocol_version"] == "1.0"
        assert "features" not in hello["capabilities"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_hello_rejects_malformed_features(tmp_path: Path) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        frames = await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": "ping"},
        )
        assert frames[-1]["error"]["code"] == -32602
    finally:
        writer.close()
        await server.close()


# Finding 2: new_session accepts cwd.


@pytest.mark.asyncio
async def test_new_session_cwd_selects_the_tool_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    call = ToolCall("pwd-call", "bash", {"command": "pwd"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["session_cwd"])
        frames = await _request(reader, writer, 2, "new_session", {"cwd": str(project)})
        assert frames[-1]["result"]["session"]["cwd"] == str(project.resolve())
        await _request(reader, writer, 3, "send", {"text": "where am I"})
        await _event(reader, "approval_request")
        await _request(reader, writer, 4, "approve", {"request_id": "pwd-call"})
        end = await _event(reader, "tool_end")
        assert _tool_stdout(end) == str(project.resolve())
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_new_session_cwd_must_be_an_existing_absolute_directory(
    tmp_path: Path,
) -> None:
    afile = tmp_path / "file.txt"
    afile.write_text("x")
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["session_cwd"])
        for index, value in enumerate(
            [str(tmp_path / "missing"), str(afile), "relative/dir", "", 7]
        ):
            frames = await _request(
                reader, writer, index + 2, "new_session", {"cwd": value}
            )
            assert frames[-1]["error"]["code"] == -32602, value
        status = (await _request(reader, writer, 20, "status"))[-1]["result"]
        assert status["session"] is None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_new_session_cwd_requires_the_negotiated_feature(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        frames = await _request(reader, writer, 2, "new_session", {"cwd": str(project)})
        assert frames[-1]["error"]["code"] == -32602
        assert "session_cwd" in frames[-1]["error"]["message"]
    finally:
        await _close(server, writer)


# Finding 3: user_message events.


@pytest.mark.asyncio
async def test_send_and_steer_emit_user_message_events(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("one"), TextContent("two")], delay=0.2)]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1][
            "result"
        ]["session"]["session_id"]
        await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "hello", "delivery_id": "send-3"},
        )
        event = await _event(reader, "user_message")
        assert event == {
            "event": "user_message",
            "session_id": session_id,
            "text": "hello",
            "mode": "send",
            "attachments": [],
            "delivery_id": "send-3",
        }
        await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "also this", "delivery_id": "steer-4"},
        )
        steer = await _event(reader, "user_message")
        assert steer["mode"] == "steer"
        assert steer["text"] == "also this"
        assert steer["attachments"] == []
        assert steer["delivery_id"] == "steer-4"
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_user_message_delivery_id_is_opt_in(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("one"), TextContent("two")], delay=0.2)]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["user_message_event", "delivery_id"])
        await _request(reader, writer, 2, "new_session", {})
        await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "hello", "delivery_id": "legacy-send"},
        )
        assert (await _event(reader, "user_message")) == {
            "event": "user_message",
            "session_id": server.runtime.session_id,
            "text": "hello",
            "mode": "send",
            "attachments": [],
        }
        await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "later", "delivery_id": "legacy-steer"},
        )
        event = await _event(reader, "user_message")
        assert "delivery_id" not in event
        await _event(reader, "agent_end")
        history = (
            await _request(
                reader,
                writer,
                5,
                "session_history",
                {"session_id": server.runtime.session_id},
            )
        )[-1]["result"]["messages"]
        assert all("delivery_id" not in row for row in history)
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_user_message_delivery_id_is_omitted_without_a_request_id(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("one"), TextContent("two")], delay=0.2)]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        await _request(reader, writer, 2, "new_session", {})
        send_frames = await _request(reader, writer, 3, "send", {"text": "hello"})
        steer_frames = await _request(reader, writer, 4, "steer", {"text": "later"})
        for _frames in (send_frames, steer_frames):
            assert "delivery_id" not in await _event(reader, "user_message")
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_queued_steer_keeps_its_delivery_id_when_another_request_is_pending(
    tmp_path: Path,
) -> None:
    call = ToolCall("queued-step", "bash", {"command": "printf ok"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        await _request(reader, writer, 2, "new_session", {})
        await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "start", "delivery_id": "send-start"},
        )
        approval = await _event(reader, "approval_request")
        await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "queued", "delivery_id": "steer-queued"},
        )
        echo = await _event(reader, "user_message")
        assert echo["delivery_id"] == "steer-queued"

        writer.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "approve",
                    "params": {"request_id": approval["request_id"]},
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        later_echoes = []
        saw_pending_reply = False
        while True:
            frame = await _read(reader)
            saw_pending_reply |= frame.get("id") == 5
            params = frame.get("params", {})
            if params.get("event") == "user_message":
                later_echoes.append(params)
            if params.get("event") == "agent_end":
                break
        assert saw_pending_reply
        assert later_echoes == []
        assert len(backend.calls) == 2
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_user_message_delivery_id_survives_restart_in_history(
    tmp_path: Path,
) -> None:
    call = ToolCall("history-step", "bash", {"command": "printf ok"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1][
            "result"
        ]["session"]["session_id"]
        await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "start", "delivery_id": "history-send"},
        )
        approval = await _event(reader, "approval_request")
        await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "queued", "delivery_id": "history-steer"},
        )
        await _request(
            reader,
            writer,
            5,
            "approve",
            {"request_id": approval["request_id"]},
        )
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)

    restarted = _server(tmp_path, FakeBackend([]), _launch(tmp_path))
    reader, writer = await _connect(restarted)
    try:
        await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        await _request(reader, writer, 6, "resume", {"session_id": session_id})
        history = (
            await _request(
                reader,
                writer,
                7,
                "session_history",
                {"session_id": session_id},
            )
        )[-1]["result"]["messages"]
        user_rows = [row for row in history if row["role"] == "user"]
        assert [row["delivery_id"] for row in user_rows] == [
            "history-send",
            "history-steer",
        ]
    finally:
        await _close(restarted, writer)


@pytest.mark.asyncio
async def test_duplicate_delivery_id_does_not_emit_a_second_user_message(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("done")]), ScriptedTurn([TextContent("done")])]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        hello = await _hello(
            reader,
            writer,
            ["user_message_event", "delivery_id", "user_message_delivery_id"],
        )
        assert hello["capabilities"]["features"] == [
            "user_message_event",
            "user_message_delivery_id",
            "delivery_id",
        ]
        await _request(reader, writer, 2, "new_session", {})
        await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "once", "delivery_id": "duplicate-send"},
        )
        first_frames = await _frames_until_event(reader, "agent_end")
        assert [
            frame["params"]["event"] for frame in first_frames if "params" in frame
        ].count("user_message") == 1
        frames = await _request(
            reader,
            writer,
            4,
            "send",
            {"text": "once", "delivery_id": "duplicate-send"},
        )
        assert frames[-1]["result"]["duplicate"] is True
        frames += await _request(
            reader, writer, 5, "send", {"text": "next", "delivery_id": "next-send"}
        )
        frames += await _frames_until_event(reader, "agent_end")
        messages = [
            frame["params"]
            for frame in frames
            if frame.get("params", {}).get("event") == "user_message"
        ]
        assert len(messages) == 1
        assert messages[0]["text"] == "next"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_send_images_emits_user_message_with_attachment_summary(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("seen")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["user_message_event"])
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1][
            "result"
        ]["session"]["session_id"]
        image = {
            "name": "shot.png",
            "mime_type": "image/png",
            "data": base64.b64encode(PNG).decode(),
        }
        await _request(
            reader,
            writer,
            3,
            "send_images",
            {"session_id": session_id, "text": "look", "images": [image]},
        )
        event = await _event(reader, "user_message")
        assert event["event"] == "user_message"
        assert event["mode"] == "send"
        assert event["text"] == "look"
        assert event["attachments"] == [
            {"name": "shot.png", "mime_type": "image/png", "size": len(PNG)}
        ]
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_model_switch_keeps_image_before_prompt_in_provider_request(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("seen")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["user_message_event"])
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1][
            "result"
        ]["session"]["session_id"]
        settings = await _request(
            reader,
            writer,
            3,
            "set_settings",
            {
                "session_id": session_id,
                "model": "claude-sonnet-5",
                "approval_mode": "ask",
            },
        )
        assert settings[-1]["result"]["model"] == "claude-sonnet-5"
        image = {
            "name": "circle.png",
            "mime_type": "image/png",
            "data": base64.b64encode(PNG).decode(),
        }
        await _request(
            reader,
            writer,
            4,
            "send_images",
            {"session_id": session_id, "text": "what shape is this?", "images": [image]},
        )
        await _event(reader, "agent_end")

        messages = backend.calls[0][0]
        anthropic = build_messages_payload(
            messages,
            [],
            model="claude-sonnet-5",
            max_tokens=4096,
            thinking_budget=2048,
        )
        codex = build_responses_payload(messages, [], model="gpt-5.6-luna")

        assert [block["type"] for block in anthropic["messages"][-1]["content"]] == [
            "image",
            "text",
        ]
        assert [block["type"] for block in codex["input"][-1]["content"]] == [
            "input_image",
            "input_text",
        ]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_user_message_event_requires_the_negotiated_feature(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("ok")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        await _request(reader, writer, 2, "new_session", {})
        await _request(reader, writer, 3, "send", {"text": "hello"})
        events = []
        while True:
            frame = await _event_frame(reader)
            events.append(frame["params"]["event"])
            if events[-1] == "agent_end":
                break
        assert "user_message" not in events
    finally:
        await _close(server, writer)


async def _event_frame(reader) -> dict:
    from tests.test_server import _read

    while True:
        frame = await _read(reader)
        if frame.get("method") == "event":
            return frame


# Finding 4: missing session_id on 1.1 extensions is invalid params.


@pytest.mark.asyncio
async def test_extensions_report_missing_session_id_as_invalid_params(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
        await _request(reader, writer, 2, "new_session", {})
        for index, params in enumerate([{}, {"session_id": ""}, {"session_id": 4}]):
            frames = await _request(reader, writer, index + 3, "session_tree", params)
            assert frames[-1]["error"]["code"] == -32602, params
        frames = await _request(
            reader, writer, 9, "session_tree", {"session_id": "other"}
        )
        assert frames[-1]["error"]["code"] == -32003
    finally:
        await _close(server, writer)


# Finding 5: documented SessionMetadata matches the serializer.


def _schema_fields(name: str) -> tuple[set[str], set[str]]:
    protocol = (Path(__file__).parents[1] / "docs" / "serve-protocol.md").read_text(
        encoding="utf-8"
    )
    block = re.search(
        rf"^{name} = \{{\n(.*?)\n\}}$", protocol, re.MULTILINE | re.DOTALL
    )
    assert block is not None, f"{name} schema block is missing"
    sections = dict(
        re.findall(r"(required|optional): \{(.*?)\}", block.group(1), re.DOTALL)
    )
    required = set(re.findall(r"(\w+):", sections.get("required", "")))
    optional = set(re.findall(r"(\w+):", sections.get("optional", "")))
    return required, optional


def test_documented_session_metadata_matches_the_serializer() -> None:
    required, optional = _schema_fields("SessionMetadata")
    common = {
        "session_id": "a" * 32,
        "provider": "codex",
        "model": "offline",
        "cwd": "/tmp",
        "retained_tail": 8,
        "compaction_budget": 1000,
    }
    plain = SessionMetadata.new(**common).to_dict()
    layered = SessionMetadata.new(
        **common, tool_allow=("read",), tool_allow_layers=(("read",), ("read", "bash"))
    ).to_dict()
    assert required == set(plain)
    assert set(layered) - set(plain) <= optional
    assert optional == (set(layered) - set(plain)) | {"first_message_preview"}


# Finding 6: abort emits turn_aborted in every running state.


@pytest.mark.asyncio
async def test_abort_emits_turn_aborted_during_approval_and_tool_execution(
    tmp_path: Path,
) -> None:
    for stage in ("approval", "tool"):
        call = ToolCall(f"sleep-{stage}", "bash", {"command": "sleep 5"})
        backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
        server = _server(tmp_path / stage, backend, _launch(tmp_path))
        reader, writer = await _connect(server)
        try:
            await _legacy_hello(reader, writer)
            await _request(reader, writer, 2, "new_session", {})
            await _request(reader, writer, 3, "send", {"text": "wait"})
            await _event(reader, "approval_request")
            if stage == "tool":
                await _request(reader, writer, 4, "approve", {"request_id": call.id})
                await _event(reader, "tool_start")
            frames = await _request(reader, writer, 5, "abort")
            assert frames[-1]["result"] == {"aborted": True}
            await _event(reader, "turn_aborted")
            status = (await _request(reader, writer, 6, "status"))[-1]["result"]
            assert status["state"] == "idle"
        finally:
            await _close(server, writer)


# Finding 7: list_sessions paging.


@pytest.mark.asyncio
async def test_list_sessions_pages_with_offset_and_limit(tmp_path: Path) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["list_sessions_paging"])
        created = []
        for index in range(3):
            frames = await _request(reader, writer, f"new-{index}", "new_session", {})
            created.append(frames[-1]["result"]["session"]["session_id"])
        everything = (await _request(reader, writer, 2, "list_sessions"))[-1]["result"]
        ordered = [item["session_id"] for item in everything["sessions"]]
        assert sorted(ordered) == sorted(created)
        assert everything["next_offset"] is None
        first = (await _request(reader, writer, 3, "list_sessions", {"limit": 2}))[-1][
            "result"
        ]
        assert [item["session_id"] for item in first["sessions"]] == ordered[:2]
        assert first["next_offset"] == 2
        second = (
            await _request(
                reader, writer, 4, "list_sessions", {"offset": 2, "limit": 2}
            )
        )[-1]["result"]
        assert [item["session_id"] for item in second["sessions"]] == ordered[2:]
        assert second["next_offset"] is None
        assert "truncated" not in second
        for index, params in enumerate(
            [{"offset": -1}, {"offset": "1"}, {"limit": 0}, {"limit": True}]
        ):
            frames = await _request(reader, writer, index + 10, "list_sessions", params)
            assert frames[-1]["error"]["code"] == -32602, params
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_list_sessions_paging_requires_the_negotiated_feature(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _legacy_hello(reader, writer)
        await _request(reader, writer, 2, "new_session", {})
        result = (await _request(reader, writer, 3, "list_sessions"))[-1]["result"]
        assert set(result) == {"sessions"}
        frames = await _request(reader, writer, 4, "list_sessions", {"offset": 1})
        assert frames[-1]["error"]["code"] == -32602
    finally:
        await _close(server, writer)


# Finding 8: ping.


@pytest.mark.asyncio
async def test_ping_answers_before_a_session_and_during_a_turn(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn([TextContent("a"), TextContent("b")], delay=0.2)]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["ping"])
        assert (await _request(reader, writer, 2, "ping"))[-1]["result"] == {
            "pong": True
        }
        await _request(reader, writer, 3, "new_session", {})
        await _request(reader, writer, 4, "send", {"text": "slow"})
        await _event(reader, "assistant_delta")
        assert (await _request(reader, writer, 5, "ping"))[-1]["result"] == {
            "pong": True
        }
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


def _documented_features() -> list[str]:
    doc = (Path(__file__).parents[1] / "docs" / "serve-protocol.md").read_text()
    table = doc.split("| feature | effect |", 1)[1].split("\n\n", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    return [row.split("`", 2)[1] for row in rows]


def test_every_advertised_feature_is_documented_and_every_documented_one_exists() -> (
    None
):
    # A feature name is a promise to clients: advertise only what works, and
    # document everything advertised.
    assert sorted(_documented_features()) == sorted(SERVER_FEATURES)


@pytest.mark.asyncio
async def test_hello_advertises_approval_history_only_when_requested(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, ["approval_history"])
        assert hello["capabilities"]["features"] == ["approval_history"]
    finally:
        await _close(server, writer)


def _host_binding(token: str, *, name: str = "gamma") -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "executable": "/usr/bin/env",
            "summary": "call the gamma session API",
            "environment": {"GAMMA_TEST_TOKEN": token},
        }
    ]


@pytest.mark.asyncio
async def test_session_host_commands_negotiation_validation_and_redaction(
    tmp_path: Path,
) -> None:
    token = "host-token-must-not-echo"
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    observer_writer = None
    try:
        hello = await _hello(reader, writer, ["session_host_commands"])
        assert "session_host_commands" in hello["capabilities"]["features"]

        invalid_bindings = [
            [_host_binding("x")[0]] * 5,
            [
                {
                    **_host_binding("x")[0],
                    "environment": {f"KEY_{index}": "x" for index in range(33)},
                }
            ],
            [{**_host_binding("x")[0], "summary": "x" * 1025}],
        ]
        for request_id, invalid in enumerate(invalid_bindings, 20):
            rejected = await _request(
                reader,
                writer,
                request_id,
                "new_session",
                {"host_commands": invalid},
            )
            assert rejected[-1]["error"]["code"] == -32602
            assert token not in json.dumps(rejected)

        malformed = _host_binding(token)
        malformed[0]["unexpected"] = token
        rejected = await _request(
            reader, writer, 2, "new_session", {"host_commands": malformed}
        )
        assert rejected[-1]["error"]["code"] == -32602
        assert token not in json.dumps(rejected)

        created = await _request(
            reader,
            writer,
            3,
            "new_session",
            {"host_commands": _host_binding(token)},
        )
        assert token not in json.dumps(created)
        session_id = created[-1]["result"]["session"]["session_id"]
        assert server.control._active_host_commands is not None
        assert token not in repr(server.control._active_host_commands)

        status = await _request(reader, writer, 4, "status")
        assert token not in json.dumps(status)

        observer_reader, observer_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _request(
            observer_reader,
            observer_writer,
            40,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "session_host_commands"],
            },
        )
        observer_rejected = await _request(
            observer_reader,
            observer_writer,
            41,
            "new_session",
            {"host_commands": _host_binding(token)},
        )
        assert observer_rejected[-1]["error"]["code"] == -32008
        assert token not in json.dumps(observer_rejected)

        resumed = await _request(
            reader, writer, 5, "resume", {"session_id": session_id}
        )
        assert token not in json.dumps(resumed)
        assert server.control._active_host_commands is not None
        assert not server.control._active_host_commands.commands
        await _request(
            reader,
            writer,
            6,
            "resume",
            {
                "session_id": session_id,
                "host_commands": _host_binding(token),
            },
        )
        emptied = await _request(
            reader,
            writer,
            7,
            "resume",
            {"session_id": session_id, "host_commands": []},
        )
        assert token not in json.dumps(emptied)
        assert server.control._active_host_commands is not None
        assert not server.control._active_host_commands.commands
    finally:
        if observer_writer is not None:
            observer_writer.close()
            await observer_writer.wait_closed()
        await _close(server, writer)

    unnegotiated = _server(tmp_path / "unnegotiated", None, _launch(tmp_path))
    reader, writer = await _connect(unnegotiated)
    try:
        await _hello(reader, writer, [])
        frames = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding(token)},
        )
        assert frames[-1]["error"]["code"] == -32602
        assert token not in json.dumps(frames)
    finally:
        await _close(unnegotiated, writer)


@pytest.mark.asyncio
async def test_session_host_commands_do_not_persist_or_reach_children(
    tmp_path: Path,
) -> None:
    token = "host-token-never-durable"
    call = ToolCall(
        "background-env-check",
        "run_background",
        {"command": 'test -n "$GAMMA_TEST_TOKEN"'},
    )
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    captured: list[object] = []
    try:
        await _hello(
            reader,
            writer,
            ["session_host_commands", "session_history_fidelity"],
        )
        captured += await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding(token)},
        )
        session_id = server.runtime.session_id
        assert session_id is not None
        captured += await _request(reader, writer, 3, "send", {"text": "check"})
        approval = await _event(reader, "approval_request")
        captured.append(approval)
        captured += await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": "background-env-check"},
        )
        tool_end = await _event(reader, "tool_end")
        captured.append(tool_end)
        turn_task = server.control._turn_task
        assert turn_task is not None
        await asyncio.wait_for(asyncio.shield(turn_task), 10)
        captured += await _request(reader, writer, 5, "status")
        captured += await _request(
            reader,
            writer,
            6,
            "session_history",
            {"session_id": session_id},
        )
        captured.append(server.runtime.metadata.to_dict())
        captured.append(server.runtime.manager.export(session_id))

        snapshots = server.runtime.loop.tool_registry.background_tasks.snapshot()
        assert snapshots
        assert token not in json.dumps(
            [
                {
                    "task_id": item.task_id,
                    "command": item.command,
                    "owner": item.owner,
                    "note": item.note,
                }
                for item in snapshots
            ]
        )
        assert token not in json.dumps(captured)
        assert all(
            token.encode() not in path.read_bytes()
            for path in (tmp_path / "home").rglob("*")
            if path.is_file()
        )
        from zeta.skills import SkillCatalog
        from zeta.tools import ToolRegistry

        child_cwd = tmp_path / "child"
        child_cwd.mkdir()
        child_registry = ToolRegistry(child_cwd, skill_catalog=SkillCatalog.empty())
        try:
            child = await child_registry.execute(
                ToolCall(
                    "child-env-check",
                    "bash",
                    {"command": 'test -z "$GAMMA_TEST_TOKEN" && printf child-pass'},
                )
            )
            assert child["structuredContent"]["stdout"] == "child-pass"
        finally:
            await child_registry.close()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_host_commands_wake_during_same_session_resume_uses_one_turn_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.protocol.types import MessageRole

    call_a = ToolCall(
        "wake-host-a",
        "bash",
        {"command": 'printf %s "$GAMMA_TEST_TOKEN"'},
    )
    call_b = ToolCall(
        "wake-host-b",
        "bash",
        {"command": 'printf %s "$GAMMA_TEST_TOKEN"'},
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call_a]),
            ScriptedTurn([TextContent("continued a")]),
            ScriptedTurn(tool_calls=[call_b]),
            ScriptedTurn([TextContent("continued b")]),
        ]
    )
    entered = [asyncio.Event(), asyncio.Event()]
    release_backend = [asyncio.Event(), asyncio.Event()]
    original_complete = backend.complete

    async def gated_complete(messages, tool_schemas):
        turn = len(backend.calls)
        if turn in {0, 2}:
            gate = turn // 2
            entered[gate].set()
            await release_backend[gate].wait()
        async for event in original_complete(messages, tool_schemas):
            yield event

    monkeypatch.setattr(backend, "complete", gated_complete)
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    started = asyncio.Event()
    release_resume = asyncio.Event()
    try:
        await _hello(reader, writer, ["session_host_commands"])
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("snapshot-a", name="gamma-a")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        real_resume = server.runtime.resume_session

        async def paused_resume(*args, **kwargs):
            started.set()
            await release_resume.wait()
            return await real_resume(*args, **kwargs)

        monkeypatch.setattr(server.runtime, "resume_session", paused_resume)
        resume = asyncio.create_task(
            _request(
                reader,
                writer,
                3,
                "resume",
                {
                    "session_id": session_id,
                    "host_commands": _host_binding("snapshot-b", name="gamma-b"),
                },
            )
        )
        await started.wait()
        server.runtime.opened.store.append_agent_notification(
            "wake-a",
            child_session_path="/wake-a",
            description="child",
            status="completed",
            text="wake a",
        )
        server.control._schedule_background_wake(session_id)
        await entered[0].wait()
        release_resume.set()
        assert "result" in (await resume)[-1]
        release_backend[0].set()
        approval = await _event(reader, "approval_request")
        assert approval["request_id"] == call_a.id
        await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": call_a.id},
        )
        assert _tool_stdout(await _event(reader, "tool_end")) == "snapshot-a"
        await _event(reader, "agent_end")
        assert any(
            isinstance(block, TextContent) and "gamma-a" in block.text
            for message in backend.calls[1][0]
            if message.role is MessageRole.SYSTEM
            for block in message.content
        )

        server.runtime.opened.store.append_agent_notification(
            "wake-b",
            child_session_path="/wake-b",
            description="child",
            status="completed",
            text="wake b",
        )
        server.control._schedule_background_wake(session_id)
        await entered[1].wait()
        release_backend[1].set()
        approval = await _event(reader, "approval_request")
        assert approval["request_id"] == call_b.id
        await _request(
            reader,
            writer,
            5,
            "approve",
            {"request_id": call_b.id},
        )
        assert _tool_stdout(await _event(reader, "tool_end")) == "snapshot-b"
        await _event(reader, "agent_end")
        assert any(
            isinstance(block, TextContent) and "gamma-b" in block.text
            for message in backend.calls[3][0]
            if message.role is MessageRole.SYSTEM
            for block in message.content
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "send_images"])
@pytest.mark.parametrize("reconnect_features", [[], ["session_host_commands"]])
async def test_session_host_commands_reject_client_turn_before_resume(
    tmp_path: Path,
    method: str,
    reconnect_features: list[str],
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("must not run")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    try:
        await _hello(reader, writer, ["session_host_commands"])
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("detached")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if server.connections.controller_metadata() is None:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")

        durable_rows = tuple(server.runtime.opened.store.entries)
        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _request(
            replay_reader,
            replay_writer,
            3,
            "hello",
            {"protocol_version": "1.2", "features": reconnect_features},
        )
        params: dict[str, object] = {"text": "must not run"}
        if method == "send_images":
            params["session_id"] = session_id
            params["images"] = [
                {
                    "name": "shot.png",
                    "mime_type": "image/png",
                    "data": base64.b64encode(PNG).decode(),
                }
            ]
        rejected = await _request(replay_reader, replay_writer, 4, method, params)

        assert rejected[-1]["error"] == {
            "code": -32003,
            "message": "session is not attached",
            "data": {"code": "host_binding_required"},
        }
        assert backend.calls == []
        assert tuple(server.runtime.opened.store.entries) == durable_rows
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_legacy_session_still_accepts_send_after_reconnect(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("legacy ok")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    try:
        await _hello(reader, writer, [])
        await _request(reader, writer, 2, "new_session", {})
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if server.connections.controller_metadata() is None:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _hello(replay_reader, replay_writer, [])
        sent = await _request(
            replay_reader, replay_writer, 3, "send", {"text": "legacy"}
        )
        assert sent[-1]["result"]["accepted"] is True
        await _event(replay_reader, "agent_end")
        assert len(backend.calls) == 1
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_detached_session_returns_existing_duplicate_delivery(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    features = ["delivery_id", "session_host_commands"]
    try:
        await _hello(reader, writer, features)
        await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("duplicate")},
        )
        first = await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "once", "delivery_id": "detached-duplicate"},
        )
        await _event(reader, "agent_end")
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if server.connections.controller_metadata() is None:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _hello(replay_reader, replay_writer, features)
        durable_rows = tuple(server.runtime.opened.store.entries)
        duplicate = await _request(
            replay_reader,
            replay_writer,
            4,
            "send",
            {"text": "once", "delivery_id": "detached-duplicate"},
        )
        assert duplicate[-1]["result"] == {
            **first[-1]["result"],
            "duplicate": True,
        }
        assert len(backend.calls) == 1
        assert tuple(server.runtime.opened.store.entries) == durable_rows
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_reconnect", [False, True])
async def test_session_host_commands_notification_after_release_has_no_binding(
    tmp_path: Path,
    approval_reconnect: bool,
) -> None:
    from zeta.protocol.types import MessageRole

    call = ToolCall(
        "released-host-call",
        "bash",
        {"command": 'printf %s "$GAMMA_TEST_TOKEN"'},
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("continued")]),
        ]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    features = ["session_host_commands"]
    if approval_reconnect:
        features.append("approval_reconnect")
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.2", "features": features},
        )
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("released-a")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if (
                server.connections is not None
                and server.connections.controller_metadata() is None
            ):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")
        assert server.control._active_host_commands is None

        server.runtime.opened.store.append_agent_notification(
            "released-wake",
            child_session_path="/released-wake",
            description="child",
            status="completed",
            text="wake while detached",
        )
        server.control._schedule_background_wake(session_id)
        assert server.control._turn_task is None
        for _ in range(10):
            await asyncio.sleep(0)
        assert backend.calls == []

        probe_reader, probe_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        probe_features = ["approval_reconnect"] if approval_reconnect else []
        await _request(
            probe_reader,
            probe_writer,
            9,
            "hello",
            {"protocol_version": "1.2", "features": probe_features},
        )
        await asyncio.sleep(0)
        assert server.control._turn_task is None
        assert backend.calls == []
        probe_writer.close()
        await probe_writer.wait_closed()
        for _ in range(100):
            if (
                server.connections is not None
                and server.connections.controller_metadata() is None
            ):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("probe controller was not released")

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _request(
            replay_reader,
            replay_writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": features},
        )
        await asyncio.sleep(0)
        assert server.control._active_host_commands is None
        assert backend.calls == []

        resumed = await _request(
            replay_reader,
            replay_writer,
            11,
            "resume",
            {
                "session_id": session_id,
                "host_commands": _host_binding("released-c", name="gamma-c"),
            },
        )
        assert "result" in resumed[-1]
        approval = await _event(replay_reader, "approval_request")
        assert approval["request_id"] == call.id
        await _request(
            replay_reader,
            replay_writer,
            12,
            "approve",
            {"request_id": call.id},
        )
        assert _tool_stdout(await _event(replay_reader, "tool_end")) == "released-c"
        await _event(replay_reader, "agent_end")
        assert any(
            isinstance(block, TextContent) and "gamma-c" in block.text
            for messages, _schemas in backend.calls
            for message in messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
        )
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_session_host_commands_live_approval_before_resume_keeps_original_snapshot(
    tmp_path: Path,
) -> None:
    call = ToolCall(
        "live-host-call",
        "bash",
        {
            "command": (
                'test "$GAMMA_TEST_TOKEN" = live-a '
                "&& printf live-pass || printf live-fail"
            )
        },
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("continued")]),
            ScriptedTurn([TextContent("next")]),
        ]
    )
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["approval_reconnect", "session_host_commands"],
            },
        )
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("live-a", name="gamma-a")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 3, "send", {"text": "call gamma"})
        request = (await _frames_until_event(reader, "approval_request"))[-1]["params"]
        assert request["request_id"] == call.id

        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if (
                server.connections is not None
                and server.connections.controller_metadata() is None
            ):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _request(
            replay_reader,
            replay_writer,
            10,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["approval_reconnect", "session_host_commands"],
            },
        )
        await _event(replay_reader, "approval_request")
        await _request(
            replay_reader,
            replay_writer,
            11,
            "approve",
            {"request_id": call.id},
        )
        tool_end = await _event(replay_reader, "tool_end")
        assert _tool_stdout(tool_end) == "live-pass"
        await _event(replay_reader, "agent_end")
        assert all(
            any(
                isinstance(block, TextContent) and "gamma-a" in block.text
                for block in message.content
            )
            for message in backend.calls[1][0]
            if message.role.value == "system"
        )

        resumed = await _request(
            replay_reader,
            replay_writer,
            12,
            "resume",
            {
                "session_id": session_id,
                "host_commands": _host_binding("live-b", name="gamma-b"),
            },
        )
        assert "result" in resumed[-1]
        await _request(replay_reader, replay_writer, 13, "send", {"text": "next"})
        await _event(replay_reader, "agent_end")
        assert any(
            isinstance(block, TextContent) and "gamma-b" in block.text
            for message in backend.calls[-1][0]
            if message.role.value == "system"
            for block in message.content
        )
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_session_host_commands_durable_approval_before_resume_uses_none_snapshot(
    tmp_path: Path,
) -> None:
    from zeta.protocol.types import (
        Message,
        MessageRole,
        StreamEvent,
        StreamEventType,
        ToolUseContent,
    )

    call = ToolCall(
        "durable-host-call",
        "bash",
        {
            "command": (
                'test -z "$GAMMA_TEST_TOKEN" '
                "&& printf durable-none || printf leaked-host-env"
            )
        },
    )
    backend = FakeBackend([ScriptedTurn([TextContent("continued")])])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    replay_writer = None
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["approval_reconnect", "session_host_commands"],
            },
        )
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("durable-a", name="gamma-a")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        server.runtime.opened.store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
        )
        await server.control._event(
            StreamEvent(StreamEventType.TOOL_APPROVAL_START, tool_call=call)
        )
        await _event(reader, "approval_request")

        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if (
                server.connections is not None
                and server.connections.controller_metadata() is None
            ):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("controller was not released")
        assert server.control._active_host_commands is None

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _request(
            replay_reader,
            replay_writer,
            10,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["approval_reconnect", "session_host_commands"],
            },
        )
        await _event(replay_reader, "approval_request")
        approved = await _request(
            replay_reader,
            replay_writer,
            11,
            "approve",
            {"request_id": call.id},
        )
        assert approved[-1]["result"]["request_id"] == call.id
        tool_end = await _event(replay_reader, "tool_end")
        assert _tool_stdout(tool_end) == "durable-none"
        turn_task = server.control._turn_task
        assert turn_task is not None
        await asyncio.wait_for(asyncio.shield(turn_task), 10)
        assert backend.calls == []

        resumed = await _request(
            replay_reader,
            replay_writer,
            12,
            "resume",
            {
                "session_id": session_id,
                "host_commands": _host_binding("durable-c", name="gamma-c"),
            },
        )
        assert "result" in resumed[-1]
        assert (
            len(
                [
                    row
                    for row in server.runtime.opened.store.entries
                    if row.type == "approval_resolution"
                    and row.data.get("request_id") == call.id
                ]
            )
            == 1
        )
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_session_host_commands_replacement_failure_matches_runtime_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, None, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["session_host_commands"])
        created = await _request(
            reader,
            writer,
            2,
            "new_session",
            {"host_commands": _host_binding("failure-a")},
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        binding_a = server.control._active_host_commands
        real_resume = server.runtime.resume_session

        async def fail_with_runtime(_session_id, *, force=False):
            assert not force
            raise RuntimeError("setup failed before replacement")

        monkeypatch.setattr(server.runtime, "resume_session", fail_with_runtime)
        failed = await _request(
            reader,
            writer,
            3,
            "resume",
            {"session_id": "different-session", "host_commands": _host_binding("b")},
        )
        assert failed[-1]["error"]["code"] == -32000
        assert server.runtime.state is not None
        assert server.control._active_host_commands is binding_a

        monkeypatch.setattr(server.control, "unhealthy", lambda: True)

        async def fail_after_close(requested, *, force=False):
            assert requested == session_id and force
            await server.runtime.close()
            raise RuntimeError("forced resume failed after close")

        monkeypatch.setattr(server.runtime, "resume_session", fail_after_close)
        forced = await _request(
            reader,
            writer,
            4,
            "resume",
            {"session_id": session_id, "host_commands": _host_binding("c")},
        )
        assert forced[-1]["error"]["code"] == -32000
        assert server.runtime.state is None
        assert server.control._active_host_commands is None
        no_send = await _request(reader, writer, 40, "send", {"text": "must fail"})
        assert no_send[-1]["error"]["code"] == -32003
        server.control._schedule_background_wake(session_id)
        assert server.control._turn_task is None

        await real_resume(session_id)
        server.control._active_host_commands = binding_a

        async def fail_replacement_after_close(requested, *, force=False):
            assert requested == "replacement-session" and not force
            await server.runtime.close()
            raise RuntimeError("replacement failed after close")

        monkeypatch.setattr(
            server.runtime, "resume_session", fail_replacement_after_close
        )
        replacement = await _request(
            reader,
            writer,
            5,
            "resume",
            {
                "session_id": "replacement-session",
                "host_commands": _host_binding("d"),
            },
        )
        assert replacement[-1]["error"]["code"] == -32000
        assert server.runtime.state is None
        assert server.control._active_host_commands is None
        no_send = await _request(reader, writer, 50, "send", {"text": "must fail"})
        assert no_send[-1]["error"]["code"] == -32003
        server.control._schedule_background_wake(session_id)
        assert server.control._turn_task is None

        monkeypatch.setattr(server.control, "unhealthy", lambda: False)
        monkeypatch.setattr(server.runtime, "resume_session", real_resume)
    finally:
        await _close(server, writer)
