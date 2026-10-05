"""Protocol fixes found while building the gamma web client."""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import pytest

from tests.test_server import (
    _close,
    _connect,
    _event,
    _request,
    _socket_path,
)
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionMetadata
from zeta.protocol.types import TextContent, ToolCall
from zeta.server import ZetaServer

FEATURES = ["session_cwd", "user_message_event", "list_sessions_paging", "ping"]
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"


def _server(tmp_path: Path, backend: FakeBackend | None, cwd: Path) -> ZetaServer:
    backend = backend or FakeBackend([])
    return ZetaServer(
        home=tmp_path / "home",
        cwd=cwd,
        provider="fake",
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
    frames = await _request(reader, writer, "hello", "hello", {"protocol_version": "1.0"})
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
async def test_resume_runs_tools_in_the_session_cwd(tmp_path: Path) -> None:
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
        resumed = await _request(reader, writer, 2, "resume", {"session_id": session_id})
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
async def test_clients_without_features_see_unchanged_capabilities(tmp_path: Path) -> None:
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
            reader, writer, 1, "hello", {"protocol_version": "1.0", "features": FEATURES}
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
            frames = await _request(reader, writer, index + 2, "new_session", {"cwd": value})
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
        await _hello(reader, writer, ["user_message_event"])
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1]["result"][
            "session"
        ]["session_id"]
        frames = await _request(reader, writer, 3, "send", {"text": "hello"})
        assert [frame.get("params", {}).get("event") for frame in frames[:-1]] == [
            "user_message"
        ]
        assert frames[0]["params"] == {
            "event": "user_message",
            "session_id": session_id,
            "text": "hello",
            "mode": "send",
            "attachments": [],
        }
        frames = await _request(reader, writer, 4, "steer", {"text": "also this"})
        steer = next(
            frame["params"]
            for frame in frames
            if frame.get("params", {}).get("event") == "user_message"
        )
        assert steer["mode"] == "steer"
        assert steer["text"] == "also this"
        assert steer["attachments"] == []
        await _event(reader, "agent_end")
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
        session_id = (await _request(reader, writer, 2, "new_session", {}))[-1]["result"][
            "session"
        ]["session_id"]
        image = {
            "name": "shot.png",
            "mime_type": "image/png",
            "data": base64.b64encode(PNG).decode(),
        }
        frames = await _request(
            reader,
            writer,
            3,
            "send_images",
            {"session_id": session_id, "text": "look", "images": [image]},
        )
        event = frames[0]["params"]
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
async def test_user_message_event_requires_the_negotiated_feature(tmp_path: Path) -> None:
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
        frames = await _request(reader, writer, 9, "session_tree", {"session_id": "other"})
        assert frames[-1]["error"]["code"] == -32003
    finally:
        await _close(server, writer)


# Finding 5: documented SessionMetadata matches the serializer.


def _schema_fields(name: str) -> tuple[set[str], set[str]]:
    protocol = (Path(__file__).parents[1] / "docs" / "serve-protocol.md").read_text(
        encoding="utf-8"
    )
    block = re.search(rf"^{name} = \{{\n(.*?)\n\}}$", protocol, re.M | re.S)
    assert block is not None, f"{name} schema block is missing"
    sections = dict(
        re.findall(r"(required|optional): \{(.*?)\}", block.group(1), re.S)
    )
    return tuple(  # type: ignore[return-value]
        set(re.findall(r"(\w+):", sections.get(key, ""))) for key in ("required", "optional")
    )


def test_documented_session_metadata_matches_the_serializer() -> None:
    required, optional = _schema_fields("SessionMetadata")
    common = {
        "session_id": "a" * 32,
        "provider": "fake",
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
            assert [frame["params"]["event"] for frame in frames[:-1]][-1] == "turn_aborted"
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
    backend = FakeBackend([ScriptedTurn([TextContent("a"), TextContent("b")], delay=0.2)])
    server = _server(tmp_path, backend, _launch(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["ping"])
        assert (await _request(reader, writer, 2, "ping"))[-1]["result"] == {"pong": True}
        await _request(reader, writer, 3, "new_session", {})
        await _request(reader, writer, 4, "send", {"text": "slow"})
        await _event(reader, "assistant_delta")
        assert (await _request(reader, writer, 5, "ping"))[-1]["result"] == {"pong": True}
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)
