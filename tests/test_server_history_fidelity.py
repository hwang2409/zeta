"""End-to-end serve history fidelity coverage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.server_client import event, request
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.server import ZetaServer


async def _open(tmp_path: Path, backend: FakeBackend | None = None):
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=Path("/tmp") / f"zeta-history-fidelity-{tmp_path.name}.sock",
        backend_factory=(
            (lambda provider, model, home: (backend, model or "offline"))
            if backend is not None
            else None
        ),
        approval_reconnect_grace=0,
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    await request(
        reader,
        writer,
        1,
        "hello",
        {"protocol_version": "1.2", "features": ["session_history_fidelity"]},
    )
    session_id = (await request(
        reader, writer, 2, "new_session", {"provider": "codex"}
    ))[-1]["result"]["session"]["session_id"]
    return server, reader, writer, session_id


async def _close(server: ZetaServer, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()
    await server.close()


async def _history(reader, writer, session_id: str):
    return (await request(
        reader, writer, 90, "session_history", {"session_id": session_id}
    ))[-1]["result"]["messages"]


@pytest.mark.asyncio
async def test_session_history_fidelity_projects_bounded_tool_data_over_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path)
    try:
        store = server.runtime.opened.store
        store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(ToolCall("bash-1", "bash", {"command": "x" * 9_000}))],
            )
        )
        store.append_message(
            with_message_origin(
                Message(
                    MessageRole.USER,
                    tool_result=ToolResult(
                        "bash-1",
                        "rendered output",
                        is_error=True,
                        structured_content={
                            "exit_code": 7,
                            "stdout": "out",
                            "stderr": "err" * 3_000,
                            "stdout_truncated": False,
                            "stderr_truncated": False,
                        },
                    ),
                ),
                MessageOrigin.USER,
            )
        )
        for call_id, stdout_truncated, stderr_truncated in (
            ("bash-complete", False, False),
            ("bash-source-capped", True, False),
        ):
            store.append_message(
                with_message_origin(
                    Message(
                        MessageRole.USER,
                        tool_result=ToolResult(
                            call_id,
                            "retained",
                            structured_content={
                                "exit_code": 0,
                                "stdout": "retained",
                                "stderr": "",
                                "stdout_truncated": stdout_truncated,
                                "stderr_truncated": stderr_truncated,
                            },
                        ),
                    ),
                    MessageOrigin.USER,
                )
            )
        store.append_message(
            with_message_origin(
                Message(
                    MessageRole.USER,
                    tool_result=ToolResult(
                        "bash-old",
                        "old",
                        structured_content={
                            "exit_code": 0,
                            "stdout": "old",
                            "stderr": "",
                        },
                    ),
                ),
                MessageOrigin.USER,
            )
        )

        rows = await _history(reader, writer, session_id)
        call = rows[0]["content"][0]["tool_call"]
        assert call["arguments_truncated"] is True
        assert call["arguments"]["command"].endswith("[truncated: 1000 bytes]")
        result = rows[1]["tool_result"]
        assert result["structured_content"]["exit_code"] == 7
        assert result["structured_content"]["stdout"] == "out"
        assert result["structured_content"]["stderr"].endswith(
            "[truncated: 1000 bytes]"
        )
        assert result["structured_content"]["stdout_truncated"] is False
        assert result["structured_content"]["stderr_truncated"] is False
        assert result["structured_content_truncated"] is True
        assert rows[2]["tool_result"]["structured_content_truncated"] is False
        assert rows[3]["tool_result"]["structured_content_truncated"] is True
        assert rows[4]["tool_result"]["structured_content_truncated"] is None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_history_fidelity_projects_tool_image_stubs_over_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path)
    try:
        image = {
            "type": "image",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg==",
            "mimeType": "image/png",
        }
        store = server.runtime.opened.store
        store.append_message(
            with_message_origin(
                Message(
                    MessageRole.USER,
                    tool_result=ToolResult(
                        "image-tool",
                        "images",
                        content_blocks=[dict(image), dict(image), dict(image)],
                    ),
                ),
                MessageOrigin.USER,
            )
        )
        blocks = store._entries[-1].data["message"]["tool_result"]["content_blocks"]
        blocks[0].update({"size": 99, "caption": "known"})
        blocks[1]["data"] = "YWJj"
        blocks[1].update({"width": 2, "height": 3})
        del blocks[2]["data"]
        blocks[2].update({"path": "/private/image.png", "annotations": {}})

        rows = await _history(reader, writer, session_id)
        assert rows[0]["tool_result"]["content_blocks"] == [
            {
                "type": "image",
                "mimeType": "image/png",
                "caption": "known",
                "size": 99,
                "data_omitted": True,
            },
            {
                "type": "image",
                "mimeType": "image/png",
                "width": 2,
                "height": 3,
                "size": 3,
                "data_omitted": True,
            },
            {
                "type": "image",
                "mimeType": "image/png",
                "data_omitted": True,
            },
        ]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_dropped_steer_is_published_then_replayed_with_terminal_status(
    tmp_path: Path,
) -> None:
    call = ToolCall("read-1", "read", {"path": str(tmp_path / "missing")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server, reader, writer, session_id = await _open(tmp_path, backend)
    try:
        await request(reader, writer, 3, "send", {"text": "read"})
        await event(reader, "approval_request")
        await request(reader, writer, 4, "steer", {"text": "also inspect tests"})
        await request(reader, writer, 5, "abort")
        live = await event(reader, "delivery_status")
        assert live["status"] == "dropped"
        assert live["reason"] == "abort"
        assert live["content"] == [{"type": "text", "text": "steer dropped: abort"}]
        status = (await request(reader, writer, 6, "status"))[-1]["result"]
        assert status["state"] == "idle"

        rows = await _history(reader, writer, session_id)
        replayed = next(row for row in rows if row.get("event") == "delivery_status")
        assert {key: value for key, value in live.items() if key != "session_id"} == replayed
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_history_without_fidelity_keeps_legacy_empty_arguments(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path)
    try:
        server.runtime.opened.store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(ToolCall("call-1", "read", {"path": "/secret"}))],
            )
        )
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        await request(reader, writer, 10, "hello", {"protocol_version": "1.2"})
        rows = await _history(reader, writer, session_id)
        assert rows[0]["content"][0]["tool_call"]["arguments"] == {}
        assert "arguments_truncated" not in rows[0]["content"][0]["tool_call"]
    finally:
        await _close(server, writer)
