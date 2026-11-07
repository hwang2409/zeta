"""End-to-end serve history fidelity coverage."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.server_client import event, request
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    with_message_origin,
)
from zeta.server import ZetaServer


async def _open(
    tmp_path: Path,
    backend: CompletionBackend | None = None,
    *,
    features: list[str] | None = None,
):
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
        {
            "protocol_version": "1.2",
            "features": features or ["session_history_fidelity"],
        },
    )
    session_id = (
        await request(reader, writer, 2, "new_session", {"provider": "codex"})
    )[-1]["result"]["session"]["session_id"]
    return server, reader, writer, session_id


async def _close(server: ZetaServer, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()
    await server.close()


async def _history(reader, writer, session_id: str):
    return (
        await request(reader, writer, 90, "session_history", {"session_id": session_id})
    )[-1]["result"]["messages"]


class _BlockingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield StreamEvent(StreamEventType.MESSAGE_START)


class _FailingBackend(CompletionBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        raise RuntimeError("provider failed")
        if False:
            yield StreamEvent(StreamEventType.MESSAGE_START)


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
        assert live["content"] == [{"type": "text", "text": "also inspect tests"}]
        status = (await request(reader, writer, 6, "status"))[-1]["result"]
        assert status["state"] == "idle"

        rows = await _history(reader, writer, session_id)
        replayed = next(row for row in rows if row.get("event") == "delivery_status")
        assert {
            key: value for key, value in live.items() if key != "session_id"
        } == replayed
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_served_turn_end_is_distinct_and_negotiated_over_socket(
    tmp_path: Path,
) -> None:
    target = tmp_path / "segment.txt"
    target.write_text("ok")
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[ToolCall("read-segment", "read", {"path": str(target)})]
            ),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server, reader, writer, session_id = await _open(
        tmp_path,
        backend,
        features=["session_history_fidelity", "served_turn_end"],
    )
    try:
        await request(reader, writer, 3, "send", {"text": "hello"})
        await event(reader, "approval_request")
        await request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": "read-segment", "scope": "once"},
        )
        provider_end = await event(reader, "turn_end")
        assert set(provider_end["data"]) >= {"turn", "tool_calls"}
        assert "reason" not in provider_end["data"]
        second_provider_end = await event(reader, "turn_end")
        assert set(second_provider_end["data"]) >= {"turn", "tool_calls"}
        await event(reader, "agent_end")

        rows = await _history(reader, writer, session_id)
        served = [row for row in rows if row.get("event") == "served_turn_end"]
        assert len(served) == 1
        assert served[0]["data"] == {"reason": "completed"}

        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        hello = await request(
            reader,
            writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": ["session_history_fidelity"]},
        )
        assert "served_turn_end" not in hello[-1]["result"]["capabilities"]["features"]
        rows = await _history(reader, writer, session_id)
        assert not any(row.get("event") == "served_turn_end" for row in rows)
    finally:
        await _close(server, writer)


@pytest.mark.parametrize(
    ("reason", "terminal"),
    [
        ("completed", "agent_end"),
        ("user_stop", "turn_aborted"),
        ("error", "error"),
        ("disconnect", None),
    ],
)
@pytest.mark.asyncio
async def test_real_served_turn_finalizers_persist_closed_outcomes(
    tmp_path: Path,
    reason: str,
    terminal: str | None,
) -> None:
    backend: CompletionBackend
    if reason in {"user_stop", "disconnect"}:
        backend = _BlockingBackend()
    elif reason == "error":
        backend = _FailingBackend()
    else:
        backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server, reader, writer, session_id = await _open(
        tmp_path, backend, features=["served_turn_end"]
    )
    finalized = asyncio.Event()
    original_finalize = server.control.deliveries.finalize_turn

    async def track_finalize(*args, **kwargs) -> None:
        await original_finalize(*args, **kwargs)
        finalized.set()

    server.control.deliveries.finalize_turn = track_finalize
    try:
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        hello = await request(
            reader,
            writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": ["served_turn_end"]},
        )
        assert "served_turn_end" in hello[-1]["result"]["capabilities"]["features"]
        await request(reader, writer, 11, "send", {"text": "finish"})
        if isinstance(backend, _BlockingBackend):
            await backend.started.wait()
        if reason == "user_stop":
            await request(reader, writer, 12, "abort")
        elif reason == "disconnect":
            writer.close()
            await writer.wait_closed()
        if terminal is not None:
            await event(reader, terminal)
        await finalized.wait()

        if reason == "disconnect":
            reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
            await request(
                reader,
                writer,
                13,
                "hello",
                {"protocol_version": "1.2", "features": ["served_turn_end"]},
            )
        rows = await _history(reader, writer, session_id)
        assert [
            row["data"]["reason"]
            for row in rows
            if row.get("event") == "served_turn_end"
        ] == [reason]
    finally:
        await _close(server, writer)


@pytest.mark.parametrize("finalizer", ["notification", "resumed_tool"])
@pytest.mark.asyncio
async def test_background_finalizers_persist_served_turn_end(
    tmp_path: Path, finalizer: str
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("notification handled")])])
    server, reader, writer, session_id = await _open(
        tmp_path, backend, features=["served_turn_end"]
    )
    finalized = asyncio.Event()
    original_finalize = server.control.deliveries.finalize_turn

    async def track_finalize(*args, **kwargs) -> None:
        await original_finalize(*args, **kwargs)
        finalized.set()

    server.control.deliveries.finalize_turn = track_finalize
    try:
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        hello = await request(
            reader,
            writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": ["served_turn_end"]},
        )
        assert "served_turn_end" in hello[-1]["result"]["capabilities"]["features"]
        store = server.runtime.opened.store
        if finalizer == "notification":
            store.append_agent_notification(
                "child",
                child_session_path="/tmp/child",
                description="child",
                status="completed",
                text="done",
            )
            server.control._schedule_background_wake(session_id)
        else:
            target = tmp_path / "resumed-tool.txt"
            target.write_text("ready")
            call = ToolCall("resumed-read", "read", {"path": str(target)})
            store.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
                [(call.id, call)],
            )
            await request(
                reader,
                writer,
                11,
                "approve",
                {"request_id": call.id, "scope": "once"},
            )
        await finalized.wait()

        rows = await _history(reader, writer, session_id)
        assert [
            row["data"]["reason"]
            for row in rows
            if row.get("event") == "served_turn_end"
        ] == ["completed"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_turn_end_commit_precedes_cleanup_and_agent_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server, reader, writer, session_id = await _open(
        tmp_path, backend, features=["served_turn_end"]
    )
    store = server.runtime.opened.store
    original_durable_write = type(store)._run_durable_write
    committed = threading.Event()
    release = threading.Event()

    def gate_after_commit(self, method_name, args, kwargs):
        result = original_durable_write(self, method_name, args, kwargs)
        if method_name == "_append_row" and args == (
            "turn_end",
            {"reason": "completed"},
        ):
            committed.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(type(store), "_run_durable_write", gate_after_commit)
    announcement: asyncio.Task[dict] | None = None
    try:
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        hello = await request(
            reader,
            writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": ["served_turn_end"]},
        )
        assert "served_turn_end" in hello[-1]["result"]["capabilities"]["features"]
        await request(reader, writer, 11, "send", {"text": "finish"})
        announcement = asyncio.create_task(event(reader, "agent_end"))
        assert await asyncio.to_thread(committed.wait, 5)

        def reopened_rows():
            from zeta.core.store import ConversationStore

            reopened = ConversationStore(
                store.root_dir, session_id=session_id, _must_exist=True
            )
            try:
                return reopened.replay()
            finally:
                reopened.close()

        durable_rows = await asyncio.to_thread(reopened_rows)
        assert any(
            row.type == "turn_end" and row.data == {"reason": "completed"}
            for row in durable_rows
        )
        assert server.runtime.state.status == "running"
        assert not announcement.done()

        release.set()
        await announcement
        rows = await _history(reader, writer, session_id)
        assert [
            row["data"]["reason"]
            for row in rows
            if row.get("event") == "served_turn_end"
        ] == ["completed"]
    finally:
        release.set()
        if announcement is not None and not announcement.done():
            announcement.cancel()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_history_projects_all_closed_served_turn_end_reasons(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(
        tmp_path, features=["served_turn_end"]
    )
    try:
        store = server.runtime.opened.store
        for reason in ("completed", "user_stop", "error", "disconnect"):
            try:
                store._append_row("turn_end", {"reason": reason})
            except ValueError:
                pass

        rows = await _history(reader, writer, session_id)
        assert [
            row["data"]["reason"]
            for row in rows
            if row.get("event") == "served_turn_end"
        ] == ["completed", "user_stop", "error", "disconnect"]
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
