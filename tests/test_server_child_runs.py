from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, PerActorFakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.server.server import ZetaServer


class _ObservedChildBackend(PerActorFakeBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.follow_up_started = asyncio.Event()

    @owned_completion
    async def complete(self, messages, tool_schemas):
        if any(
            isinstance(block, TextContent) and block.text == "continue"
            for message in messages
            for block in message.content
        ):
            self.follow_up_started.set()
        stream = PerActorFakeBackend.complete.__wrapped__(
            self, messages, tool_schemas
        )
        async for event in stream:
            yield event


async def _start_client(server: ZetaServer) -> ServeClient:
    await await_or_fail(server.start(), what="serve start")
    reader, writer = await await_or_fail(
        asyncio.open_unix_connection(str(server.socket_path)),
        what="Unix socket connect",
    )
    return ServeClient(reader=reader, writer=writer)


def _finished_child(
    root: ConversationStore,
    path: Path,
    *,
    local_id: str,
    run_id: str,
    depth: int,
) -> ConversationStore:
    child = ConversationStore(path, session_id=local_id)
    child.start_agent_lifecycle(
        handle=run_id,
        started_at="2026-11-08T00:00:00+00:00",
        depth=depth,
        agent_type="general",
        description=run_id,
    )
    child.finish_agent_lifecycle(
        "completed",
        final_result="done",
        turns_used=1,
        finished_at="2026-11-08T00:00:01+00:00",
    )
    call_id = f"call-{run_id}"
    call = ToolCall(
        call_id,
        "agent",
        {"prompt": "inspect", "description": run_id, "background": True},
    )
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    root.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("done")],
            tool_result=ToolResult(
                call_id,
                "done",
                structured_content={
                    "child_session_path": str(child.session_dir),
                    "child_instance_id": run_id,
                },
            ),
        )
    )
    return child


def test_child_run_rows_keep_top_level_and_parent_identity(tmp_path: Path) -> None:
    from zeta.core.child_runs import child_run_rows

    root = ConversationStore(tmp_path / "sessions", session_id="top")
    direct = _finished_child(
        root,
        root.session_dir / "agents",
        local_id="1",
        run_id="top:1",
        depth=1,
    )
    nested = _finished_child(
        root,
        direct.session_dir / "agents",
        local_id="1",
        run_id="top:1:1",
        depth=2,
    )
    try:
        rows = child_run_rows(root)
        assert [(row["run_id"], row["parent_run_id"]) for row in rows] == [
            ("top:1", None),
            ("top:1:1", "top:1"),
        ]
        assert {row["session_id"] for row in rows} == {"top"}
    finally:
        nested.close()
        direct.close()
        root.close()


def test_child_run_tool_count_is_cumulative_across_follow_up_segments(
    tmp_path: Path,
) -> None:
    from zeta.core.child_runs import child_run_rows

    root = ConversationStore(tmp_path / "sessions", session_id="top")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="top:1",
        started_at="2026-11-08T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="worker",
    )
    call = ToolCall(
        "call-top:1",
        "agent",
        {"prompt": "inspect", "description": "worker", "background": True},
    )
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    root.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("ok")],
            tool_result=ToolResult(
                "call-top:1",
                "ok",
                structured_content={
                    "child_session_path": str(child.session_dir),
                    "child_instance_id": "top:1",
                },
            ),
        )
    )
    running = frozenset({"top:1"})
    try:
        child.update_agent_lifecycle(
            turns_used=1, tool_calls=2, current_step="first", state="running"
        )
        [first] = child_run_rows(root, running_handles=running)
        assert first["tool_count"] == 2
        # a follow-up segment adds more tool calls; the whole-run total is
        # cumulative and must never decrease across the segment boundary.
        child.update_agent_lifecycle(
            turns_used=2, tool_calls=3, current_step="second", state="running"
        )
        [second] = child_run_rows(root, running_handles=running)
        assert second["tool_count"] == 5
        assert second["tool_count"] >= first["tool_count"]
    finally:
        child.close()
        root.close()


@pytest.mark.asyncio
async def test_child_run_methods_require_negotiated_feature_over_real_transport(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        provider="codex",
        home=tmp_path / "home",
        cwd=tmp_path,
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )
    client = await _start_client(server)
    try:
        await client.request("hello", "hello", {"protocol_version": "1.2"})
        listed = await client.request(
            "list",
            "child_run_list",
            {"session_id": "unused"},
        )
        sent = await client.request(
            "send",
            "child_run_send",
            {"session_id": "unused", "run_id": "unused:1", "text": "no"},
        )
        assert listed["error"]["code"] == -32601
        assert sent["error"]["code"] == -32601
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_child_run_list_and_safe_point_send_over_real_transport(
    tmp_path: Path,
) -> None:
    child_call = ToolCall(
        "child",
        "agent",
        {
            "prompt": "child prompt",
            "description": "waiting child",
            "background": True,
        },
    )
    backend = _ObservedChildBackend(
        [
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("parent done")]),
        ],
        {
            "child prompt": [
                ScriptedTurn([TextContent("first answer")]),
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            "finish",
                            "finish",
                            {"status": "done", "summary": "follow-up received"},
                        )
                    ]
                ),
            ]
        },
        parent_prompt="start child",
        gated_child_call=1,
        parent_waits_for_children=False,
    )
    server = ZetaServer(
        provider="codex",
        home=tmp_path / "home",
        cwd=tmp_path,
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    client = await _start_client(server)
    observer_reader, observer_writer = await await_or_fail(
        asyncio.open_unix_connection(str(server.socket_path)),
        what="observer Unix socket connect",
    )
    observer = ServeClient(reader=observer_reader, writer=observer_writer)
    try:
        hello = await client.request(
            "hello",
            "hello",
            {"protocol_version": "1.2", "features": ["child_runs"]},
        )
        await observer.request(
            "observer-hello",
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": ["serve_observers", "child_runs"],
            },
        )
        created = await client.request(
            "new", "new_session", {"provider": "codex"}
        )
        session_id = created["result"]["session"]["session_id"]

        await client.request("start", "send", {"text": "start child"})
        await await_or_fail(
            backend.child_started["child prompt"].wait(),
            what="child model safe point",
        )
        run_id = f"{session_id}:1"

        listed = await client.request(
            "list-before",
            "child_run_list",
            {"session_id": session_id},
        )
        observed = await observer.request(
            "observer-list",
            "child_run_list",
            {"session_id": session_id},
        )
        observer_sent = await observer.request(
            "observer-send-child",
            "child_run_send",
            {"session_id": session_id, "run_id": run_id, "text": "continue"},
        )
        sent = await client.request(
            "send-child",
            "child_run_send",
            {"session_id": session_id, "run_id": run_id, "text": "continue"},
        )
        assert "result" in listed and "result" in sent, {
            "list": listed,
            "send": sent,
        }
        assert hello["result"]["capabilities"]["features"] == ["child_runs"]
        assert observed["result"].keys() == listed["result"].keys()
        assert [
            {key: value for key, value in row.items() if key != "elapsed_seconds"}
            for row in observed["result"]["runs"]
        ] == [
            {key: value for key, value in row.items() if key != "elapsed_seconds"}
            for row in listed["result"]["runs"]
        ]
        assert observer_sent["error"] == {
            "code": -32008,
            "message": "method is unavailable to read-only observers",
            "data": {"code": "observer_read_only", "method": "child_run_send"},
        }
        [before] = listed["result"]["runs"]
        assert before["run_id"] == run_id
        assert before["session_id"] == session_id
        assert before["parent_run_id"] is None
        assert before["accepts_follow_ups"] is True
        assert "handle" not in before

        assert sent["result"] == {"accepted": True, "run_id": run_id}
        backend.release_children.set()
        await await_or_fail(
            backend.follow_up_started.wait(),
            what="safe-point child delivery",
        )

        listed_again = await client.request(
            "list-after",
            "child_run_list",
            {"session_id": session_id},
        )
        [after] = listed_again["result"]["runs"]
        assert after["run_id"] == run_id
        assert after["turns_used"] > before["turns_used"]
        assert after["tool_count"] >= before["tool_count"]
    finally:
        backend.release_children.set()
        await observer.close()
        await client.close()
        await server.close()
