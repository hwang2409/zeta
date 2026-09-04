import json
import time
from pathlib import Path

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools.agent import agent_result
from zeta.tui.render import render_event
from zeta.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)


async def _collect(events):
    return [event async for event in events]


def _result(store: ConversationStore, call_id: str) -> ToolResult:
    return next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
        and message.tool_result.tool_call_id == call_id
    )


@pytest.mark.asyncio
async def test_agent_output_reads_finished_child_with_roles_and_pages(
    tmp_path: Path,
) -> None:
    call = ToolCall(
        "agent-output",
        "agent",
        {"prompt": "inspect", "description": "research"},
    )
    store = ConversationStore(tmp_path)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn([TextContent("child answer")]),
        ]
    )
    loop = AgentLoop(backend, store, max_turns=1)
    await _collect(loop.run_turn("start"))
    handle = _result(store, call.id).structured_content["child_instance_id"]

    first = await loop.tool_registry.execute(
        ToolCall("output-1", "agent_output", {"handle": handle, "limit": 8})
    )
    assert first["isError"] is False
    assert first["structuredContent"]["truncated"] is True
    assert first["structuredContent"]["next_offset"] > 0
    assert first["content"][0]["text"] == "user: in"

    second = await loop.tool_registry.execute(
        ToolCall(
            "output-2",
            "agent_output",
            {
                "handle": handle,
                "offset": first["structuredContent"]["next_offset"],
            },
        )
    )
    assert "assistant: child answer" in second["content"][0]["text"]
    assert second["structuredContent"]["truncated"] is False


@pytest.mark.asyncio
async def test_agent_output_rejects_handle_not_on_active_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(FakeBackend([]), store, max_turns=1)

    result = await loop.tool_registry.execute(
        ToolCall("output-missing", "agent_output", {"handle": "old:1"})
    )

    assert result["isError"] is True
    assert "unknown child handle" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_reads_new_live_tail_on_each_call(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("running")],
            tool_result=ToolResult(
                "agent-call",
                "running",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(child.session_dir),
                    "status": "running",
                },
            ),
        )
    )
    loop = AgentLoop(FakeBackend([]), store, max_turns=1)

    before = await loop.tool_registry.execute(
        ToolCall("output-live-1", "agent_output", {"handle": "parent:1"})
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("new tail")]))
    after = await loop.tool_registry.execute(
        ToolCall("output-live-2", "agent_output", {"handle": "parent:1"})
    )

    assert "new tail" not in before["content"][0]["text"]
    assert "assistant: new tail" in after["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_stats_are_in_provider_visible_receipt_and_status_text(
    tmp_path: Path,
) -> None:
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    handle = "parent:1"
    child.start_agent_lifecycle(
        handle=handle,
        started_at="2026-09-04T10:00:00+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description="stats",
    )
    child.update_agent_lifecycle(tool_calls=1, turns_used=2)
    child.finish_agent_lifecycle(
        "completed",
        final_result="done",
        turns_used=2,
        finished_at="2026-09-04T10:00:01+00:00",
    )
    parent.append_message(Message(MessageRole.USER, [TextContent("start")]))
    parent.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("done")],
            tool_result=ToolResult(
                "agent-call",
                "done",
                structured_content={
                    "child_instance_id": handle,
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    receipt = agent_result(
        "done",
        error=False,
        turns_used=2,
        child_session_path=str(child.session_dir),
        status="completed",
        child_instance_id=handle,
    )
    receipt_text = receipt["content"][0]["text"]
    assert "2 turns" in receipt_text
    assert "1 tool calls" in receipt_text
    assert "error=false" in receipt_text
    assert "canceled=false" in receipt_text

    status = await AgentLoop(FakeBackend([]), parent, max_turns=1).tool_registry.execute(
        ToolCall("status-call", "agent_status", {"handle": handle})
    )
    status_text = status["content"][0]["text"]
    assert "2 turns" in status_text
    assert "1 tool calls" in status_text
    assert "error=false" in status_text
    assert "canceled=false" in status_text


@pytest.mark.asyncio
async def test_background_notification_carries_structured_stats(tmp_path: Path) -> None:
    call = ToolCall(
        "agent-background-stats",
        "agent",
        {"prompt": "inspect", "description": "stats", "background": True},
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        FakeBackend(
            [
                ScriptedTurn(tool_calls=[call]),
                ScriptedTurn([TextContent("child done")]),
            ]
        ),
        store,
        max_turns=1,
    )

    await _collect(loop.run_turn("start"))
    await loop._background_owner.wait()
    stats = store.agent_notifications(pending_only=False)[0].data["stats"]

    assert stats == {
        "turns_used": 1,
        "elapsed": stats["elapsed"],
        "tool_calls": 0,
        "error": False,
        "canceled": False,
    }
    assert stats["elapsed"] >= 0


def test_foreground_receipt_shows_lifecycle_stats(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-04T10:00:00+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description="stats",
    )
    child.update_agent_lifecycle(tool_calls=2, turns_used=1)
    child.finish_agent_lifecycle(
        "completed",
        final_result="done",
        turns_used=1,
        finished_at="2026-09-04T10:00:01+00:00",
    )
    call = ToolCall(
        "agent-receipt-stats",
        "agent",
        {"prompt": "inspect", "description": "stats"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={
                "child_session_path": str(child.session_dir),
                "turns_used": 1,
            },
        ),
    )

    rendered = render_event(event)
    assert rendered is not None
    assert "2 tool calls" in rendered.plain


@pytest.mark.asyncio
async def test_agent_output_response_has_one_total_byte_bound(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(Message(MessageRole.USER, [TextContent("x" * 10_000)]))
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("done")],
            tool_result=ToolResult(
                "agent-call",
                "done",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    loop = AgentLoop(FakeBackend([]), store, max_turns=1)

    result = await loop.tool_registry.execute(
        ToolCall("output-large", "agent_output", {"handle": "parent:1"})
    )
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 10_000


@pytest.mark.asyncio
async def test_agent_output_reads_snapshot_without_lock_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("saved")]))
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("running")],
            tool_result=ToolResult(
                "agent-call",
                "running",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    child.path.write_bytes(child.path.read_bytes() + b'{"incomplete":')
    before = child.path.read_bytes()
    before_mtime = child.path.stat().st_mtime_ns

    def lock_is_forbidden() -> None:
        raise AssertionError("agent_output acquired the child append lock")

    monkeypatch.setattr(ConversationStore, "_append_lock", lock_is_forbidden)
    started = time.monotonic()
    result = await AgentLoop(FakeBackend([]), store, max_turns=1).tool_registry.execute(
        ToolCall("output-snapshot", "agent_output", {"handle": "parent:1"})
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert result["isError"] is False
    assert "assistant: saved" in result["content"][0]["text"]
    assert "incomplete" not in result["content"][0]["text"]
    assert child.path.read_bytes() == before
    assert child.path.stat().st_mtime_ns == before_mtime


@pytest.mark.asyncio
async def test_agent_output_error_total_cap_handles_multibyte_handle(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path), max_turns=1)
    result = await loop.tool_registry.execute(
        ToolCall("output-emoji", "agent_output", {"handle": "😀" * 20_000})
    )

    assert result["isError"] is True
    assert "unknown child handle" in result["content"][0]["text"]
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 10_000
