import asyncio
import json
import os
from pathlib import Path

import pytest

from zeta.core.abort import AbortGenerationRegistry
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools.agent import MAX_AGENT_STATUS_RESULT, MAX_AGENT_STATUS_STEP
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


async def _status(loop: AgentLoop, handle: str | None = None) -> dict[str, object]:
    arguments = {} if handle is None else {"handle": handle}
    return await loop.tool_registry.execute(
        ToolCall("status-call", "agent_status", arguments)
    )


@pytest.mark.asyncio
async def test_agent_status_round_trip_and_live_snapshot(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    foreground_call = ToolCall(
        "foreground-agent",
        "agent",
        {"prompt": "inspect", "description": "foreground"},
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[foreground_call]),
            ScriptedTurn([TextContent("foreground done")]),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "background-agent",
                        "agent",
                        {
                            "prompt": "inspect",
                            "description": "background",
                            "background": True,
                        },
                    )
                ]
            ),
            ScriptedTurn([TextContent("background done")], delay=0.05),
        ]
    )
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start foreground"))
    foreground_result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result and message.tool_result.tool_call_id == foreground_call.id
    )
    assert foreground_result.structured_content is not None
    foreground_handle = foreground_result.structured_content["child_instance_id"]
    assert type(foreground_handle) is str

    foreground_status = await _status(loop, foreground_handle)
    foreground_child = foreground_status["structuredContent"]["children"][0]
    assert foreground_child["handle"] == foreground_handle
    assert foreground_child["state"] == "completed"
    assert foreground_child["final_result"] == "foreground done"
    assert foreground_child["turns_used"] == 1
    assert foreground_child["tree_budget"] == 25
    assert foreground_child["depth"] == 1
    assert foreground_child["agent_type"] == "general"
    assert foreground_child["finished_at"] is not None
    assert foreground_child["elapsed"] >= 0

    await _collect(loop.run_turn("start background"))
    background_result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result
        and message.tool_result.structured_content is not None
        and message.tool_result.structured_content.get("status") == "running"
    )
    assert background_result.structured_content is not None
    background_handle = background_result.structured_content["child_instance_id"]
    assert type(background_handle) is str

    live_status = await _status(loop, background_handle)
    live_child = live_status["structuredContent"]["children"][0]
    assert live_child["state"] == "running"
    assert live_child["finished_at"] is None
    assert live_child["current_step"] == "turn 1: thinking"

    await asyncio.sleep(0.1)
    all_status = await _status(loop)
    children = all_status["structuredContent"]["children"]
    assert {child["handle"] for child in children} == {
        foreground_handle,
        background_handle,
    }
    assert all("final_result" not in child for child in children)
    background_status = await _status(loop, background_handle)
    assert background_status["structuredContent"]["children"][0]["final_result"] == (
        "background done"
    )

    unknown = await _status(loop, "missing-child")
    assert unknown["isError"] is True
    assert "unknown child handle" in unknown["content"][0]["text"]
    await loop.close()


@pytest.mark.asyncio
async def test_parent_ownership_survives_lifecycle_start_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall(
        "agent-1",
        "agent",
        {"prompt": "inspect", "description": "child"},
    )
    store = ConversationStore(tmp_path)

    def crash_before_lifecycle(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("lifecycle start crashed")

    monkeypatch.setattr(
        ConversationStore,
        "start_agent_lifecycle",
        crash_before_lifecycle,
    )
    loop = AgentLoop(FakeBackend([]), store, max_turns=1)
    with pytest.raises(RuntimeError, match="lifecycle start crashed"):
        await loop._run_agent_tool(
            call,
            call.arguments,
            AbortGenerationRegistry().new_generation(),
            None,
        )

    assert store.agent_children()
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_lifecycle() is None

    monkeypatch.undo()
    resumed = ConversationStore(tmp_path, session_id=store.session_id)
    AgentLoop(FakeBackend([]), resumed, max_turns=1)
    assert not resumed.agent_children()
    assert (
        ConversationStore(store.session_dir / "agents", session_id="1").agent_canceled()
        is not None
    )


async def _collect(events):
    return [event async for event in events]


def _persist_finished_receipt(
    store: ConversationStore,
    child: ConversationStore,
    call_id: str,
    handle: str,
) -> None:
    call = ToolCall(
        call_id,
        "agent",
        {"prompt": "inspect", "description": call_id},
    )
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("done")],
            tool_result=ToolResult(
                call_id,
                "done",
                structured_content={
                    "child_session_path": str(child.session_dir),
                    "child_instance_id": handle,
                },
            ),
        )
    )


def _new_finished_child(
    store: ConversationStore,
    index: int,
    *,
    result: str = "done",
    step: str = "finished",
) -> tuple[ConversationStore, str]:
    child = ConversationStore(store.session_dir / "agents", session_id=str(index))
    handle = f"{store.session_id}:{index}"
    child.start_agent_lifecycle(
        handle=handle,
        started_at="2026-09-04T10:00:00+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description=f"child {index}",
    )
    child.update_agent_lifecycle(current_step=step)
    child.finish_agent_lifecycle(
        "completed",
        final_result=result,
        turns_used=1,
        finished_at="2026-09-04T10:00:01+00:00",
    )
    return child, handle


def test_terminal_lifecycle_write_is_immutable_after_resume(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path, session_id="child")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-04T10:00:00+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description="child",
    )
    child.finish_agent_lifecycle(
        "completed",
        final_result="first result",
        turns_used=1,
        finished_at="2026-09-04T10:00:01+00:00",
    )
    first = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))

    reopened = ConversationStore(tmp_path, session_id="child")
    reopened.finish_agent_lifecycle(
        "failed",
        final_result="replacement result",
        turns_used=9,
        finished_at="2026-09-04T14:35:15+00:00",
    )

    assert json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8")) == first


@pytest.mark.asyncio
async def test_elapsed_uses_monotonic_time_when_wall_clock_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(store, 1)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle.update(
        {
            "state": "running",
            "finished_at": None,
            "started_at": "2030-01-01T00:00:00+00:00",
            "started_monotonic": 100.0,
            "monotonic_pid": os.getpid(),
        }
    )
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")
    _persist_finished_receipt(store, child, "agent-1", handle)
    monkeypatch.setattr("zeta.tools.agent.time.monotonic", lambda: 105.0)

    status_loop = AgentLoop(FakeBackend([]), store, max_turns=1)
    status = await _status(status_loop, handle)
    assert status["structuredContent"]["children"][0]["elapsed"] == 5.0


@pytest.mark.asyncio
async def test_list_all_uses_active_branch_and_bounds_payload(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    checkpoint = store.append_checkpoint("before children")
    child, handle = _new_finished_child(
        store,
        1,
        result="r" * 20_000,
        step="s" * 2_000,
    )
    _persist_finished_receipt(store, child, "agent-1", handle)
    store.append_fork(checkpoint.data["label"])
    loop = AgentLoop(FakeBackend([]), store, max_turns=1)

    all_status = await _status(loop)
    assert all_status["structuredContent"]["children"] == []


@pytest.mark.asyncio
async def test_list_all_stays_bounded_for_many_children(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    for index in range(1, 51):
        child, handle = _new_finished_child(
            store,
            index,
            result="r" * 20_000,
            step="s" * 2_000,
        )
        _persist_finished_receipt(store, child, f"agent-{index}", handle)

    status = await _status(AgentLoop(FakeBackend([]), store, max_turns=1))
    payload = json.dumps(status["structuredContent"])
    assert len(payload) < 100_000
    assert all(
        "final_result" not in child
        for child in status["structuredContent"]["children"]
    )


@pytest.mark.asyncio
async def test_requested_status_bounds_result_and_step(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(
        store,
        1,
        result="r" * 20_000,
        step="s" * 2_000,
    )
    _persist_finished_receipt(store, child, "agent-1", handle)
    status = await _status(AgentLoop(FakeBackend([]), store, max_turns=1), handle)
    item = status["structuredContent"]["children"][0]
    assert len(item["final_result"]) <= MAX_AGENT_STATUS_RESULT
    assert len(item["current_step"]) <= MAX_AGENT_STATUS_STEP
    assert "[truncated]" in item["final_result"]
    assert "[truncated]" in item["current_step"]
