import asyncio
from pathlib import Path

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.types import TextContent, ToolCall


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
    assert next(child for child in children if child["handle"] == background_handle)[
        "final_result"
    ] == "background done"

    unknown = await _status(loop, "missing-child")
    assert unknown["isError"] is True
    assert "unknown child handle" in unknown["content"][0]["text"]
    await loop.close()


async def _collect(events):
    return [event async for event in events]
