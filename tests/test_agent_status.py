import asyncio
import json
import os
import random
from pathlib import Path

import pytest

from zeta.agent.receipt import encode_json
from zeta.core.abort import AbortGenerationRegistry
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


async def _status(
    loop: AgentLoop,
    handle: str | None = None,
    **arguments: object,
) -> dict[str, object]:
    if handle is not None:
        arguments["handle"] = handle
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
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start foreground"))
    foreground_result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result
        and message.tool_result.tool_call_id == foreground_call.id
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

    live_list_status = await _status(loop)
    live_list_child = live_list_status["structuredContent"]["children"][0]
    assert live_list_child["handle"] == background_handle
    assert live_list_child["state"] == "running"
    assert live_list_child["finished_at"] is None
    assert live_list_status["structuredContent"]["finished_omitted"] == 1
    assert (
        "1 finished children omitted (query by handle for results)"
        in live_list_status["content"][0]["text"]
    )

    await asyncio.sleep(0.1)
    all_status = await _status(loop)
    children = all_status["structuredContent"]["children"]
    assert children == []
    assert all_status["structuredContent"]["finished_omitted"] == 2
    assert (
        "2 finished children omitted (query by handle for results)"
        in all_status["content"][0]["text"]
    )
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
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
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
    AgentLoop(FakeBackend([]), resumed, max_turns=1, skill_catalog=SkillCatalog.empty())
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


def _new_live_child(
    store: ConversationStore,
    index: int,
    *,
    step: str = "running",
) -> tuple[ConversationStore, str]:
    child = ConversationStore(store.session_dir / "agents", session_id=str(index))
    handle = f"{store.session_id}:{index}"
    child.start_agent_lifecycle(
        handle=handle,
        started_at=f"2026-09-04T10:00:{index:02d}+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description=f"child {index}",
    )
    child.update_agent_lifecycle(current_step=step)
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

    status_loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
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
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    all_status = await _status(loop)
    assert all_status["structuredContent"]["children"] == []
    assert "finished_omitted" not in all_status["structuredContent"]
    assert "finished children omitted" not in all_status["content"][0]["text"]


@pytest.mark.asyncio
async def test_list_status_counts_only_live_children_for_pagination(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    finished_child, finished_handle = _new_finished_child(store, 1)
    _persist_finished_receipt(store, finished_child, "agent-1", finished_handle)
    first_live_child, first_live_handle = _new_live_child(store, 2)
    _persist_finished_receipt(store, first_live_child, "agent-2", first_live_handle)
    second_live_child, second_live_handle = _new_live_child(store, 3)
    _persist_finished_receipt(store, second_live_child, "agent-3", second_live_handle)
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    first_page = await _status(loop, limit=1)
    assert [
        child["handle"] for child in first_page["structuredContent"]["children"]
    ] == [first_live_handle]
    assert first_page["structuredContent"]["total"] == 2
    assert first_page["structuredContent"]["next_offset"] == 1
    assert first_page["structuredContent"]["finished_omitted"] == 1

    second_page = await _status(loop, offset=1, limit=1)
    assert [
        child["handle"] for child in second_page["structuredContent"]["children"]
    ] == [second_live_handle]
    assert second_page["structuredContent"]["total"] == 2
    assert second_page["structuredContent"]["truncated"] is False
    assert second_page["structuredContent"]["finished_omitted"] == 1


@pytest.mark.asyncio
async def test_list_status_all_finished_reports_omission_summary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    for index in range(1, 4):
        child, handle = _new_finished_child(store, index)
        _persist_finished_receipt(store, child, f"agent-{index}", handle)

    status = await _status(
        AgentLoop(
            FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
        )
    )
    assert status["structuredContent"]["children"] == []
    assert status["structuredContent"]["finished_omitted"] == 3
    assert status["content"][0]["text"] == (
        "agent status: 0 children\n"
        "3 finished children omitted (query by handle for results)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("missing", "could not read child state"),
        ("invalid-json", "could not read child state"),
        ("non-dict", "lifecycle data is not an object"),
        ("terminal-without-finished-at", "terminal state has no finished_at"),
        ("finished-at-while-working", "non-terminal state has finished_at"),
    ],
)
async def test_list_status_keeps_damaged_children_visible(
    tmp_path: Path, damage: str, reason: str
) -> None:
    store = ConversationStore(tmp_path)
    damaged_child, damaged_handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, damaged_child, "agent-1", damaged_handle)
    healthy_child, healthy_handle = _new_live_child(store, 2)
    _persist_finished_receipt(store, healthy_child, "agent-2", healthy_handle)

    lifecycle_path = damaged_child.agent_lifecycle_path
    if damage == "missing":
        lifecycle_path.unlink()
    elif damage == "invalid-json":
        lifecycle_path.write_text("{", encoding="utf-8")
    elif damage == "non-dict":
        lifecycle_path.write_text("[]", encoding="utf-8")
    else:
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        if damage == "terminal-without-finished-at":
            lifecycle["state"] = "completed"
            lifecycle["finished_at"] = None
        else:
            lifecycle["state"] = "running"
            lifecycle["finished_at"] = "2026-09-04T10:00:01+00:00"
        lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    structured = status["structuredContent"]
    children = {child["handle"]: child for child in structured["children"]}
    assert structured["total"] == 2
    assert structured["truncated"] is False
    assert children[damaged_handle]["state"] == "unknown"
    assert children[damaged_handle]["finished_at"] is None
    assert reason in children[damaged_handle]["reason"]
    assert children[healthy_handle]["state"] == "running"
    assert f"child {damaged_handle}: state: unknown" in status["content"][0]["text"]
    if damage in {"missing", "invalid-json"}:
        explicit = await _status(loop, damaged_handle)
        assert explicit["isError"] is False
        assert explicit["structuredContent"]["children"][0]["state"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "state"),
    [
        ("started_at", "", "running"),
        ("started_at", "not-a-time", "running"),
        ("started_at", None, "running"),
        ("finished_at", "", "completed"),
        ("finished_at", "not-a-time", "completed"),
        ("finished_at", 42, "completed"),
    ],
)
async def test_list_status_keeps_invalid_timestamps_visible(
    tmp_path: Path, field: str, value: object, state: str
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["state"] = state
    lifecycle[field] = value
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    structured = status["structuredContent"]
    item = structured["children"][0]
    assert item["state"] == "unknown"
    assert "invalid child timestamps" in item["reason"]
    assert "finished_omitted" not in structured
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("turns_used", "tool_calls", "tree_budget", "depth")
        for value in (None, {}, [])
    ],
)
async def test_list_status_keeps_invalid_numeric_metadata_visible(
    tmp_path: Path, field: str, value: object
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle[field] = value
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]
    assert item["state"] == "unknown"
    assert f"invalid child metadata {field}" in item["reason"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_list_status_keeps_oversized_invalid_metadata_visible(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["tree_budget"] = "x" * 20_000
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]
    assert item["state"] == "unknown"
    assert "invalid child metadata tree_budget" in item["reason"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "agent_type",
        "description",
        "current_step",
    ],
)
async def test_list_status_bounds_displayed_string_metadata(
    tmp_path: Path, field: str
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle[field] = "x" * 20_000
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]
    assert item["state"] == "running"
    assert item[field]
    assert "[truncated]" in item[field]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_list_all_stays_bounded_for_many_live_children(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    for index in range(1, 51):
        child, handle = _new_live_child(store, index, step="s" * 2_000)
        _persist_finished_receipt(store, child, f"agent-{index}", handle)

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    structured = status["structuredContent"]
    assert structured["truncated"] is True
    assert structured["next_offset"] == len(structured["children"])
    assert structured["total"] == 50
    assert len(structured["children"]) < structured["total"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


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
    status = await _status(
        AgentLoop(
            FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ),
        handle,
    )
    item = status["structuredContent"]["children"][0]
    assert "[truncated]" in item["final_result"]
    assert "[truncated]" in item["current_step"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["started_at", "finished_at"])
async def test_list_status_keeps_oversized_timestamps_visible(
    tmp_path: Path, field: str
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle[field] = "2026-09-04T10:00:00." + ("0" * 20_000) + "+00:00"
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop, handle)
    item = status["structuredContent"]["children"][0]
    assert status["isError"] is False
    assert item["state"] in {"completed", "unknown"}
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [None, {}, [], -1, float("inf"), float("nan")])
async def test_list_status_keeps_invalid_elapsed_visible(
    tmp_path: Path, elapsed: object
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["elapsed"] = elapsed
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop, handle)
    item = status["structuredContent"]["children"][0]
    assert status["isError"] is False
    assert item["state"] in {"completed", "unknown"}
    assert "finished_omitted" not in status["structuredContent"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
@pytest.mark.parametrize("final_result", [None, {}, []])
async def test_list_status_keeps_non_string_final_result_visible(
    tmp_path: Path, final_result: object
) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["final_result"] = final_result
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]
    assert item["state"] == "unknown"
    assert "invalid child final_result" in item["reason"]
    assert "finished_omitted" not in status["structuredContent"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_list_status_closes_the_displayed_field_set(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    damaged_child, _ = _new_finished_child(store, 1)
    garbage = "x" * 20_000
    _persist_finished_receipt(store, damaged_child, "agent-1", garbage)
    lifecycle = json.loads(
        damaged_child.agent_lifecycle_path.read_text(encoding="utf-8")
    )
    lifecycle.update(
        {
            "handle": garbage,
            "state": garbage,
            "started_at": garbage,
            "finished_at": garbage,
            "elapsed": garbage,
            "turns_used": garbage,
            "tool_calls": garbage,
            "tree_budget": garbage,
            "current_step": garbage,
            "depth": garbage,
            "agent_type": garbage,
            "description": garbage,
            "final_result": garbage,
            "started_monotonic": garbage,
            "monotonic_pid": garbage,
        }
    )
    damaged_child.agent_lifecycle_path.write_text(
        json.dumps(lifecycle), encoding="utf-8"
    )
    healthy_child, healthy_handle = _new_finished_child(store, 2)
    _persist_finished_receipt(store, healthy_child, "agent-2", healthy_handle)

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    structured = status["structuredContent"]
    unknown = [child for child in structured["children"] if child["state"] == "unknown"]
    assert len(unknown) == 1
    assert unknown[0]["handle"]
    assert structured["finished_omitted"] == 1
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("handle", "wrong-handle"),
        ("state", []),
        ("started_at", None),
        ("finished_at", {}),
        ("elapsed", {}),
        ("turns_used", {}),
        ("tool_calls", []),
        ("tree_budget", None),
        ("current_step", {}),
        ("depth", []),
        ("agent_type", None),
        ("description", []),
        ("final_result", None),
    ],
)
async def test_projection_damage_keeps_the_child_visible(
    tmp_path: Path, field: str, value: object
) -> None:
    store = ConversationStore(tmp_path)
    finished = field in {"elapsed", "final_result"}
    child, handle = (
        _new_finished_child(store, 1) if finished else _new_live_child(store, 1)
    )
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle[field] = value
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop, handle if finished else None)

    assert status["isError"] is False
    assert len(status["structuredContent"]["children"]) == 1
    assert status["structuredContent"]["children"][0]["state"] == "unknown"
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_projection_damage_keeps_generated_reason_visible(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    child.agent_lifecycle_path.write_text("{", encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]

    assert status["isError"] is False
    assert item["state"] == "unknown"
    assert item["reason"] == "could not read child state"
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_large_turn_count_degrades_one_row(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["turns_used"] = int("9" * 4_200)
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]

    assert status["isError"] is False
    assert item["state"] == "unknown"
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_huge_monotonic_value_degrades_one_row(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_live_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle.update(
        {
            "started_monotonic": int("9" * 4_200),
            "monotonic_pid": os.getpid(),
        }
    )
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop)
    item = status["structuredContent"]["children"][0]

    assert status["isError"] is False
    assert item["state"] == "unknown"
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_unicode_final_result_is_bounded_by_encoded_bytes(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child, handle = _new_finished_child(store, 1)
    _persist_finished_receipt(store, child, "agent-1", handle)
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["final_result"] = "漢字🙂" * 5_000
    child.agent_lifecycle_path.write_text(
        json.dumps(lifecycle, ensure_ascii=False), encoding="utf-8"
    )

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    status = await _status(loop, handle)
    item = status["structuredContent"]["children"][0]

    assert status["isError"] is False
    assert item["state"] == "completed"
    assert item["final_result"]
    assert "[truncated]" in item["final_result"]
    assert len(encode_json(status)) <= loop.tool_registry.max_output_chars


@pytest.mark.asyncio
async def test_seeded_garbage_keeps_all_children_accounted_for(tmp_path: Path) -> None:
    randomizer = random.Random(147)
    store = ConversationStore(tmp_path)
    fields = (
        "handle",
        "state",
        "started_at",
        "finished_at",
        "elapsed",
        "turns_used",
        "tool_calls",
        "tree_budget",
        "current_step",
        "depth",
        "agent_type",
        "description",
        "final_result",
    )
    for index in range(1, 201):
        child, handle = _new_live_child(store, index)
        _persist_finished_receipt(store, child, f"agent-{index}", handle)
        lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
        for field in fields:
            if randomizer.random() < 0.55:
                size = randomizer.randrange(0, 50_001)
                lifecycle[field] = randomizer.choice(
                    [
                        None,
                        {},
                        [],
                        randomizer.randrange(-10, 11),
                        "x" * size,
                        ("漢字🙂\x00" * ((size // 4) + 1))[:size],
                    ]
                )
        child.agent_lifecycle_path.write_text(
            json.dumps(lifecycle, ensure_ascii=False), encoding="utf-8"
        )

    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    seen: list[str] = []
    offset = 0
    first_finished_omitted: int | None = None
    while True:
        status = await _status(loop, offset=offset)
        assert status["isError"] is False
        assert len(encode_json(status)) <= loop.tool_registry.max_output_chars
        structured = status["structuredContent"]
        page = structured["children"]
        seen.extend(child["handle"] for child in page)
        first_finished_omitted = structured.get(
            "finished_omitted", first_finished_omitted or 0
        )
        if not structured["truncated"]:
            break
        offset = structured["next_offset"]

    assert len(seen) == len(set(seen))
    assert len(seen) + (first_finished_omitted or 0) == 200
