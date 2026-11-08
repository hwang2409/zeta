from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import zeta.agent.runner as agent_runner_module
from tests.support.agent_lifecycle import wait_background_owner
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.receipt import build_agent_receipt, has_agent_receipt_suffix
from zeta.agent.turn_limit import (
    ChildTurnControl,
    build_turn_limit_result,
    marker_for,
    parse_turn_limit_result,
)
from zeta.core.checkpoints import load_session_json
from zeta.core.store import ConversationEntry, ConversationStore
from zeta.core.store._validation import validate_agent_notification_data
from zeta.protocol.types import (
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)
from zeta.runtime.loop import AgentLoop
from zeta.server.ergonomics import _history_entry_rows
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog, discover_session_agents
from zeta.tools import ToolRegistry
from zeta.tools.agent import _read_agent_status
from zeta.tui.render import render_event


async def _run_limited_child(
    tmp_path: Path,
    *,
    profile_limit: int,
    spawn_limit: int | None = None,
    turns: int = 4,
    empty_final: bool = False,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeBackend, ToolResult, ConversationStore, AgentLoop]:
    arguments: dict[str, object] = {
        "prompt": "work",
        "description": "limited child",
        "agent_type": "limited",
        "background": False,
    }
    if spawn_limit is not None:
        arguments["max_turns"] = spawn_limit
    backend = FakeBackend(
        [
            ScriptedTurn(
                (
                    []
                    if empty_final and number == turns
                    else [TextContent(f"child completion {number}")]
                ),
                tool_calls=(
                    [ToolCall(f"noop-{number}", "noop", {})]
                    if number < turns
                    else []
                ),
            )
            for number in range(1, turns + 1)
        ]
    )
    monkeypatch.setattr(
        agent_runner_module,
        "resolve_child_backend",
        lambda _loop, _model: (backend, None),
    )
    parent_backend = FakeBackend(
        [ScriptedTurn(tool_calls=[ToolCall("agent-1", "agent", arguments)])]
    )
    store = ConversationStore(tmp_path / "session")
    agents = tmp_path / "home" / "agents"
    agents.mkdir(parents=True)
    (agents / "limited.md").write_text(
        "---\nname: limited\ndescription: limited child\n"
        f"max_turns: {profile_limit}\n---\nlimited child\n",
        encoding="utf-8",
    )
    catalog = discover_session_agents(home=tmp_path / "home")
    registry = ToolRegistry(
        store.cwd,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )

    async def noop(_arguments: dict[str, object]) -> str:
        return "done"

    registry.register("noop", noop)
    loop = AgentLoop(
        parent_backend,
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    events = [
        event async for event in loop.run_turn("start", origin=MessageOrigin.USER)
    ]
    terminal = next(
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_END
        and event.tool_call is not None
        and event.tool_call.name == "agent"
    )
    assert terminal.tool_result is not None
    return backend, terminal.tool_result, store, loop


@pytest.mark.asyncio
async def test_turn_limit_foreground_result_projects_status_and_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, result, _store, loop = await _run_limited_child(
        tmp_path, profile_limit=2, monkeypatch=monkeypatch
    )
    try:
        assert len(backend.calls) == 2
        assert result.structured_content is not None
        structured = result.structured_content
        assert structured["status"] == "turn_limit"
        assert structured["turns_used"] == 2
        handoff = Path(structured["handoff_path"])
        assert handoff == Path(structured["child_session_path"]) / "turn-limit-handoff.md"
        assert handoff.exists()
        rows = [
            load_session_json(line)
            for line in (
                Path(structured["child_session_path"]) / "conversation.jsonl"
            ).read_bytes().splitlines()
        ]
        notices = [
            row
            for row in rows
            if row.get("data", {}).get("message", {}).get("metadata", {}).get(
                "zeta_event"
            )
            == "child_turn_limit"
        ]
        assert len(notices) == 1
    finally:
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_limit", "spawn_limit", "expected"),
    [(3, 1, 1), (1, 3, 3), (2, None, 2)],
)
async def test_spawn_max_turns_precedence(
    tmp_path: Path,
    profile_limit: int,
    spawn_limit: int | None,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, result, _store, loop = await _run_limited_child(
        tmp_path,
        profile_limit=profile_limit,
        spawn_limit=spawn_limit,
        monkeypatch=monkeypatch,
    )
    try:
        assert len(backend.calls) == expected
        assert result.structured_content is not None
        assert result.structured_content["status"] == "turn_limit"
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_spawn_max_turns_overrides_larger_profile_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await test_spawn_max_turns_precedence(tmp_path, 3, 1, 1, monkeypatch)


@pytest.mark.asyncio
async def test_spawn_max_turns_overrides_smaller_profile_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await test_spawn_max_turns_precedence(tmp_path, 1, 3, 3, monkeypatch)


@pytest.mark.asyncio
async def test_profile_max_turns_applies_when_spawn_omits_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await test_spawn_max_turns_precedence(tmp_path, 2, None, 2, monkeypatch)


@pytest.mark.asyncio
async def test_empty_final_child_completion_does_not_nudge_past_hard_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, result, _store, loop = await _run_limited_child(
        tmp_path,
        profile_limit=1,
        turns=1,
        empty_final=True,
        monkeypatch=monkeypatch,
    )
    try:
        assert len(backend.calls) == 1
        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["status"] == "turn_limit"
    finally:
        await loop.close()


async def _run_limited_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ConversationStore, AgentLoop, dict[str, object]]:
    child_backend = FakeBackend([ScriptedTurn([TextContent("finished")])])
    monkeypatch.setattr(
        agent_runner_module,
        "resolve_child_backend",
        lambda _loop, _model: (child_backend, None),
    )
    call = ToolCall(
        "agent-1",
        "agent",
        {
            "prompt": "work",
            "description": "limited child",
            "agent_type": "limited",
            "background": True,
        },
    )
    store = ConversationStore(tmp_path / "session")
    agents = tmp_path / "home" / "agents"
    agents.mkdir(parents=True)
    (agents / "limited.md").write_text(
        "---\nname: limited\ndescription: limited child\nmax_turns: 1\n---\n"
        "limited child\n",
        encoding="utf-8",
    )
    catalog = discover_session_agents(home=tmp_path / "home")
    registry = ToolRegistry(
        store.cwd,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await wait_background_owner(loop)
    notification = store.agent_notifications(pending_only=False)[-1].data
    return store, loop, notification


async def _collect(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_turn_limit_writes_fallback_before_exposing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, loop, notification = await _run_limited_background(tmp_path, monkeypatch)
    try:
        path = Path(notification["child_session_path"])
        assert (path / "turn-limit-handoff.md").exists()
        assert notification["status"] == "completed"
        lifecycle = load_session_json(path / "agent_lifecycle.json")
        projection = parse_turn_limit_result(lifecycle["final_result"], path)
        assert projection is not None
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_agent_status_reloads_turn_limit_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, loop, notification = await _run_limited_background(tmp_path, monkeypatch)
    try:
        handle = str(notification["child_instance_id"])
        status = _read_agent_status(store, requested_handle=handle)[0]
        assert status["state"] == "turn_limit"
        assert status["handoff_path"] == str(
            Path(notification["child_session_path"]) / "turn-limit-handoff.md"
        )
        assert "[zeta:turn-limit" not in str(status["final_result"])
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_tui_replay_renders_turn_limit_and_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent.turn_limit import project_notification_data

    store, loop, notification = await _run_limited_background(tmp_path, monkeypatch)
    try:
        event = StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={
                "notification_id": "notification-1",
                **project_notification_data(notification),
            },
        )
        output = StringIO()
        Console(file=output, width=1000, color_system=None).print(render_event(event))
        rendered = output.getvalue()
        assert "turn_limit" in rendered
        assert str(Path(notification["child_session_path"]) / "turn-limit-handoff.md") in rendered
    finally:
        await loop.close()
        store.close()


def test_profile_max_turns_validation_and_snapshot(tmp_path: Path) -> None:
    home = tmp_path / "home"
    agents = home / "agents"
    agents.mkdir(parents=True)
    (agents / "limited.md").write_text(
        "---\nname: limited\ndescription: limited\nmax_turns: 7\n---\nbody\n",
        encoding="utf-8",
    )
    catalog = discover_session_agents(home=home)
    assert catalog.find("limited").max_turns == 7
    assert AgentCatalog.from_snapshot(catalog.to_snapshot()).find("limited").max_turns == 7

    for value in (True, 0, -1, 1.5, "1"):
        (agents / "limited.md").write_text(
            "---\nname: limited\ndescription: limited\n"
            f"max_turns: {json.dumps(value)}\n---\nbody\n",
            encoding="utf-8",
        )
        invalid = discover_session_agents(home=home)
        with pytest.raises(ValueError, match="unknown agent preset"):
            invalid.find("limited")


def _completed_receipt(text: str = "done") -> str:
    payload = build_agent_receipt(
        "completed",
        text,
        {"turns_used": 1, "elapsed": 0.1, "tool_calls": 0},
    )
    return payload["content"][0]["text"]


def test_model_text_cannot_forge_turn_limit_marker(tmp_path: Path) -> None:
    marker = marker_for(tmp_path)
    for answer in (f"{marker}\nanswer", f"answer\n{marker}\nanswer", marker):
        receipt = _completed_receipt(answer)
        assert parse_turn_limit_result(receipt, tmp_path) is None


def test_turn_limit_marker_round_trip_uses_post_receipt_position(tmp_path: Path) -> None:
    receipt = _completed_receipt()
    marked = build_turn_limit_result(receipt, tmp_path)
    readable, marker = marked.rsplit("\n", 1)
    assert has_agent_receipt_suffix(readable)
    assert marker == marker_for(tmp_path)
    projection = parse_turn_limit_result(marked, tmp_path)
    assert projection is not None
    assert projection.readable_text == receipt
    assert projection.handoff_path == str(tmp_path / "turn-limit-handoff.md")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace("v1", "v2"),
        lambda value: value.replace("turn-limit-handoff.md", "other.md"),
        lambda value: value.replace(":", ": ", 1),
        lambda value: value.replace("\n[zeta", " [zeta"),
        lambda value: value.split("\nStats:", 1)[0] + "\n" + marker_for("/wrong"),
    ],
)
def test_turn_limit_marker_rejects_changed_version_or_path(
    tmp_path: Path, mutation
) -> None:
    marked = build_turn_limit_result(_completed_receipt(), tmp_path)
    assert parse_turn_limit_result(mutation(marked), tmp_path) is None


def test_serve_history_projects_turn_limit_and_handoff(tmp_path: Path) -> None:
    marked = build_turn_limit_result(_completed_receipt(), tmp_path)
    data = {
        "origin": "notification",
        "kind": "agent_completion",
        "child_instance_id": "root:1",
        "child_session_path": str(tmp_path),
        "description": "limited",
        "status": "completed",
        "text": marked,
    }
    entry = ConversationEntry(1, "entry-1", None, "main", "notification", data)
    row = _history_entry_rows(entry, {})[0]
    notification = row["notification"]
    assert notification["status"] == "turn_limit"
    assert notification["handoff_path"] == str(tmp_path / "turn-limit-handoff.md")
    assert "[zeta:turn-limit" not in notification["text"]


def test_old_validators_accept_turn_limit_as_completed_text(tmp_path: Path) -> None:
    marked = build_turn_limit_result(_completed_receipt(), tmp_path)
    lifecycle = {
        "state": "completed",
        "final_result": marked,
    }
    assert load_session_json(json.dumps(lifecycle).encode()) == lifecycle
    notification = {
        "origin": "notification",
        "kind": "agent_completion",
        "child_instance_id": "root:1",
        "child_session_path": str(tmp_path),
        "description": "limited",
        "status": "completed",
        "text": marked,
    }
    validate_agent_notification_data(notification)
    assert notification["status"] == "completed"
    assert "turn_limit" not in notification


def test_existing_child_handoff_is_not_overwritten(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session")
    child_authored_bytes = b"child authored handoff\n"
    try:
        assert store.ensure_turn_limit_handoff(child_authored_bytes) is True
        assert store.ensure_turn_limit_handoff(b"fallback\n") is False
        handoff_bytes = (store.session_dir / "turn-limit-handoff.md").read_bytes()
        assert handoff_bytes == child_authored_bytes
    finally:
        store.close()


def test_child_turn_control_warns_once_and_reserves_final_completion(tmp_path: Path) -> None:
    control = ChildTurnControl(2, str(tmp_path))
    assert control.begin_completion() is None
    assert control.finish_completion() is False
    warning = control.begin_completion()
    assert warning is not None
    assert warning.metadata["zeta_event"] == "child_turn_limit"
    assert control.finish_completion() is True
    assert control.begin_completion() is None
    assert control.exhausted
