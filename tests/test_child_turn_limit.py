from __future__ import annotations

import asyncio
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

import zeta.agent.runner as agent_runner_module
from tests.support.agent_lifecycle import wait_background_owner
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail, owned_app
from zeta.agent import pending_delivery as pending_delivery_module
from zeta.agent.background import (
    BackgroundAgentOwner,
    finish_background_child,
    recover_agent_children,
)
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
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.server.ergonomics import _history_entry_rows
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog, discover_session_agents
from zeta.tools import ToolRegistry
from zeta.tools.agent import _read_agent_status
from zeta.tui.app import TUIApp
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
                    [ToolCall(f"noop-{number}", "noop", {})] if number < turns else []
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
    backend, result, store, loop = await _run_limited_child(
        tmp_path, profile_limit=2, monkeypatch=monkeypatch
    )
    try:
        assert len(backend.calls) == 2
        assert result.structured_content is not None
        structured = result.structured_content
        assert structured["status"] == "turn_limit"
        assert structured["turns_used"] == 2
        handoff = Path(structured["handoff_path"])
        assert (
            handoff == Path(structured["child_session_path"]) / "turn-limit-handoff.md"
        )
        assert handoff.exists()
        rows = [
            load_session_json(line)
            for line in (Path(structured["child_session_path"]) / "conversation.jsonl")
            .read_bytes()
            .splitlines()
        ]
        notices = [
            row
            for row in rows
            if row.get("data", {})
            .get("message", {})
            .get("metadata", {})
            .get("zeta_event")
            == "child_turn_limit"
        ]
        assert len(notices) == 1
    finally:
        await loop.close()
        store.close()


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
    backend, result, store, loop = await _run_limited_child(
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
        store.close()


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
async def test_nested_children_keep_independent_limits_after_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_call = ToolCall(
        "child-call",
        "agent",
        {
            "prompt": "child work",
            "description": "limited child",
            "agent_type": "child",
            "background": False,
        },
    )
    grandchild_call = ToolCall(
        "grandchild-call",
        "agent",
        {
            "prompt": "grandchild work",
            "description": "limited grandchild",
            "agent_type": "grandchild",
            "background": True,
        },
    )
    child_backend = FakeBackend([ScriptedTurn(tool_calls=[grandchild_call])])
    grandchild_started = asyncio.Event()
    release_grandchild = asyncio.Event()

    class PausedGrandchildBackend(FakeBackend):
        @owned_completion
        async def complete(self, messages, tool_schemas):
            if not self.calls:
                grandchild_started.set()
                await asyncio.wait_for(release_grandchild.wait(), timeout=10)
            async for event in FakeBackend.complete.__wrapped__(
                self, messages, tool_schemas
            ):
                yield event

    grandchild_backend = PausedGrandchildBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("noop-call", "noop", {})]),
            ScriptedTurn([TextContent("grandchild finished")]),
        ]
    )
    monkeypatch.setattr(
        agent_runner_module,
        "resolve_child_backend",
        lambda parent_loop, _model: (
            child_backend if parent_loop.agent_depth == 0 else grandchild_backend,
            None,
        ),
    )
    agents = tmp_path / "home" / "agents"
    agents.mkdir(parents=True)
    (agents / "child.md").write_text(
        "---\nname: child\ndescription: child\nmax_turns: 1\n---\nchild prompt\n",
        encoding="utf-8",
    )
    (agents / "grandchild.md").write_text(
        "---\nname: grandchild\ndescription: grandchild\n"
        "max_turns: 2\n---\ngrandchild prompt\n",
        encoding="utf-8",
    )
    store = ConversationStore(tmp_path / "session")
    registry = ToolRegistry(
        store.cwd,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=discover_session_agents(home=tmp_path / "home"),
    )

    async def noop(_arguments: dict[str, object]) -> str:
        return "done"

    registry.register("noop", noop, requires_approval=False)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[child_call])]),
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        events = [
            event async for event in loop.run_turn("start", origin=MessageOrigin.USER)
        ]
        child_terminal = next(
            event
            for event in events
            if event.type is StreamEventType.TOOL_EXECUTION_END
            and event.tool_call is not None
            and event.tool_call.id == "child-call"
        )
        assert child_terminal.tool_result is not None
        await await_or_fail(
            grandchild_started.wait(),
            what="grandchild backend start",
        )
        assert child_terminal.tool_result.structured_content is not None
        assert child_terminal.tool_result.structured_content["status"] == "turn_limit"
        assert len(child_backend.calls) == 1
        grandchild_id = f"{store.session_id}:1:1"
        assert loop._background_owner._parent_ids[grandchild_id] is None

        release_grandchild.set()
        await asyncio.wait_for(wait_background_owner(loop), timeout=10)
        assert len(grandchild_backend.calls) == 2
        notifications = store.agent_notifications(pending_only=False)
        grandchild_notification = next(
            item
            for item in notifications
            if item.data["child_instance_id"] == grandchild_id
        )
        from zeta.agent.turn_limit import project_notification_data

        assert (
            project_notification_data(grandchild_notification.data)["status"]
            == "turn_limit"
        )
        grandchild_store = ConversationStore(
            store.session_dir / "agents" / "1" / "agents", session_id="1"
        )
        try:
            lifecycle = grandchild_store.agent_lifecycle()
            assert lifecycle is not None
            assert lifecycle["turns_used"] == 2
            assert (
                parse_turn_limit_result(
                    lifecycle["final_result"], grandchild_store.session_dir
                )
                is not None
            )
        finally:
            grandchild_store.close()
    finally:
        release_grandchild.set()
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_empty_final_child_completion_does_not_nudge_past_hard_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, result, store, loop = await _run_limited_child(
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
        store.close()


@pytest.mark.asyncio
async def test_profile_turn_limit_allows_final_tool_batch_then_stops(
    tmp_path: Path,
) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    durable_effect = tmp_path / "effect"

    async def recording_tool(arguments: dict[str, object]) -> str:
        del arguments
        tool_started.set()
        await asyncio.wait_for(release_tool.wait(), timeout=10)
        durable_effect.write_text("finished", encoding="utf-8")
        return "tool complete"

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("record-1", "record", {})]),
            ScriptedTurn([TextContent("must not run")]),
        ]
    )
    store = ConversationStore(tmp_path / "session")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.register("record", recording_tool)
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        max_turns=3,
        agent_instance_id="root:1",
        child_turn_control=ChildTurnControl(1, str(store.session_dir)),
        skill_catalog=SkillCatalog.empty(),
    )
    turn = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=10)
        release_tool.set()
        events = await asyncio.wait_for(turn, timeout=10)
        assert durable_effect.read_text(encoding="utf-8") == "finished"
        second_completion_started = len(backend.calls) > 1
        assert second_completion_started is False
        assert (
            sum(event.type is StreamEventType.TOOL_EXECUTION_START for event in events)
            == 1
        )
    finally:
        release_tool.set()
        if not turn.done():
            turn.cancel()
            await asyncio.wait_for(
                asyncio.gather(turn, return_exceptions=True), timeout=10
            )
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_follow_up_during_final_tool_batch_is_terminally_rejected(
    tmp_path: Path,
) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    final_tool_finished = False

    async def wait_tool(arguments: dict[str, object]) -> str:
        nonlocal final_tool_finished
        del arguments
        tool_started.set()
        try:
            await asyncio.wait_for(release_tool.wait(), timeout=10)
            final_tool_finished = True
            return "tool complete"
        finally:
            release_tool.set()

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("wait-1", "wait", {})]),
            ScriptedTurn([TextContent("must not run")]),
        ]
    )
    store = ConversationStore(tmp_path / "session")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.register("wait", wait_tool)
    control = ChildTurnControl(1, str(store.session_dir))
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        max_turns=3,
        agent_instance_id="root:1",
        child_turn_control=control,
        skill_catalog=SkillCatalog.empty(),
    )
    turn = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=10)
        await loop.step_delivery.produce(
            pending_delivery_module.PromptInput(
                "next step", MessageOrigin.AGENT_SEND, None
            )
        )
        release_tool.set()
        await asyncio.wait_for(turn, timeout=10)
        assert final_tool_finished is True
        follow_up_processed = len(backend.calls) > 1
        second_completion_started = len(backend.calls) > 1
        assert follow_up_processed is False
        assert second_completion_started is False
    finally:
        release_tool.set()
        if not turn.done():
            turn.cancel()
            await asyncio.wait_for(
                asyncio.gather(turn, return_exceptions=True), timeout=10
            )
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_finish_gate_with_one_lifetime_completion_left_uses_only_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_called = False

    async def gate(*args, **kwargs):
        nonlocal gate_called
        del args, kwargs
        gate_called = True
        raise AssertionError("finish gate must not run after the hard limit")

    monkeypatch.setattr(agent_runner_module, "run_finish_gate", gate)
    backend, result, store, loop = await _run_limited_child(
        tmp_path,
        profile_limit=1,
        turns=1,
        monkeypatch=monkeypatch,
    )
    try:
        assert len(backend.calls) == 1
        assert gate_called is False
        assert result.structured_content is not None
        assert result.structured_content["status"] == "turn_limit"
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_top_level_pending_followup_still_runs_after_soft_max_turns(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn([]), ScriptedTurn([TextContent("final visible answer")])]
    )
    store = ConversationStore(tmp_path / "top-level")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        events = await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        empty_nudge_started = len(backend.calls) == 2
        assert empty_nudge_started is True
        assert len(backend.calls) == 2
        assert any(
            event.message is not None
            and event.message.role.value == "assistant"
            and any(
                isinstance(block, TextContent) and block.text == "final visible answer"
                for block in event.message.content
            )
            for event in events
        )
        assert not any(
            event.type is StreamEventType.ERROR
            and event.error is not None
            and event.error.code == "max_turns"
            for event in events
        )
    finally:
        await loop.close()
        store.close()


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


def test_agent_status_projects_nested_turn_limit_child_path(tmp_path: Path) -> None:
    child_store = ConversationStore(tmp_path / "child")
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    try:
        grandchild_call = ToolCall(
            "grandchild-call", "agent", {"description": "grandchild"}
        )
        child_store.register_agent_child(
            grandchild_call,
            child_session_path=str(grandchild_store.session_dir),
            description="grandchild",
            child_instance_id="root:1:1",
        )
        grandchild_store.mark_agent_parent(grandchild_call.id)
        grandchild_store.start_agent_lifecycle(
            handle="root:1:1",
            started_at="2026-01-01T00:00:00+00:00",
            depth=2,
            agent_type="grandchild",
            description="grandchild",
            cwd=str(tmp_path),
        )
        grandchild_store.finish_agent_lifecycle(
            "completed",
            final_result=build_turn_limit_result(
                _completed_receipt(), grandchild_store.session_dir
            ),
            turns_used=1,
            canonical_receipt=True,
        )

        status = _read_agent_status(child_store, requested_handle="root:1:1")[0]

        assert status["state"] == "turn_limit"
        assert status["handoff_path"] == str(
            grandchild_store.session_dir / "turn-limit-handoff.md"
        )
    finally:
        grandchild_store.close()
        child_store.close()


@pytest.mark.asyncio
async def test_tui_replay_renders_child_turn_limit_notice_once(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session")
    control = ChildTurnControl(10, str(store.session_dir))
    warning = None
    for _ in range(9):
        warning = control.begin_completion() or warning
        control.finish_completion()
    assert warning is not None
    store.append_message(warning)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    async with owned_app(app):
        app._rebuild_transcript()
        rendered = " ".join(Text.from_ansi(output.getvalue()).plain.split())
        assert rendered.count("child turn limit") == 1
        assert rendered.count("2 model completion(s) remain, including this one") == 1
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
        assert (
            str(Path(notification["child_session_path"]) / "turn-limit-handoff.md")
            in rendered
        )
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_phase",
    [
        "handoff_write",
        "descendant_register",
        "descendant_finish",
        "lifecycle_commit",
        "publication",
    ],
)
async def test_turn_limit_finalization_defers_cancel_at_each_await(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_phase: str,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(tmp_path / "child")
    call = ToolCall("child-call", "agent", {"description": "limited"})
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="limited",
        child_instance_id="child-1",
    )
    child_store.mark_agent_parent(call.id)
    child_store.start_agent_lifecycle(
        handle="child-1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="limited",
    )
    grandchild_call = ToolCall(
        "grandchild-call", "agent", {"description": "grandchild"}
    )
    child_store.register_agent_child(
        grandchild_call,
        child_session_path=str(tmp_path / "grandchild"),
        description="grandchild",
        child_instance_id="child-1:1",
    )
    owner = BackgroundAgentOwner(parent_store)
    owner.conversation_channel.register_delivery_flow(
        None, pending_delivery_module.StepDeliveryFlow(parent_store)
    )
    owner.conversation_channel.register_delivery_flow(
        "child-1", pending_delivery_module.StepDeliveryFlow(child_store)
    )
    control = ChildTurnControl(1, str(child_store.session_dir))
    control.begin_completion()
    control.finish_completion()
    child_loop = SimpleNamespace(
        child_turn_control=control,
        tool_registry=SimpleNamespace(
            background_tasks=SimpleNamespace(snapshot=lambda: ())
        ),
    )
    phase_started = asyncio.Event()
    release_phase = asyncio.Event()
    handoff_exists_at_publish = False
    original_child_write = child_store._to_thread_agent_state
    original_parent_write = parent_store._to_thread_agent_state
    original_publish = owner.conversation_channel.publish_completion

    async def gate(phase: str) -> None:
        if phase != cancel_phase:
            return
        phase_started.set()
        await asyncio.wait_for(release_phase.wait(), timeout=10)

    async def child_write(operation, *args, **kwargs):
        phases = {
            "ensure_turn_limit_handoff": "handoff_write",
            "finish_agent_child": "descendant_finish",
            "finish_agent_lifecycle": "lifecycle_commit",
        }
        await gate(phases.get(operation.__name__, ""))
        return await original_child_write(operation, *args, **kwargs)

    async def parent_write(operation, *args, **kwargs):
        if operation.__name__ == "register_agent_child":
            await gate("descendant_register")
        return await original_parent_write(operation, *args, **kwargs)

    async def pause_publish(**kwargs):
        nonlocal handoff_exists_at_publish
        handoff_exists_at_publish = (
            child_store.session_dir / "turn-limit-handoff.md"
        ).exists()
        await gate("publication")
        return await original_publish(**kwargs)

    # test-contract: seam force cancellation at each durable write await.
    monkeypatch.setattr(child_store, "_to_thread_agent_state", child_write)
    # test-contract: seam force cancellation at the descendant registration await.
    monkeypatch.setattr(parent_store, "_to_thread_agent_state", parent_write)
    monkeypatch.setattr(owner.conversation_channel, "publish_completion", pause_publish)

    async def close_child(_cause: str) -> tuple[()]:
        return ()

    def terminal_result(text: str, error: bool, status: str, stats=None, **kwargs):
        state = status if status in {"completed", "failed", "canceled"} else "completed"
        return build_agent_receipt(
            state,
            text,
            stats,
            structured_content={"status": status},
        )

    watcher = asyncio.create_task(
        finish_background_child(
            child_task=asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=terminal_result(
                        "done",
                        False,
                        "completed",
                        stats={"turns_used": 1, "elapsed": 0.0, "tool_calls": 0},
                    ),
                )
            ),
            child_store=child_store,
            parent_store=parent_store,
            tool_call=call,
            child_instance_id="child-1",
            child_path=str(child_store.session_dir),
            description="limited",
            child_turns=lambda: 1,
            build_result=terminal_result,
            validate_result=lambda result, tool_call_id: ToolResult(
                tool_call_id,
                result["content"][0]["text"],
                is_error=result["isError"],
                structured_content=result["structuredContent"],
            ),
            publish_event=lambda event: None,
            cleanup=lambda: owner.unregister("child-1"),
            close_child=close_child,
            error_message=str,
            background_owner=owner,
            child_loop=child_loop,
            finish_gate_state=SimpleNamespace(terminal_admitted=True),
        )
    )
    release_grandchild_watcher = asyncio.Event()
    grandchild_watcher = asyncio.create_task(
        await_or_fail(
            release_grandchild_watcher.wait(), what="grandchild watcher release"
        )
    )
    owner.register(
        "child-1:1",
        grandchild_watcher.cancel,
        grandchild_watcher,
        child_store,
        parent_instance_id="child-1",
    )
    owner.register("child-1", watcher.cancel, watcher, parent_store)
    try:
        await asyncio.wait_for(phase_started.wait(), timeout=10)
        watcher.cancel()
        release_phase.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(watcher, timeout=10)
        assert handoff_exists_at_publish is True
        lifecycle = child_store.agent_lifecycle()
        assert lifecycle is not None and lifecycle["state"] == "completed"
        assert (
            parse_turn_limit_result(
                lifecycle["final_result"], str(child_store.session_dir)
            )
            is not None
        )
        assert len(parent_store.agent_notifications(pending_only=False)) == 1
        assert "child-1:1" in parent_store.agent_children()
        assert "child-1:1" not in child_store.agent_children()
        assert owner._parent_ids["child-1:1"] is None
    finally:
        release_phase.set()
        if not watcher.done():
            watcher.cancel()
            await asyncio.wait_for(
                asyncio.gather(watcher, return_exceptions=True), timeout=10
            )
        owner.unregister("child-1:1")
        release_grandchild_watcher.set()
        await await_or_fail(grandchild_watcher, what="grandchild watcher cleanup")
        child_store.close()
        parent_store.close()


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
    assert (
        AgentCatalog.from_snapshot(catalog.to_snapshot()).find("limited").max_turns == 7
    )

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


def test_turn_limit_marker_round_trip_uses_post_receipt_position(
    tmp_path: Path,
) -> None:
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


def test_serve_replay_maps_turn_limit_receipt(tmp_path: Path) -> None:
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


@pytest.mark.asyncio
async def test_turn_limit_recovery_after_lifecycle_commit_publishes_once(
    tmp_path: Path,
) -> None:
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    call = ToolCall("child-call", "agent", {"description": "limited"})
    parent.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="limited",
        background=True,
        child_instance_id="root:1",
    )
    child.mark_agent_parent(call.id)
    child.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="limited",
    )
    handoff_bytes = b"child-authored recovery handoff\n"
    child.ensure_turn_limit_handoff(handoff_bytes)
    child.finish_agent_lifecycle(
        "completed",
        final_result=build_turn_limit_result(_completed_receipt(), child.session_dir),
        turns_used=1,
        canonical_receipt=True,
    )
    child.close()

    owner = BackgroundAgentOwner(parent)
    loop = SimpleNamespace(store=parent, _background_owner=owner)
    try:
        recover_agent_children(loop)
        notifications = parent.agent_notifications(pending_only=False)
        assert len(notifications) == 1
        from zeta.agent.turn_limit import project_notification_data

        projected = project_notification_data(notifications[0].data)
        assert projected["status"] == "turn_limit"
        assert projected["handoff_path"] == str(
            child.session_dir / "turn-limit-handoff.md"
        )
        assert not parent.agent_children()
        assert child.agent_lifecycle()["state"] == "completed"
        assert "[zeta:turn-limit:v1]" in child.agent_lifecycle()["final_result"]
        assert (
            child.session_dir / "turn-limit-handoff.md"
        ).read_bytes() == handoff_bytes

        recover_agent_children(loop)
        assert len(parent.agent_notifications(pending_only=False)) == 1
        assert (
            child.session_dir / "turn-limit-handoff.md"
        ).read_bytes() == handoff_bytes
    finally:
        parent.close()


def test_nested_turn_limit_recovery_matches_existing_mid_publication_behavior(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path / "root")
    parent = ConversationStore(root.session_dir / "agents", session_id="1")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    call = ToolCall("grandchild-call", "agent", {"description": "limited"})
    parent.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="limited",
        background=True,
        child_instance_id="root:1:1",
    )
    child.mark_agent_parent(call.id)
    child.start_agent_lifecycle(
        handle="root:1:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=2,
        agent_type="worker",
        description="limited",
    )
    handoff_bytes = b"nested crash handoff\n"
    child.ensure_turn_limit_handoff(handoff_bytes)
    marked = build_turn_limit_result(_completed_receipt(), child.session_dir)
    child.finish_agent_lifecycle(
        "completed",
        final_result=marked,
        turns_used=1,
        canonical_receipt=True,
    )
    root.append_agent_notification(
        "root:1:1",
        child_session_path=str(child.session_dir),
        description="limited",
        status="completed",
        text=marked,
        display_only=True,
    )
    child.close()

    owner = BackgroundAgentOwner(root)
    loop = SimpleNamespace(store=parent, _background_owner=owner)
    try:
        recover_agent_children(loop)
        assert len(root.agent_notifications(pending_only=False)) == 1
        assert len(parent.agent_notifications(pending_only=False)) == 1
        assert not parent.agent_children()
        assert child.agent_lifecycle()["state"] == "completed"
        assert "[zeta:turn-limit:v1]" in child.agent_lifecycle()["final_result"]
        assert (
            child.session_dir / "turn-limit-handoff.md"
        ).read_bytes() == handoff_bytes
    finally:
        parent.close()
        root.close()


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


def test_child_turn_control_warns_once_and_reserves_final_completion(
    tmp_path: Path,
) -> None:
    control = ChildTurnControl(2, str(tmp_path))
    assert control.begin_completion() is None
    assert control.finish_completion() is False
    warning = control.begin_completion()
    assert warning is not None
    assert warning.metadata["zeta_event"] == "child_turn_limit"
    assert control.finish_completion() is True
    assert control.begin_completion() is None
    assert control.exhausted
