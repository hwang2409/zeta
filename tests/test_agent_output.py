import asyncio
import json
import multiprocessing
import os
import threading
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

import zeta.tools.agent as agent_tools
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.adoption import adopt_agent_children
from zeta.agent.receipt import (
    build_agent_receipt,
    encode_json,
    receipt_message_size,
    receipt_tool_result,
)
from zeta.agent.tool_results import validated_tool_result
from zeta.core import agent_progress, tool_dispatch
from zeta.core.agent_progress import remove_live_agent_lifecycle
from zeta.core.store import ConversationStore
from zeta.core.tool_dispatch import dispatch_tool_calls
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools.agent import _active_agent_receipts, agent_result
from zeta.tools.registry import ToolRegistry
from zeta.tui.cards.agent import AgentCard
from zeta.tui.render import render_event

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")


async def _collect(events):
    return [event async for event in events]


def _result(store: ConversationStore, call_id: str) -> ToolResult:
    return next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
        and message.tool_result.tool_call_id == call_id
    )


def test_receipt_compaction_preserves_killed_task_provenance() -> None:
    ids = [f"task-{index:02d}" for index in range(64)]
    result = build_agent_receipt(
        "completed",
        "answer",
        {"turns_used": 1, "elapsed": 0.1, "tool_calls": 0},
        structured_content={
            "killed_task_ids": ids,
            "killed_task_count": 100,
            "killed_task_ids_truncated": True,
        },
        max_bytes=1_000,
    )
    metadata = result["structuredContent"]
    assert isinstance(metadata, dict)
    assert metadata["killed_task_count"] == 100
    assert metadata["killed_task_ids_truncated"] is True
    assert len(metadata["killed_task_ids"]) < 64


@pytest.mark.parametrize(
    ("ids", "count"),
    [
        (["x" * 64], 1),
        ([f"task-{index}" for index in range(64)], 65),
        (["x" * 64] + [f"task-{index}" for index in range(63)], 65),
    ],
)
def test_killed_task_metadata_is_bounded_and_counted(ids: list[str], count: int) -> None:
    result = build_agent_receipt(
        "completed",
        "answer",
        {"turns_used": 1, "elapsed": 0.1, "tool_calls": 0},
        structured_content={
            "killed_task_ids": ids,
            "killed_task_count": count,
            "killed_task_ids_truncated": True,
        },
    )
    metadata = result["structuredContent"]
    assert isinstance(metadata, dict)
    assert metadata["killed_task_count"] == count
    assert metadata["killed_task_ids_truncated"] is True
    assert all(len(item) <= 64 for item in metadata["killed_task_ids"])
    assert len(metadata["killed_task_ids"]) <= 64


def test_small_receipt_limit_uses_effective_minimum() -> None:
    result = build_agent_receipt(
        "completed",
        "answer",
        {"turns_used": 1, "elapsed": 0.1, "tool_calls": 0},
        max_bytes=200,
    )
    assert result["content"][0]["text"].startswith("answer")
    assert len(encode_json(result)) <= 1_000
    assert 200 < receipt_message_size(result) <= 1_000


@pytest.mark.asyncio
async def test_agent_setup_exception_uses_failed_receipt(tmp_path: Path) -> None:
    call = ToolCall(
        "setup-failure",
        "agent",
        {"prompt": "inspect", "description": "setup"},
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        ConversationStore(tmp_path),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    setup_calls = 0

    async def fail_child_setup() -> None:
        nonlocal setup_calls
        setup_calls += 1
        if setup_calls == 2:
            raise RuntimeError("setup exploded")

    loop._ensure_mcp_servers = fail_child_setup
    events = await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    result = _result(loop.store, call.id)

    assert result.is_error is True
    assert result.is_canceled is False
    assert "agent error: setup exploded" in result.content
    assert "error=true · canceled=false" in result.content
    terminal = next(
        event for event in events if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content == result.content


@pytest.mark.asyncio
async def test_agent_setup_failure_finishes_child_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall(
        "late-setup-failure",
        "agent",
        {"prompt": "inspect", "description": "late setup"},
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    def fail_clone(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("registry clone exploded")

    loop_thread = threading.get_ident()
    read_threads: list[int] = []
    lifecycle_read = tool_dispatch.read_agent_lifecycle

    def tracked_lifecycle_read(path: str | Path) -> object:
        read_threads.append(threading.get_ident())
        return lifecycle_read(path)

    monkeypatch.setattr(loop.tool_registry, "clone_for_session", fail_clone)
    monkeypatch.setattr(tool_dispatch, "read_agent_lifecycle", tracked_lifecycle_read)
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    result = _result(store, call.id)
    assert result.is_error is True
    assert result.structured_content is not None
    assert "0 turns" in result.content
    assert "0 tool calls" in result.content
    assert "error=true · canceled=false" in result.content
    assert loop_thread not in read_threads
    handle = result.structured_content["child_instance_id"]
    assert type(handle) is str

    status = await loop.tool_registry.execute(
        ToolCall("status-after-setup-failure", "agent_status", {"handle": handle})
    )
    child = status["structuredContent"]["children"][0]
    assert child["state"] == "failed"
    assert child["final_result"].startswith("agent error: registry clone exploded")
    assert child["elapsed"] >= 0


@pytest.mark.asyncio
async def test_post_clone_setup_failure_releases_registry_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall(
        "post-clone-setup-failure",
        "agent",
        {"prompt": "inspect", "description": "post-clone setup"},
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        ConversationStore(tmp_path),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    captured_fd: int | None = None
    clone_for_session = loop.tool_registry.clone_for_session

    def capture_clone(store: ConversationStore, **kwargs: object):
        nonlocal captured_fd
        registry = clone_for_session(store, **kwargs)
        captured_fd = registry.background_tasks._directory_fd
        return registry

    def fail_child_loop_setup(self: AgentLoop, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise RuntimeError("child loop setup exploded")

    monkeypatch.setattr(loop.tool_registry, "clone_for_session", capture_clone)
    monkeypatch.setattr(AgentLoop, "__init__", fail_child_loop_setup)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert captured_fd is not None
    with pytest.raises(OSError):
        os.fstat(captured_fd)
    await loop.close()


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
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
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


def test_active_agent_receipt_discovery_does_not_scan_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    for index in range(50):
        store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent(f"message {index}")]),
                MessageOrigin.USER,
            )
        )
    call = ToolCall("agent-call", "agent", {})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("running")],
            tool_result=ToolResult(
                call.id,
                "running",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(store.session_dir / "agents" / "1"),
                },
            ),
        )
    )
    parsed = 0
    from_dict = Message.from_dict

    def counting_from_dict(data: dict[str, object]) -> Message:
        nonlocal parsed
        parsed += 1
        return from_dict(data)

    monkeypatch.setattr(Message, "from_dict", staticmethod(counting_from_dict))

    receipts = _active_agent_receipts(store)

    assert receipts == {"parent:1": store.session_dir / "agents" / "1"}
    assert parsed == 0

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    parsed = 0
    assert _active_agent_receipts(reopened) == receipts
    assert parsed == 0


@pytest.mark.asyncio
async def test_agent_output_rejects_handle_not_on_active_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    result = await loop.tool_registry.execute(
        ToolCall("output-missing", "agent_output", {"handle": "old:1"})
    )

    assert result["isError"] is True
    assert "unknown child handle" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_rejects_forged_and_out_of_tree_receipts(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    other = ConversationStore(tmp_path, session_id="other")
    victim = ConversationStore(other.session_dir / "agents", session_id="1")
    victim.append_message(Message(MessageRole.ASSISTANT, [TextContent("secret")]))
    real_call = ToolCall("real-agent", "agent", {})
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(real_call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("forged")],
            tool_result=ToolResult(
                "not-agent",
                "forged",
                structured_content={
                    "child_instance_id": "other:1",
                    "child_session_path": str(victim.session_dir),
                },
            ),
        )
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("out of tree")],
            tool_result=ToolResult(
                real_call.id,
                "out of tree",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(victim.session_dir),
                },
            ),
        )
    )
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    forged = await loop.tool_registry.execute(
        ToolCall("output-forged", "agent_output", {"handle": "other:1"})
    )
    out_of_tree = await loop.tool_registry.execute(
        ToolCall("output-outside", "agent_output", {"handle": "parent:1"})
    )

    assert forged["isError"] is True
    assert out_of_tree["isError"] is True
    assert "unknown child handle" in forged["content"][0]["text"]
    assert "unknown child handle" in out_of_tree["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_reads_new_live_tail_on_each_call(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("agent-call", "agent", {}))],
        )
    )
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
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

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
    parent = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    handle = "parent:1"
    child.start_agent_lifecycle(
        handle=handle,
        started_at="2026-09-04T10:00:00+00:00",
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
    parent.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    parent.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("agent-call", "agent", {}))],
        )
    )
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
        stats=child.agent_lifecycle() or {},
    )
    receipt_text = receipt["content"][0]["text"]
    assert "2 turns" in receipt_text
    assert "1 tool calls" in receipt_text
    assert "error=false" in receipt_text
    assert "canceled=false" in receipt_text

    status = await AgentLoop(
        FakeBackend([]),
        parent,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    ).tool_registry.execute(ToolCall("status-call", "agent_status", {"handle": handle}))
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
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
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
            lifecycle_projection=child.agent_lifecycle(),
        ),
    )

    rendered = render_event(event)
    assert rendered is not None
    assert "2 tool calls" in rendered.plain


async def _terminal_agent_event(
    tmp_path: Path,
    child: ConversationStore,
) -> tuple[AgentLoop, StreamEvent]:
    remove_live_agent_lifecycle(child.agent_lifecycle_path)
    registry = ToolRegistry(
        tmp_path,
        session_store=ConversationStore(tmp_path / "parent"),
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.register(
        "agent",
        lambda arguments: agent_result(
            "done",
            error=False,
            turns_used=1,
            child_session_path=str(child.session_dir),
            status="completed",
            stats=child.agent_lifecycle(),
        ),
    )
    loop = AgentLoop(
        FakeBackend([]),
        registry.session_store,
        registry=registry,
        skill_catalog=SkillCatalog.empty(),
    )
    events = [
        event
        async for event in dispatch_tool_calls(
            loop,
            [ToolCall("agent-terminal", "agent", {})],
            validated_tool_result,
        )
    ]
    return loop, next(
        event for event in events if event.type is StreamEventType.TOOL_EXECUTION_END
    )


@pytest.mark.asyncio
async def test_concurrent_status_calls_cannot_regress_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="child",
    )
    child.update_agent_lifecycle(tool_calls=2)
    child.finish_agent_lifecycle(
        "completed",
        final_result="done",
        turns_used=1,
        finished_at="2026-01-01T00:00:01+00:00",
    )
    loop, event = await _terminal_agent_event(tmp_path, child)
    lifecycle = child.agent_lifecycle()
    assert lifecycle is not None
    older = {**lifecycle, "tool_calls": 1}
    newer = {**lifecycle, "tool_calls": 2}
    older_started = threading.Event()
    release_older = threading.Event()
    lock = threading.Lock()
    calls = 0

    def controlled_read(*args: object) -> bytes:
        nonlocal calls
        with lock:
            calls += 1
            call = calls
        if call == 1:
            older_started.set()
            assert release_older.wait(1)
            return json.dumps(older).encode()
        release_older.set()
        return json.dumps(newer).encode()

    monkeypatch.setattr(agent_tools, "_read_child_file", controlled_read)
    first = asyncio.create_task(
        asyncio.to_thread(
            agent_tools._project_agent_status,
            object(),
            "parent:1",
            child.session_dir,
            notifications=(),
        )
    )
    assert await asyncio.to_thread(older_started.wait, 1)
    second = asyncio.create_task(
        asyncio.to_thread(
            agent_tools._project_agent_status,
            object(),
            "parent:1",
            child.session_dir,
            notifications=(),
        )
    )
    newer_result, older_result = await asyncio.gather(second, first)
    assert newer_result["tool_calls"] == 2
    assert older_result["tool_calls"] == 1
    rendered = AgentCard.render_receipt(event)
    assert rendered is not None
    assert "2 tool calls" in rendered.plain
    await loop.close()
    child.close()


@pytest.mark.asyncio
async def test_terminal_result_construction_never_reads_lifecycle_on_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="child",
    )
    child.finish_agent_lifecycle(
        "completed", final_result="done", turns_used=1, finished_at="2026-01-01T00:00:01+00:00"
    )
    loop_thread = threading.get_ident()
    read_threads: list[int] = []
    core_load = agent_progress.load_session_json
    tool_load = agent_tools.load_session_json
    def tracked_core_load(*args: object, **kwargs: object) -> object:
        read_threads.append(threading.get_ident())
        return core_load(*args, **kwargs)

    def tracked_tool_load(*args: object, **kwargs: object) -> object:
        read_threads.append(threading.get_ident())
        return tool_load(*args, **kwargs)

    monkeypatch.setattr(agent_progress, "load_session_json", tracked_core_load)
    monkeypatch.setattr(agent_tools, "load_session_json", tracked_tool_load)
    loop, _ = await _terminal_agent_event(tmp_path, child)
    assert read_threads and loop_thread not in read_threads
    await loop.close()
    child.close()


@pytest.mark.asyncio
async def test_full_child_runs_never_read_lifecycle_on_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle_read = tool_dispatch.read_agent_lifecycle
    loop_reads: list[str] = []

    def guarded_lifecycle_read(path: str | Path) -> object:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            loop_reads.append(str(path))
        return lifecycle_read(path)

    monkeypatch.setattr(tool_dispatch, "read_agent_lifecycle", guarded_lifecycle_read)

    async def run_child(case: str) -> None:
        call = ToolCall(
            f"child-{case}",
            "agent",
            {"prompt": "inspect", "description": case},
        )
        turns = [ScriptedTurn(tool_calls=[call])]
        if case == "success":
            turns.append(ScriptedTurn([TextContent("done")]))
        elif case == "cancel":
            turns.append(ScriptedTurn([TextContent("late")], delay=10))
        store = ConversationStore(tmp_path / case)
        loop = AgentLoop(
            FakeBackend(turns),
            store,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        )
        if case == "setup-failure":
            def fail_clone(*args: object, **kwargs: object) -> None:
                del args, kwargs
                raise RuntimeError("registry clone exploded")

            monkeypatch.setattr(loop.tool_registry, "clone_for_session", fail_clone)
        task = asyncio.create_task(
            _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        )
        if case == "cancel":
            while len(loop.backend.calls) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        await loop.close()

    for case in ("success", "failure", "cancel", "setup-failure"):
        await run_child(case)
    assert loop_reads == []


@pytest.mark.asyncio
async def test_tui_receipt_render_never_reads_lifecycle_on_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="child",
    )
    child.finish_agent_lifecycle(
        "completed", final_result="done", turns_used=1, finished_at="2026-01-01T00:00:01+00:00"
    )
    loop, event = await _terminal_agent_event(tmp_path, child)
    monkeypatch.setattr(
        "zeta.tui.cards.agent._read_lifecycle",
        lambda path: pytest.fail("terminal render read lifecycle storage"),
    )
    rendered = AgentCard.render_receipt(event)
    assert rendered is not None
    await loop.close()
    child.close()


@pytest.mark.asyncio
async def test_durable_terminal_receipt_survives_repeated_renders(
    tmp_path: Path,
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="child",
    )
    child.update_agent_lifecycle(tool_calls=3)
    child.finish_agent_lifecycle(
        "completed", final_result="done", turns_used=1, finished_at="2026-01-01T00:00:01+00:00"
    )
    loop, event = await _terminal_agent_event(tmp_path, child)
    assert event.tool_call is not None
    card = AgentCard(event.tool_call)
    first = card.finish(event)
    second = card.terminal_render()
    assert first is not None and second is not None
    assert "3 tool calls" in first.plain
    assert second.plain == first.plain
    await loop.close()
    child.close()


@pytest.mark.asyncio
async def test_agent_output_response_has_one_total_byte_bound(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("x" * 10_000)]), MessageOrigin.USER))
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("agent-call", "agent", {}))],
        )
    )
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
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    result = await loop.tool_registry.execute(
        ToolCall("output-large", "agent_output", {"handle": "parent:1"})
    )
    assert len(encode_json(result)) <= 10_000


@pytest.mark.asyncio
async def test_agent_output_pages_unicode_with_persistence_size(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("😀" * 1_800)]), MessageOrigin.USER))
    call = ToolCall("emoji-agent", "agent", {})
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("running")],
            tool_result=ToolResult(
                call.id,
                "running",
                structured_content={
                    "child_instance_id": "parent:1",
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    loop = AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    offset = 0
    pages: list[dict[str, object]] = []
    while True:
        page = await loop.tool_registry.execute(
            ToolCall(
                f"output-emoji-{offset}",
                "agent_output",
                {"handle": "parent:1", "offset": offset},
            )
        )
        assert page["isError"] is False
        assert len(encode_json(page)) <= 10_000
        pages.append(page)
        structured = page["structuredContent"]
        if not structured["truncated"]:
            break
        offset = structured["next_offset"]

    text = "".join(page["content"][0]["text"] for page in pages)
    assert len(pages) > 1
    assert text.count("😀") == 1_800


@pytest.mark.asyncio
async def test_agent_output_reads_adopted_background_descendant_notification(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("grandchild output")])
    )
    grandchild_call = ToolCall(
        "grandchild-call",
        "agent",
        {"prompt": "inspect", "description": "grandchild"},
    )
    grandchild.mark_agent_parent(grandchild_call.id)
    child.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(grandchild_call)])
    )
    child.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        background=True,
        child_instance_id="parent:1:1",
    )
    child.update_agent_child_turns("parent:1:1", 4)
    sibling_call = ToolCall(
        "sibling-call",
        "agent",
        {"prompt": "inspect", "description": "sibling"},
    )
    child.register_agent_child(
        sibling_call,
        child_session_path=str(child.session_dir / "agents" / "2"),
        description="sibling",
        child_instance_id="parent:1:2",
    )
    adopt_agent_children(child, root)
    reopened_root = ConversationStore(tmp_path, session_id="parent")
    assert reopened_root.agent_children()["parent:1:1"]["turns_used"] == 4
    reopened_root.close()
    root.append_agent_notification(
        "parent:1:1",
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        status="completed",
        text="grandchild complete",
    )
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    result = await loop.tool_registry.execute(
        ToolCall(
            "output-adopted-grandchild",
            "agent_output",
            {"handle": "parent:1:1"},
        )
    )

    assert result["isError"] is False
    assert "assistant: grandchild output" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_reads_snapshot_without_lock_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("saved")]))
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("agent-call", "agent", {}))],
        )
    )
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
    result = await AgentLoop(
        FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty()
    ).tool_registry.execute(
        ToolCall("output-snapshot", "agent_output", {"handle": "parent:1"})
    )

    assert result["isError"] is False
    assert "assistant: saved" in result["content"][0]["text"]
    assert "incomplete" not in result["content"][0]["text"]
    assert child.path.read_bytes() == before
    assert child.path.stat().st_mtime_ns == before_mtime


@pytest.mark.parametrize("max_bytes", [1, 64, 128, 512, 999])
def test_terminal_receipt_uses_documented_effective_minimum(max_bytes: int) -> None:
    result = build_agent_receipt("completed", "done", {}, max_bytes=max_bytes)
    assert receipt_message_size(result) <= 1_000
    assert result["content"][0]["text"]


def test_terminal_receipt_truncates_report_but_preserves_follow_up_reply() -> None:
    result = build_agent_receipt(
        "completed",
        "ignored: compatibility answer argument",
        {"turns_used": 1, "elapsed": 0.0, "tool_calls": 0},
        tool_call_id="receipt-follow-up",
        max_bytes=1_500,
        report="old report " * 2_000,
        reply="FINAL FOLLOW-UP REPLY",
    )

    text = result["content"][0]["text"]
    assert "[earlier report truncated]" in text
    assert "FINAL FOLLOW-UP REPLY" in text
    assert receipt_message_size(result, "receipt-follow-up") <= 1_500


def test_terminal_receipt_round_trips_after_store_reopen(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="receipt-parent")
    result = build_agent_receipt(
        "completed",
        "final answer",
        {"turns_used": 2, "elapsed": 1.5, "tool_calls": 1},
        structured_content={
            "child_instance_id": "receipt-parent:1",
            "child_session_path": str(tmp_path / "child"),
            "status": "completed",
        },
        tool_call_id="receipt-call",
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(result["content"][0]["text"])],
            tool_result=receipt_tool_result("receipt-call", result),
        )
    )
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    saved = reopened.messages()[0].tool_result
    assert saved is not None
    assert saved.content == result["content"][0]["text"]
    assert saved.structured_content == result["structuredContent"]
    assert saved.is_error is False
    assert saved.is_canceled is False


def test_terminal_receipt_keeps_legacy_status_flags_compatible(tmp_path: Path) -> None:
    common = {
        "turns_used": 1,
        "child_session_path": str(tmp_path / "child"),
        "child_instance_id": "parent:1",
        "stats": {"turns_used": 1, "elapsed": 0.0, "tool_calls": 0},
    }
    completed = agent_result("done", error=False, status="completed", **common)
    failed = agent_result("broken", error=True, status="error", **common)
    canceled = agent_result(
        "stopped", error=False, status="canceled", canceled=True, **common
    )

    assert completed["isError"] is False
    assert failed["isError"] is True
    assert canceled["isError"] is False
    assert canceled["isCanceled"] is True
    for result, status in (
        (completed, "completed"),
        (failed, "error"),
        (canceled, "canceled"),
    ):
        assert result["structuredContent"]["status"] == status
        assert "error=" in result["content"][0]["text"]
        assert "canceled=" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_error_total_cap_handles_multibyte_handle(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    result = await loop.tool_registry.execute(
        ToolCall("output-emoji", "agent_output", {"handle": "😀" * 20_000})
    )

    assert result["isError"] is True
    assert "unknown child handle" in result["content"][0]["text"]
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 10_000


def _record_running_child(
    parent: ConversationStore,
    child: ConversationStore,
    *,
    call: ToolCall,
    handle: str,
    depth: int,
) -> None:
    parent.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    parent.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description=handle,
        background=True,
        child_instance_id=handle,
    )
    child.mark_agent_parent(call.id)
    child.start_agent_lifecycle(
        handle=handle,
        started_at=f"2026-11-08T00:00:0{depth}+00:00",
        depth=depth,
        agent_type="general",
        description=handle,
    )


@pytest.mark.asyncio
async def test_root_reads_unadopted_grandchild_status_and_output(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    _record_running_child(
        root,
        child,
        call=ToolCall("child-call", "agent", {"prompt": "child"}),
        handle="root:1",
        depth=1,
    )
    _record_running_child(
        child,
        grandchild,
        call=ToolCall("grandchild-call", "agent", {"prompt": "grandchild"}),
        handle="root:1:1",
        depth=2,
    )
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("nested answer")])
    )
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    status = await loop.tool_registry.execute(
        ToolCall("nested-status", "agent_status", {"handle": "root:1:1"})
    )
    output = await loop.tool_registry.execute(
        ToolCall("nested-output", "agent_output", {"handle": "root:1:1"})
    )

    assert status["structuredContent"]["children"][0]["handle"] == "root:1:1"
    assert "assistant: nested answer" in output["content"][0]["text"]


@pytest.mark.asyncio
async def test_subtree_reads_reject_exact_path_marker_without_parent_call(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    forged = ToolCall("forged-call", "agent", {"prompt": "forged"})
    root.register_agent_child(
        forged,
        child_session_path=str(child.session_dir),
        description="forged",
        background=True,
        child_instance_id="root:1",
    )
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    status = await loop.tool_registry.execute(
        ToolCall("forged-status", "agent_status", {"handle": "root:1"})
    )
    output = await loop.tool_registry.execute(
        ToolCall("forged-output", "agent_output", {"handle": "root:1"})
    )

    assert status["isError"] is True
    assert "unknown child handle" in status["content"][0]["text"]
    assert output["isError"] is True
    assert "unknown child handle" in output["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_output_reads_and_fits_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    call = ToolCall("output-thread-call", "agent", {"prompt": "work"})
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    root.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("running")],
            tool_result=ToolResult(
                call.id,
                "running",
                structured_content={
                    "child_instance_id": "root:1",
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("answer")]))
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    event_loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    read_threads: list[int] = []
    original = agent_tools._read_child_file

    def controlled_read(*args: object, **kwargs: object) -> bytes:
        read_threads.append(threading.get_ident())
        asyncio_loop.call_soon_threadsafe(entered.set)
        assert release.wait(2)
        return original(*args, **kwargs)

    asyncio_loop = asyncio.get_running_loop()
    monkeypatch.setattr(agent_tools, "_read_child_file", controlled_read)
    task = asyncio.create_task(
        loop.tool_registry.execute(
            ToolCall("output-thread", "agent_output", {"handle": "root:1"})
        )
    )
    try:
        await entered.wait()
        assert read_threads[0] != event_loop_thread
        assert not task.done()
    finally:
        release.set()
    result = await task
    assert result["isError"] is False


@pytest.mark.asyncio
async def test_agent_status_projects_and_fits_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    _record_running_child(
        root,
        child,
        call=ToolCall("status-thread-call", "agent", {"prompt": "work"}),
        handle="root:1",
        depth=1,
    )
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    event_loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    projection_threads: list[int] = []
    original = agent_tools._status_response
    blocked = False

    def controlled_projection(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal blocked
        projection_threads.append(threading.get_ident())
        if not blocked:
            blocked = True
            asyncio_loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
        return original(*args, **kwargs)

    asyncio_loop = asyncio.get_running_loop()
    monkeypatch.setattr(agent_tools, "_status_response", controlled_projection)
    task = asyncio.create_task(
        loop.tool_registry.execute(ToolCall("status-thread", "agent_status", {}))
    )
    try:
        await entered.wait()
        assert projection_threads[0] != event_loop_thread
        assert not task.done()
    finally:
        release.set()
    result = await task
    assert result["isError"] is False


def _create_nested_child_in_process(
    root_dir: str, ready: Connection
) -> None:
    child = ConversationStore(Path(root_dir) / "root" / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    call = ToolCall("process-grandchild", "agent", {"prompt": "nested"})
    _record_running_child(
        child,
        grandchild,
        call=call,
        handle="root:1:1",
        depth=2,
    )
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("cross-process answer")])
    )
    ready.send("ready")
    ready.recv()
    grandchild.close()
    child.close()
    ready.close()


@pytest.mark.asyncio
async def test_resident_root_discovers_descendant_created_by_second_process(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    _record_running_child(
        root,
        child,
        call=ToolCall("process-child", "agent", {"prompt": "child"}),
        handle="root:1",
        depth=1,
    )
    child.close()
    parent_pipe, child_pipe = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_create_nested_child_in_process,
        args=(str(tmp_path), child_pipe),
    )
    process.start()
    child_pipe.close()
    try:
        assert await asyncio.to_thread(parent_pipe.recv) == "ready"
        loop = AgentLoop(
            FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
        )
        status = await loop.tool_registry.execute(
            ToolCall("process-status", "agent_status", {"handle": "root:1:1"})
        )
        output = await loop.tool_registry.execute(
            ToolCall("process-output", "agent_output", {"handle": "root:1:1"})
        )

        assert status["structuredContent"]["children"][0]["handle"] == "root:1:1"
        assert "cross-process answer" in output["content"][0]["text"]
    finally:
        parent_pipe.send("close")
        await asyncio.to_thread(process.join, 5)
        if process.is_alive():
            process.terminate()
            process.join()
        parent_pipe.close()
    assert process.exitcode == 0
    process.close()
    root.close()


@pytest.mark.parametrize(
    "handle",
    ["root:1", "root:2", "other:1:1", "root:1:0", "root:10:2", "root:1:bad"],
)
@pytest.mark.asyncio
async def test_nested_reader_rejects_outside_and_malformed_handles_before_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handle: str,
) -> None:
    from zeta.tools.agent import subtree

    child = ConversationStore(tmp_path / "root" / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-11-08T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="child",
    )
    opens = 0

    def forbidden_open(path: Path) -> ConversationStore:
        del path
        nonlocal opens
        opens += 1
        raise AssertionError("invalid handle opened a session")

    monkeypatch.setattr(subtree, "_open_snapshot", forbidden_open)
    loop = AgentLoop(
        FakeBackend([]), child, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    result = await loop.tool_registry.execute(
        ToolCall("invalid-output", "agent_output", {"handle": handle})
    )

    assert result["isError"] is True
    assert "unknown child handle" in result["content"][0]["text"]
    assert opens == 0


@pytest.mark.asyncio
async def test_damaged_intermediate_is_unknown_and_healthy_sibling_survives(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    damaged = ConversationStore(root.session_dir / "agents", session_id="1")
    hidden = ConversationStore(damaged.session_dir / "agents", session_id="1")
    healthy = ConversationStore(root.session_dir / "agents", session_id="2")
    _record_running_child(
        root,
        damaged,
        call=ToolCall("damaged", "agent", {"prompt": "damaged"}),
        handle="root:1",
        depth=1,
    )
    _record_running_child(
        damaged,
        hidden,
        call=ToolCall("hidden", "agent", {"prompt": "hidden"}),
        handle="root:1:1",
        depth=2,
    )
    _record_running_child(
        root,
        healthy,
        call=ToolCall("healthy", "agent", {"prompt": "healthy"}),
        handle="root:2",
        depth=1,
    )
    remove_live_agent_lifecycle(damaged.agent_lifecycle_path)
    damaged.close()
    damaged.path.write_text("not json\n")
    loop = AgentLoop(
        FakeBackend([]), root, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    status = await loop.tool_registry.execute(
        ToolCall("damaged-status", "agent_status", {})
    )
    output = await loop.tool_registry.execute(
        ToolCall("damaged-output", "agent_output", {"handle": "root:1"})
    )

    rows = {
        row["handle"]: row for row in status["structuredContent"]["children"]
    }
    assert rows["root:1"]["state"] == "unknown"
    assert rows["root:2"]["state"] == "running"
    assert "root:1:1" not in rows
    assert output["isError"] is True
    assert "could not read child transcript" in output["content"][0]["text"]
