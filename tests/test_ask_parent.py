from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.background import BackgroundAgentOwner
from zeta.agent.runner import consume_run
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.skills.catalog import SkillCatalog
from zeta.tools.agent_send import send_to_run
from zeta.tools.ask_parent import register_ask_parent
from zeta.tools.registry import ToolRegistry


def _setup(tmp_path: Path):
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1", cwd=tmp_path)
    handle = "parent:1"
    parent.register_agent_child(
        ToolCall("agent-1", "agent", {"prompt": "work", "description": "worker"}),
        child_session_path=str(child.session_dir),
        description="worker",
        agent_type="worker",
        background=True,
        child_instance_id=handle,
        accepts_follow_ups=True,
    )
    owner = BackgroundAgentOwner(parent)
    watcher = asyncio.create_task(asyncio.sleep(60))
    owner.register(
        handle,
        watcher.cancel,
        watcher,
        parent_store=parent,
        active_store=child,
    )
    channel = getattr(owner, "conversation_channel", None)
    if channel is not None:
        channel.register_loop(handle)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        session_store=child,
        skill_catalog=SkillCatalog.empty(),
    )
    if channel is None:
        register_ask_parent(
            registry,
            parent_store=parent,
            child_store=child,
            child_instance_id=handle,
            notify_parent=lambda: None,
        )
    else:
        register_ask_parent(
            registry,
            channel=channel,
            child_instance_id=handle,
        )
    return parent, child, registry, handle, owner


@pytest.mark.asyncio
async def test_ask_parent_returns_immediately_and_persists_once(tmp_path: Path) -> None:
    parent, _child, registry, handle, owner = _setup(tmp_path)
    result = await asyncio.wait_for(
        registry.execute(
            ToolCall(
                "ask-1",
                "ask_parent",
                {"question": "Which approach?", "options": ["A", "B"]},
            )
        ),
        timeout=0.2,
    )

    questions = parent.agent_notifications()
    assert result["isError"] is False
    assert "continue your work" in result["content"][0]["text"]
    assert len(questions) == 1
    assert questions[0].data["kind"] == "child_question"
    assert questions[0].data["child_instance_id"] == handle
    owner.cancel_all()


@pytest.mark.asyncio
async def test_ordinary_agent_send_is_not_consumed_as_question_answer(
    tmp_path: Path,
) -> None:
    parent, child, registry, handle, owner = _setup(tmp_path)
    ask = asyncio.create_task(
        registry.execute(
            ToolCall("ask-1", "ask_parent", {"question": "Can I proceed?"})
        )
    )
    for _ in range(100):
        if parent.agent_notifications():
            break
        await asyncio.sleep(0.01)

    assert send_to_run(parent, handle, "Proceed") is None
    await asyncio.wait_for(ask, timeout=0.2)
    assert [entry.data["text"] for entry in child.pending_prompts()] == ["Proceed"]
    assert len(parent.agent_notifications()) == 1
    owner.cancel_all()


@pytest.mark.asyncio
async def test_ask_parent_schema_and_handler_reject_oversized_question(
    tmp_path: Path,
) -> None:
    _parent, _child, registry, _handle, owner = _setup(tmp_path)
    schema = next(item for item in registry.schemas if item["name"] == "ask_parent")
    assert schema["parameters"]["properties"]["question"]["maxLength"] == 4_000

    result = await registry.execute(
        ToolCall("ask-1", "ask_parent", {"question": "x" * 4_001})
    )
    assert result["isError"] is True
    assert "too long" in result["content"][0]["text"]
    owner.cancel_all()


@pytest.mark.asyncio
async def test_child_finish_and_cancel_withdraw_pending_questions(
    tmp_path: Path,
) -> None:
    parent, _child, _registry, handle, owner = _setup_without_loop(tmp_path)
    channel = owner.conversation_channel
    for reason in ("finished", "canceled"):
        channel.publish_question(
            child_instance_id=handle,
            question_id=reason,
            question=f"Question before {reason}?",
            options=None,
        )
        assert channel.close_questions(handle, reason=reason) == 1

    pending = parent.agent_notifications()
    assert [entry.data["kind"] for entry in pending] == [
        "child_question_withdrawn",
        "child_question_withdrawn",
    ]
    assert [entry.data["text"] for entry in pending] == [
        "question withdrawn: child finished",
        "question withdrawn: child canceled",
    ]


@pytest.mark.asyncio
async def test_grandchild_question_wakes_idle_direct_parent_loop(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path / "root")
    parent = ConversationStore(root.session_dir / "agents", session_id="1", cwd=tmp_path)
    owner = BackgroundAgentOwner(root)
    parent_id = "root:1"
    grandchild_id = "root:1:1"
    owner.register(
        parent_id,
        lambda: None,
        None,  # type: ignore[arg-type] - consume_run is driven directly below
        parent_store=root,
        active_store=parent,
    )
    owner.register(
        grandchild_id,
        lambda: None,
        None,  # type: ignore[arg-type] - only live ownership is relevant
        parent_store=parent,
        parent_instance_id=parent_id,
    )
    owner.conversation_channel.register_loop(parent_id)
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("initial work done")]),
            ScriptedTurn([TextContent("question observed")]),
        ]
    )
    loop = AgentLoop(
        backend,
        parent,
        max_turns=None,
        skill_catalog=SkillCatalog.empty(),
        agent_depth=1,
        agent_instance_id=parent_id,
        background_owner=owner,
    )
    turns = 0

    def update_turns(value: int) -> None:
        nonlocal turns
        turns = value

    task = asyncio.create_task(
        consume_run(
            loop,
            "work",
            child_store=parent,
            child_path=str(parent.session_dir),
            publish=lambda _status: None,
            child_turns=lambda: turns,
            update_turns=update_turns,
            update_step=lambda _step: None,
            update_tool_calls=lambda _count: None,
            finish_lifecycle=lambda _state, _text: {},
            publish_lifecycle=lambda *_args, **_kwargs: None,
            child_result=lambda text, **_kwargs: {
                "content": [{"type": "text", "text": text}],
                "isError": False,
                "structuredContent": None,
            },
            error_message=str,
        )
    )
    for _ in range(100):
        if backend.calls:
            break
        await asyncio.sleep(0.01)
    assert len(backend.calls) == 1

    owner.conversation_channel.publish_question(
        child_instance_id=grandchild_id,
        question_id="wake-parent",
        question="Need input",
        options=None,
    )
    for _ in range(100):
        if len(backend.calls) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(backend.calls) == 2
    assert any(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in backend.calls[1][0]
    )

    owner.unregister(grandchild_id)
    owner.conversation_channel.wake(parent_id)
    result = await asyncio.wait_for(task, timeout=1)
    assert result["isError"] is False


def test_resume_closes_question_for_child_that_no_longer_runs_once(
    tmp_path: Path,
) -> None:
    parent, child, _registry, handle, owner = _setup_without_loop(tmp_path)
    child.mark_agent_parent("agent-1", agent_type="worker")
    owner.conversation_channel.publish_question(
        child_instance_id=handle,
        question_id="resume-question",
        question="Still there?",
        options=None,
    )
    parent_path = parent.session_dir
    parent.close()
    child.close()

    resumed = ConversationStore(
        parent_path.parent, session_id=parent_path.name, cwd=tmp_path
    )
    loop = AgentLoop(
        FakeBackend([]), resumed, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    pending = resumed.agent_notifications()
    assert [entry.data["kind"] for entry in pending] == [
        "child_question_withdrawn",
        "agent_completion",
    ]

    resumed.close()
    reopened = ConversationStore(
        parent_path.parent, session_id=parent_path.name, cwd=tmp_path
    )
    second = AgentLoop(
        FakeBackend([]), reopened, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    assert len(reopened.agent_notifications()) == 2
    assert loop is not second


def test_resume_with_no_child_marker_withdraws_orphaned_question(
    tmp_path: Path,
) -> None:
    parent, child, _registry, handle, owner = _setup_without_loop(tmp_path)
    owner.conversation_channel.publish_question(
        child_instance_id=handle,
        question_id="orphan",
        question="Anyone there?",
        options=None,
    )
    parent.finish_agent_child(handle)
    parent_path = parent.session_dir
    parent.close()
    child.close()

    resumed = ConversationStore(
        parent_path.parent, session_id=parent_path.name, cwd=tmp_path
    )
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())
    pending = resumed.agent_notifications()
    assert len(pending) == 1
    assert pending[0].data["kind"] == "child_question_withdrawn"
    assert pending[0].data["text"] == "question withdrawn: child canceled"


@pytest.mark.asyncio
async def test_grandchild_question_wakes_direct_parent_then_routes_to_live_ancestor(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path / "root")
    parent = ConversationStore(root.session_dir / "agents", session_id="1", cwd=tmp_path)
    grandchild = ConversationStore(
        parent.session_dir / "agents", session_id="1", cwd=tmp_path
    )
    owner = BackgroundAgentOwner(root)
    parent_task = asyncio.create_task(asyncio.sleep(60))
    child_task = asyncio.create_task(asyncio.sleep(60))
    owner.register(
        "root:1",
        parent_task.cancel,
        parent_task,
        parent_store=root,
        active_store=parent,
    )
    owner.register(
        "root:1:1",
        child_task.cancel,
        child_task,
        parent_store=parent,
        active_store=grandchild,
        parent_instance_id="root:1",
    )
    channel = owner.conversation_channel
    channel.register_loop("root:1")

    channel.publish_question(
        child_instance_id="root:1:1",
        question_id="direct",
        question="Direct?",
        options=None,
    )
    await asyncio.wait_for(channel.wait("root:1"), timeout=0.1)
    assert [entry.data["question_id"] for entry in parent.agent_notifications()] == [
        "direct"
    ]
    assert root.agent_notifications() == []

    channel.unregister_loop("root:1")
    owner.unregister("root:1")
    channel.publish_question(
        child_instance_id="root:1:1",
        question_id="adopted",
        question="Ancestor?",
        options=None,
    )
    routed = root.agent_notifications()
    assert [entry.data["question_id"] for entry in routed] == ["adopted"]
    assert "next live ancestor" in routed[0].data["routing_note"]
    child_task.cancel()


def _setup_without_loop(tmp_path: Path):
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1", cwd=tmp_path)
    handle = "parent:1"
    parent.register_agent_child(
        ToolCall("agent-1", "agent", {"prompt": "work", "description": "worker"}),
        child_session_path=str(child.session_dir),
        description="worker",
        agent_type="worker",
        background=True,
        child_instance_id=handle,
        accepts_follow_ups=True,
    )
    owner = BackgroundAgentOwner(parent)
    owner.register(
        handle,
        lambda: None,
        None,  # type: ignore[arg-type] - watcher state is not used by this store test
        parent_store=parent,
        active_store=child,
    )
    return parent, child, None, handle, owner
