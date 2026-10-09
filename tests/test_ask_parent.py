from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills.catalog import SkillCatalog
from zeta.tools.agent_send import send_to_run
from zeta.tools.registry import ToolRegistry


def _setup(tmp_path: Path):
    from zeta.tools.ask_parent import register_ask_parent

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
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        session_store=child,
        skill_catalog=SkillCatalog.empty(),
    )
    wakes: list[None] = []
    register_ask_parent(
        registry,
        parent_store=parent,
        child_store=child,
        child_instance_id=handle,
        notify_parent=lambda: wakes.append(None),
    )
    return parent, child, registry, handle, wakes


async def _wait_for_question(parent: ConversationStore):
    for _ in range(100):
        notifications = parent.agent_notifications()
        if notifications:
            return notifications[-1]
        await asyncio.sleep(0.01)
    raise AssertionError("question was not persisted")


@pytest.mark.asyncio
async def test_ask_parent_returns_answer_received_during_wait(tmp_path: Path) -> None:
    parent, _child, registry, handle, wakes = _setup(tmp_path)
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "ask-1",
                "ask_parent",
                {"question": "Which approach?", "options": ["A", "B"], "wait_seconds": 2},
            )
        )
    )
    notification = await _wait_for_question(parent)
    question_id = notification.data["question_id"]
    assert send_to_run(parent, handle, "Use B", question_id) is None

    result = await task

    assert result["isError"] is False
    assert result["content"][0]["text"] == "Use B"
    assert result["structuredContent"] == {
        "question_id": question_id,
        "answered": True,
        "answer": "Use B",
    }
    assert wakes == [None]


@pytest.mark.asyncio
async def test_ask_parent_timeout_leaves_later_answer_as_follow_up(tmp_path: Path) -> None:
    parent, child, registry, handle, _wakes = _setup(tmp_path)
    result = await registry.execute(
        ToolCall(
            "ask-1",
            "ask_parent",
            {"question": "Can I proceed?", "wait_seconds": 0},
        )
    )
    question = parent.agent_notifications()[0]

    assert result["isError"] is False
    assert (
        result["content"][0]["text"]
        == "no answer yet — proceed with a stated assumption"
    )
    assert send_to_run(parent, handle, "Proceed", question.data["question_id"]) is None
    assert [entry.data["text"] for entry in child.pending_prompts()] == ["Proceed"]


@pytest.mark.asyncio
async def test_ask_parent_rejects_a_second_concurrent_question(tmp_path: Path) -> None:
    parent, _child, registry, handle, _wakes = _setup(tmp_path)
    first = asyncio.create_task(
        registry.execute(
            ToolCall(
                "ask-1",
                "ask_parent",
                {"question": "First?", "wait_seconds": 2},
            )
        )
    )
    notification = await _wait_for_question(parent)
    second = await registry.execute(
        ToolCall(
            "ask-2",
            "ask_parent",
            {"question": "Second?", "wait_seconds": 0},
        )
    )

    assert second["isError"] is True
    assert "already has a pending parent question" in second["content"][0]["text"]
    assert len(parent.agent_notifications()) == 1
    assert send_to_run(parent, handle, "Answer", notification.data["question_id"]) is None
    await first


@pytest.mark.asyncio
async def test_child_question_is_persisted_once_and_replays_on_resume(tmp_path: Path) -> None:
    parent, _child, registry, _handle, _wakes = _setup(tmp_path)
    await registry.execute(
        ToolCall(
            "ask-1",
            "ask_parent",
            {"question": "Persist me", "wait_seconds": 0},
        )
    )
    original = parent.agent_notifications()
    parent_path = parent.session_dir
    parent.close()

    resumed = ConversationStore(
        parent_path.parent, session_id=parent_path.name, cwd=tmp_path
    )
    replayed = resumed.agent_notifications()

    assert len(original) == len(replayed) == 1
    assert replayed[0].id == original[0].id
    assert replayed[0].data["kind"] == "child_question"


def test_agent_send_rejects_reviewer_and_canceled_child(tmp_path: Path) -> None:
    parent, _child, _registry, handle, _wakes = _setup(tmp_path)
    marker = parent.agent_children()[handle]
    parent.finish_agent_child(handle)
    parent.register_agent_child(
        ToolCall.from_dict(marker["tool_call"]),
        child_session_path=str(marker["child_session_path"]),
        description="reviewer",
        agent_type="reviewer",
        background=True,
        child_instance_id=handle,
        accepts_follow_ups=False,
    )

    reviewer_error = send_to_run(parent, handle, "another pass")
    assert reviewer_error is not None
    assert "reviewers are one-shot; start a fresh reviewer" in reviewer_error

    parent.finish_agent_child(handle)
    canceled_error = send_to_run(parent, handle, "another pass")
    assert canceled_error is not None
    assert "canceled, finished, or never started" in canceled_error
