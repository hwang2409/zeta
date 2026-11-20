"""Restart recovery of a child whose parent receipt landed before its lifecycle."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.agent.receipt import agent_stats, build_agent_receipt, receipt_tool_result
from zeta.core.store import ConversationIntegrityError, ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

_D5_TO_D1 = [("completed", "completed"), ("failed", "failed"), ("canceled", "canceled")]


def _running_child(parent: ConversationStore, call: ToolCall, index: str) -> Path:
    child = ConversationStore(parent.session_dir / "agents", session_id=index)
    child.mark_agent_parent(call.id)
    child.start_agent_lifecycle(
        handle=f"{parent.session_id}:{index}",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="work",
    )
    parent.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    parent.allocate_agent_index()
    parent.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="work",
        child_instance_id=f"{parent.session_id}:{index}",
    )
    path = child.session_dir
    child.close()
    return path


def _append_receipt(parent: ConversationStore, call: ToolCall, state: str) -> None:
    result = receipt_tool_result(
        call.id,
        build_agent_receipt(
            state,  # type: ignore[arg-type]
            f"child {state}",
            agent_stats(None, status=state, turns_used=1),
        ),
    )
    parent.append_message(
        Message(
            MessageRole.TOOL_RESULT, [TextContent(result.content)], tool_result=result
        )
    )


def _lifecycle_state(path: Path) -> str:
    store = ConversationStore(path.parent, session_id=path.name)
    try:
        return store.agent_lifecycle()["state"]
    finally:
        store.close()


async def _restart(tmp_path: Path) -> tuple[AgentLoop, ConversationStore]:
    store = await asyncio.to_thread(ConversationStore, tmp_path, session_id="root")
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
    try:
        await loop.activate()
    except BaseException:
        await loop.close()
        store.close()
        raise
    return loop, store


@pytest.mark.asyncio
@pytest.mark.parametrize(("receipt", "expected"), _D5_TO_D1)
async def test_restart_repairs_running_foreground_child_from_parent_receipt(
    tmp_path: Path, receipt: str, expected: str
) -> None:
    def seed() -> Path:
        root = ConversationStore(tmp_path, session_id="root")
        call = ToolCall("agent-1", "agent", {"prompt": "w", "description": "work"})
        path = _running_child(root, call, "1")
        _append_receipt(root, call, receipt)
        root.close()
        return path

    child_path = await asyncio.to_thread(seed)
    loop, store = await _restart(tmp_path)
    try:
        assert store.agent_children() == {}
        assert await asyncio.to_thread(_lifecycle_state, child_path) == expected
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("receipt", "expected"), _D5_TO_D1)
async def test_restart_repairs_running_nested_child_from_parent_receipt(
    tmp_path: Path, receipt: str, expected: str
) -> None:
    def seed() -> Path:
        root = ConversationStore(tmp_path, session_id="root")
        call = ToolCall("agent-1", "agent", {"prompt": "w", "description": "work"})
        child_path = _running_child(root, call, "1")
        child = ConversationStore(child_path.parent, session_id=child_path.name)
        grand_call = ToolCall("agent-2", "agent", {"prompt": "w", "description": "g"})
        grandchild_path = _running_child(child, grand_call, "1")
        _append_receipt(child, grand_call, receipt)
        child.close()
        root.close()
        return grandchild_path

    grandchild_path = await asyncio.to_thread(seed)
    loop, store = await _restart(tmp_path)
    try:
        assert store.agent_children() == {}
        assert await asyncio.to_thread(_lifecycle_state, grandchild_path) == expected
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", ["{not json", '{"state": 5}'])
async def test_restart_leaves_child_with_broken_lifecycle_file(
    tmp_path: Path, broken: str, caplog: pytest.LogCaptureFixture
) -> None:
    def seed() -> Path:
        root = ConversationStore(tmp_path, session_id="root")
        call = ToolCall("agent-1", "agent", {"prompt": "w", "description": "work"})
        path = _running_child(root, call, "1")
        root.close()
        (path / "agent_lifecycle.json").write_text(broken)
        return path

    child_path = await asyncio.to_thread(seed)
    lifecycle_file = child_path / "agent_lifecycle.json"
    with caplog.at_level(logging.WARNING, logger="zeta.agent.recovery_lease"):
        try:
            loop, store = await _restart(tmp_path)
        except ConversationIntegrityError as exc:
            pytest.fail(f"restart crashed on a broken child lifecycle: {exc}")
    try:
        assert list(store.agent_children()) == ["root:1"]
        assert lifecycle_file.read_text() == broken
        assert store.tool_result("agent-1") is None
        assert any(str(child_path) in r.getMessage() for r in caplog.records)
    finally:
        await loop.close()
        store.close()
