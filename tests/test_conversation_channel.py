from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.background import BackgroundAgentOwner
from zeta.agent.notifications import NotificationWake
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, TextContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


def _publish_completion(
    owner: BackgroundAgentOwner,
    child_instance_id: str,
) -> None:
    owner.conversation_channel.publish_completion(
        child_instance_id=child_instance_id,
        marker_key="child-call",
        child_session_path=f"/sessions/{child_instance_id}",
        description="nested work",
        status="completed",
        text="nested work complete",
        stats=None,
        killed_task_ids=None,
        killed_task_count=None,
        killed_task_ids_truncated=False,
    )


@pytest.mark.asyncio
async def test_grandchild_completion_is_display_only_for_root_while_parent_lives(
    tmp_path: Path,
) -> None:
    root_store = ConversationStore(tmp_path, session_id="root")
    parent_store = ConversationStore(root_store.session_dir / "agents", session_id="1")
    owner = BackgroundAgentOwner(root_store)
    parent_id = "root:1"
    grandchild_id = "root:1:1"
    watcher = asyncio.current_task()
    assert watcher is not None
    owner.register(parent_id, lambda: None, watcher, parent_store=root_store)
    owner.register(
        grandchild_id,
        lambda: None,
        watcher,
        parent_store=parent_store,
        parent_instance_id=parent_id,
    )
    owner.conversation_channel.register_loop(parent_id)
    root_wakes: list[None] = []
    owner.set_wake_callback(lambda: root_wakes.append(None))

    _publish_completion(owner, grandchild_id)
    await asyncio.wait_for(owner.conversation_channel.wait(parent_id), timeout=1)

    parent_wake = NotificationWake(parent_store)
    parent_message = parent_wake.pending_message()
    assert parent_message is not None
    assert parent_message.metadata["notifications"][0]["child_instance_id"] == (
        grandchild_id
    )
    root_entries = root_store.agent_notifications(pending_only=False)
    assert len(root_entries) == 1
    assert root_entries[0].data["display_only"] is True
    assert root_store.agent_notifications() == []
    assert NotificationWake(root_store).pending_message() is None
    assert root_wakes == []

    backend = FakeBackend([ScriptedTurn([TextContent("root reply")])])
    root_loop = AgentLoop(
        backend,
        root_store,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    _ = [
        event
        async for event in root_loop.run_turn(
            "next root turn", origin=MessageOrigin.USER
        )
    ]
    await root_loop.close()
    assert not any(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in backend.calls[0][0]
    )


@pytest.mark.asyncio
async def test_grandchild_completion_reaches_root_when_parent_is_not_live(
    tmp_path: Path,
) -> None:
    root_store = ConversationStore(tmp_path, session_id="root")
    parent_store = ConversationStore(root_store.session_dir / "agents", session_id="1")
    owner = BackgroundAgentOwner(root_store)
    parent_id = "root:1"
    grandchild_id = "root:1:1"
    watcher = asyncio.current_task()
    assert watcher is not None
    owner.register(parent_id, lambda: None, watcher, parent_store=root_store)
    owner.register(
        grandchild_id,
        lambda: None,
        watcher,
        parent_store=parent_store,
        parent_instance_id=parent_id,
    )
    root_wakes: list[None] = []
    owner.set_wake_callback(lambda: root_wakes.append(None))

    _publish_completion(owner, grandchild_id)

    root_message = NotificationWake(root_store).pending_message()
    assert root_message is not None
    assert root_message.metadata["notifications"][0]["child_instance_id"] == (
        grandchild_id
    )
    assert "display_only" not in root_store.agent_notifications()[0].data
    assert root_wakes == [None]


@pytest.mark.asyncio
async def test_direct_child_completion_remains_pending_for_root(tmp_path: Path) -> None:
    root_store = ConversationStore(tmp_path, session_id="root")
    owner = BackgroundAgentOwner(root_store)
    child_id = "root:1"
    watcher = asyncio.current_task()
    assert watcher is not None
    owner.register(child_id, lambda: None, watcher, parent_store=root_store)
    root_wakes: list[None] = []
    owner.set_wake_callback(lambda: root_wakes.append(None))

    _publish_completion(owner, child_id)

    notifications = root_store.agent_notifications()
    assert len(notifications) == 1
    assert "display_only" not in notifications[0].data
    assert root_wakes == [None]
