from __future__ import annotations

import os
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text

from zeta.agent.background import recover_agent_children
from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.loop.agent import AgentLoop
from zeta.skills.catalog import SkillCatalog
from zeta.tui.app import TUIApp


def _append_notification(
    store: ConversationStore, child_instance_id: str
) -> str:
    return store.append_agent_notification(
        child_instance_id,
        child_session_path=f"/tmp/{child_instance_id}",
        description=f"notification {child_instance_id}",
        status="completed",
        text="done",
    ).id


def test_recovery_scans_notification_store_once_for_many_children(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="recovery-scan")
    for index in range(8):
        child_id = f"recovery-scan:{index}"
        call = ToolCall(f"call-{index}", "agent", {"prompt": "work"})
        store.register_agent_child(
            call,
            child_session_path=str(store.session_dir / "missing" / str(index)),
            description=f"child {index}",
            background=True,
            child_instance_id=child_id,
        )
        _append_notification(store, child_id)

    loop = SimpleNamespace(
        store=store,
        _background_owner=SimpleNamespace(notification_store=store),
    )
    with patch.object(
        store, "agent_notifications", wraps=store.agent_notifications
    ) as notifications:
        recover_agent_children(loop)

    assert notifications.call_count == 1
    assert store.agent_children() == {}


@pytest.mark.asyncio
async def test_tui_replay_scans_notifications_constant_times_and_batches_markers(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="tui-scan")
    notification_ids = [_append_notification(store, f"child-{index}") for index in range(8)]
    store.mark_agent_notification_presented_to_tui(notification_ids[0])
    store.acknowledge_agent_notification(notification_ids[1])
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
        resumed=True,
    )

    with patch.object(
        store, "replay_readonly", wraps=store.replay_readonly
    ) as replay_readonly:
        assert await app._rebuild_transcript_async(batch_size=3)

    assert replay_readonly.call_count == 1
    rendered = Text.from_ansi(output.getvalue()).plain
    assert "notification child-0" not in rendered
    assert "notification child-1" not in rendered
    for index in range(2, 8):
        assert f"notification child-{index}" in rendered

    reopened = ConversationStore(tmp_path, session_id="tui-scan")
    rows = reopened.replay()
    presented = {
        entry.data["notification_id"]
        for entry in rows
        if entry.type == "notification_tui_presented"
    }
    acknowledged = {
        entry.data["notification_id"]
        for entry in rows
        if entry.type == "notification_ack"
    }
    assert presented == {notification_ids[0], *notification_ids[2:]}
    assert acknowledged == {notification_ids[1]}

    second_output = StringIO()
    second_app = TUIApp(
        AgentLoop(FakeBackend([]), reopened, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=second_output, force_terminal=False),
        resumed=True,
    )
    assert await second_app._rebuild_transcript_async(batch_size=3)
    assert "notification child-" not in Text.from_ansi(second_output.getvalue()).plain


@pytest.mark.asyncio
async def test_batched_presentation_markers_are_visible_after_reopen(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="presentation-batch")
    notification_ids = [_append_notification(store, f"child-{index}") for index in range(4)]
    store.acknowledge_agent_notification(notification_ids[0])
    event_loop_thread = threading.get_ident()
    fsync_threads: list[int] = []
    real_fsync = os.fsync

    def observed_fsync(fd: int) -> None:
        fsync_threads.append(threading.get_ident())
        real_fsync(fd)

    with patch("zeta.core.store._log.os.fsync", side_effect=observed_fsync) as fsync:
        await store.mark_agent_notifications_presented_to_tui_async(notification_ids)

    fsync.assert_called_once()
    assert fsync_threads[0] != event_loop_thread
    reopened = ConversationStore(tmp_path, session_id="presentation-batch")
    rows = reopened.replay()
    assert {
        entry.data["notification_id"]
        for entry in rows
        if entry.type == "notification_tui_presented"
    } == set(notification_ids)
    assert [
        entry.data["notification_id"]
        for entry in rows
        if entry.type == "notification_ack"
    ] == [notification_ids[0]]
