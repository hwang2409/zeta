from __future__ import annotations

import asyncio
import os
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from tests.support.branch_reads import branch_entries
from tests.support.fake_backend import FakeBackend
from zeta.agent.background_lifecycle import recover_agent_children
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ToolCall,
)
from zeta.runtime.loop.agent import AgentLoop
from zeta.skills.catalog import SkillCatalog
from zeta.storage_format import WRITER_EPOCH
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


def _register_missing_background_child(
    store: ConversationStore, index: int
) -> str:
    child_id = f"{store.session_id}:{index}"
    store.register_agent_child(
        ToolCall(f"call-{index}", "agent", {"prompt": "work"}),
        child_session_path=str(store.session_dir / "missing" / str(index)),
        description=f"child {index}",
        background=True,
        child_instance_id=child_id,
    )
    return child_id


@pytest.mark.asyncio
async def test_recovery_does_not_duplicate_notification_appended_after_indexing(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="recovery-race")
    child_1 = _register_missing_background_child(store, 1)
    child_2 = _register_missing_background_child(store, 2)
    external = ConversationStore(tmp_path, session_id="recovery-race")
    original_sync = store.sync_agent_completion_notification

    def sync_after_external_completion(child_instance_id: str):
        if child_instance_id == child_2:
            _append_notification(external, child_2)
        return original_sync(child_instance_id)

    loop = SimpleNamespace(
        store=store,
        _background_owner=SimpleNamespace(notification_store=store),
    )
    with patch.object(
        store,
        "sync_agent_completion_notification",
        side_effect=sync_after_external_completion,
    ):
        await asyncio.to_thread(recover_agent_children, loop, WRITER_EPOCH)

    notifications = [
        entry
        for entry in store.agent_notifications(pending_only=False)
        if entry.data["child_instance_id"] == child_2
    ]
    assert len(notifications) == 1
    assert notifications[0].data["status"] == "completed"
    assert any(
        entry.data["child_instance_id"] == child_1
        for entry in store.agent_notifications(pending_only=False)
    )


@pytest.mark.asyncio
async def test_nested_recovery_refreshes_both_notification_stores_per_child(
    tmp_path: Path,
) -> None:
    root_store = ConversationStore(tmp_path / "root", session_id="root")
    nested_store = ConversationStore(tmp_path / "nested", session_id="nested")
    child_paths: dict[int, Path] = {}
    for index in (1, 2):
        child_path = nested_store.session_dir / "agents" / str(index)
        child_paths[index] = child_path
        with ConversationStore(child_path.parent, session_id=child_path.name) as child:
            child.mark_agent_parent(f"call-{index}")
        nested_store.register_agent_child(
            ToolCall(f"call-{index}", "agent", {"prompt": "work"}),
            child_session_path=str(child_path),
            description=f"child {index}",
            background=True,
            child_instance_id=f"nested:{index}",
        )

    external_nested = ConversationStore(tmp_path / "nested", session_id="nested")
    original_sync = root_store.sync_agent_completion_notification

    def sync_after_indexes(child_instance_id: str):
        if child_instance_id == "nested:2":
            _append_notification(external_nested, "nested:2")
        return original_sync(child_instance_id)

    loop = SimpleNamespace(
        store=nested_store,
        _background_owner=SimpleNamespace(notification_store=root_store),
    )
    with patch.object(
        root_store,
        "sync_agent_completion_notification",
        side_effect=sync_after_indexes,
    ):
        await asyncio.to_thread(recover_agent_children, loop, WRITER_EPOCH)

    with ConversationStore(child_paths[2].parent, session_id="2") as child_2:
        assert child_2.agent_canceled() is None
    assert [
        (entry.data["child_instance_id"], entry.data["status"])
        for entry in root_store.agent_notifications(pending_only=False)
    ] == [("nested:1", "canceled"), ("nested:2", "completed")]
    assert [
        (entry.data["child_instance_id"], entry.data["status"])
        for entry in nested_store.agent_notifications(pending_only=False)
    ] == [("nested:1", "canceled"), ("nested:2", "completed")]


@pytest.mark.asyncio
async def test_nested_recovery_uses_notification_returned_by_second_store(
    tmp_path: Path,
) -> None:
    root_store = ConversationStore(tmp_path / "root", session_id="root")
    nested_store = ConversationStore(tmp_path / "nested", session_id="nested")
    child_path = nested_store.session_dir / "agents" / "1"
    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        child.mark_agent_parent("call-1")
    nested_store.register_agent_child(
        ToolCall("call-1", "agent", {"prompt": "work"}),
        child_session_path=str(child_path),
        description="child 1",
        background=True,
        child_instance_id="nested:1",
    )

    external_nested = ConversationStore(tmp_path / "nested", session_id="nested")
    original_sync = nested_store.sync_agent_completion_notification

    def sync_after_fallback(child_instance_id: str):
        _append_notification(external_nested, child_instance_id)
        return original_sync(child_instance_id)

    loop = SimpleNamespace(
        store=nested_store,
        _background_owner=SimpleNamespace(notification_store=root_store),
    )
    with patch.object(
        nested_store,
        "sync_agent_completion_notification",
        side_effect=sync_after_fallback,
    ):
        await asyncio.to_thread(recover_agent_children, loop, WRITER_EPOCH)

    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        assert child.agent_canceled() is None
    assert root_store.agent_notifications(pending_only=False)[0].data["status"] == (
        "completed"
    )
    assert nested_store.agent_notifications(pending_only=False)[0].data["status"] == (
        "completed"
    )


def test_recovery_notification_index_keeps_first_duplicate(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="recovery-first")
    child_id = "recovery-first:1"
    child_path = store.session_dir / "agents" / "1"
    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        child.mark_agent_parent("call-1")
    store.register_agent_child(
        ToolCall("call-1", "agent", {"prompt": "work"}),
        child_session_path=str(child_path),
        description="child 1",
        background=True,
        child_instance_id=child_id,
    )
    _append_notification(store, child_id)
    store.append_agent_notification(
        child_id,
        child_session_path=str(child_path),
        description="duplicate canceled notification",
        status="canceled",
        text="canceled",
    )

    recover_agent_children(
        SimpleNamespace(
            store=store,
            _background_owner=SimpleNamespace(notification_store=store),
        ),
        WRITER_EPOCH,
    )

    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        assert child.agent_canceled() is None


def test_recovery_uses_point_notification_queries_for_many_children(
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
    with (
        patch.object(
            store,
            "agent_completion_notifications_by_child",
            wraps=store.agent_completion_notifications_by_child,
        ) as full_scan,
        patch.object(
            store,
            "sync_agent_completion_notification",
            wraps=store.sync_agent_completion_notification,
        ) as point_query,
    ):
        recover_agent_children(loop, WRITER_EPOCH)

    # test-contract: complexity recovery uses point queries, not full scans
    assert full_scan.call_count == 0
    # test-contract: complexity one point query per child keeps work linear
    assert point_query.call_count == 8
    assert store.agent_children() == {}


def _presented_notification_ids(store: ConversationStore) -> set[str]:
    return {
        entry.data["notification_id"]
        for entry in branch_entries(store)
        if entry.type == "notification_tui_presented"
    }


def _notification_replay_app(
    tmp_path: Path, session_id: str
) -> tuple[TUIApp, ConversationStore, list[str]]:
    store = ConversationStore(tmp_path, session_id=session_id)
    notification_ids = [
        _append_notification(store, f"child-{index}") for index in range(2)
    ]
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        resumed=True,
    )
    return app, store, notification_ids


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
    rows = branch_entries(reopened)
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
