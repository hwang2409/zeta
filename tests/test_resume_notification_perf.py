from __future__ import annotations

import asyncio
import os
import threading
from contextlib import contextmanager
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
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
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


def test_recovery_does_not_duplicate_notification_appended_after_indexing(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="recovery-race")
    child_1 = _register_missing_background_child(store, 1)
    child_2 = _register_missing_background_child(store, 2)
    external = ConversationStore(tmp_path, session_id="recovery-race")
    original_append_lock = store._append_lock
    staged = False

    @contextmanager
    def lock_after_external_completion():
        nonlocal staged
        if not staged:
            staged = True
            _append_notification(external, child_2)
        with original_append_lock():
            yield

    loop = SimpleNamespace(
        store=store,
        _background_owner=SimpleNamespace(notification_store=store),
    )
    with patch.object(store, "_append_lock", lock_after_external_completion):
        recover_agent_children(loop)

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


def test_nested_recovery_refreshes_both_notification_stores_per_child(
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
    original_append = root_store.append_agent_notification_if_absent
    staged = False

    def append_after_indexes(*args: object, **kwargs: object):
        nonlocal staged
        if not staged:
            staged = True
            _append_notification(external_nested, "nested:2")
        return original_append(*args, **kwargs)

    loop = SimpleNamespace(
        store=nested_store,
        _background_owner=SimpleNamespace(notification_store=root_store),
    )
    with patch.object(
        root_store,
        "append_agent_notification_if_absent",
        side_effect=append_after_indexes,
    ):
        recover_agent_children(loop)

    with ConversationStore(child_paths[2].parent, session_id="2") as child_2:
        assert child_2.agent_canceled() is None
    assert [
        (entry.data["child_instance_id"], entry.data["status"])
        for entry in root_store.agent_notifications(pending_only=False)
    ] == [("nested:1", "canceled")]
    assert [
        (entry.data["child_instance_id"], entry.data["status"])
        for entry in nested_store.agent_notifications(pending_only=False)
    ] == [("nested:2", "completed"), ("nested:1", "canceled")]


def test_nested_recovery_uses_notification_returned_by_second_store(
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
    original_append = nested_store.append_agent_notification_if_absent
    staged = False

    def append_after_fallback(*args: object, **kwargs: object):
        nonlocal staged
        if not staged:
            staged = True
            _append_notification(external_nested, "nested:1")
        return original_append(*args, **kwargs)

    loop = SimpleNamespace(
        store=nested_store,
        _background_owner=SimpleNamespace(notification_store=root_store),
    )
    with patch.object(
        nested_store,
        "append_agent_notification_if_absent",
        side_effect=append_after_fallback,
    ):
        recover_agent_children(loop)

    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        assert child.agent_canceled() is None
    assert root_store.agent_notifications(pending_only=False)[0].data["status"] == (
        "canceled"
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
        )
    )

    with ConversationStore(child_path.parent, session_id=child_path.name) as child:
        assert child.agent_canceled() is None


def test_completion_notification_query_returns_detached_first_match(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="notification-query")
    child_id = "notification-query:1"
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("hello")],
            metadata={"nested": {"value": "original"}},
        )
    )
    store.append_agent_notification(
        child_id,
        child_session_path="/tmp/child",
        description="original",
        status="completed",
        text="done",
        stats={
            "turns_used": 1,
            "elapsed": 0.1,
            "tool_calls": 0,
            "error": False,
            "canceled": False,
        },
    )
    store.append_agent_notification(
        child_id,
        child_session_path="/tmp/child",
        description="duplicate",
        status="canceled",
        text="canceled",
    )

    first = store.agent_completion_notifications_by_child()
    first[child_id].data["description"] = "mutated"
    first[child_id].data["stats"]["turns_used"] = 99
    first.clear()
    replay_entries = store.tui_replay_entries()
    replay_message = next(entry for entry in replay_entries if entry.type == "message")
    assert replay_message.message is not None
    replay_message.message.metadata["nested"]["value"] = "mutated"
    replay_notification = next(
        entry for entry in replay_entries if entry.type == "notification"
    )
    replay_notification.data["description"] = "mutated again"
    replay_notification.data["stats"]["turns_used"] = 100

    subsequent = store.agent_completion_notifications_by_child()
    assert subsequent[child_id].data["description"] == "original"
    assert subsequent[child_id].data["stats"]["turns_used"] == 1
    assert store.agent_notifications(pending_only=False)[0].data["description"] == "original"
    replayed_message = next(
        entry.message
        for entry in store.tui_replay_entries()
        if entry.type == "message"
    )
    assert replayed_message is not None
    assert replayed_message.metadata["nested"]["value"] == "original"


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
        store,
        "agent_completion_notifications_by_child",
        wraps=store.agent_completion_notifications_by_child,
    ) as notification_index:
        recover_agent_children(loop)

    assert notification_index.call_count == 1
    assert store.agent_children() == {}


def _presented_notification_ids(store: ConversationStore) -> set[str]:
    return {
        entry.data["notification_id"]
        for entry in store.replay()
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
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        resumed=True,
    )
    return app, store, notification_ids


def test_sync_tui_replay_persists_rendered_notifications_before_error(
    tmp_path: Path,
) -> None:
    app, store, notification_ids = _notification_replay_app(tmp_path, "sync-error")
    render_calls = 0

    def fail_second_render(*args: object, **kwargs: object) -> None:
        nonlocal render_calls
        render_calls += 1
        if render_calls == 2:
            raise RuntimeError("render failed")

    with (
        patch.object(app, "_print_unit", side_effect=fail_second_render),
        pytest.raises(RuntimeError, match="render failed"),
    ):
        app._rebuild_transcript()

    assert render_calls == 2
    assert _presented_notification_ids(store) == {notification_ids[0]}


@pytest.mark.asyncio
async def test_async_tui_replay_persists_rendered_notifications_before_error(
    tmp_path: Path,
) -> None:
    app, store, notification_ids = _notification_replay_app(tmp_path, "async-error")
    render_calls = 0

    def fail_second_render(*args: object, **kwargs: object) -> None:
        nonlocal render_calls
        render_calls += 1
        if render_calls == 2:
            raise RuntimeError("render failed")

    with (
        patch.object(app, "_print_unit", side_effect=fail_second_render),
        pytest.raises(RuntimeError, match="render failed"),
    ):
        await app._rebuild_transcript_async(batch_size=2)

    assert render_calls == 2
    assert _presented_notification_ids(store) == {notification_ids[0]}


@pytest.mark.asyncio
async def test_async_tui_replay_persists_rendered_notifications_when_cancelled(
    tmp_path: Path,
) -> None:
    app, store, notification_ids = _notification_replay_app(tmp_path, "async-cancel")
    replay_task: asyncio.Task[bool] | None = None
    render_calls = 0

    def cancel_after_first_render(*args: object, **kwargs: object) -> None:
        nonlocal render_calls
        render_calls += 1
        assert replay_task is not None
        asyncio.get_running_loop().call_soon(replay_task.cancel)

    with patch.object(app, "_print_unit", side_effect=cancel_after_first_render):
        replay_task = asyncio.create_task(app._rebuild_transcript_async(batch_size=1))
        with pytest.raises(asyncio.CancelledError):
            await replay_task

    assert render_calls == 1
    assert _presented_notification_ids(store) == {notification_ids[0]}


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
        store, "tui_replay_entries", wraps=store.tui_replay_entries
    ) as replay_entries:
        assert await app._rebuild_transcript_async(batch_size=3)

    assert replay_entries.call_count == 1
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
