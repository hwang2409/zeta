from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zeta.cli.main import main
from zeta.core.checkpoints import ConversationEntry, ConversationIntegrityError
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.loop.persistence import DraftPersistence
from zeta.storage_format import StorageFormatError
from zeta.tools._shared.process import BackgroundTaskRegistry


def _inject(path: Path) -> None:
    value = json.loads(path.read_text())
    value["future"] = {"nested": {"value": 7}}
    path.write_text(json.dumps(value))


def _assert_future(path: Path) -> None:
    assert json.loads(path.read_text())["future"] == {"nested": {"value": 7}}


def test_every_session_mapping_writer_preserves_unknown_nested_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    session_dir = store.session_dir

    metadata_path = session_dir / "meta.json"
    state_path = session_dir / "session_state.json"
    _inject(metadata_path)
    _inject(state_path)
    manager.touch(opened.metadata)
    store.set_bash_cwd(tmp_path / "next")
    _assert_future(metadata_path)
    _assert_future(state_path)

    store.start_agent_lifecycle(
        handle="agent-1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="test",
    )
    lifecycle_path = session_dir / "agent_lifecycle.json"
    _inject(lifecycle_path)
    store.finish_agent_lifecycle("completed", final_result="done")
    _assert_future(lifecycle_path)
    store.close()


def test_keyed_agent_child_delete_survives_reload_and_keeps_unknown_data(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    for child_id in ("child-1", "child-2"):
        store.register_agent_child(
            ToolCall(child_id, "agent", {"description": child_id}),
            child_session_path=str(store.session_dir / "agents" / child_id),
            description=child_id,
            child_instance_id=child_id,
        )

    state_path = store.state_path
    state = json.loads(state_path.read_text())
    state["agent_children"]["child-2"]["future"] = {"nested": True}
    future_child = dict(state["agent_children"]["child-2"])
    future_child["future_only"] = True
    state["agent_children"]["future-child"] = future_child
    state_path.write_text(json.dumps(state))

    store.finish_agent_child("child-1")
    store.close()
    reopened = ConversationStore(
        manager.sessions_dir, session_id=opened.metadata.session_id
    )
    persisted = json.loads(state_path.read_text())["agent_children"]
    assert set(reopened.agent_children()) == {"child-2", "future-child"}
    assert "child-1" not in persisted
    assert persisted["child-2"]["future"] == {"nested": True}
    assert persisted["future-child"]["future_only"] is True
    reopened.close()


@pytest.mark.asyncio
async def test_remaining_session_publishers_preserve_unknown_nested_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = await asyncio.to_thread(
        manager.create, provider="codex", model="offline", cwd=tmp_path
    )
    store = opened.store

    workspace = WorkspaceSnapshotStore(
        store.session_dir,
        store.session_id,
        storage_lease=store._storage_lease,
    )
    workspace.state_path.write_text(json.dumps({}))
    _inject(workspace.state_path)
    workspace._persist()
    _assert_future(workspace.state_path)
    workspace.close()

    background_path = store.session_dir / "background_tasks.json"
    background_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "command": "true",
                    "pid": 1,
                    "running": False,
                    "future": {"nested": True},
                },
                {
                    "task_id": "future-task",
                    "future_shape": {"version": 2},
                },
            ]
        )
    )
    background = BackgroundTaskRegistry(
        session_dir=store.session_dir,
        directory_fd=store.directory_fd,
        storage_lease=store._storage_lease,
    )
    background._persist()
    background_rows = {
        row["task_id"]: row for row in json.loads(background_path.read_text())
    }
    assert background_rows["task-1"]["future"] == {"nested": True}
    assert "future-task" in background_rows
    assert background_rows["future-task"] == {
        "task_id": "future-task",
        "future_shape": {"version": 2},
    }
    await background.close()

    from zeta.core.session_links import persist_pending_root_link

    persist_pending_root_link(
        manager.sessions_dir,
        store.session_id,
        "project-1",
        "session",
        None,
        store._storage_lease,
    )
    link_path = store.session_dir / "project_link_pending.json"
    _inject(link_path)
    persist_pending_root_link(
        manager.sessions_dir,
        store.session_id,
        "project-1",
        "session",
        None,
        store._storage_lease,
    )
    _assert_future(link_path)

    draft_path = store.session_dir / "draft"
    draft = DraftPersistence(
        draft_path,
        delay=0,
        directory_fd=store.directory_fd,
        storage_lease=store._storage_lease,
    )
    draft.schedule("first")
    await draft.flush_pending()
    _inject(draft_path)
    draft.schedule("second")
    await draft.flush_pending()
    _assert_future(draft_path)
    store.close()


def test_session_manager_refuses_too_new_home_without_changing_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    metadata = tmp_path / "sessions" / "session-1" / "meta.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b'{"future":"metadata"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        SessionManager(tmp_path)

    assert metadata.read_bytes() == b'{"future":"metadata"}\n'


def test_conversation_store_refuses_too_new_home_without_changing_session_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    state = tmp_path / "sessions" / "session-1" / "session_state.json"
    state.parent.mkdir(parents=True)
    state.write_bytes(b'{"bash_cwd":"/future","future":"session state"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        ConversationStore(tmp_path / "sessions", session_id="session-1")

    assert state.read_bytes() == b'{"bash_cwd":"/future","future":"session state"}\n'


def test_workspace_store_refuses_too_new_home_without_changing_snapshots(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    session_dir = tmp_path / "sessions" / "session-1"
    snapshots = session_dir / "workspace_snapshots.json"
    snapshots.parent.mkdir(parents=True)
    snapshots.write_bytes(b'{"future":"snapshots"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        WorkspaceSnapshotStore(session_dir, "session-1")

    assert snapshots.read_bytes() == b'{"future":"snapshots"}\n'


def test_too_new_refusal_is_visible_in_tui_and_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":999}\n')
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    assert main(["--provider", "codex", "--model", "offline"]) == 1
    assert "newer than writer capability" in capsys.readouterr().err
    assert main(["serve", "--provider", "codex", "--model", "offline"]) == 2
    assert "newer than writer capability" in capsys.readouterr().err


def test_unknown_conversation_variant_fails_closed() -> None:
    with pytest.raises(
        ConversationIntegrityError, match="unsupported conversation entry type"
    ):
        ConversationEntry.from_dict(
            {
                "seq": 1,
                "id": "entry-1",
                "parent_id": None,
                "lane": "main",
                "type": "future_variant",
                "data": {},
            }
        )
