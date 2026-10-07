from __future__ import annotations

import asyncio
import time
from pathlib import Path

from zeta.attention import AttentionStore, create_discussion_fork
from zeta.cli.panel import PanelApplication
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.tools.resolve_attention import register_fork


def _project_session(home: Path, project_id: str):
    return SessionManager(home).create(
        provider="fake",
        model="fake",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )


def test_resolve_attention_targets_only_original_and_resolves_record(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    other = _project_session(tmp_path, project.project_id)
    anchor = original.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Pick one",
        why="Only the user can choose.",
    )
    fork_id = create_discussion_fork(tmp_path, original.store.session_id, record.id)
    assert (
        create_discussion_fork(tmp_path, original.store.session_id, record.id)
        == fork_id
    )
    fork = manager.open(fork_id)
    registry = ToolRegistry(
        tmp_path,
        session_store=fork.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
        tool_allow=fork.metadata.tool_allow,
    )
    register_fork(registry)
    assert "read" in registry.registered_names
    assert "resolve_attention" in registry.registered_names
    assert {"edit", "bash", "agent", "run_background"}.isdisjoint(
        registry.registered_names
    )

    result = asyncio.run(
        registry.definitions_by_name["resolve_attention"].handler(
            {"decision": "Use SQLite."}
        )
    )

    assert result["isError"] is False
    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    assert (
        len(inbox.new_ids(project.project_id, session_id=original.store.session_id))
        == 1
    )
    assert inbox.new_ids(project.project_id, session_id=other.store.session_id) == ()
    message_id = inbox.new_ids(
        project.project_id, session_id=original.store.session_id
    )[0]
    assert inbox.claim(project.project_id, message_id, other.store.session_id) is None
    message = inbox.claim(project.project_id, message_id, original.store.session_id)
    assert message is not None
    assert fork_id in message["body"]
    assert record.created_at in message["body"]
    resolved = AttentionStore(original.store.session_dir).get(record.id)
    assert resolved.status == "resolved"
    assert resolved.fork_session_id == fork_id
    assert resolved.decision == "Use SQLite."
    assert (
        create_discussion_fork(tmp_path, original.store.session_id, record.id)
        == fork_id
    )
    asyncio.run(registry.close())
    fork.store.close()
    other.store.close()
    original.store.close()


def test_panel_refresh_runs_storage_scan_off_event_loop(
    tmp_path: Path, monkeypatch
) -> None:
    def slow_snapshot(_home: Path):
        time.sleep(0.15)
        from zeta.attention import PanelSnapshot

        return PanelSnapshot(())

    monkeypatch.setattr("zeta.cli.panel.panel_snapshot", slow_snapshot)

    async def exercise() -> int:
        panel = PanelApplication(tmp_path)
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(8):
                await asyncio.sleep(0.02)
                ticks += 1

        await asyncio.gather(panel.refresh(), ticker())
        return ticks

    assert asyncio.run(exercise()) >= 6
