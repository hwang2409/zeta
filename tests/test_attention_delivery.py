from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from zeta.attention_forks import create_discussion_fork
from zeta.attention_records import AttentionStore
from zeta.cli.panel import PanelApplication
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.tools.resolve_attention import register_fork


def _project_session(home: Path, project_id: str):
    return SessionManager(home).create(
        provider="codex",
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
    repeated = asyncio.run(
        registry.definitions_by_name["resolve_attention"].handler(
            {"decision": "Use SQLite."}
        )
    )

    assert result["isError"] is False
    assert repeated["isError"] is False
    assert repeated["structuredContent"]["already_resolved"] is True
    assert (
        repeated["structuredContent"]["message_id"]
        == result["structuredContent"]["message_id"]
    )
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
        from zeta.attention_panel import PanelSnapshot

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


def test_forged_fork_cannot_resolve_unbound_attention(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
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
    forged = _project_session(tmp_path, project.project_id)
    (forged.store.session_dir / "attention_fork.json").write_text(
        json.dumps(
            {
                "forked_from_session": original.store.session_id,
                "forked_at_entry": anchor.id,
                "attention_id": record.id,
            }
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=forged.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    register_fork(registry)

    with pytest.raises(ValueError, match="not bound"):
        asyncio.run(
            registry.definitions_by_name["resolve_attention"].handler(
                {"decision": "Forged decision."}
            )
        )

    assert AttentionStore(original.store.session_dir).get(record.id).status == "open"
    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    assert inbox.new_ids(project.project_id, session_id=original.store.session_id) == ()
    asyncio.run(registry.close())
    forged.store.close()
    original.store.close()


def test_panel_keeps_dead_target_visible_while_decision_delivery_is_pending(
    tmp_path: Path,
) -> None:
    from zeta.attention_panel import panel_snapshot
    from zeta.cli.panel import format_snapshot

    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
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
    result = asyncio.run(
        registry.definitions_by_name["resolve_attention"].handler(
            {"decision": "Use SQLite."}
        )
    )
    assert result["isError"] is False
    asyncio.run(registry.close())
    fork.store.close()
    original.store.close()

    snapshot = panel_snapshot(tmp_path)
    sessions = snapshot.projects[0].sessions
    source = next(
        session
        for session in sessions
        if session.session_id == original.store.session_id
    )
    assert source.live is False
    assert len(source.attention) == 1
    assert source.attention[0].status == "delivery pending"
    output = format_snapshot(snapshot)
    assert "inactive" in output
    assert "delivery pending" in output
