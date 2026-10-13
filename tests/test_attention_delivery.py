from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from zeta.attention_forks import create_discussion_fork, deliver_attention_decision
from zeta.attention_records import AttentionStore
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


def test_orchestrator_can_resolve_own_attention_without_inbox(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Pick one",
        why="Only the user can choose.",
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=original.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    definition = registry.definitions_by_name["resolve_attention"]
    assert "answered the decision in chat" in definition.description
    result = asyncio.run(
        definition.handler({"attention_id": record.id, "decision": "Use SQLite."})
    )
    repeated = asyncio.run(
        definition.handler({"attention_id": record.id, "decision": "Use SQLite."})
    )
    resolved = AttentionStore(original.store.session_dir).get(record.id)
    assert result["structuredContent"]["already_resolved"] is False
    assert repeated["structuredContent"]["already_resolved"] is True
    assert resolved.status == "resolved"
    assert resolved.decision == "Use SQLite."
    assert ProjectInbox(
        manager.project_registry, sessions_root=tmp_path / "sessions"
    ).new_ids(project.project_id, session_id=original.store.session_id) == ()
    asyncio.run(registry.close())
    original.store.close()


def test_orchestrator_resolve_attention_rejects_unknown_and_foreign_items(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    other = _project_session(tmp_path, project.project_id)
    record = AttentionStore(other.store.session_dir).request(
        session_id=other.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Pick one",
        why="Only the user can choose.",
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=original.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    own_record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Pick two",
        why="Only the user can choose.",
    )
    handler = registry.definitions_by_name["resolve_attention"].handler
    with pytest.raises(FileNotFoundError):
        asyncio.run(handler({"attention_id": record.id, "decision": "no"}))
    with pytest.raises(FileNotFoundError):
        asyncio.run(handler({"attention_id": "0" * 32, "decision": "no"}))
    with pytest.raises(ValueError, match="nonempty"):
        asyncio.run(handler({"attention_id": own_record.id, "decision": "   "}))
    assert "resolve_attention" in registry.registered_names
    asyncio.run(registry.close())
    other.store.close()
    original.store.close()


def test_popup_and_orchestrator_resolution_are_first_writer_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    answerer = _project_session(tmp_path, project.project_id)
    record = AttentionStore(original.store.session_dir).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Pick one",
        why="Only the user can choose.",
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=original.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    delivery_blocked = threading.Event()
    release_delivery = threading.Event()
    tool_started = threading.Event()
    original_send = ProjectInbox.send
    original_resolve = AttentionStore.resolve

    def gated_send(self, *args, **kwargs):
        delivery_blocked.set()
        assert release_delivery.wait(2)
        return original_send(self, *args, **kwargs)

    def watched_resolve(self, attention_id, *, session_id, decision, **kwargs):
        if decision == "chat answer":
            tool_started.set()
        return original_resolve(
            self,
            attention_id,
            session_id=session_id,
            decision=decision,
            **kwargs,
        )

    monkeypatch.setattr(ProjectInbox, "send", gated_send)
    monkeypatch.setattr(AttentionStore, "resolve", watched_resolve)

    async def race():
        popup = asyncio.create_task(
            asyncio.to_thread(
                deliver_attention_decision,
                tmp_path,
                record,
                "popup answer",
                from_session=answerer.store.session_id,
            )
        )
        assert await asyncio.to_thread(delivery_blocked.wait, 2)
        tool = asyncio.create_task(
            registry.definitions_by_name["resolve_attention"].handler(
                {"attention_id": record.id, "decision": "chat answer"}
            )
        )
        assert await asyncio.to_thread(tool_started.wait, 2)
        release_delivery.set()
        return await asyncio.gather(popup, tool)

    popup_result, tool_result = asyncio.run(race())
    message_id = popup_result[0]
    resolved = AttentionStore(original.store.session_dir).get(record.id)
    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")

    assert popup_result[-1] is False
    assert tool_result["structuredContent"] == {
        "attention_id": record.id,
        "already_resolved": True,
        "decision": "popup answer",
    }
    assert resolved.decision == "popup answer"
    assert inbox.new_ids(
        project.project_id, session_id=original.store.session_id
    ) == (message_id,)
    message = inbox.claim(project.project_id, message_id, original.store.session_id)
    assert message is not None
    assert "Decision: popup answer" in message["body"]

    asyncio.run(registry.close())
    answerer.store.close()
    original.store.close()
