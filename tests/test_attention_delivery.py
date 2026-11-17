from __future__ import annotations

import asyncio
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from zeta.attention_forks import create_discussion_fork
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.tools.resolve_attention import register_fork


def _project_session(home: Path, project_id: str, *, parent_session_id: str | None = None):
    return SessionManager(home).create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project_id,
        parent_session_id=parent_session_id,
        auto_project=False,
    )


def _hold_session_open(home: str, session_id: str, ready, release) -> None:
    opened = SessionManager(Path(home)).open(session_id)
    try:
        ready.set()
        if not release.wait(10):
            raise AssertionError("timed out waiting to release the session owner")
    finally:
        opened.store.close()


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
    record = AttentionStore(original.store.session_dir, writer_epoch=5).request(
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
    message = inbox.list(project.project_id)["new"][0]
    assert message["origin"] == "agent"
    assert message["authority"] == "information"
    assert "Decision: Use SQLite." in message["body"]
    resolved = AttentionStore(original.store.session_dir, writer_epoch=5).get(record.id)
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
    record = AttentionStore(original.store.session_dir, writer_epoch=5).request(
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

    assert AttentionStore(original.store.session_dir, writer_epoch=5).get(record.id).status == "open"
    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    assert inbox.new_ids(project.project_id, session_id=original.store.session_id) == ()
    asyncio.run(registry.close())
    forged.store.close()
    original.store.close()


def test_orchestrator_can_resolve_own_attention_without_inbox(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    record = AttentionStore(original.store.session_dir, writer_epoch=5).request(
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
    resolved = AttentionStore(original.store.session_dir, writer_epoch=5).get(record.id)
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
    record = AttentionStore(other.store.session_dir, writer_epoch=5).request(
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
    own_record = AttentionStore(original.store.session_dir, writer_epoch=5).request(
        session_id=original.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Pick two",
        why="Only the user can choose.",
    )
    handler = registry.definitions_by_name["resolve_attention"].handler
    with pytest.raises(ValueError, match="live session"):
        asyncio.run(handler({"attention_id": record.id, "decision": "no"}))
    with pytest.raises(ValueError, match="dead predecessor"):
        asyncio.run(handler({"attention_id": "0" * 32, "decision": "no"}))
    with pytest.raises(ValueError, match="nonempty"):
        asyncio.run(handler({"attention_id": own_record.id, "decision": "   "}))
    assert "resolve_attention" in registry.registered_names
    asyncio.run(registry.close())
    other.store.close()
    original.store.close()


def test_list_and_resolve_attention_recover_dead_predecessor(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    current = _project_session(tmp_path, project.project_id)
    predecessor = _project_session(tmp_path, project.project_id)
    record = AttentionStore(predecessor.store.session_dir, writer_epoch=5).request(
        session_id=predecessor.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Recover this choice",
        why="The predecessor stopped before the user answered.",
    )
    predecessor.store.close()
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    listed = asyncio.run(
        registry.definitions_by_name["list_attention"].handler({})
    )
    item = listed["structuredContent"]["items"][0]
    assert item["id"] == record.id
    assert item["kind"] == "decision"
    assert item["title"] == record.title
    assert item["owning_session"] == record.session_id
    resolved = asyncio.run(
        registry.definitions_by_name["resolve_attention"].handler(
            {"attention_id": record.id, "decision": "Use the fallback."}
        )
    )
    assert resolved["isError"] is False
    assert AttentionStore(
        current.store.session_dir.parent / record.session_id, writer_epoch=5
    ).get(record.id).status == "resolved"
    asyncio.run(registry.close())
    current.store.close()


def test_list_attention_pages_and_keeps_broken_timestamps_visible(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    current = _project_session(tmp_path, project.project_id)
    predecessor = _project_session(tmp_path, project.project_id)
    attention = AttentionStore(predecessor.store.session_dir, writer_epoch=5)
    records = [
        attention.request(
            session_id=predecessor.store.session_id,
            project_id=project.project_id,
            entry_id=None,
            entry_seq=None,
            title=f"Choice {index:03d}",
            why="The predecessor stopped.",
        )
        for index in range(101)
    ]
    for record, created_at in zip(records[:3], (None, "not-a-time", "2026-01-01T00:00:00"), strict=True):
        path = attention.directory / f"{record.id}.json"
        value = json.loads(path.read_text())
        value["created_at"] = created_at
        path.write_text(json.dumps(value))
    predecessor.store.close()
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )

    first = asyncio.run(registry.definitions_by_name["list_attention"].handler({}))
    second = asyncio.run(
        registry.definitions_by_name["list_attention"].handler({"offset": 100})
    )
    first_items = first["structuredContent"]["items"]
    second_items = second["structuredContent"]["items"]
    # mutation: omit Broken rows from _list_attention_sync(); the visibility assertion fails.
    assert all(item["broken"] is True for item in first_items[:3])
    # mutation: change the list page size from 100 to 101; the paging assertions fail.
    assert first["structuredContent"]["total"] == 101
    assert first["structuredContent"]["next_offset"] == 100
    assert len(first_items) == 100 and len(second_items) == 1
    assert [item["id"] for item in first_items[:3]] == sorted(
        record.id for record in records[:3]
    )
    assert "next_offset" not in second["structuredContent"]
    asyncio.run(registry.close())
    current.store.close()


def test_list_attention_omits_nonopen_and_out_of_scope_owners(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    other_project = manager.project_registry.create_project("beta", "Beta")
    current = _project_session(tmp_path, project.project_id)
    predecessor = _project_session(tmp_path, project.project_id)
    child = _project_session(
        tmp_path, project.project_id, parent_session_id=predecessor.store.session_id
    )
    other = _project_session(tmp_path, other_project.project_id)
    predecessor_store = AttentionStore(predecessor.store.session_dir, writer_epoch=5)
    visible = predecessor_store.request(
        session_id=predecessor.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Visible",
        why="Still open.",
    )
    answered = predecessor_store.request(
        session_id=predecessor.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Answered",
        why="Already done.",
    )
    dismissed = predecessor_store.request(
        session_id=predecessor.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Dismissed",
        why="Already dismissed.",
    )
    predecessor_store.replace(replace(answered, status="resolved", decision="yes"))
    predecessor_store.replace(replace(dismissed, status="resolved", decision=None))
    child_record = AttentionStore(child.store.session_dir, writer_epoch=5).request(
        session_id=child.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Child",
        why="Child-owned.",
    )
    other_record = AttentionStore(other.store.session_dir, writer_epoch=5).request(
        session_id=other.store.session_id,
        project_id=other_project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Other project",
        why="Out of scope.",
    )
    child.store.close()
    other.store.close()
    predecessor.store.close()
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )

    listed = asyncio.run(registry.definitions_by_name["list_attention"].handler({}))
    ids = {item["id"] for item in listed["structuredContent"]["items"]}
    assert ids == {visible.id}
    handler = registry.definitions_by_name["resolve_attention"].handler
    for hidden in (child_record, other_record):
        with pytest.raises(ValueError, match="dead predecessor"):
            asyncio.run(
                handler({"attention_id": hidden.id, "decision": "not allowed"})
            )
    asyncio.run(registry.close())
    current.store.close()


def test_resolve_attention_rejects_ambiguous_predecessor_id(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    current = _project_session(tmp_path, project.project_id)
    first = _project_session(tmp_path, project.project_id)
    second = _project_session(tmp_path, project.project_id)
    record = AttentionStore(first.store.session_dir, writer_epoch=5).request(
        session_id=first.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Duplicate",
        why="Copied after a crash.",
    )
    AttentionStore(second.store.session_dir, writer_epoch=5).replace(
        replace(record, session_id=second.store.session_id), create_directory=True
    )
    first.store.close()
    second.store.close()
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )

    # mutation: bypass the len(matches) ambiguity refusal; this assertion fails after a write.
    with pytest.raises(ValueError, match="ambiguous") as error:
        asyncio.run(
            registry.definitions_by_name["resolve_attention"].handler(
                {"attention_id": record.id, "decision": "no"}
            )
        )
    assert first.store.session_id in str(error.value)
    assert second.store.session_id in str(error.value)
    asyncio.run(registry.close())
    current.store.close()


def test_resolve_attention_rejects_current_and_predecessor_duplicate_id(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    current = _project_session(tmp_path, project.project_id)
    predecessor = _project_session(tmp_path, project.project_id)
    current_store = AttentionStore(current.store.session_dir, writer_epoch=5)
    record = current_store.request(
        session_id=current.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Current copy",
        why="This id was copied into a predecessor.",
    )
    AttentionStore(predecessor.store.session_dir, writer_epoch=5).replace(
        replace(record, session_id=predecessor.store.session_id), create_directory=True
    )
    predecessor.store.close()
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )

    # mutation: return after resolving own before _foreign_matches(); this assertion fails after a write.
    with pytest.raises(ValueError, match="ambiguous") as error:
        asyncio.run(
            registry.definitions_by_name["resolve_attention"].handler(
                {"attention_id": record.id, "decision": "must not write"}
            )
        )
    assert current.store.session_id in str(error.value)
    assert predecessor.store.session_id in str(error.value)
    assert current_store.get(record.id).status == "open"
    assert AttentionStore(
        tmp_path / "sessions" / predecessor.store.session_id, writer_epoch=5
    ).get(record.id).status == "open"
    asyncio.run(registry.close())
    current.store.close()


def test_resolve_attention_refuses_owner_that_starts_after_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    current = _project_session(tmp_path, project.project_id)
    predecessor = _project_session(tmp_path, project.project_id)
    record = AttentionStore(predecessor.store.session_dir, writer_epoch=5).request(
        session_id=predecessor.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Race",
        why="The old owner may restart.",
    )
    owner_id = predecessor.store.session_id
    predecessor.store.close()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_session_open,
        args=(str(tmp_path), owner_id, ready, release),
    )
    from zeta.tools import resolve_attention as resolve_module

    real_matches = resolve_module._foreign_matches

    def start_owner_after_lookup(registry, attention_id):
        matches = real_matches(registry, attention_id)
        process.start()
        assert ready.wait(10), "timed out waiting for the predecessor to restart"
        return matches

    # test-contract: seam force the predecessor startup after exact-id lookup
    monkeypatch.setattr(resolve_module, "_foreign_matches", start_owner_after_lookup)
    registry = ToolRegistry(
        tmp_path,
        session_store=current.store,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=manager.project_registry,
    )
    try:
        # mutation: remove SessionExecutionLease.acquire(); this assertion fails without the live-owner refusal.
        with pytest.raises(ValueError, match="live session"):
            asyncio.run(
                registry.definitions_by_name["resolve_attention"].handler(
                    {"attention_id": record.id, "decision": "must not write"}
                )
            )
        assert AttentionStore(
            tmp_path / "sessions" / owner_id, writer_epoch=5
        ).get(record.id).status == "open"
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        asyncio.run(registry.close())
        current.store.close()
    assert process.exitcode == 0


def test_popup_and_orchestrator_resolution_conflict_without_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    original = _project_session(tmp_path, project.project_id)
    answerer = _project_session(tmp_path, project.project_id)
    record = AttentionStore(original.store.session_dir, writer_epoch=5).request(
        session_id=original.store.session_id, project_id=project.project_id,
        entry_id=None, entry_seq=None, title="Pick one", why="Only the user can choose.",
    )
    registry = ToolRegistry(tmp_path, session_store=original.store,
        skill_catalog=SkillCatalog.empty(), project_id=project.project_id,
        project_registry=manager.project_registry)
    replies = AttentionReplies(tmp_path, original.store.session_id,
        registry_state=registry.project_registry_state)
    replies.stage(record.id, "popup answer")
    with pytest.raises(ValueError, match="TUI decisions popup"):
        asyncio.run(registry.definitions_by_name["resolve_attention"].handler(
            {"attention_id": record.id, "decision": "chat answer"}))
    replies.flush(answerer.store.session_id)
    resolved = AttentionStore(original.store.session_dir, writer_epoch=5).get(record.id)
    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    message_id = inbox.new_ids(project.project_id, session_id=original.store.session_id)[0]
    assert resolved.decision == "popup answer"
    message = inbox.claim(project.project_id, message_id, original.store.session_id)
    assert message is not None and "Decision: popup answer" in message["body"]
    asyncio.run(registry.close())
    answerer.store.close()
    original.store.close()
