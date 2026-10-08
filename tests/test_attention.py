from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import zeta
from zeta.attention_forks import (
    ATTENTION_FORK_POLICY,
    create_discussion_fork,
    read_attention_fork,
    validate_attention_fork,
)
from zeta.attention_panel import panel_snapshot
from zeta.attention_records import AttentionStore
from zeta.config.tool_policy import ToolPolicy
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
)
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolDefinition, ToolRegistry


def _session(home: Path, *, project_id: str | None = None):
    manager = SessionManager(home)
    return manager.create(
        provider="fake",
        model="fake",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )


def test_request_attention_writes_atomic_record(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    entry = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    registry = ToolRegistry(
        tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
    )
    definition = registry.definitions_by_name["request_attention"]
    assert isinstance(definition, ToolDefinition)
    assert (
        "Use once when a decision only the user can make is pending"
        in definition.description
    )
    assert "then continue other work" in definition.description
    assert "Do not repeat that you are waiting on the user" in definition.description
    assert (
        "assuming the user has read nothing since their last message"
        in definition.description
    )
    identity = Path(zeta.__file__).parent / "prompts" / "identity.md"
    assert "request_attention" not in identity.read_text(encoding="utf-8")

    result = asyncio.run(
        definition.handler(
            {
                "title": "Choose database",
                "why": "The implementation needs one durable database.",
                "options": ["SQLite", "Postgres"],
                "recommendation": "SQLite",
            }
        )
    )

    records = list((opened.store.session_dir / "attention").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert result["content"][0]["text"].startswith("Attention requested:")
    assert record["entry_id"] == entry.id
    assert record["entry_seq"] == entry.seq
    assert record["status"] == "open"
    assert not list(records[0].parent.glob("*.tmp"))
    asyncio.run(registry.close())
    opened.store.close()


def test_request_attention_is_not_available_to_children(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    registry = ToolRegistry(
        tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty()
    )
    child = registry.clone_for_session(
        opened.store, exclude_names={"request_attention"}
    )
    assert "request_attention" in registry.registered_names
    assert "request_attention" not in child.registered_names
    asyncio.run(child.close())
    asyncio.run(registry.close())
    opened.store.close()


def test_panel_list_is_read_only_and_uses_session_lease(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    opened = _session(tmp_path, project_id=project.project_id)
    attention_store = AttentionStore(opened.store.session_dir)
    attention_store.request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Need direction",
        why="Choose a route.",
    )
    opened.store.register_agent_child(
        ToolCall("child-call", "agent", {}),
        child_session_path="agents/1",
        description="Review implementation",
        background=True,
    )
    child = ConversationStore(opened.store.session_dir / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-10-07T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="Review implementation",
    )
    second = _session(tmp_path, project_id=project.project_id)
    resolved_store = AttentionStore(second.store.session_dir)
    resolved = resolved_store.request(
        session_id=second.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Already decided",
        why="Historical item.",
    )
    resolved_store.replace(
        replace(resolved, status="resolved", resolved_at=resolved.created_at)
    )
    (opened.store.session_dir / "background_tasks.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "command": "pytest",
                    "pid": 42,
                    "running": True,
                    "started_at": 1.0,
                    "ended_at": 4.5,
                }
            ]
        )
    )
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    snapshot = panel_snapshot(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    from zeta.cli.main import main as cli_main

    assert cli_main(["panel", "--list"]) == 0
    output = capsys.readouterr().out
    after = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert snapshot.projects[0].name == "alpha"
    assert "alpha" in output
    assert "Review implementation" in output
    assert "Need direction" in output
    assert "Already decided" in output
    sessions = snapshot.projects[0].sessions
    assert any(
        session.tasks[0].label == "pytest" and session.tasks[0].elapsed_seconds == 3.5
        for session in sessions
        if session.tasks
    )
    assert len(sessions) == 2
    assert any(
        session.lanes[0].label == "Review implementation"
        for session in sessions
        if session.lanes
    )
    assert {item.status for session in sessions for item in session.attention} == {
        "open",
        "resolved",
    }
    assert any(
        lane.elapsed_seconds is not None
        for session in sessions
        for lane in session.lanes
        if lane.status == "running"
    )
    assert before == after
    child.close()
    second.store.close()
    opened.store.close()
    assert panel_snapshot(tmp_path).projects == ()


def test_attention_fork_policy_is_action_aware_and_read_only() -> None:
    assert isinstance(ATTENTION_FORK_POLICY, ToolPolicy)
    assert ATTENTION_FORK_POLICY.allows("read")
    assert ATTENTION_FORK_POLICY.allows("resolve_attention")
    assert not ATTENTION_FORK_POLICY.allows("edit")
    assert not ATTENTION_FORK_POLICY.allows("bash")
    assert not ATTENTION_FORK_POLICY.allows("agent")
    assert not ATTENTION_FORK_POLICY.allows("run_background")


def test_fork_copies_active_branch_through_anchor_without_modifying_source(
    tmp_path: Path,
) -> None:
    opened = _session(tmp_path)
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("one")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("two")])
    )
    opened.store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("three")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    attention = AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    original = {
        p.name: p.read_bytes()
        for p in opened.store.session_dir.iterdir()
        if p.is_file()
    }

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)

    assert {
        p.name: p.read_bytes()
        for p in opened.store.session_dir.iterdir()
        if p.is_file()
    } == original
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    assert [
        entry.data["message"]["content"][0]["text"] for entry in fork.store.replay()[:2]
    ] == ["one", "two"]
    assert len(fork.store.replay()) == 3  # harness note follows copied branch
    fork_metadata = read_attention_fork(fork.store.session_dir)
    assert fork_metadata is not None
    assert fork_metadata.forked_from_session == opened.store.session_id
    assert fork_metadata.forked_at_entry == anchor.id
    assert fork_metadata.attention_id == attention.id
    assert fork.metadata.tool_allow == (
        "read",
        "fetch",
        "websearch",
        "recall_history",
        "project",
        "mcp_discover",
        "resolve_attention",
    )
    fork.store.close()
    opened.store.close()


def test_concurrent_fork_creators_reuse_one_usable_session(
    tmp_path: Path, monkeypatch
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    opened = _session(tmp_path, project_id=project.project_id)
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    attention = AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    sessions_before = {path.name for path in manager.sessions_dir.iterdir()}
    original_create = SessionManager.create

    def delayed_create(self, *args, **kwargs):
        time.sleep(0.1)
        return original_create(self, *args, **kwargs)

    monkeypatch.setattr(SessionManager, "create", delayed_create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fork_ids = tuple(
            pool.map(
                lambda _: create_discussion_fork(
                    tmp_path, opened.store.session_id, attention.id
                ),
                range(2),
            )
        )

    assert fork_ids[0] == fork_ids[1]
    assert {path.name for path in manager.sessions_dir.iterdir()} == sessions_before | {
        fork_ids[0]
    }
    fork = manager.open(fork_ids[0], _read_only=True)
    validated = validate_attention_fork(
        home=tmp_path,
        current_session_id=fork_ids[0],
        current_project_id=project.project_id,
        directory_fd=fork.store.directory_fd,
    )
    assert validated.record.fork_session_id == fork_ids[0]
    fork.store.close()
    opened.store.close()


def test_failed_fork_binding_removes_created_session(
    tmp_path: Path, monkeypatch
) -> None:
    opened = _session(tmp_path)
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Choose")])
    )
    attention = AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    manager = SessionManager(tmp_path)
    sessions_before = {path.name for path in manager.sessions_dir.iterdir()}

    def fail_replace(self, record, *, create_directory=False):
        raise OSError("binding publication failed")

    monkeypatch.setattr(AttentionStore, "replace", fail_replace)
    with pytest.raises(OSError, match="binding publication failed"):
        create_discussion_fork(tmp_path, opened.store.session_id, attention.id)

    assert {path.name for path in manager.sessions_dir.iterdir()} == sessions_before
    opened.store.close()
