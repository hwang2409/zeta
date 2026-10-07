from __future__ import annotations

import asyncio
import json
from pathlib import Path

from zeta.attention import AttentionStore, create_discussion_fork, panel_snapshot
from zeta.core.session import SessionManager
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry


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
    registry = ToolRegistry(tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty())
    definition = registry.definitions_by_name["request_attention"]

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
    registry = ToolRegistry(tmp_path, session_store=opened.store, skill_catalog=SkillCatalog.empty())
    child = registry.clone_for_session(opened.store, exclude_names={"request_attention"})
    assert "request_attention" in registry.registered_names
    assert "request_attention" not in child.registered_names
    asyncio.run(child.close())
    asyncio.run(registry.close())
    opened.store.close()


def test_panel_snapshot_is_read_only_and_uses_session_lease(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "Alpha")
    opened = _session(tmp_path, project_id=project.project_id)
    AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="Need direction",
        why="Choose a route.",
    )
    (opened.store.session_dir / "background_tasks.json").write_text(
        json.dumps([{"task_id": "task-1", "command": "pytest", "pid": 42, "running": True}])
    )
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    snapshot = panel_snapshot(tmp_path)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert snapshot.projects[0].name == "alpha"
    assert snapshot.projects[0].sessions[0].tasks[0].label == "pytest"
    assert snapshot.projects[0].sessions[0].attention[0].title == "Need direction"
    assert before == after
    opened.store.close()
    assert panel_snapshot(tmp_path).projects == ()


def test_fork_copies_active_branch_through_anchor_without_modifying_source(tmp_path: Path) -> None:
    opened = _session(tmp_path)
    opened.store.append_message(Message(MessageRole.USER, [TextContent("one")]))
    anchor = opened.store.append_message(Message(MessageRole.ASSISTANT, [TextContent("two")]))
    opened.store.append_message(Message(MessageRole.USER, [TextContent("three")]))
    attention = AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=None,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="Question",
        why="Need a choice.",
    )
    original = {p.name: p.read_bytes() for p in opened.store.session_dir.iterdir() if p.is_file()}

    fork_id = create_discussion_fork(tmp_path, opened.store.session_id, attention.id)

    assert {p.name: p.read_bytes() for p in opened.store.session_dir.iterdir() if p.is_file()} == original
    fork = SessionManager(tmp_path).open(fork_id, _read_only=True)
    assert [entry.data["message"]["content"][0]["text"] for entry in fork.store.replay()[:2]] == ["one", "two"]
    assert len(fork.store.replay()) == 3  # harness note follows copied branch
    assert fork.metadata.forked_from_session == opened.store.session_id
    assert fork.metadata.forked_at_entry == anchor.id
    assert fork.metadata.attention_id == attention.id
    assert fork.metadata.tool_allow == (
        "read", "fetch", "websearch", "recall_history", "project", "mcp_discover", "resolve_attention"
    )
    fork.store.close()
    opened.store.close()
