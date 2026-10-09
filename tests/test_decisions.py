from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zeta.attention_decisions import DecisionItem, open_decisions
from zeta.attention_forks import (
    create_discussion_fork,
    deliver_attention_decision,
    release_discussion_fork,
)
from zeta.attention_records import AttentionStore
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.tui.agent_card import AgentNavigation
from zeta.tui.cards.decisions import DecisionsPanel
from zeta.tui.render import format_status


def _plain(lines: list[list[tuple[str, str]]]) -> str:
    return "\n".join("".join(text for _style, text in line) for line in lines)


def _session(home: Path, project_id: str):
    return SessionManager(home).create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )


def _request(opened, project_id, title, why, **kw):
    return AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=project_id,
        entry_id=kw.get("entry_id"),
        entry_seq=kw.get("entry_seq"),
        title=title,
        why=why,
        options=kw.get("options", ()),
    )


# -- status-bar count -------------------------------------------------------


def test_status_bar_shows_decisions_only_when_present() -> None:
    with_count = format_status("codex", "fake", "idle", decisions_count=2).plain
    assert "● 2 decisions" in with_count
    singular = format_status("codex", "fake", "idle", decisions_count=1).plain
    assert "● 1 decision" in singular
    none = format_status("codex", "fake", "idle", decisions_count=0).plain
    assert "decision" not in none


# -- scan across live sessions ---------------------------------------------


def test_open_decisions_lists_live_open_items_from_two_sessions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    first = _session(tmp_path, project.project_id)
    second = _session(tmp_path, project.project_id)
    _request(first, project.project_id, "Pick DB", "Choose one.", options=["SQLite"])
    resolved = _request(second, project.project_id, "Old", "Historical.")
    AttentionStore(second.store.session_dir).replace(
        replace(resolved, status="resolved", resolved_at=resolved.created_at)
    )
    _request(second, project.project_id, "Pick region", "Where to deploy.")

    items = open_decisions(tmp_path)
    titles = {item.record.title for item in items}
    assert titles == {"Pick DB", "Pick region"}
    assert {item.session_id for item in items} == {
        first.store.session_id,
        second.store.session_id,
    }
    assert all(item.project_name == "alpha" for item in items)

    first.store.close()
    second.store.close()
    # A closed (not live) session no longer contributes a pending decision.
    assert open_decisions(tmp_path) == ()


# -- popup rendering --------------------------------------------------------


def _item(session_id: str, title: str, why: str, options=()) -> DecisionItem:
    from zeta.attention_records import AttentionRecord

    record = AttentionRecord(
        id="a" * 32,
        created_at="2026-01-01T00:00:00Z",
        session_id=session_id,
        project_id="p",
        entry_id=None,
        entry_seq=None,
        lane="orchestrator",
        title=title,
        why=why,
        options=tuple(options),
        recommendation=None,
        status="open",
    )
    return DecisionItem(session_id, "p", "proj", record)


def test_decisions_popup_lists_items_and_offers_options() -> None:
    panel = DecisionsPanel()
    one = _item("1" * 32, "Pick DB", "Choose a durable store.", ["SQLite", "Postgres"])
    two = _item("2" * 32, "Pick region", "Where to deploy the worker.")
    # Distinct ids so selection is stable.
    two = DecisionItem(
        two.session_id, "p", "proj", replace(two.record, id="b" * 32)
    )
    panel.set_items([one, two])
    text = _plain(panel.render_lines())
    assert "Pick DB" in text
    assert "Pick region" in text
    assert "1. SQLite" in text
    assert "2. Postgres" in text
    assert panel.pick_option(1) == "SQLite"
    assert panel.pick_option(9) is None


def test_decisions_popup_answer_mode_captures_text() -> None:
    panel = DecisionsPanel()
    panel.set_items([_item("1" * 32, "Pick DB", "Choose.")])
    assert panel.enter_answer() is True
    assert panel.mode == "answer"
    panel.set_answer("use sqlite")
    assert "use sqlite" in _plain(panel.render_lines())
    panel.exit_answer()
    assert panel.mode == "list"


# -- quick answer delivery --------------------------------------------------


def test_quick_answer_delivers_to_asking_session_and_closes(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    asker = _session(tmp_path, project.project_id)
    answerer = _session(tmp_path, project.project_id)
    record = _request(asker, project.project_id, "Pick DB", "Choose one.")

    message_id, already = deliver_attention_decision(
        tmp_path, record, "Use SQLite.", from_session=answerer.store.session_id
    )
    assert already is False

    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    new = inbox.new_ids(project.project_id, session_id=asker.store.session_id)
    assert new == (message_id,)
    message = inbox.claim(project.project_id, message_id, asker.store.session_id)
    assert "Use SQLite." in message["body"]
    assert AttentionStore(asker.store.session_dir).get(record.id).status == "resolved"

    # A repeat is idempotent and reports the prior delivery.
    _, again = deliver_attention_decision(
        tmp_path, record, "Use SQLite.", from_session=answerer.store.session_id
    )
    assert again is True

    asker.store.close()
    answerer.store.close()


# -- discussion fork in the agent view -------------------------------------


def test_fork_context_adds_main_and_relabels_root(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    asker = _session(tmp_path, project.project_id)
    anchor = asker.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = _request(
        asker,
        project.project_id,
        "Pick DB",
        "Choose one.",
        entry_id=anchor.id,
        entry_seq=anchor.seq,
    )
    fork_id = create_discussion_fork(tmp_path, asker.store.session_id, record.id)
    fork = manager.open(fork_id)
    try:
        nav = AgentNavigation(fork.store)
        returned: list[bool] = []
        nav.set_fork_context(
            title=record.title,
            main_path=asker.store.session_dir,
            on_return=lambda: returned.append(True),
        )
        labels = [entry.label for entry in nav.entries]
        assert f"discussion: {record.title}" in labels
        assert "(main)" in labels

        # Selecting (main) returns to the orchestrator; esc-style exit does not.
        main_index = labels.index("(main)")
        nav.selected_index = main_index
        nav.open_selected()
        assert returned == [True]

        nav.exit_navigation()
        assert returned == [True]  # exit/esc never triggers a return
    finally:
        fork.store.close()
        asker.store.close()


def test_return_without_decision_unbinds_so_item_can_be_rediscussed(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    asker = _session(tmp_path, project.project_id)
    anchor = asker.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = _request(
        asker,
        project.project_id,
        "Pick DB",
        "Choose one.",
        entry_id=anchor.id,
        entry_seq=anchor.seq,
    )
    fork_id = create_discussion_fork(tmp_path, asker.store.session_id, record.id)
    assert AttentionStore(asker.store.session_dir).get(record.id).fork_session_id == fork_id

    source = release_discussion_fork(tmp_path, fork_id)
    assert source == asker.store.session_id
    # Open item: the binding is cleared so a fresh fork starts next time.
    assert AttentionStore(asker.store.session_dir).get(record.id).fork_session_id is None
    new_fork = create_discussion_fork(tmp_path, asker.store.session_id, record.id)
    assert new_fork != fork_id
    asker.store.close()


def test_return_after_decision_keeps_resolved_binding(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    asker = _session(tmp_path, project.project_id)
    anchor = asker.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = _request(
        asker,
        project.project_id,
        "Pick DB",
        "Choose one.",
        entry_id=anchor.id,
        entry_seq=anchor.seq,
    )
    fork_id = create_discussion_fork(tmp_path, asker.store.session_id, record.id)
    deliver_attention_decision(
        tmp_path,
        AttentionStore(asker.store.session_dir).get(record.id),
        "Use SQLite.",
        from_session=fork_id,
    )
    source = release_discussion_fork(tmp_path, fork_id)
    assert source == asker.store.session_id
    resolved = AttentionStore(asker.store.session_dir).get(record.id)
    assert resolved.status == "resolved"
    assert resolved.fork_session_id == fork_id  # binding kept for a closed item
    asker.store.close()


# -- removal ----------------------------------------------------------------


def test_zeta_panel_command_is_removed(monkeypatch, tmp_path: Path) -> None:
    from zeta.cli.main import main as cli_main

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli_main(["panel"])
    assert exc.value.code != 0

    import zeta

    assert not (Path(zeta.__file__).parent / "attention_panel.py").exists()
    assert not (Path(zeta.__file__).parent / "cli" / "panel.py").exists()
