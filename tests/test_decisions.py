from __future__ import annotations

from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.attention_decisions import DecisionItem, open_decisions
from zeta.attention_forks import (
    create_discussion_fork,
    deliver_attention_decision,
    release_discussion_fork,
)
from zeta.attention_records import AttentionStore
from zeta.core.loop import AgentLoop
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.tools.resolve_attention import register_fork
from zeta.tui.agent_card import AgentNavigation
from zeta.tui.app import TUIApp
from zeta.tui.cards.decisions import DecisionsPanel
from zeta.tui.overlay import OverlayControl
from zeta.tui.render import format_status
from zeta.tui.slash_handlers.decisions_panel import DecisionsMixin


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


def test_status_bar_shows_main_indicator_during_discussion() -> None:
    shown = format_status("codex", "fake", "idle", main_active=True).plain
    assert "◆ main" in shown
    hidden = format_status("codex", "fake", "idle", main_active=False).plain
    assert "◆ main" not in hidden


# -- session-scoped scan ----------------------------------------------------


def _decisions_app(home: Path, opened) -> TUIApp:
    return TUIApp(
        AgentLoop(
            FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty()
        ),
        provider="codex",
        model="fake",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )


@pytest.mark.asyncio
async def test_decisions_tick_counts_only_current_root_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    current = _session(tmp_path, project.project_id)
    other = _session(tmp_path, project.project_id)
    _request(current, project.project_id, "Current", "This one.")
    _request(other, project.project_id, "Other", "Not this one.")

    app = _decisions_app(tmp_path, current)
    child = ConversationStore(
        current.store.session_dir / "agents", session_id="child"
    )
    app._agent_navigation.current_path = child.session_dir
    await app._decisions_tick()

    assert app.decisions_count == 1

    await app.loop.tool_registry.close()
    child.close()
    current.store.close()
    other.store.close()


@pytest.mark.asyncio
async def test_decisions_tick_clears_after_resolve_attention(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    current = _session(tmp_path, project.project_id)
    other = _session(tmp_path, project.project_id)
    _request(other, project.project_id, "Other", "Not this one.")
    anchor = current.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = _request(
        current,
        project.project_id,
        "Current",
        "This one.",
        entry_id=anchor.id,
        entry_seq=anchor.seq,
    )
    app = _decisions_app(tmp_path, current)
    await app._decisions_tick()
    assert app.decisions_count == 1

    fork_id = create_discussion_fork(tmp_path, current.store.session_id, record.id)
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
    result = await registry.definitions_by_name["resolve_attention"].handler(
        {"decision": "Use SQLite."}
    )
    assert result["isError"] is False

    await app._decisions_tick()
    assert app.decisions_count == 0

    await registry.close()
    fork.store.close()
    await app.loop.tool_registry.close()
    current.store.close()
    other.store.close()


def test_open_decisions_lists_live_open_items_for_one_session(tmp_path: Path) -> None:
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

    first_items = open_decisions(tmp_path, first.store.session_id)
    assert {item.record.title for item in first_items} == {"Pick DB"}
    assert all(item.project_name == "alpha" for item in first_items)

    second_items = open_decisions(tmp_path, second.store.session_id)
    assert {item.record.title for item in second_items} == {"Pick region"}

    first.store.close()
    second.store.close()
    # A closed (not live) session no longer contributes a pending decision.
    assert open_decisions(tmp_path, first.store.session_id) == ()


def test_decisions_slash_lists_only_current_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    current = _session(tmp_path, project.project_id)
    other = _session(tmp_path, project.project_id)
    _request(current, project.project_id, "Current", "This one.")
    _request(other, project.project_id, "Other", "Not this one.")

    app = SimpleNamespace(
        _home=tmp_path,
        _decisions_session_id=current.store.session_id,
        loop=current,
    )
    output = DecisionsMixin.slash_decisions(app, "")
    assert "Current" in output
    assert "Other" not in output
    assert current.store.session_id[:8] not in output

    current.store.close()
    other.store.close()


def test_discussion_fork_decisions_use_source_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = manager.project_registry.create_project("alpha", "git")
    source = _session(tmp_path, project.project_id)
    other = _session(tmp_path, project.project_id)
    record = _request(source, project.project_id, "Source", "Discuss this.")
    _request(other, project.project_id, "Other", "Not this one.")
    app = SimpleNamespace(
        _fork_context=SimpleNamespace(source_session_id=source.store.session_id),
        loop=SimpleNamespace(store=SimpleNamespace(session_id="fork")),
    )
    session_id = DecisionsMixin._decisions_session_id.fget(app)
    assert session_id == source.store.session_id
    assert [item.record.title for item in open_decisions(tmp_path, session_id)] == [
        record.title
    ]
    source.store.close()
    other.store.close()


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


def test_decisions_popup_accepts_fragments_with_mouse_handlers() -> None:
    panel = DecisionsPanel()
    panel.set_items(
        [
            _item(
                "1" * 32,
                "Pick DB",
                "Choose a durable store.",
                ["SQLite", "Postgres"],
            )
        ]
    )
    lines = panel.render_lines()
    style, text = lines[2][0]
    lines[2][0] = (style, text, lambda mouse_event: None)
    control = OverlayControl()
    control.set_lines(lines)

    assert control.preferred_width(120) > 0
    content = control.create_content(width=72, height=10)
    assert content.line_count == 10


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

    message_id, _, already = deliver_attention_decision(
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
    _, _, again = deliver_attention_decision(
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
