from __future__ import annotations

import asyncio
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.attention_decisions import DecisionItem, open_decisions
from zeta.attention_forks import create_discussion_fork, release_discussion_fork
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.runtime.loop import AgentLoop
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
    return AttentionStore(opened.store.session_dir, writer_epoch=5).request(
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
    assert "● 2 attention items" in with_count
    singular = format_status("codex", "fake", "idle", decisions_count=1).plain
    assert "● 1 attention item" in singular
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
        AgentLoop(FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="fake",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )


@pytest.mark.asyncio
async def test_decisions_tick_counts_only_current_root_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(manager.project_registry.create_project, "alpha", "git")
    current = await asyncio.to_thread(lambda: _session(tmp_path, project.project_id))
    other = await asyncio.to_thread(lambda: _session(tmp_path, project.project_id))
    await asyncio.to_thread(_request, current, project.project_id, "Current", "This one.")
    await asyncio.to_thread(_request, other, project.project_id, "Other", "Not this one.")

    app = _decisions_app(tmp_path, current)
    child = ConversationStore(current.store.session_dir / "agents", session_id="child")
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
    project = await asyncio.to_thread(manager.project_registry.create_project, "alpha", "git")
    current = await asyncio.to_thread(lambda: _session(tmp_path, project.project_id))
    other = await asyncio.to_thread(lambda: _session(tmp_path, project.project_id))
    await asyncio.to_thread(_request, other, project.project_id, "Other", "Not this one.")
    anchor = current.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("Need a choice")])
    )
    record = await asyncio.to_thread(
        _request,
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

    fork_id = await asyncio.to_thread(
        lambda: create_discussion_fork(tmp_path, current.store.session_id, record.id)
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
    AttentionStore(second.store.session_dir, writer_epoch=5).replace(
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


@pytest.mark.asyncio
async def test_decisions_slash_lists_only_current_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(manager.project_registry.create_project, "alpha", "git")
    current = await asyncio.to_thread(_session, tmp_path, project.project_id)
    other = await asyncio.to_thread(_session, tmp_path, project.project_id)
    await asyncio.to_thread(
        _request, current, project.project_id, "Current", "This one."
    )
    await asyncio.to_thread(
        _request, other, project.project_id, "Other", "Not this one."
    )

    app = SimpleNamespace(
        _home=tmp_path,
        _decisions_session_id=current.store.session_id,
        loop=current,
        _run_attention_operation=lambda operation, *args: asyncio.to_thread(
            operation, *args
        ),
        _attention_replies=lambda session_id: AttentionReplies(tmp_path, session_id),
    )
    output = await DecisionsMixin.slash_decisions(app, "")
    assert "Current" in output
    assert "Other" not in output
    assert current.store.session_id[:8] not in output

    await asyncio.to_thread(current.store.close)
    await asyncio.to_thread(other.store.close)


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


def _item(
    session_id: str, title: str, why: str, options=(), recommendation=None
) -> DecisionItem:
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
        recommendation=recommendation,
        status="open",
    )
    return DecisionItem(session_id, "p", "proj", record)


def test_decisions_popup_lists_items_and_offers_options() -> None:
    panel = DecisionsPanel()
    one = _item("1" * 32, "Pick DB", "Choose a durable store.", ["SQLite", "Postgres"])
    two = _item("2" * 32, "Pick region", "Where to deploy the worker.")
    # Distinct ids so selection is stable.
    two = DecisionItem(two.session_id, "p", "proj", replace(two.record, id="b" * 32))
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

    outcome = AttentionReplies(tmp_path, asker.store.session_id).reply(
        record.id, "Use SQLite.", from_session=answerer.store.session_id
    )
    assert not outcome.already_resolved
    assert outcome.message_id is not None

    inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
    new = inbox.new_ids(project.project_id, session_id=asker.store.session_id)
    assert new == (outcome.message_id,)
    message = inbox.claim(
        project.project_id, outcome.message_id, asker.store.session_id
    )
    assert "Use SQLite." in message["body"]
    assert AttentionStore(asker.store.session_dir, writer_epoch=5).get(record.id).status == "resolved"

    # A repeat is idempotent and reports the prior delivery.
    again = AttentionReplies(tmp_path, asker.store.session_id).reply(
        record.id, "Use SQLite.", from_session=answerer.store.session_id
    )
    assert again.already_resolved

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
    assert (
        AttentionStore(asker.store.session_dir, writer_epoch=5).get(record.id).fork_session_id
        == fork_id
    )

    source = release_discussion_fork(tmp_path, fork_id)
    assert source == asker.store.session_id
    # Open item: the binding is cleared so a fresh fork starts next time.
    assert (
        AttentionStore(asker.store.session_dir, writer_epoch=5).get(record.id).fork_session_id is None
    )
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
    AttentionReplies(tmp_path, asker.store.session_id).reply(
        record.id, "Use SQLite.", from_session=fork_id
    )
    source = release_discussion_fork(tmp_path, fork_id)
    assert source == asker.store.session_id
    resolved = AttentionStore(asker.store.session_dir, writer_epoch=5).get(record.id)
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


class _FixedSizeOutput(DummyOutput):
    def __init__(self, width: int, height: int) -> None:
        self._size = Size(rows=height, columns=width)

    def get_size(self) -> Size:
        return self._size


def _screen_text(screen: Screen, width: int, height: int) -> str:
    return "\n".join(
        "".join(screen.data_buffer[y][x].char for x in range(width))
        for y in range(height)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "height"), [(80, 24), (200, 50)])
async def test_decisions_overflow_indicator_uses_real_float_render(
    tmp_path: Path, width: int, height: int
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    try:
        session = app._make_session()
        app._active_session = session
        app._install_full_screen_layout(session)
        why = (
            "\n".join(
                f"why paragraph {index} has useful context " + "useful context " * 12
                for index in range(17)
            )
            + "\ntail-marker"
        )
        item = _item(
            "1" * 32,
            "Which deployment should we use?",
            why,
            options=[
                "first option with enough detail to wrap",
                "second option with enough detail to wrap",
                "third option with enough detail to wrap",
            ],
            recommendation="Use the first option because it is safest.",
        )
        app._decisions_panel.set_items([item])
        await app._open_decisions_panel()

        async def render() -> tuple[Screen, MouseHandlers]:
            session.app.output = _FixedSizeOutput(width, height)
            screen = Screen()
            handlers = MouseHandlers()
            session.layout.update_parents_relations()
            with set_app(session.app):
                session.layout.container.write_to_screen(
                    screen,
                    handlers,
                    WritePosition(0, 0, width, height),
                    "",
                    False,
                    None,
                )
                screen.draw_all_floats()
            return screen, handlers

        def press(key: str) -> None:
            with set_app(session.app):
                session.app.key_processor.feed(KeyPress(key))
                session.app.key_processor.process_keys()

        first_screen, _ = await render()
        first = _screen_text(first_screen, width, height)
        assert "↓ more · pgdn" in first
        for _ in range(20):
            press(Keys.PageDown)
        bottom_screen, _ = await render()
        bottom = _screen_text(bottom_screen, width, height)
        assert "Recommendation:" in bottom
        assert "↓ more · pgdn" not in bottom
        assert "↑ more · pgup" in bottom

        offset_at_bottom = app._decisions_panel_control.offset
        press(Keys.PageUp)
        assert app._decisions_panel_control.offset < offset_at_bottom
        press(Keys.PageDown)
        assert app._decisions_panel_control.offset == offset_at_bottom

        # The answer input owns focus, but page keys must still scroll the popup.
        press("a")
        assert app._decisions_panel.mode == "answer"
        app._decisions_panel_control.top()
        for _ in range(20):
            press(Keys.PageDown)
        answer_bottom_screen, _ = await render()
        answer_bottom = _screen_text(answer_bottom_screen, width, height)
        assert "tail-marker" in answer_bottom
        assert "↑ more · pgup" in answer_bottom
        answer_offset = app._decisions_panel_control.offset
        press("x")
        assert app._decisions_panel_control.offset == answer_offset

        # The rendered popup installs the wheel handler in MouseHandlers.
        press(Keys.Escape)
        app._decisions_panel_control.top()
        initial_screen, handlers = await render()
        initial = _screen_text(initial_screen, width, height)
        assert "↓ more · pgdn" in initial
        wheel_event = MouseEvent(
            position=Point(0, 0),
            event_type=MouseEventType.SCROLL_DOWN,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
        wheel_handler = None
        with set_app(session.app):
            for row in range(height):
                for column in range(width):
                    candidate = handlers.mouse_handlers[row][column]
                    if candidate is None:
                        continue
                    wheel_event.position = Point(column, row)
                    candidate(wheel_event)
                    if app._decisions_panel_control.offset > 0:
                        wheel_handler = candidate
                        break
                if wheel_handler is not None:
                    break
            assert wheel_handler is not None
            for _ in range(30):
                wheel_handler(wheel_event)
        wheel_screen, _ = await render()
        wheel_text = _screen_text(wheel_screen, width, height)
        assert "tail-marker" in wheel_text
        assert "↑ more · pgup" in wheel_text

        press(Keys.PageUp)
        assert (
            app._decisions_panel_control.offset
            < app._decisions_panel_control._max_offset()
        )
        second = _item("2" * 32, "Other decision", "short", ["Keep"])
        app._decisions_panel.set_items([second])
        app._refresh_decisions_panel()
        assert app._decisions_panel_control.offset == 0
    finally:
        await app.close()


def test_decisions_short_detail_has_no_overflow_indicator() -> None:
    panel = DecisionsPanel()
    panel.set_items([_item("1" * 32, "Short", "Why", ["One"])])
    control = OverlayControl()
    control.set_lines(panel.render_lines())
    content = control.create_content(width=80, height=24)
    text = "\\n".join(
        "".join(fragment[1] for fragment in content.get_line(i))
        for i in range(content.line_count)
    )
    assert "more" not in text


def test_decisions_wraps_wide_unbroken_tokens_by_cells() -> None:
    panel = DecisionsPanel()
    panel.set_items([_item("1" * 32, "界界界界界界界界", "Why")])
    lines = panel.render_lines()
    question = [line for line in lines if "界" in _plain([line])]
    assert question
    assert all(
        sum(max(0, get_cwidth(c)) for c in _plain([line])) <= 66 for line in question
    )


def test_decisions_detail_keeps_full_text_and_recommendation() -> None:
    panel = DecisionsPanel()
    panel.set_items(
        [
            _item(
                "1" * 32,
                "A very long decision title that must remain readable in the detail view",
                "A long explanation that should wrap across multiple rows instead of being truncated.",
                ["The first option has enough text to wrap too."],
                recommendation="Prefer the first option because it is safer.",
            )
        ]
    )
    text = _plain(panel.render_lines())
    assert "A very long decision title that must remain readable" in text
    assert "truncated" in text
    assert "Recommendation:" in text
    assert "Prefer the first option" in text
