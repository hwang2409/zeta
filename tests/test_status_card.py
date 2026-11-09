import asyncio
import contextlib
import json
from dataclasses import fields
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from tests.support.fake_backend import FakeBackend
from tests.support.submissions import await_submission_pipeline
from zeta.attention_decisions import DecisionItem
from zeta.attention_records import AttentionStore
from zeta.core.commands.slash_registry import SlashCommand
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.overlay import OverlayControl
from zeta.tui.slash_handlers import decisions_panel as decisions_panel_module


def _lines(texts: list[str]) -> list[list[tuple[str, str]]]:
    return [[("", text)] for text in texts]


def test_status_card_is_focusable_and_scrolls_with_a_bounded_offset() -> None:
    card = OverlayControl()
    card.set_lines(_lines([f"line {index}" for index in range(20)]))

    assert card.is_focusable
    # The box keeps its top and bottom borders fixed, so a 5-row float scrolls
    # the 3 rows between them: the offset tops out at 20 - 3 = 17.
    card.create_content(width=40, height=5)
    card.scroll(3)
    assert card.offset == 3
    card.scroll(100)
    assert card.offset == 17
    card.page(-1)
    assert card.offset == 15
    card.top()
    assert card.offset == 0
    card.bottom()
    assert card.offset == 17


@pytest.mark.asyncio
async def test_status_card_open_close_preserves_composer_and_transcript(
    tmp_path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._transcript.append(Text("existing transcript"))
    units = app._transcript.units
    session.default_buffer.set_document(Document("draft text", 5))
    app.open_status_card(app.slash_status())

    assert app.status_card_active
    assert app._transcript.units == units
    assert session.layout.has_focus(app._status_card_window)
    app._status_card.create_content(width=40, height=4)
    app._status_card.bottom()
    assert app._status_card.offset > 0

    app.close_status_card()
    assert not app.status_card_active
    assert session.default_buffer.text == "draft text"
    assert session.default_buffer.cursor_position == 5
    assert session.layout.has_focus(session.default_buffer)
    assert app._transcript.units == units


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "title"),
    [
        ("status", "completion_client_setups"),
        ("mcp", "MCP servers"),
        ("tasks", "Background tasks"),
        ("decisions", "Attention"),
    ],
)
async def test_bare_overlay_commands_dispatch_and_render_through_registry(
    tmp_path, command: str, title: str
) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    existing_tasks = asyncio.all_tasks()
    dispatched: list[str] = []
    for name in ("help", command):
        original = app._slash_commands._commands.get(name)
        assert original is not None

        async def traced(session, args, *, _command=original):
            dispatched.append(_command.name)
            return await _command.run(session, args, session.slash_lock)

        app._slash_commands._commands[name] = SlashCommand.loop_safe(
            name,
            traced,
            original.description,
            serve=original.serve,
            reason="trace dispatch",
        )
    try:
        app._submit_input("/help")
        await await_submission_pipeline(app)
        rendered_help = "\n".join(app._transcript.lines(100))
        assert "/status — show session status" in rendered_help
        assert "/accounts — list, switch, or label accounts" in rendered_help
        assert dispatched == ["help"]

        app._submit_input(f"/{command}")
        await await_submission_pipeline(app)
        active = {
            "status": app.status_card_active,
            "mcp": app.status_card_active and app._mcp_manager_open,
            "tasks": app.tasks_panel_active,
            "decisions": app.decisions_panel_active,
        }
        assert active[command], active
        assert dispatched == ["help", command]
        screen = Screen(initial_width=80, initial_height=24)
        with set_app(session.app):
            session.layout.update_parents_relations()
            session.layout.container.write_to_screen(
                screen,
                MouseHandlers(),
                WritePosition(0, 0, 80, 24),
                "",
                False,
                None,
            )
            screen.draw_all_floats()
        screen_text = "\n".join(
            "".join(screen.data_buffer[y][x].char for x in range(80))
            for y in range(24)
        )
        assert title in screen_text
        assert session.default_buffer.text == ""
    finally:
        if command in {"status", "mcp"}:
            app.close_status_card()
        elif command == "tasks":
            app.close_tasks_panel()
        else:
            app.close_decisions_panel()
        await app._submissions.close()
        await app.loop.close()
        spawned = asyncio.all_tasks() - existing_tasks - {asyncio.current_task()}
        for task in spawned:
            task.cancel()
        if spawned:
            await asyncio.wait_for(
                asyncio.gather(*spawned, return_exceptions=True), timeout=30
            )


class _FixedSizeOutput(DummyOutput):
    def __init__(self, width: int, height: int) -> None:
        self._size = Size(rows=height, columns=width)

    def get_size(self) -> Size:
        return self._size


async def _render_status(session, width: int, height: int) -> Screen:
    session.app.output = _FixedSizeOutput(width, height)
    screen = Screen()
    handlers = MouseHandlers()
    session.layout.update_parents_relations()
    with set_app(session.app):
        session.layout.container.write_to_screen(
            screen, handlers, WritePosition(0, 0, width, height), "", False, None
        )
        screen.draw_all_floats()
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("with_fact_floor", [False, True])
async def test_status_card_renders_compaction_with_or_without_fact_floor(
    tmp_path, with_fact_floor: bool
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    source = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("source")])
    )
    aggregate = {
        "fact": {"type": "pr", "pr": 7},
        "first_seq": source.seq,
        "last_seq": source.seq,
        "value_rank": [source.seq, 0],
    }
    store.append_compaction_marker(
        "summary",
        source.seq,
        source.seq,
        kind="evict",
        view=[
            {
                "seq": source.seq,
                "message": Message(
                    MessageRole.ASSISTANT, [TextContent("receipt")]
                ).to_dict(),
            }
        ],
        fact_floor=(
            {"version": 1, "oldest": [aggregate], "newest": [aggregate]}
            if with_fact_floor
            else None
        ),
    )
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
        app.open_status_card(app.slash_status())

        await _render_status(session, 100, 30)
        app._status_card.bottom()
        screen = await _render_status(session, 100, 30)
        text = "\n".join(
            "".join(screen.data_buffer[y][x].char for x in range(100))
            for y in range(30)
        )
        assert "compaction_history:" in text
        assert "turn 1: 1 entries" in text
        assert "retained typed facts" not in text
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_decisions_popup_renders_kinds_and_dismisses_information(
    tmp_path, monkeypatch
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    attention = AttentionStore(store.session_dir)
    decision = attention.request(
        session_id=store.session_id,
        project_id=None,
        entry_id=None,
        entry_seq=None,
        title="choose",
        why="choose a value " + "with supporting context " * 20,
    )
    information = attention.request(
        session_id=store.session_id,
        project_id=None,
        entry_id=None,
        entry_seq=None,
        title="answer",
        why="the answer is ready " + "with supporting context " * 20,
    )
    information_path = attention.directory / f"{information.id}.json"
    information_value = json.loads(information_path.read_text())
    information_value["kind"] = "answer"
    information_path.write_text(json.dumps(information_value))
    if hasattr(information, "kind"):
        information = attention.get(information.id)
    else:
        information = SimpleNamespace(
            **{field.name: getattr(information, field.name) for field in fields(information)},
            kind="answer",
        )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        zeta_home=tmp_path,
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._decisions_panel.set_items(
        [
            DecisionItem(store.session_id, None, "Unassigned", decision),
            DecisionItem(store.session_id, None, "Unassigned", information),
        ]
    )
    await app._open_decisions_panel()
    try:
        with set_app(session.app):
            session.app.key_processor.feed(KeyPress("j"))
            session.app.key_processor.process_keys()
        screen = await _render_status(session, 100, 50)
        rows = [
            "".join(screen.data_buffer[y][x].char for x in range(100))
            for y in range(50)
        ]
        text = "\n".join(rows)
        assert "decision" in text and "answer" in text
        assert text.index("decision") < text.index("answer")
        decision_badge = next(
            style
            for style, text in app._decisions_panel._list_row(
                app._decisions_panel.items[0]
            )
            if text.startswith("  decision ·")
        )
        answer_badge = next(
            style
            for style, text in app._decisions_panel._list_row(
                app._decisions_panel.items[1]
            )
            if text.startswith("  answer ·")
        )
        def foreground(style: str) -> str:
            return style.split(" on ", 1)[0].split(" bg:", 1)[0]

        assert foreground(decision_badge) != foreground(answer_badge)
        assert "d dismiss" in text
        assert app._decisions_panel.selected.record.kind == "answer"
        await _render_status(session, 100, 24)
        with set_app(session.app):
            for _ in range(20):
                session.app.key_processor.feed(KeyPress(Keys.PageDown))
                session.app.key_processor.process_keys()
        offset_before_dismiss = app._decisions_panel_control.offset
        assert offset_before_dismiss > 0

        app._stop_decisions_refresh()
        tick_done = asyncio.Event()
        dismissal_task: asyncio.Task[None] | None = None
        original_create_task = asyncio.create_task

        def capture_task(coro):
            nonlocal dismissal_task
            dismissal_task = original_create_task(coro)
            return dismissal_task

        monkeypatch.setattr(decisions_panel_module.asyncio, "create_task", capture_task)
        original_tick = app._decisions_tick

        async def wait_for_dismissal() -> None:
            await original_tick()
            tick_done.set()

        app._decisions_tick = wait_for_dismissal
        with set_app(session.app):
            session.app.key_processor.feed(KeyPress("d"))
            session.app.key_processor.process_keys()
        try:
            await asyncio.wait_for(tick_done.wait(), timeout=10)
        except TimeoutError:
            if dismissal_task is not None:
                dismissal_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(dismissal_task, 10)
            raise
        assert information.id not in {item.record.id for item in app._decisions_panel.items}
        dismissed = attention.get(information.id)
        assert dismissed.status == "resolved"
        assert dismissed.decision is None
        assert app._decisions_panel_control.offset > 0
    finally:
        app.close_decisions_panel()
        await app.close()


@pytest.mark.asyncio
async def test_empty_decisions_popup_renders_and_closes(tmp_path) -> None:
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
        await app._open_decisions_panel()

        screen = await _render_status(session, 80, 24)
        text = "\n".join(
            "".join(screen.data_buffer[y][x].char for x in range(80))
            for y in range(24)
        )
        assert "No open attention items." in text
        assert "esc close" in text

        app.close_decisions_panel()
        # Closed floats are still measured during redraws.
        await _render_status(session, 80, 24)
        assert not app.decisions_panel_active
        assert session.layout.has_focus(session.default_buffer)
    finally:
        await app.close()


def _card_bounds(screen: Screen, width: int, height: int) -> tuple[int, int, int, int]:
    cells = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if "class:overlay" in screen.data_buffer[y][x].style
    ]
    assert cells
    return (
        min(x for x, _ in cells),
        max(x for x, _ in cells),
        min(y for _, y in cells),
        max(y for _, y in cells),
    )


@pytest.mark.asyncio
async def test_status_card_render_is_centered_and_bounded(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._status_card.set_lines(_lines(["status", "──────", "x" * 200]))
    app._status_card_open = True

    screen = await _render_status(session, 100, 30)
    left, right, top, bottom = _card_bounds(screen, 100, 30)
    assert right - left + 1 == 72
    assert bottom - top + 1 < 30


@pytest.mark.asyncio
async def test_status_card_small_terminal_scrolls_and_clamps(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._status_card.set_lines(_lines([f"line {index}" for index in range(20)]))
    app._status_card_open = True

    screen = await _render_status(session, 24, 7)
    left, right, top, bottom = _card_bounds(screen, 24, 7)
    assert right - left + 1 == 20
    assert bottom - top + 1 == 3

    app._status_card.bottom()
    assert app._status_card.offset == 19
    app._status_card.scroll(100)
    assert app._status_card.offset == 19
    app._status_card.top()
    assert app._status_card.offset == 0


def test_status_card_content_is_width_bounded() -> None:
    card = OverlayControl()
    card.set_lines(_lines(["x" * 100]))
    content = card.create_content(width=12, height=3)

    rendered = "".join(text for _, text in content.get_line(1))
    assert len(rendered) == 12
    assert rendered.startswith("│")
    assert rendered.endswith("│")
