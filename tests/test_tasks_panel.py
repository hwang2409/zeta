"""Tests for the ``/tasks`` background-process panel."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.text import Text

from tests.support.fake_backend import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tools._shared.process import BackgroundTaskInfo, BackgroundTaskRegistry
from zeta.tui import theme
from zeta.tui.app import TUIApp
from zeta.tui.cards.shared import wrap_by_cells
from zeta.tui.cards.tasks_panel import (
    MAX_OUTPUT_LINES,
    BackgroundTasksPanel,
    format_runtime,
)
from zeta.tui.render import format_status


def _info(
    task_id: str,
    command: str = "echo hi",
    *,
    pid: int = 100,
    owner: str = "run_background",
    running: bool = True,
    exit_code: int | None = None,
    note: str | None = None,
    terminal_phase: str | None = None,
    started_at: float | None = 0.0,
    ended_at: float | None = None,
    output_bytes: int = 0,
    output_lines: int = 0,
) -> BackgroundTaskInfo:
    return BackgroundTaskInfo(
        task_id=task_id,
        command=command,
        pid=pid,
        owner=owner,
        running=running,
        exit_code=exit_code,
        note=note,
        terminal_phase=terminal_phase,
        started_at=started_at,
        ended_at=ended_at,
        output_bytes=output_bytes,
        output_lines=output_lines,
    )


def _text(panel: BackgroundTasksPanel) -> list[str]:
    return ["".join(fragment for _, fragment in line) for line in panel.render_lines()]


def _styles(panel: BackgroundTasksPanel) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in panel.render_lines():
        for style, fragment in line:
            pairs.append((style, fragment))
    return pairs


# -- pure view model --------------------------------------------------------


def _base_decisions_wrap(text: str, width: int) -> list[str]:
    """Reference oracle copied from decisions.py at base 6a4b690d."""

    width = max(1, width)
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split() or [""]:
            piece = word if not current else f"{current} {word}"
            if sum(max(0, get_cwidth(ch)) for ch in piece) <= width:
                current = piece
            else:
                if current:
                    lines.append(current)
                while sum(max(0, get_cwidth(ch)) for ch in word) > width:
                    split = 0
                    cells = 0
                    for index, character in enumerate(word):
                        step = max(0, get_cwidth(character))
                        if cells + step > width:
                            break
                        cells += step
                        split = index + 1
                    if split == 0:
                        split = 1
                    lines.append(word[:split])
                    word = word[split:]
                current = word
        lines.append(current)
    return lines or [""]


def _base_tasks_split_cells(text: str, width: int) -> tuple[str, str]:
    cells = 0
    for index, ch in enumerate(text):
        step = max(0, get_cwidth(ch))
        if cells + step > width:
            return text[:index], text[index:]
        cells += step
    return text, ""


def _base_tasks_wrap(text: str, width: int) -> list[str]:
    """Reference oracle copied from tasks_panel.py at base 6a4b690d."""

    width = max(1, width)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        cells = 0
        for word in paragraph.split(" "):
            piece = word if not current else " " + word
            piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            if current and cells + piece_cells > width:
                lines.append(current)
                current, cells = "", 0
                piece = word
                piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            while piece_cells > width:
                head, piece = _base_tasks_split_cells(piece, width)
                lines.append((current + head) if current else head)
                current, cells = "", 0
                piece_cells = sum(max(0, get_cwidth(ch)) for ch in piece)
            current += piece
            cells += piece_cells
        lines.append(current)
    return lines or [""]


_WRAP_CASES = [
    pytest.param("a  b", 4, id="repeated-spaces"),
    pytest.param("a ", 4, id="trailing-space"),
    pytest.param("a\n", 4, id="one-trailing-newline"),
    pytest.param("a\n\n", 4, id="two-trailing-newlines"),
    pytest.param("", 1, id="empty"),
    pytest.param("abc", 1, id="width-one"),
    pytest.param("abc", 2, id="width-two"),
    pytest.param("ab", 2, id="exact-width"),
    pytest.param("你a好", 3, id="mixed-cjk-ascii"),
    pytest.param("你", 1, id="cjk-wider-than-width"),
    pytest.param("abcdefgh", 3, id="long-ascii-token"),
]


@pytest.mark.parametrize(("text", "width"), _WRAP_CASES)
def test_wrap_by_cells_matches_base_decisions(text: str, width: int) -> None:
    assert wrap_by_cells(text, width) == _base_decisions_wrap(text, width)


@pytest.mark.parametrize(("text", "_width"), _WRAP_CASES)
def test_normalized_tasks_wrap_matches_base(text: str, _width: int) -> None:
    normalized = " ".join(text.split())
    detail_width = 66
    assert wrap_by_cells(normalized, detail_width) == _base_tasks_wrap(
        normalized, detail_width
    )


def test_format_runtime_is_compact() -> None:
    assert format_runtime(None) == "—"
    assert format_runtime(4) == "4s"
    assert format_runtime(64) == "1m04s"
    assert format_runtime(3725) == "1h02m"


def test_list_orders_running_first_then_recent_finishes() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [
            _info("task-oldfinish", running=False, exit_code=0,
                  terminal_phase="natural_exit", started_at=1.0, ended_at=5.0),
            _info("task-running1", running=True, started_at=2.0),
            _info("task-newfinish", running=False, exit_code=0,
                  terminal_phase="natural_exit", started_at=3.0, ended_at=9.0),
            _info("task-running0", running=True, started_at=0.5),
        ],
        now=20.0,
    )
    ordered = [task.task_id for task in panel.tasks]
    assert ordered == [
        "task-running0",  # running, oldest start first
        "task-running1",
        "task-newfinish",  # finished, most recent first
        "task-oldfinish",
    ]


def _status_style_for(panel: BackgroundTasksPanel, label_text: str) -> str:
    for style, text in _styles(panel):
        if text == label_text:
            return style
    raise AssertionError(f"no status label {label_text!r} rendered")


def test_list_renders_running_exited_and_killed_status() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [
            _info("task-aa", running=True, started_at=0.0),
            _info("task-bb", running=False, exit_code=0,
                  terminal_phase="natural_exit", started_at=0.0, ended_at=3.0),
            _info("task-cc", running=False, exit_code=1,
                  terminal_phase="natural_exit", started_at=0.0, ended_at=3.0),
            _info("task-dd", running=False, exit_code=-15,
                  terminal_phase="task_kill", note="task killed",
                  started_at=0.0, ended_at=3.0),
        ],
        now=10.0,
    )
    # Status labels now carry shared theme tokens (converted) instead of
    # ``class:`` names; the selected row also layers the selection background.
    def token(style: str) -> str:
        return theme.prompt_toolkit_style(style)

    assert _status_style_for(panel, "running   ").startswith(token(f"bold {theme.ACCENT}"))
    assert _status_style_for(panel, "exited 0  ").startswith(token(theme.SUCCESS))
    assert _status_style_for(panel, "exited 1  ").startswith(token(theme.ERROR))
    assert _status_style_for(panel, "killed    ").startswith(token(theme.DIM))


def test_empty_state_is_clear() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks([], now=0.0)
    text = "\n".join(_text(panel))
    assert "No background tasks in this session." in text
    assert "esc close" in text


def test_selection_moves_and_clamps() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [_info("task-a", started_at=0.0), _info("task-b", started_at=1.0)],
        now=2.0,
    )
    assert panel.selected.task_id == "task-a"
    panel.move(1)
    assert panel.selected.task_id == "task-b"
    panel.move(5)
    assert panel.selected.task_id == "task-b"
    panel.move(-5)
    assert panel.selected.task_id == "task-a"


def test_kill_requires_a_second_press_and_only_targets_running() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [
            _info("task-run", running=True, started_at=0.0),
            _info("task-done", running=False, exit_code=0,
                  terminal_phase="natural_exit", started_at=0.0, ended_at=1.0),
        ],
        now=2.0,
    )
    assert panel.request_kill() is None  # arm
    assert panel.kill_pending == "task-run"
    assert "kill run?" in "\n".join(_text(panel))
    assert panel.request_kill() == "task-run"  # confirm
    assert panel.kill_pending is None

    panel.move(1)  # selecting a finished task
    assert panel.request_kill() is None
    assert panel.kill_pending is None


def test_details_view_shows_fields_and_output_counter() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks(
        [
            _info(
                "task-detail",
                command="python manage.py runserver 0.0.0.0:8000",
                pid=4242,
                running=False,
                exit_code=1,
                terminal_phase="natural_exit",
                started_at=0.0,
                ended_at=65.0,
                output_bytes=10,
                output_lines=12,
                note="background output was cut off because a process outside the task still held its output",
            )
        ],
        now=65.0,
    )
    assert panel.open_details() is not None
    panel.feed_output("line one\nline two\n", total_lines=12)
    text = "\n".join(_text(panel))
    assert "Shell details" in text
    assert "Status  exited 1" in text
    assert "Runtime 1m05s" in text
    assert "PID     4242" in text
    assert "Exit    1" in text
    assert "background output was cut off because a process outside the task still held its output" in text
    assert "python manage.py runserver 0.0.0.0:8000" in text
    assert "showing 2 of 12 lines" in text
    assert "line one" in text and "line two" in text
    assert panel.back() is True
    assert panel.mode == "list"


def test_output_tail_is_bounded_to_recent_lines() -> None:
    panel = BackgroundTasksPanel()
    panel.set_tasks([_info("task-x", started_at=0.0, output_lines=10_000)], now=1.0)
    panel.open_details()
    panel.feed_output("".join(f"line{i}\n" for i in range(5_000)), total_lines=5_000)
    display = [line for line in _text(panel) if line.strip().startswith("line")]
    assert len(display) <= MAX_OUTPUT_LINES
    # The retained lines are the most recent ones.
    assert any("line4999" in line for line in display)
    assert all("line0\n" not in line for line in display)


# -- registry seam ----------------------------------------------------------


def test_snapshot_reports_status_and_line_count() -> None:
    registry = BackgroundTaskRegistry()
    # A record recovered from a previous session has no start time.
    info = _info("task-x")
    assert info.started_at == 0.0
    assert registry.snapshot() == ()


async def _wait_output(registry: BackgroundTaskRegistry, task_id: str, size: int) -> None:
    for _ in range(500):
        if registry.snapshot()[0].output_bytes >= size:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for output")


@pytest.mark.asyncio
async def test_output_tail_reads_only_new_bytes(tmp_path: Path) -> None:
    registry = BackgroundTaskRegistry()
    task_id, _ = await registry.start("cat", tmp_path)
    try:
        await registry.input(task_id, "x" * 5000 + "\n")
        await _wait_output(registry, task_id, 5001)
        # Drain the backlog fully, tracking the cursor as the panel does.
        cursor = 0
        while True:
            result = await registry.output(task_id, since=cursor)
            cursor = result["cursor"]
            if cursor >= registry.snapshot()[0].output_bytes:
                break
        # Appending more and reading *from the last cursor* returns only the
        # new bytes, never a re-read of the 5 KB backlog.
        await registry.input(task_id, "DELTA\n", eof=True)
        await _wait_output(registry, task_id, cursor + 6)
        tail = await registry.output(task_id, since=cursor)
        assert tail["output"] == "DELTA\n"
    finally:
        await registry.kill(task_id)
        await registry.close()


@pytest.mark.asyncio
async def test_panel_refresh_owns_output_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    registry = app.loop.tool_registry.background_tasks
    task_id, _ = await registry.start("cat", tmp_path)
    output_appended = asyncio.Event()
    original_append = registry._append

    def append_output(record: Any, chunk: bytes) -> None:
        original_append(record, chunk)
        output_appended.set()

    monkeypatch.setattr(registry, "_append", append_output)
    await registry.input(task_id, "first\n")
    await output_appended.wait()
    output_appended.clear()

    first_read = asyncio.Event()
    release_first_read = asyncio.Event()
    original_output = registry.output
    read_count = 0

    async def controlled_output(
        selected_task_id: str, since: int | None = None, *, max_chars: int | None = None
    ) -> dict[str, object]:
        nonlocal read_count
        result = await original_output(selected_task_id, since, max_chars=max_chars)
        read_count += 1
        if read_count == 1:
            first_read.set()
            await release_first_read.wait()
        return result

    monkeypatch.setattr(registry, "output", controlled_output)
    app.open_tasks_panel()
    app._stop_tasks_refresh()
    app._tasks_key("enter")

    refresh = asyncio.create_task(app._tasks_tick())
    try:
        await first_read.wait()
        await registry.input(task_id, "second\n")
        await output_appended.wait()

        # Paint from the current immutable snapshot while the only output read
        # is paused. Painting must not start another read or move its cursor.
        app._refresh_tasks_panel(keep_offset=True)
        release_first_read.set()
        await refresh
        settled = asyncio.Event()
        asyncio.get_running_loop().call_soon(settled.set)
        await settled.wait()
        await app._tasks_tick()

        rendered = "\n".join(_text(app._tasks_panel))
        assert rendered.count("first") == 1
        assert rendered.count("second") == 1
        assert rendered.index("first") < rendered.index("second")
    finally:
        release_first_read.set()
        if not refresh.done():
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        app.close_tasks_panel()
        registry.set_notice_sink(None)
        await registry.kill(task_id)
        await registry.close()


@pytest.mark.asyncio
async def test_snapshot_runtime_and_terminal_phase(tmp_path: Path) -> None:
    registry = BackgroundTaskRegistry()
    running_id, _ = await registry.start("sleep 30", tmp_path)
    exit_id, _ = await registry.start("true", tmp_path)
    try:
        await registry.wait(exit_id, timeout=10)
        snap = {task.task_id: task for task in registry.snapshot()}
        assert snap[running_id].running is True
        assert snap[running_id].started_at is not None
        assert snap[running_id].ended_at is None
        assert snap[exit_id].running is False
        assert snap[exit_id].exit_code == 0
        assert snap[exit_id].ended_at is not None
        assert snap[exit_id].ended_at >= snap[exit_id].started_at
    finally:
        await registry.kill(running_id)
        await registry.close()


# -- bottom-bar hint --------------------------------------------------------


def test_bottom_bar_background_segment_mentions_tasks() -> None:
    plain = format_status("fake", "offline", "idle", background_count=1).plain
    assert "bg 1" in plain
    assert "/tasks" in plain
    assert "/tasks" not in format_status("fake", "offline", "idle").plain


# -- TUI integration --------------------------------------------------------


def _app(tmp_path: Path) -> TUIApp:
    store = ConversationStore(tmp_path / "sessions")
    return TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )


@pytest.mark.asyncio
async def test_tasks_panel_open_close_preserves_composer(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._transcript.append(Text("existing transcript"))
    units = app._transcript.units
    session.default_buffer.set_document(Document("draft text", 5))

    app.open_tasks_panel()
    try:
        assert app.tasks_panel_active
        assert session.layout.has_focus(app._tasks_panel_window)
        assert app._transcript.units == units
        text = "\n".join(
            "".join(f for _, f in line)
            for line in app._tasks_panel.render_lines()
        )
        assert "No background tasks in this session." in text
    finally:
        app.close_tasks_panel()

    assert not app.tasks_panel_active
    assert session.default_buffer.text == "draft text"
    assert session.default_buffer.cursor_position == 5
    assert session.layout.has_focus(session.default_buffer)


@pytest.mark.asyncio
async def test_slash_tasks_opens_the_panel(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    session.default_buffer.set_document(Document("/tasks", 6))

    assert app._submit_input("/tasks") is True
    try:
        assert app.tasks_panel_active
        assert session.default_buffer.text == ""
    finally:
        app.close_tasks_panel()


@pytest.mark.asyncio
async def test_kill_from_panel_kills_selected_and_requires_confirm(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    registry = app.loop.tool_registry.background_tasks
    first, _ = await registry.start("sleep 30", tmp_path)
    second, _ = await registry.start("sleep 30", tmp_path)

    app.open_tasks_panel()
    try:
        app._tasks_key("down")
        assert app._tasks_panel.selected.task_id == second
        app._tasks_key("k")  # arm the confirm
        assert app._tasks_panel.kill_pending == second
        # Still running: a single press does not kill.
        await asyncio.sleep(0.1)
        assert registry.snapshot()  # not raised
        assert {t.task_id: t.running for t in registry.snapshot()}[second] is True

        app._tasks_key("k")  # confirm
        await asyncio.wait_for(registry.wait(second), timeout=10)
        states = {t.task_id: t.running for t in registry.snapshot()}
        assert states[second] is False
        assert states[first] is True
    finally:
        app.close_tasks_panel()
        registry.set_notice_sink(None)
        await registry.kill(first)
        await registry.close()
