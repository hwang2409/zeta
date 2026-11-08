from __future__ import annotations

import asyncio
import threading
import time
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail, close_owned_app
from zeta.attention_decisions import open_decisions
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp


def _session(home: Path, project_id: str):
    return SessionManager(home).create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )


def _request(opened, project_id: str, title: str, *, kind: str = "decision"):
    return AttentionStore(opened.store.session_dir).request(
        session_id=opened.store.session_id,
        project_id=project_id,
        entry_id=None,
        entry_seq=None,
        title=title,
        why="choose",
        options=("yes",) if kind == "decision" else (),
        kind=kind,
    )


def _records(owner, project_id: str):
    return [_request(owner, project_id, f"Pick {index}") for index in range(3)]


def _wait_for_records(home: Path, session_id: str, record_ids: set[str]) -> None:
    store = AttentionStore(home / "sessions" / session_id)
    deadline = time.monotonic() + 9
    while time.monotonic() < deadline:
        try:
            resolved = all(
                store.get(record_id).status == "resolved" for record_id in record_ids
            )
        except (FileNotFoundError, ValueError):
            # Atomic replacement can unlink the inode after this reader opens it.
            resolved = False
        if resolved:
            return
        threading.Event().wait(0.01)
    raise AssertionError("attention records did not resolve before deadline")


class _FixedOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=30, columns=100)


async def _make_app(
    home: Path, opened, source_session_id: str, *, open_panel: bool = True
) -> TUIApp:
    app = TUIApp(
        AgentLoop(FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="fake",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._decisions_panel.set_items(open_decisions(home, source_session_id))
    await app.loop._activate_project_inbox()
    if app.loop._project_inbox_publication_event is None:
        app.loop._project_inbox_owner_loop = asyncio.get_running_loop()
        app.loop._project_inbox_publication_event = asyncio.Event()
        watcher = asyncio.create_task(app.loop._watch_project_inbox())
        app.loop._tracked_tasks.add(watcher)
        watcher.add_done_callback(app.loop._tracked_tasks.discard)
    if open_panel:
        opening = app.open_decisions_panel()
        if opening is not None:
            await await_or_fail(opening, what="open decisions panel")
    return app


def _press(app: TUIApp, key: str) -> None:
    key_value = Keys.Escape if key == "escape" else key
    with set_app(app._active_session.app):
        app._active_session.app.key_processor.feed(KeyPress(key_value))
        app._active_session.app.key_processor.process_keys()


@pytest.mark.asyncio
async def test_tui_popup_waits_for_recovery_before_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(
        manager.project_registry.create_project, "alpha", "git"
    )
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    records = await await_or_fail(
        asyncio.to_thread(_records, owner, project.project_id),
        what="create attention records",
    )
    app = await _make_app(tmp_path, owner, owner.store.session_id, open_panel=False)
    app.close_decisions_panel()
    await await_or_fail(
        asyncio.to_thread(lambda: not app.decisions_panel_active), what="close popup"
    )
    gate = threading.Event()
    started = threading.Event()
    real_recover = AttentionReplies.recover

    def gated_recover(replies: AttentionReplies):
        started.set()
        if not gate.wait(10):
            raise AssertionError("recovery gate did not open")
        return real_recover(replies)

    monkeypatch.setattr(AttentionReplies, "recover", gated_recover)
    try:
        opening = app.open_decisions_panel()
        await await_or_fail(
            asyncio.to_thread(started.wait, 10), what="start gated recovery"
        )
        assert not app.decisions_panel_active
        _press(app, "1")
        assert all(
            AttentionStore(owner.store.session_dir).get(r.id).status == "open"
            for r in records
        )
        gate.set()
        if opening is not None:
            await await_or_fail(opening, what="popup recovery")
        else:
            await await_or_fail(
                asyncio.to_thread(lambda: app.decisions_panel_active),
                what="popup recovery",
            )
        assert [item.record.id for item in app._decisions_panel.items] == [
            r.id for r in records
        ]
    finally:
        gate.set()
        owner.store.close()
        await await_or_fail(close_owned_app(app), what="TUIApp.close")


@pytest.mark.asyncio
async def test_tui_batches_three_answers_only_on_last_enter_and_wakes_once(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(
        manager.project_registry.create_project, "alpha", "git"
    )
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    answerer = await asyncio.to_thread(_session, tmp_path, project.project_id)
    records = await await_or_fail(
        asyncio.to_thread(_records, owner, project.project_id),
        what="create attention records",
    )
    app = await _make_app(tmp_path, owner, owner.store.session_id)
    wakes: list[bool] = []
    wake_event = asyncio.Event()

    async def record_wake(*, periodic: bool = False) -> None:
        wakes.append(True)
        wake_event.set()

    app.loop._check_project_inbox = record_wake
    try:
        app._active_session.app.output = _FixedOutput()
        screen = Screen()
        handlers = MouseHandlers()
        with set_app(app._active_session.app):
            app._active_session.layout.container.write_to_screen(
                screen, handlers, WritePosition(0, 0, 100, 30), "", False, None
            )
            screen.draw_all_floats()
        assert any(handlers.mouse_handlers)
        _press(app, "1")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert all(
            AttentionStore(owner.store.session_dir).get(r.id).status == "open"
            for r in records
        )
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "1")
        await asyncio.sleep(0)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "1")
        await asyncio.sleep(0)
        await asyncio.wait_for(
            asyncio.to_thread(
                _wait_for_records,
                tmp_path,
                owner.store.session_id,
                {r.id for r in records},
            ),
            10,
        )
        inbox = ProjectInbox(
            manager.project_registry, sessions_root=tmp_path / "sessions"
        )
        assert (
            len(inbox.new_ids(project.project_id, session_id=owner.store.session_id))
            == 1
        )
        await await_or_fail(wake_event.wait(), what="same-session wake")
        assert len(wakes) == 1
    finally:
        owner.store.close()
        answerer.store.close()
        await await_or_fail(close_owned_app(app), what="TUIApp.close")


@pytest.mark.asyncio
async def test_tui_escape_flushes_two_answers_and_dismisses_non_decision(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(
        manager.project_registry.create_project, "alpha", "git"
    )
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    answerer = await asyncio.to_thread(_session, tmp_path, project.project_id)
    records = await await_or_fail(
        asyncio.to_thread(_records, owner, project.project_id),
        what="create attention records",
    )
    answer = await asyncio.to_thread(
        _request, owner, project.project_id, "Status", kind="answer"
    )
    app = await _make_app(tmp_path, owner, owner.store.session_id)
    wakes: list[bool] = []
    wake_event = asyncio.Event()

    async def record_wake(*, periodic: bool = False) -> None:
        wakes.append(True)
        wake_event.set()

    app.loop._check_project_inbox = record_wake
    try:
        _press(app, "1")
        await asyncio.sleep(0)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "1")
        await asyncio.sleep(0)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "d")
        await asyncio.sleep(0)
        await asyncio.wait_for(
            asyncio.to_thread(
                _wait_for_records, tmp_path, owner.store.session_id, {answer.id}
            ),
            10,
        )
        _press(app, "escape")
        await asyncio.wait_for(
            asyncio.to_thread(
                _wait_for_records,
                tmp_path,
                owner.store.session_id,
                {r.id for r in records[:2]},
            ),
            10,
        )
        assert (
            AttentionStore(owner.store.session_dir).get(records[2].id).status == "open"
        )
        inbox = ProjectInbox(
            manager.project_registry, sessions_root=tmp_path / "sessions"
        )
        assert (
            len(inbox.new_ids(project.project_id, session_id=owner.store.session_id))
            == 1
        )
        await await_or_fail(wake_event.wait(), what="same-session wake")
        assert len(wakes) == 1
    finally:
        owner.store.close()
        answerer.store.close()
        await await_or_fail(close_owned_app(app), what="TUIApp.close")


@pytest.mark.asyncio
async def test_cancelled_tui_adapter_still_checks_late_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(
        manager.project_registry.create_project, "cancel tui", "test"
    )
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    record = await asyncio.to_thread(
        _request, owner, project.project_id, "late publication"
    )
    app = await _make_app(tmp_path, owner, owner.store.session_id, open_panel=False)
    release = threading.Event()
    worker_started = threading.Event()
    worker_done = threading.Event()
    loop = app.loop
    checks: list[bool] = []

    async def count_check(*, periodic: bool = False) -> None:
        checks.append(periodic)

    real_send = ProjectInbox.send

    def blocked_send(inbox: ProjectInbox, **kwargs):
        worker_started.set()
        assert release.wait(10), "TUI publication gate was not released"
        try:
            return real_send(inbox, **kwargs)
        finally:
            worker_done.set()

    # test-contract: seam hold the real TUI worker before durable publication
    monkeypatch.setattr(ProjectInbox, "send", blocked_send)
    # test-contract: seam count admission owned by the project-inbox watcher
    monkeypatch.setattr(loop, "_check_project_inbox", count_check)
    try:
        replies = app._attention_replies(owner.store.session_id)
        task = asyncio.create_task(
            app._run_attention_operation(
                replies.reply, record.id, "yes", owner.store.session_id
            )
        )
        assert await await_or_fail(
            asyncio.to_thread(worker_started.wait, 10), what="TUI worker start"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert checks == []
        release.set()
        assert await await_or_fail(
            asyncio.to_thread(worker_done.wait, 10), what="TUI worker finish"
        )
        for _ in range(10):
            if checks:
                break
            await asyncio.sleep(0)
        assert checks.count(False) == 1, (
            "cancelled TUI adapter lost its late publication check"
        )
    finally:
        release.set()
        owner.store.close()
        await await_or_fail(close_owned_app(app), what="TUIApp.close")
