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
from tests.support.owned import await_or_fail
from zeta.attention_decisions import open_decisions
from zeta.attention_records import AttentionStore
from zeta.runtime.loop import AgentLoop
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp


def _session(home: Path, project_id: str):
    return SessionManager(home).create(
        provider="codex", model="fake", cwd=home, project_id=project_id, auto_project=False
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
        if all(store.get(record_id).status == "resolved" for record_id in record_ids):
            return
        threading.Event().wait(0.01)
    raise AssertionError("attention records did not resolve before deadline")


class _FixedOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=30, columns=100)


async def _make_app(home: Path, opened, source_session_id: str) -> TUIApp:
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
    app.open_decisions_panel()
    return app


def _press(app: TUIApp, key: str) -> None:
    key_value = Keys.Escape if key == "escape" else key
    with set_app(app._active_session.app):
        app._active_session.app.key_processor.feed(KeyPress(key_value))
        app._active_session.app.key_processor.process_keys()


@pytest.mark.asyncio
async def test_tui_batches_three_answers_only_on_last_enter_and_wakes_once(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(manager.project_registry.create_project, "alpha", "git")
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    answerer = await asyncio.to_thread(_session, tmp_path, project.project_id)
    records = await await_or_fail(
        asyncio.to_thread(_records, owner, project.project_id),
        what="create attention records",
    )
    app = await _make_app(tmp_path, owner, owner.store.session_id)
    wakes: list[bool] = []
    app._schedule_background_wake = lambda: wakes.append(True)
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
        assert all(AttentionStore(owner.store.session_dir).get(r.id).status == "open" for r in records)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "1")
        await asyncio.sleep(0)
        _press(app, "j")
        await asyncio.sleep(0)
        _press(app, "1")
        await asyncio.sleep(0)
        await asyncio.wait_for(
            asyncio.to_thread(_wait_for_records, tmp_path, owner.store.session_id, {r.id for r in records}),
            10,
        )
        inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
        assert len(inbox.new_ids(project.project_id, session_id=owner.store.session_id)) == 1
        assert len(wakes) == 3
    finally:
        owner.store.close()
        answerer.store.close()
        await app.loop.close()


@pytest.mark.asyncio
async def test_tui_escape_flushes_two_answers_and_dismisses_non_decision(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    project = await asyncio.to_thread(manager.project_registry.create_project, "alpha", "git")
    owner = await asyncio.to_thread(_session, tmp_path, project.project_id)
    answerer = await asyncio.to_thread(_session, tmp_path, project.project_id)
    records = await await_or_fail(
        asyncio.to_thread(_records, owner, project.project_id),
        what="create attention records",
    )
    answer = await asyncio.to_thread(_request, owner, project.project_id, "Status", kind="answer")
    app = await _make_app(tmp_path, owner, owner.store.session_id)
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
            asyncio.to_thread(_wait_for_records, tmp_path, owner.store.session_id, {answer.id}),
            10,
        )
        _press(app, "escape")
        await asyncio.wait_for(
            asyncio.to_thread(_wait_for_records, tmp_path, owner.store.session_id, {r.id for r in records[:2]}),
            10,
        )
        assert AttentionStore(owner.store.session_dir).get(records[2].id).status == "open"
        inbox = ProjectInbox(manager.project_registry, sessions_root=tmp_path / "sessions")
        assert len(inbox.new_ids(project.project_id, session_id=owner.store.session_id)) == 1
    finally:
        owner.store.close()
        answerer.store.close()
        await app.loop.close()
