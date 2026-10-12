"""Rendering failures must restore the terminal and preserve resumable state."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.cli import main as cli
from zeta.core.session import SessionManager
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.persistence import DraftPersistence
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.key_bindings import FullScreenPromptSession


@pytest.mark.parametrize(
    ("phase", "ephemeral"),
    [("startup", False), ("redraw", False), ("shutdown", True)],
)
def test_render_failure_exits_without_input_and_restores_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    ephemeral: bool,
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "ephemeral" if ephemeral else home
    monkeypatch.setenv("ZETA_HOME", str(home))
    manager = SessionManager(root)
    opened = manager.create(
        provider="codex", model="gpt-5.6-luna", cwd=tmp_path, auto_project=False,
    )
    store = opened.store
    saved = with_message_origin(
        Message(MessageRole.USER, [TextContent("saved message")]), MessageOrigin.USER,
    )
    store.append_message(saved)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty(), skip_mcp_mount=True),
        provider="codex",
        model="gpt-5.6-luna",
        console=Console(file=StringIO(), force_terminal=False),
        ephemeral_root=root if ephemeral else None,
    )
    output = StringIO()
    broken = phase == "startup"
    failures = 0
    exception_prompts: list[object] = []
    startup_canceled = False
    preferred_width = app._status_card.preferred_width

    def broken_width(width: int) -> int:
        nonlocal failures
        if broken:
            failures += 1
            raise ValueError("injected broken overlay")
        return preferred_width(width)

    def unexpected_exception_prompt(application, loop, context) -> None:
        # Avoid hanging the test in prompt-toolkit's Enter-to-continue prompt.
        exception_prompts.append(context)
        if application.future is not None and not application.future.done():
            application.exit(exception=RuntimeError("unexpected exception prompt"))

    async def stalled_replay() -> None:
        nonlocal startup_canceled
        try:
            await asyncio.Event().wait()
        finally:
            startup_canceled = True

    original_run = cli._run_tui

    async def bounded_run(app, args) -> None:
        await asyncio.wait_for(original_run(app, args), timeout=5)

    monkeypatch.setattr(cli, "create_app", lambda args: app)
    monkeypatch.setattr(cli, "_run_tui", bounded_run)
    monkeypatch.setattr(app._status_card, "preferred_width", broken_width)
    monkeypatch.setattr(Application, "_handle_exception", unexpected_exception_prompt)
    if phase == "startup":
        monkeypatch.setattr(app, "_rebuild_transcript_async", stalled_replay)

    with create_pipe_input() as pipe:
        def make_session() -> FullScreenPromptSession:
            session = FullScreenPromptSession(
                input=pipe,
                output=Vt100_Output(output, lambda: Size(rows=24, columns=80), enable_cpr=False),
                mouse_support=True,
            )

            def before_render(application) -> None:
                # Fail before the debounce can persist this newly typed draft.
                session.default_buffer.text = "unsent draft"

            def after_render(application) -> None:
                nonlocal broken
                if broken:
                    return
                broken = True
                if phase == "shutdown":
                    application.exit()
                else:
                    application.invalidate()

            session.app.before_render += before_render
            session.app.after_render += after_render
            return session

        monkeypatch.setattr(app, "_make_session", make_session)
        assert cli.main(["--provider", "codex"]) == 1

    error = capsys.readouterr().err
    assert error.count("injected broken overlay") == 1
    assert "Press ENTER" not in error
    assert exception_prompts == []
    assert failures == 1
    terminal = output.getvalue()
    assert "\x1b[?1049h" in terminal
    assert terminal.rfind("\x1b[?1049l") > terminal.rfind("\x1b[?1049h")
    assert "\x1b[?1000l" in terminal
    assert "\x1b[?25h" in terminal
    assert app._closed and app.loop._closed
    assert not store._release_lease.alive
    assert app._draft._scheduled is None
    if phase == "startup":
        assert startup_canceled
    if ephemeral:
        assert "resume with:" not in error
        assert not root.exists()
    else:
        assert f"resume with: zeta --resume {store.session_id}" in error
        reopened = manager.open(store.session_id)
        try:
            assert reopened.store.messages() == [saved]
            assert DraftPersistence(reopened.store.session_dir / "draft").load() == "unsent draft"
        finally:
            reopened.store.close()
