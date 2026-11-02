from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from tests.support.fake_backend import FakeBackend
from zeta.cli import main as cli_main
from zeta.cli.main import _exec_restart, _restart_argv
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.composer import FullScreenPromptSession


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["--resume", "root-session"]),
        (
            ["--provider", "codex", "--continue", "--model", "gpt-5.4"],
            ["--provider", "codex", "--model", "gpt-5.4", "--resume", "root-session"],
        ),
        (
            ["-c", "--no-yolo", "--no-session", "--token-budget", "120000"],
            ["--no-yolo", "--token-budget", "120000", "--resume", "root-session"],
        ),
        (
            ["--resume", "old-session", "--force-provider", "--verbose"],
            ["--force-provider", "--verbose", "--resume", "root-session"],
        ),
        (
            ["--resume=old-session", "--tools", "bash,read"],
            ["--tools", "bash,read", "--resume", "root-session"],
        ),
        (
            ["--resume", "--provider", "claude"],
            ["--provider", "claude", "--resume", "root-session"],
        ),
    ],
)
def test_restart_argv_replaces_session_selection(
    argv: list[str], expected: list[str]
) -> None:
    assert _restart_argv("/usr/local/bin/zeta", argv, "root-session") == [
        "/usr/local/bin/zeta",
        *expected,
    ]


def test_exec_restart_failure_prints_manual_resume_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_entrypoint: str, _argv: list[str]) -> None:
        raise FileNotFoundError("missing zeta")

    monkeypatch.setattr("zeta.cli.main.os.execv", fail)

    assert _exec_restart("/missing/zeta", ["--provider", "codex"], "session-123") == 1

    assert capsys.readouterr().err.splitlines() == [
        "restart failed: missing zeta",
        "resume manually: zeta --resume session-123",
    ]


def test_restart_slash_closes_once_restores_terminal_then_execs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    store = ConversationStore(home / "sessions", session_id="root-session", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        zeta_home=home,
    )
    session = FullScreenPromptSession(input=DummyInput(), output=DummyOutput())
    app._session = session
    events: list[str] = []
    close_count = 0
    original_close_runtime = app._close_runtime

    async def close_runtime() -> None:
        nonlocal close_count
        close_count += 1
        await original_close_runtime()
        events.append("close")

    async def run_prompt_toolkit(_session: object, current: TUIApp) -> None:
        while current._startup_replay_active:
            await asyncio.sleep(0)
        current._submit_input("/restart")
        while not current._exit_requested:
            await asyncio.sleep(0)

    def restore_terminal(_session: FullScreenPromptSession) -> None:
        events.append("restore")

    def execv(entrypoint: str, argv: list[str]) -> None:
        assert close_count == 1
        assert app._closed is True
        assert app._terminal_restored is True
        assert events == ["restore", "close"]
        assert entrypoint == "/Users/henry/.local/bin/zeta"
        assert argv == [
            entrypoint,
            "--provider",
            "codex",
            "--model",
            "offline",
            "--resume",
            "root-session",
        ]
        events.append("exec")

    monkeypatch.setattr(app, "_close_runtime", close_runtime)
    monkeypatch.setattr(app, "_rebuild_transcript_async", lambda: asyncio.sleep(0, result=True))
    monkeypatch.setattr("zeta.tui.app.run_prompt_toolkit", run_prompt_toolkit)
    monkeypatch.setattr(FullScreenPromptSession, "restore_terminal", restore_terminal)
    monkeypatch.setattr(cli_main, "create_app", lambda _args: app)
    monkeypatch.setattr(cli_main.os, "execv", execv)
    monkeypatch.setattr(sys, "argv", ["/Users/henry/.local/bin/zeta"])

    assert cli_main.main(
        ["--provider", "codex", "--continue", "--model", "offline"]
    ) == 0
    assert events == ["restore", "close", "exec"]
