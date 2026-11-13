from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail, owned_app
from zeta.cli import main as cli_main
from zeta.cli.main import _exec_restart, _restart_argv
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui import layout
from zeta.tui.app import TUIApp
from zeta.tui.composer import FullScreenPromptSession
from zeta.tui.fork_session import ForkStackController


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


@pytest.mark.asyncio
async def test_status_footer_composes_update_and_restart_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    monitor = SimpleNamespace(current=None)
    monkeypatch.setattr(layout, "process_update_monitor", lambda _package: monitor)

    async with owned_app(app):
        neither = "".join(text for _style, text in layout.status_toolbar(app, 120))
        assert "zeta updated · /restart" not in neither
        assert "restart pending" not in neither

        restart_pending = False
        monkeypatch.setattr(
            TUIApp, "restart_pending", property(lambda _app: restart_pending)
        )
        restart_pending = True
        restart_only = "".join(
            text for _style, text in layout.status_toolbar(app, 120)
        )
        assert "restart pending" in restart_only
        assert "zeta updated · /restart" not in restart_only

        monitor.current = ("old", "new")
        both = "".join(text for _style, text in layout.status_toolbar(app, 120))
        assert "restart pending · zeta updated · /restart" in both

        restart_pending = False
        update_only = "".join(
            text for _style, text in layout.status_toolbar(app, 120)
        )
        assert update_only.startswith("zeta updated · /restart")


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
            await await_or_fail(asyncio.sleep(0), what="startup replay")
        current._submit_input("/restart")
        while not current._exit_requested:
            await await_or_fail(asyncio.sleep(0), what="restart exit")

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
    monkeypatch.setattr("zeta.tui.prompt_runtime.run_prompt_toolkit", run_prompt_toolkit)
    monkeypatch.setattr(FullScreenPromptSession, "restore_terminal", restore_terminal)
    # test-contract: seam Patch the CLI TUI application lifecycle seam.
    monkeypatch.setattr(cli_main, "_create_tui_app", lambda *_args: app)
    monkeypatch.setattr(cli_main.os, "execv", execv)
    monkeypatch.setattr(sys, "argv", ["/Users/henry/.local/bin/zeta"])

    assert cli_main.main(
        ["--provider", "codex", "--continue", "--model", "offline"]
    ) == 0
    assert events == ["restore", "close", "exec"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ("main-arm", "fork-waits", "macro", "composer-blocks", "now", "bad-arg"),
)
async def test_restart_controller_covers_fork_and_argument_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    states: dict[TUIApp, tuple[bool, bool, bool]] = {}

    def running(app: TUIApp) -> bool:
        return states[app][0]

    def active(app: TUIApp) -> bool:
        return states[app][1]

    def composer_empty(app: TUIApp) -> bool:
        return states[app][2]

    monkeypatch.setattr(
        TUIApp,
        "restart_has_running_work",
        property(running),
    )
    monkeypatch.setattr(TUIApp, "active", property(active))
    monkeypatch.setattr(TUIApp, "composer_empty", property(composer_empty))
    scheduled: list[object] = []

    def schedule(_delay: float, callback: object) -> object:
        handle = type("Handle", (), {"cancel": lambda self: setattr(self, "cancelled", True)})()
        handle.cancelled = False
        scheduled.append((callback, handle))
        return handle

    def tick() -> None:
        callback, _handle = scheduled.pop(0)
        callback()

    main = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "main", session_id="main"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    fork = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "fork", session_id="fork"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    controller = ForkStackController(
        main, lambda _target: asyncio.sleep(0, result=fork), schedule
    )
    controller._stack.append(fork)
    fork.set_fork_controller(controller)
    states[main] = (
        case in {"main-arm", "fork-waits", "composer-blocks", "now"},
        case == "macro",
        True,
    )
    states[fork] = (False, False, case != "composer-blocks")
    try:
        if case == "main-arm":
            assert await main._slash_commands.dispatch_async(main, "/restart") == (
                "restart pending · waiting for running work · /restart now to cancel it"
            )
            states[main] = (False, False, True)
            tick()
            assert controller.restart_requested and fork._exit_requested
        elif case == "macro":
            assert await main._slash_commands.dispatch_async(main, "/restart") == (
                "restart pending · waiting for running work · /restart now to cancel it"
            )
            states[main] = (False, False, True)
            tick()
            assert controller.restart_requested
        elif case == "fork-waits":
            assert await fork._slash_commands.dispatch_async(fork, "/restart") == (
                "restart pending · waiting for running work · /restart now to cancel it"
            )
            tick()
            assert not controller.restart_requested and not main._exit_requested
            assert await fork._slash_commands.dispatch_async(fork, "/restart cancel") == (
                "restart canceled"
            )
        elif case == "composer-blocks":
            assert await main._slash_commands.dispatch_async(main, "/restart")
            states[main] = (False, False, True)
            tick()
            assert not controller.restart_requested
            states[fork] = (False, False, True)
            tick()
            assert controller.restart_requested and fork._exit_requested
        elif case == "now":
            assert await fork._slash_commands.dispatch_async(fork, "/restart now") == (
                "restarting from the local checkout..."
            )
            assert controller.restart_requested and controller.restart_cancels_work
        else:
            assert "use /restart" in await fork._slash_commands.dispatch_async(
                fork, "/restart later"
            )
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_fork_resume_binds_one_restart_idle_composer_callback(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    session = FullScreenPromptSession(input=DummyInput(), output=DummyOutput())
    try:
        for _ in range(3):
            app._bind_restart_idle_buffer(session)
        assert len(session.app.current_buffer.on_text_changed._handlers) == 1
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_restart_when_idle_polls_background_task_end(tmp_path: Path) -> None:
    scheduled: list[tuple[float, object, object]] = []

    def schedule(delay: float, callback: object) -> object:
        handle = type("Handle", (), {"cancel": lambda self: setattr(self, "cancelled", True)})()
        handle.cancelled = False
        scheduled.append((delay, callback, handle))
        return handle

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", session_id="root-session"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    controller = ForkStackController(app, lambda _target: asyncio.sleep(0), schedule)
    task_id, _pid = await app.loop.tool_registry.background_tasks.start(
        "sleep 10", tmp_path, notify_on_exit=False, owner="macro"
    )
    events: list[str] = []
    app.request_exit = lambda: events.append("restart")
    try:
        assert await app._slash_commands.dispatch_async(app, "/restart") == (
            "restart pending · waiting for running work · /restart now to cancel it"
        )
        assert len(scheduled) == 1
        await app.loop.tool_registry.background_tasks.kill(
            task_id, cause="test", tui_presented=True
        )
        scheduled[0][1]()
        assert events == ["restart"]
        assert controller.restart_requested
        assert controller._restart_idle_handle is None
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_restart_when_idle_cancel_cancels_poll_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled: list[object] = []

    def schedule(_delay: float, _callback: object) -> object:
        handle = type("Handle", (), {"cancel": lambda self: setattr(self, "cancelled", True)})()
        handle.cancelled = False
        scheduled.append(handle)
        return handle

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="codex",
        model="offline",
    )
    controller = ForkStackController(app, lambda _target: asyncio.sleep(0), schedule)
    monkeypatch.setattr(TUIApp, "restart_has_running_work", property(lambda _app: True))
    try:
        assert await app._slash_commands.dispatch_async(app, "/restart")
        assert await app._slash_commands.dispatch_async(app, "/restart cancel") == (
            "restart canceled"
        )
        assert scheduled[0].cancelled
        assert controller._restart_idle_handle is None
    finally:
        await controller.close()
