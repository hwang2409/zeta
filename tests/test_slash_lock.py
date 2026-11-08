"""The session slash lock: worker /status, visible waiting, deferred cancel."""

from __future__ import annotations

import ast
import asyncio
import threading
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from rich.console import Console

import zeta.core.slash as core_slash
import zeta.runtime.status as runtime_status
from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail, owned_app
from tests.support.storage_homes import initialize_storage_home
from zeta.core.checkpoints import active_branch
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

SRC = Path(__file__).parents[1] / "src" / "zeta"


class _Printed:
    """Record system messages and let a test await one by content."""

    def __init__(self, app: TUIApp) -> None:
        self.lines: list[str] = []
        self._changed = asyncio.Event()
        original = app._print_system

        def record(output: str) -> None:
            original(output)
            self.lines.append(output)
            self._changed.set()

        app._print_system = record  # type: ignore[method-assign]

    def count(self, text: str) -> int:
        return sum(text in line for line in self.lines)

    def index(self, text: str) -> int:
        return next(i for i, line in enumerate(self.lines) if text in line)

    async def wait_for(self, text: str, times: int = 1) -> None:
        async def seen() -> None:
            while self.count(text) < times:
                self._changed.clear()
                await self._changed.wait()

        await await_or_fail(seen(), what=f"system message {text!r}")


def _app(tmp_path: Path) -> TUIApp:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    return TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )


@pytest.mark.asyncio
async def test_tui_status_worker_waits_visibly_publishes_then_cancels(
    tmp_path, monkeypatch
):
    loop_thread = threading.get_ident()
    event_loop = asyncio.get_running_loop()
    started = event_loop.create_future()
    ran_on_loop = event_loop.create_future()
    release = threading.Event()

    def gated(entries):
        if threading.get_ident() == loop_thread:
            if not ran_on_loop.done():
                ran_on_loop.set_result(None)
        else:
            event_loop.call_soon_threadsafe(
                lambda: started.done() or started.set_result(None)
            )
            release.wait(30)
        return active_branch(entries)

    # The transcript-sized /status work calls active_branch on both bases.
    monkeypatch.setattr(core_slash, "active_branch", gated, raising=False)
    monkeypatch.setattr(runtime_status, "active_branch", gated, raising=False)
    app = _app(tmp_path)
    async with owned_app(app) as owner:
        owner.release_with(release.set)
        printed = _Printed(app)
        pipeline = app._submissions
        status = pipeline.submit("/status")
        await await_or_fail(
            asyncio.wait({started, ran_on_loop}, return_when=asyncio.FIRST_COMPLETED),
            what="/status transcript work",
        )
        assert not ran_on_loop.done(), "/status transcript work ran on the event loop"

        pipeline.submit("/vim off")
        await printed.wait_for("waiting for /status to finish")
        assert app.vim_mode is True, "/vim entered while /status held the slash lock"

        task = pipeline._slash_tasks[status.id]
        pipeline.abort(status.id)
        await printed.wait_for("waiting for /status to finish", times=2)
        assert not task.done(), "cancel did not wait for the started worker"
        assert printed.count("session_id:") == 0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(asyncio.shield(task), what="cancelled /status")
        assert printed.count("session_id:") == 1, "publish did not run before cancel"
        await printed.wait_for("vim mode: off")
        assert printed.index("session_id:") < printed.index("vim mode: off")
        assert app.vim_mode is False
        assert app.slash_lock.holder is None


class _Session:
    def __init__(self, lock) -> None:
        self.slash_lock = lock


def _registry(*commands):
    from zeta.core.commands.slash_registry import SlashCommandRegistry

    registry = SlashCommandRegistry()
    for command in commands:
        registry.register(command)
    return registry


def _status_snapshot():
    return core_slash.SlashStatusSnapshot(
        core_slash.SlashStatus("s", "codex", "offline", 0, 0, None, 0, ()),
        (),
        _module_counter,
    )


def _module_counter(_message) -> int:
    return 0


@pytest.mark.asyncio
async def test_worker_prepare_rejects_live_state_before_submission():
    """Interface check (head-only): only frozen inputs reach the worker."""

    from zeta.core.commands.slash_registry import SlashCommand, SlashLock

    ran: list[object] = []
    live: dict[str, object] = {}
    local = 1
    for value in (live, [1], (1, {"a": 1}), lambda _m: local, object()):
        command = SlashCommand.worker(
            "probe", lambda _s, _a, value=value: value, ran.append, None
        )
        with pytest.raises(TypeError, match="prepare returned unsupported/live state"):
            await _registry(command).dispatch_async(_Session(SlashLock()), "/probe")
    assert ran == []

    accepted = SlashCommand.worker(
        "probe",
        lambda _s, _a: (Path("/x"), _status_snapshot(), b"x", None),
        lambda inputs: "ok",
        None,
    )
    assert await _registry(accepted).dispatch_async(_Session(SlashLock()), "/probe") == "ok"


@pytest.mark.asyncio
async def test_async_worker_and_sync_awaitable_handlers_are_rejected():
    """Interface check (head-only)."""

    from zeta.core.commands.slash_registry import SlashCommand, SlashLock

    async def run(_inputs):
        return "never"

    with pytest.raises(TypeError, match="worker steps must be synchronous"):
        SlashCommand.worker("probe", None, run, None)
    with pytest.raises(TypeError, match="use SlashCommand.worker"):
        SlashCommand("probe", object())  # type: ignore[arg-type]

    async def coroutine():
        return "never"

    pending = coroutine()
    command = SlashCommand.loop_safe(
        "probe", lambda _s, _a: pending, reason="review probe"
    )
    with pytest.raises(TypeError, match="make it async def"):
        await _registry(command).dispatch_async(_Session(SlashLock()), "/probe")
    assert pending.cr_frame is None, "the rejected coroutine was not closed"


@pytest.mark.asyncio
async def test_nested_dispatch_fails_before_waiting_for_the_lock():
    """Interface check (head-only): no slash command dispatches another."""

    from zeta.core.commands.slash_registry import SlashCommand, SlashLock

    registry = _registry()

    async def outer(session, _args):
        return await registry.dispatch_async(session, "/inner")

    registry.register(SlashCommand.loop_safe("outer", outer, reason="probe"))
    registry.register(
        SlashCommand.loop_safe("inner", lambda _s, _a: "inner", reason="probe")
    )
    with pytest.raises(
        RuntimeError, match="/outer cannot dispatch /inner while holding the slash lock"
    ):
        await await_or_fail(
            registry.dispatch_async(_Session(SlashLock()), "/outer"),
            what="nested dispatch",
        )


def test_slash_lock_is_taken_only_by_registry_dispatch():
    """Lock order (head-only): one acquisition seam, on an event-loop task."""

    callers = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "hold"
                and "lock" in ast.unparse(node.func.value)
            ):
                callers.append(path.relative_to(SRC).as_posix())
    assert callers == ["core/commands/slash_registry.py"]


@pytest.mark.asyncio
async def test_tui_mcp_auth_cancel_is_immediate_and_frees_the_lock(
    tmp_path, monkeypatch
):
    from zeta.mcp import oauth
    from zeta.mcp.management import MCPManagementService

    home = tmp_path / "home"
    project = tmp_path / "repo"
    project.mkdir()
    initialize_storage_home(home)
    await MCPManagementService(home=home, project_dir=project).add(
        "live", scope="user", url="https://mcp.test", oauth=True, enabled=False
    )
    event_loop = asyncio.get_running_loop()
    browser = event_loop.create_future()

    async def resource(*_args, **_kwargs):
        return oauth.ProtectedResourceMetadata("https://mcp.test", "https://auth.test")

    async def auth_server(*_args, **_kwargs):
        return oauth.AuthServerMetadata(
            "https://auth.test",
            "https://auth.test/authorize",
            "https://auth.test/token",
            None,
            (),
        )

    async def register(*_args, **_kwargs):
        return "client", None

    def open_browser(url: str) -> None:
        # The callback gate: the browser never redirects back.
        event_loop.call_soon_threadsafe(
            lambda: browser.done() or browser.set_result(url)
        )

    monkeypatch.setattr(oauth, "discover_protected_resource", resource)
    monkeypatch.setattr(oauth, "discover_auth_server", auth_server)
    monkeypatch.setattr(oauth, "register_client", register)
    monkeypatch.setattr(oauth, "_default_browser_opener", open_browser)
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        zeta_home=home,
    )
    async with owned_app(app):
        printed = _Printed(app)
        pipeline = app._submissions
        auth = pipeline.submit("/mcp auth live")
        url = await await_or_fail(browser, what="/mcp auth browser open")
        redirect = parse_qs(urlparse(url).query)["redirect_uri"][0]
        port = urlparse(redirect).port

        pipeline.submit("/vim off")
        waiting = asyncio.ensure_future(
            printed.wait_for("waiting for /mcp auth to finish")
        )
        done, _ = await asyncio.wait([waiting], timeout=10)
        if not done:
            waiting.cancel()
        assert done, "the second command showed no 'waiting for /mcp auth to finish'"
        assert app.vim_mode is True

        task = pipeline._slash_tasks[auth.id]
        pipeline.abort(auth.id)
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(asyncio.shield(task), what="cancelled /mcp auth")
        assert not browser.cancelled()
        await printed.wait_for("vim mode: off")
        assert app.vim_mode is False
        assert app.slash_lock.holder is None
        with pytest.raises(OSError):
            _reader, writer = await await_or_fail(
                asyncio.open_connection("127.0.0.1", port), what="closed listener"
            )
            writer.close()
        assert not (home / "mcp-tokens" / "live.json").exists()
