import asyncio
import fcntl
import inspect
import json
import sys
import threading
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application.current import create_app_session, get_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import input_dialog as real_input_dialog
from prompt_toolkit.shortcuts import radiolist_dialog as real_radiolist_dialog
from prompt_toolkit.shortcuts import yes_no_dialog as real_yes_no_dialog
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.mcp import management as mcp_management
from zeta.mcp.management import ManagedServer, MCPManagementService
from zeta.mcp.oauth_store import MCPOAuthToken
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.cards.mcp_manager import MCPAddDraft, MCPManager
from zeta.tui.slash_handlers.mcp_manager import MCPManagerMixin


def service(tmp_path, mount=None):
    return MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "repo", mount=mount
    )


@pytest.mark.asyncio
async def test_slash_mcp_opens_primary_tui_manager(tmp_path):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)

    app._submit_input("/mcp")
    assert app.status_card_active
    assert app._mcp_manager_open
    assert app._status_card._lines[0][0][1] == "MCP servers"
    await loop.ensure_mcp_servers()
    await loop.close()


@pytest.mark.asyncio
async def test_manager_renders_every_status_glyph_and_redacts(tmp_path):
    manager = service(tmp_path)
    await manager.add(
        "disabled", scope="user", url="https://example.test", enabled=False
    )
    await manager.add("pending", scope="project", command=sys.executable)
    raw = json.loads(manager.path("user").read_text())
    raw["servers"]["disabled"]["headers"] = {"Authorization": "secret"}
    manager.path("user").write_text(json.dumps(raw))
    view = MCPManager(manager)

    rendered = "\n".join(
        "".join(text for _style, text in line) for line in view.render()
    )

    assert "○ disabled" in rendered
    assert "◌ pending" in rendered
    assert "secret" not in rendered
    assert "scope" in rendered and "transport" in rendered and "auth" in rendered


def test_status_glyph_mapping_is_complete():
    base = {"name": "x", "scope": "user", "config": {"transport": "stdio"}}
    assert (
        MCPManager.glyph(
            ManagedServer(**base, enabled=True, trusted=True, status="connected")
        )
        == "●"
    )
    assert (
        MCPManager.glyph(
            ManagedServer(**base, enabled=False, trusted=True, status="disabled")
        )
        == "○"
    )
    assert (
        MCPManager.glyph(
            ManagedServer(**base, enabled=True, trusted=True, auth="unauthorized")
        )
        == "!"
    )
    assert MCPManager.glyph(ManagedServer(**base, enabled=True, trusted=False)) == "◌"
    assert (
        MCPManager.glyph(
            ManagedServer(**base, enabled=True, trusted=True, status="degraded")
        )
        == "×"
    )


@pytest.mark.asyncio
async def test_add_wizard_opens_from_tui_key_path(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )
    assert hasattr(app._mcp_manager.service, "snapshot")

    class StubDialog:
        def __init__(self, value):
            self.value = value

        async def run_async(self):
            return self.value

    input_calls = 0

    def input_factory(*, title, text):
        nonlocal input_calls
        input_calls += 1
        if input_calls == 1:
            return real_input_dialog(title=title, text=text)
        return StubDialog("" if input_calls == 2 else "/bin/false")

    radio_values = iter(("project", "stdio"))
    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.input_dialog", input_factory
    )
    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.radiolist_dialog",
        lambda **_kwargs: StubDialog(next(radio_values)),
    )

    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        session = app._make_session()
        app._active_session = session
        app._install_full_screen_layout(session)
        prompt = asyncio.create_task(session.prompt_async())
        async with asyncio.timeout(5):
            while not get_app().is_running:
                await asyncio.sleep(0)
        session.default_buffer.text = "draft text"
        session.default_buffer.cursor_position = 5
        app.open_mcp_manager()
        assert app._mcp_manager_open
        pipe.send_text("a")
        async with asyncio.timeout(5):
            while not app._mcp_wizard_dialog_active:
                await asyncio.sleep(0)
        pipe.send_text("draft-from-key-path\t\r")
        async with asyncio.timeout(5):
            while not any(
                item.name == "draft-from-key-path"
                for item in app._mcp_manager.service.snapshot.entries
            ):
                await asyncio.sleep(0)
        assert (
            app._mcp_manager.service.show("draft-from-key-path", scope="project").name
            == "draft-from-key-path"
        )
        assert "RuntimeError" not in str(app._mcp_manager.last_result)

        async def send_manager_key(key: str, completed) -> None:
            pipe.send_text(key)
            async with asyncio.timeout(5):
                while not completed():
                    await asyncio.sleep(0)

        def item():
            return app._mcp_manager.service.show("draft-from-key-path")

        await send_manager_key("T", lambda: item().trusted)
        await send_manager_key("T", lambda: not item().trusted)
        await send_manager_key("e", lambda: not item().enabled)
        await send_manager_key("e", lambda: item().enabled)
        await send_manager_key(
            "d", lambda: app._mcp_manager.service.snapshot.entries == ()
        )
        assert not prompt.done()
        assert app._mcp_manager_open
        assert session.layout.current_window is app._status_card_window
        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(5):
            while app._mcp_manager_open:
                await asyncio.sleep(0.01)
        assert session.default_buffer.text == "draft text"
        assert session.default_buffer.cursor_position == 5
        pipe.send_text("!")
        async with asyncio.timeout(5):
            while session.default_buffer.text != "draft! text":
                await asyncio.sleep(0.01)
        prompt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prompt
    await loop.close()


@pytest.mark.asyncio
async def test_real_prompt_lock_contention_stays_responsive_and_renders_busy(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    project = tmp_path / "repo"
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=home, project_dir=project)
    home.mkdir()
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "contended": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "enabled": False,
                    }
                }
            }
        )
    )
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=home,
    )
    refreshed = asyncio.Event()
    original_refresh = app._refresh_mcp_manager

    def record_refresh() -> None:
        original_refresh()
        refreshed.set()

    app._refresh_mcp_manager = record_refresh
    lock_path = app._mcp_manager.service.path("user").with_name(".mcp.json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    prompt = None
    try:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            session = app._make_session()
            app._active_session = session
            app._install_full_screen_layout(session)
            prompt = asyncio.create_task(session.prompt_async())
            async with asyncio.timeout(5):
                while not get_app().is_running:
                    await asyncio.sleep(0)
            app.open_mcp_manager()
            async with asyncio.timeout(10):
                await refreshed.wait()

            mcp_management.fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            contended = threading.Event()
            attempted = []
            original_flock = mcp_management.fcntl.flock
            loop_thread = threading.get_ident()

            def observe_flock(fd, operation):
                if operation & fcntl.LOCK_EX:
                    attempted.append((threading.get_ident(), operation))
                    contended.set()
                    if (
                        threading.get_ident() == loop_thread
                        or not operation & fcntl.LOCK_NB
                    ):
                        raise BlockingIOError
                return original_flock(fd, operation)

            monkeypatch.setattr(mcp_management.fcntl, "flock", observe_flock)
            refreshed.clear()
            pipe.send_text("e")
            async with asyncio.timeout(5):
                assert await asyncio.to_thread(contended.wait, 5)
            assert attempted
            assert all(thread != loop_thread for thread, _operation in attempted)
            assert all(operation & fcntl.LOCK_NB for _thread, operation in attempted)

            responsive = asyncio.Event()
            asyncio.get_running_loop().call_soon(responsive.set)
            async with asyncio.timeout(5):
                await responsive.wait()
            assert not refreshed.is_set()

            async with asyncio.timeout(10):
                await refreshed.wait()
            rendered = "\n".join(
                "".join(text for _style, text in line)
                for line in app._status_card._lines
            )
            assert (
                "mcp error: mcp busy: configuration is being updated; retry" in rendered
            )
            assert not app._mcp_manager.service.show("contended", scope="user").enabled
    finally:
        mcp_management.fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        if prompt is not None:
            prompt.cancel()
            with pytest.raises(asyncio.CancelledError):
                await prompt
        await loop.close()


@pytest.mark.asyncio
async def test_add_wizard_escape_does_not_leak_to_parent_manager(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )

    terminal_entered = asyncio.Event()
    terminal_release = asyncio.Event()
    escape_dispatched = asyncio.Event()

    original_close_status_card = app.close_status_card

    def observed_close_status_card() -> None:
        escape_dispatched.set()
        original_close_status_card()

    monkeypatch.setattr(app, "close_status_card", observed_close_status_card)

    @asynccontextmanager
    async def gated_terminal():
        terminal_entered.set()
        await terminal_release.wait()
        yield

    async def cancelled_collect():
        return None

    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.in_terminal", gated_terminal
    )
    monkeypatch.setattr(app, "_collect_mcp_add_draft", cancelled_collect)

    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        session = app._make_session()
        app._active_session = session
        app._install_full_screen_layout(session)
        prompt = asyncio.create_task(session.prompt_async())
        async with asyncio.timeout(5):
            while not get_app().is_running:
                await asyncio.sleep(0)
        session.default_buffer.text = "keep me"
        session.default_buffer.cursor_position = 4
        app.open_mcp_manager()
        pipe.send_text("a")
        async with asyncio.timeout(5):
            while not terminal_entered.is_set():
                await asyncio.sleep(0)
        assert app._mcp_wizard_active
        assert not app._mcp_wizard_dialog_active

        # Escape arrives while the child is transitioning into the terminal.
        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(5):
            await escape_dispatched.wait()
        assert app._mcp_manager_open
        assert session.layout.current_window is app._status_card_window

        terminal_release.set()
        async with asyncio.timeout(5):
            while app._mcp_wizard_active:
                await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert app._mcp_manager_open
        assert not prompt.done()

        # A manager action is still accepted after the transition window.
        selected = app._mcp_manager.selected
        await app._dispatch_mcp_manager("j")
        assert app._mcp_manager_open
        assert app._mcp_manager.selected == selected
        prompt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prompt
    await loop.close()


@pytest.mark.asyncio
async def test_wizard_cancelled_before_first_run_resets_state(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(loop, provider="codex", model="offline", zeta_home=tmp_path / "home")
    app._mcp_manager_open = True

    async def wizard():
        raise AssertionError("must not start")

    monkeypatch.setattr(app, "_run_mcp_add_wizard", wizard)
    app._status_action("a")
    task = app._mcp_wizard_task
    assert task is not None
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not app._mcp_wizard_active
    assert not app._mcp_wizard_dialog_active
    assert app._mcp_wizard_task is None
    await loop.close()


@pytest.mark.asyncio
async def test_wizard_exception_resets_state(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(loop, provider="codex", model="offline", zeta_home=tmp_path / "home")
    app._mcp_manager_open = True

    async def wizard():
        raise LookupError("boom")

    # test-contract: seam force the real wizard task's exception callback
    monkeypatch.setattr(app, "_run_mcp_add_wizard", wizard)
    app._status_action("a")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not app._mcp_wizard_active
    assert not app._mcp_wizard_dialog_active
    assert app._mcp_wizard_task is None
    await loop.close()


@pytest.mark.asyncio
async def test_second_add_key_ignored_while_wizard_active(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(loop, provider="codex", model="offline", zeta_home=tmp_path / "home")
    app._mcp_manager_open = True
    gate = asyncio.Event()
    started = 0

    async def wizard():
        nonlocal started
        started += 1
        await gate.wait()

    monkeypatch.setattr(app, "_run_mcp_add_wizard", wizard)
    app._status_action("a")
    first = app._mcp_wizard_task
    app._status_action("a")
    assert app._mcp_wizard_task is first
    await asyncio.sleep(0)
    assert started == 1
    gate.set()
    await first
    await loop.close()


@pytest.mark.asyncio
async def test_app_close_cancels_running_wizard(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(loop, provider="codex", model="offline", zeta_home=tmp_path / "home")
    app._mcp_manager_open = True
    cancelled = asyncio.Event()

    async def wizard():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    # test-contract: seam keep the real wizard task open through app shutdown
    monkeypatch.setattr(app, "_run_mcp_add_wizard", wizard)
    app._status_action("a")
    await asyncio.sleep(0)
    await app.close()
    assert cancelled.is_set()
    assert app._mcp_wizard_task is None
    assert not app._mcp_wizard_active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    ("name", "scope", "transport", "environment", "command", "url", "headers", "oauth"),
)
async def test_add_wizard_escape_cancels_each_dialog_stage(monkeypatch, stage):
    class StubDialog:
        def __init__(self, value):
            self.value = value

        async def run_async(self):
            return self.value

    http_path = stage in {"url", "headers", "oauth"}

    def input_factory(*, title, text):
        if text == "Server name:":
            current, value = "name", "server"
        elif text.startswith("Environment references"):
            current, value = "environment", ""
        elif text == "Command and arguments:":
            current, value = "command", sys.executable
        elif text == "Server URL:":
            current, value = "url", "https://example.test"
        else:
            current, value = "headers", ""
        return (
            real_input_dialog(title=title, text=text)
            if stage == current
            else StubDialog(value)
        )

    def radio_factory(**kwargs):
        current = "scope" if kwargs["text"] == "Configuration scope:" else "transport"
        value = (
            "user"
            if current == "scope"
            else ("streamable-http" if http_path else "stdio")
        )
        return (
            real_radiolist_dialog(**kwargs) if stage == current else StubDialog(value)
        )

    def yes_no_factory(**kwargs):
        return real_yes_no_dialog(**kwargs) if stage == "oauth" else StubDialog(False)

    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.input_dialog", input_factory
    )
    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.radiolist_dialog", radio_factory
    )
    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.yes_no_dialog", yes_no_factory
    )

    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        task = asyncio.create_task(MCPManagerMixin._collect_mcp_add_draft())
        async with asyncio.timeout(5):
            while not get_app().is_running:
                await asyncio.sleep(0)
        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(2):
            assert await task is None


@pytest.mark.asyncio
async def test_add_wizard_persists_env_references_without_secret(tmp_path):
    manager = service(tmp_path)
    view = MCPManager(manager)

    added = await view.add(
        MCPAddDraft(
            name="linear",
            scope="project",
            transport="stdio",
            command=sys.executable,
            args=("server.py",),
            env_refs={"API_KEY": "LINEAR_API_KEY"},
        )
    )

    assert added.config["env"] == {"API_KEY": "${LINEAR_API_KEY}"}
    assert "LINEAR_API_KEY" in manager.path("project").read_text()


@pytest.mark.asyncio
async def test_remove_warns_before_unshadowing_user_definition(tmp_path):
    manager = service(tmp_path)
    await manager.add("shared", scope="user", url="https://user.example")
    await manager.add("shared", scope="project", url="https://project.example")
    view = MCPManager(manager)

    result = await view.dispatch("d")
    assert "unshadow" in str(result)
    assert manager.show("shared").scope == "project"
    await view.dispatch("d")
    assert manager.show("shared").scope == "user"


@pytest.mark.asyncio
async def test_manager_dispatches_enable_remove_test_trust_and_navigation(
    tmp_path, monkeypatch
):
    manager = service(tmp_path)
    assert inspect.iscoroutinefunction(manager.add)
    await manager.add("one", scope="project", command=sys.executable)
    await manager.add("two", scope="user", url="https://example.test", enabled=False)
    view = MCPManager(manager)
    called = []
    storage_threads = []
    loop_thread = threading.get_ident()
    original_read_text = Path.read_text

    def gated_read_text(path, *args, **kwargs):
        storage_threads.append(threading.get_ident())
        assert threading.get_ident() != loop_thread
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", gated_read_text)
    view.render()

    async def fake_test(name, *, scope="effective"):
        called.append((name, scope))
        return {"name": name, "tools": 0, "status": "ok"}

    monkeypatch.setattr(manager, "test", fake_test)
    await view.dispatch("a")
    assert view.wizard_active
    await view.dispatch("down")
    assert view.selected == 1
    await view.dispatch("up")
    assert view.selected == 0
    view.selected = next(
        index for index, item in enumerate(view.entries) if item.name == "one"
    )
    await view.dispatch("T")
    assert manager.show("one", scope="project").trusted
    await view.dispatch("T")
    assert not manager.show("one", scope="project").trusted
    await view.dispatch("T")
    assert manager.show("one", scope="project").trusted
    await view.dispatch("t")
    assert called == [("one", "project")]
    await view.dispatch("d")
    assert [item.name for item in manager.snapshot.entries] == ["two"]
    assert storage_threads and all(thread != loop_thread for thread in storage_threads)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "reason", "description"),
    [
        pytest.param("{broken", "malformed", "malformed", id="malformed"),
        pytest.param(
            json.dumps({"access_token": "x" * (70 * 1024)}),
            "oversized",
            "too large",
            id="oversized",
        ),
    ],
)
async def test_real_prompt_mcp_credential_evidence_recovery(
    tmp_path, monkeypatch, record, reason, description
):
    home = tmp_path / "home"
    project = tmp_path / "repo"
    project.mkdir()
    manager = MCPManagementService(home=home, project_dir=project)
    await manager.add(
        "live",
        scope="user",
        url="https://mcp.test",
        oauth=True,
        enabled=False,
    )
    credential_file = home / "mcp-tokens" / "live.json"
    credential_file.parent.mkdir()
    credential_file.write_text(record)
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
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=home,
    )
    expected_text = (
        f"MCP credential file is {description} for 'live': {credential_file.absolute()}. "
        "Move this file aside, then run '/mcp auth live'."
    )
    refreshed = asyncio.Event()
    original_refresh = app._refresh_mcp_manager

    def record_refresh() -> None:
        original_refresh()
        refreshed.set()

    async def fake_authorize(**_kwargs):
        return MCPOAuthToken(
            access_token="recovered-token",
            refresh_token="refresh-token",
            expires_at=None,
            token_type="Bearer",
            scope=None,
            authorization_server="https://auth.test",
            token_endpoint="https://auth.test/token",
            authorization_endpoint="https://auth.test/authorize",
            client_id="client",
            client_secret=None,
            redirect_uri="http://127.0.0.1/callback",
            resource="https://mcp.test",
        )

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    app._refresh_mcp_manager = record_refresh

    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        session = app._make_session()
        app._active_session = session
        app._install_full_screen_layout(session)
        prompt = asyncio.create_task(session.prompt_async())
        async with asyncio.timeout(5):
            while not get_app().is_running:
                await asyncio.sleep(0)
        pipe.send_text("/mcp\r")
        await asyncio.wait_for(refreshed.wait(), timeout=10)
        for key, operation in (("l", "load"), ("o", "delete")):
            refreshed.clear()
            pipe.send_text(key)
            await asyncio.wait_for(refreshed.wait(), timeout=10)
            error = getattr(app._mcp_manager, "credential_store_error", None)
            assert error == {
                "code": "mcp_credential_store",
                "reason": reason,
                "server": "live",
                "credential_file": str(credential_file.absolute()),
                "operation": operation,
                "recovery": "move_aside_then_auth",
                "text": expected_text,
            }
            rendered = "\n".join(
                "".join(text for _style, text in line)
                for line in app._status_card._lines
            )
            assert expected_text in rendered
            assert credential_file.read_text() == record

        moved_evidence = credential_file.with_suffix(".evidence")
        credential_file.rename(moved_evidence)
        refreshed.clear()
        pipe.send_text("l")
        await asyncio.wait_for(refreshed.wait(), timeout=10)
        saved = json.loads(credential_file.read_text())
        assert saved["token"]["access_token"] == "recovered-token"
        assert moved_evidence.read_text() == record
        assert app._mcp_manager.last_result == "authorized live."
        assert app._mcp_manager.credential_store_error is None
        prompt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prompt
    await loop.close()
