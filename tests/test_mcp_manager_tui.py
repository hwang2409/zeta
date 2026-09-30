import asyncio
import json
import sys
from io import StringIO

import pytest
from prompt_toolkit.application.current import create_app_session, get_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import input_dialog as real_input_dialog
from prompt_toolkit.shortcuts import radiolist_dialog as real_radiolist_dialog
from prompt_toolkit.shortcuts import yes_no_dialog as real_yes_no_dialog
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.mcp.management import ManagedServer, MCPManagementService
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
        FakeBackend([]), ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop, provider="fake", model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)

    app._submit_input("/mcp")
    assert app.status_card_active
    assert app._mcp_manager_open
    assert app._status_card._lines[0] == "MCP servers"
    await loop.ensure_mcp_servers()
    await loop.close()


def test_manager_renders_every_status_glyph_and_redacts(tmp_path):
    manager = service(tmp_path)
    manager.add("disabled", scope="user", url="https://example.test", enabled=False)
    manager.add("pending", scope="project", command=sys.executable)
    raw = json.loads(manager.path("user").read_text())
    raw["servers"]["disabled"]["headers"] = {"Authorization": "secret"}
    manager.path("user").write_text(json.dumps(raw))
    view = MCPManager(manager)

    rendered = view.render()

    assert "○ disabled" in rendered
    assert "◌ pending" in rendered
    assert "secret" not in rendered
    assert "scope" in rendered and "transport" in rendered and "auth" in rendered


def test_status_glyph_mapping_is_complete():
    base = {"name": "x", "scope": "user", "config": {"transport": "stdio"}}
    assert MCPManager.glyph(ManagedServer(**base, enabled=True, trusted=True, status="connected")) == "●"
    assert MCPManager.glyph(ManagedServer(**base, enabled=False, trusted=True, status="disabled")) == "○"
    assert MCPManager.glyph(ManagedServer(**base, enabled=True, trusted=True, auth="unauthorized")) == "!"
    assert MCPManager.glyph(ManagedServer(**base, enabled=True, trusted=False)) == "◌"
    assert MCPManager.glyph(ManagedServer(**base, enabled=True, trusted=True, status="degraded")) == "×"


@pytest.mark.asyncio
async def test_add_wizard_opens_from_tui_key_path(tmp_path, monkeypatch):
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop, provider="fake", model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )

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
        return StubDialog("" if input_calls == 2 else sys.executable)

    radio_values = iter(("user", "stdio"))
    monkeypatch.setattr("zeta.tui.slash_handlers.mcp_manager.input_dialog", input_factory)
    monkeypatch.setattr(
        "zeta.tui.slash_handlers.mcp_manager.radiolist_dialog",
        lambda **_kwargs: StubDialog(next(radio_values)),
    )

    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
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
            while not app._mcp_manager.service.path("user").exists():
                await asyncio.sleep(0.01)
        assert app._mcp_manager.service.show(
            "draft-from-key-path", scope="user"
        ).name == "draft-from-key-path"
        assert "RuntimeError" not in str(app._mcp_manager.last_result)
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
async def test_add_wizard_escape_does_not_leak_to_parent_manager(tmp_path):
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")
    app = TUIApp(
        loop, provider="fake", model="offline",
        console=Console(file=StringIO(), force_terminal=True),
        zeta_home=tmp_path / "home",
    )

    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
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
            while not app._mcp_wizard_dialog_active:
                await asyncio.sleep(0)
        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(5):
            while not app._mcp_manager_open:
                await asyncio.sleep(0)

        assert not prompt.done()
        assert app._mcp_manager_open
        assert session.layout.current_window is app._status_card_window

        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(5):
            while app._mcp_manager_open:
                await asyncio.sleep(0.01)
        assert session.default_buffer.text == "keep me"
        assert session.default_buffer.cursor_position == 4
        pipe.send_text("!")
        async with asyncio.timeout(5):
            while session.default_buffer.text != "keep! me":
                await asyncio.sleep(0.01)
        prompt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prompt
    await loop.close()


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
        return real_input_dialog(title=title, text=text) if stage == current else StubDialog(value)

    def radio_factory(**kwargs):
        current = "scope" if kwargs["text"] == "Configuration scope:" else "transport"
        value = "user" if current == "scope" else (
            "streamable-http" if http_path else "stdio"
        )
        return real_radiolist_dialog(**kwargs) if stage == current else StubDialog(value)

    def yes_no_factory(**kwargs):
        return real_yes_no_dialog(**kwargs) if stage == "oauth" else StubDialog(False)

    monkeypatch.setattr("zeta.tui.slash_handlers.mcp_manager.input_dialog", input_factory)
    monkeypatch.setattr("zeta.tui.slash_handlers.mcp_manager.radiolist_dialog", radio_factory)
    monkeypatch.setattr("zeta.tui.slash_handlers.mcp_manager.yes_no_dialog", yes_no_factory)

    with create_pipe_input() as pipe, create_app_session(
        input=pipe, output=DummyOutput()
    ):
        task = asyncio.create_task(MCPManagerMixin._collect_mcp_add_draft())
        async with asyncio.timeout(5):
            while not get_app().is_running:
                await asyncio.sleep(0)
        pipe.send_bytes(b"\x1b")
        async with asyncio.timeout(2):
            assert await task is None


def test_add_wizard_persists_env_references_without_secret(tmp_path):
    manager = service(tmp_path)
    view = MCPManager(manager)

    added = view.add(
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
    manager.add("shared", scope="user", url="https://user.example")
    manager.add("shared", scope="project", url="https://project.example")
    view = MCPManager(manager)

    result = await view.dispatch("d")
    assert "unshadow" in str(result)
    assert manager.show("shared").scope == "project"
    await view.dispatch("d")
    assert manager.show("shared").scope == "user"


@pytest.mark.asyncio
async def test_manager_dispatches_enable_remove_test_trust_and_navigation(tmp_path, monkeypatch):
    manager = service(tmp_path)
    manager.add("one", scope="project", command=sys.executable)
    manager.add("two", scope="user", url="https://example.test", enabled=False)
    view = MCPManager(manager)
    called = []

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
    await view.dispatch("t")
    assert called == [("one", "project")]
    await view.dispatch("d")
    assert [item.name for item in manager.list()] == ["two"]
