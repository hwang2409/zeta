import json
import sys
from io import StringIO

import pytest
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.mcp.management import ManagedServer, MCPManagementService
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.cards.mcp_manager import MCPAddDraft, MCPManager


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
