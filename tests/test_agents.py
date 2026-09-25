from __future__ import annotations

from pathlib import Path

import pytest

from zeta.agent import runner as agent_runner
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.runtime.unattended import build_unattended_loop
from zeta.skills.agent_catalog import (
    AgentCatalog,
    discover_packaged_agents,
    discover_session_agents,
    load_agent,
)
from zeta.skills.catalog import SkillCatalog


def _write_agent(path: Path, name: str, description: str, body: str, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n",
        encoding="utf-8",
    )


def test_agent_discovery_precedence_and_malformed_warnings(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_agent(home / "agents" / "same.md", "same", "home", "home body")
    _write_agent(
        project / ".zeta" / "agents" / "same.md", "same", "project", "project body"
    )
    _write_agent(
        home / "agents" / "explore.md", "explore", "override", "custom explore"
    )
    _write_agent(home / "agents" / "unknown.md", "unknown", "unknown model", "body", "model: made-up\n")
    _write_agent(
        home / "agents" / "plain.md",
        "plain",
        "no optional fields",
        "plain body",
        "color: blue\n",
    )
    _write_agent(
        home / "agents" / "modelled.md",
        "modelled",
        "known model",
        "modelled body",
        "model: gpt-5.4\n",
    )
    (home / "agents" / "bad.md").parent.mkdir(parents=True, exist_ok=True)
    (home / "agents" / "bad.md").write_text("not frontmatter", encoding="utf-8")

    catalog = discover_session_agents(home=home, project_dir=project)

    assert catalog.find("same").source == "project"
    assert catalog.find("same").prompt_suffix == "project body"
    assert catalog.find("unknown").model is None
    assert catalog.find("plain").model is None
    assert catalog.find("plain").tool_names is None
    assert catalog.find("modelled").model == "gpt-5.4"
    assert any("unknown model" in notice for notice in catalog.notices)
    assert any("overrides packaged preset" in notice for notice in catalog.notices)
    assert any("missing YAML frontmatter" in notice for notice in catalog.notices)


def test_agent_discovery_rejects_within_tier_duplicates_and_external_symlinks(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    agents = home / "agents"
    _write_agent(agents / "a.md", "duplicate", "one", "body")
    _write_agent(agents / "b.md", "duplicate", "two", "body")
    with pytest.raises(ValueError, match="duplicate agent name"):
        discover_session_agents(home=home)

    symlink_home = tmp_path / "symlink-home"
    symlink_agents = symlink_home / "agents"
    outside = tmp_path / "outside.md"
    _write_agent(outside, "outside", "outside", "body")
    symlink_agents.mkdir(parents=True)
    (symlink_agents / "link.md").symlink_to(outside)
    catalog = discover_session_agents(home=symlink_home)
    assert "outside" not in catalog.names()
    assert any("outside agents root" in notice for notice in catalog.notices)


def test_agent_discovery_accepts_official_claude_scalar_tools(
    tmp_path: Path,
) -> None:
    path = tmp_path / "home" / "agents" / "reader.md"
    _write_agent(
        path,
        "reader",
        "read-only child",
        "body",
        "tools: Read, Glob, Grep\n",
    )

    catalog = discover_session_agents(home=tmp_path / "home")

    assert catalog.find("reader").tool_names == frozenset({"read"})
    assert sum("unknown tool name" in notice for notice in catalog.notices) == 2


def test_agent_snapshot_omits_body_and_loads_current_file(tmp_path: Path) -> None:
    path = tmp_path / "home" / "agents" / "custom.md"
    _write_agent(path, "custom", "custom", "original body")
    catalog = discover_session_agents(home=tmp_path / "home")
    snapshot = catalog.to_snapshot()
    custom_snapshot = next(item for item in snapshot if item["name"] == "custom")

    assert "prompt_suffix" not in custom_snapshot
    assert "original body" not in str(custom_snapshot)

    path.write_text(
        "---\nname: custom\ndescription: custom\n---\nupdated body\n",
        encoding="utf-8",
    )
    assert load_agent(catalog.find("custom")) == "updated body"
    path.unlink()
    with pytest.raises(ValueError, match="no longer exists"):
        load_agent(AgentCatalog.from_snapshot(snapshot).find("custom"))


def test_agent_snapshot_restores_the_same_set() -> None:
    original = discover_packaged_agents()
    restored = AgentCatalog.from_snapshot(original.to_snapshot())
    assert restored == original
    assert restored.names() == original.names()


def test_session_metadata_restores_agent_snapshot(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_agent(home / "agents" / "custom.md", "custom", "custom", "body")
    catalog = discover_session_agents(home=home)
    opened = SessionManager(home).create(
        provider="fake",
        model="fake",
        cwd=tmp_path,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )
    stored = AgentCatalog.from_snapshot(opened.metadata.agent_catalog)
    (home / "agents" / "custom.md").unlink()
    reopened = SessionManager(home).open(opened.metadata.session_id)
    assert AgentCatalog.from_snapshot(reopened.metadata.agent_catalog) == stored
    opened.store.close()
    reopened.store.close()


@pytest.mark.asyncio
async def test_unattended_runtime_uses_packaged_agents_only(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_agent(home / "agents" / "custom.md", "custom", "custom", "body")
    custom_catalog = discover_session_agents(home=home)
    session = SessionManager(home).create(
        provider="fake",
        model="fake",
        cwd=tmp_path,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=custom_catalog,
        system_prompt="system",
    )
    loop = build_unattended_loop(
        session,
        home=home,
        allow=(),
        backend=FakeBackend([]),
    )
    schema = next(item for item in loop.tool_schemas if item["name"] == "agent")
    assert "custom" not in schema["parameters"]["properties"]["agent_type"]["enum"]
    assert schema["parameters"]["properties"]["agent_type"]["enum"] == (
        discover_packaged_agents().names()
    )
    await loop.close()
    session.store.close()


@pytest.mark.asyncio
async def test_custom_agent_body_and_tool_allowlist_reach_child(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_agent(
        home / "agents" / "reader.md",
        "reader",
        "read-only child",
        "Follow the repository reading rules.",
        "tools: [read]\n",
    )
    catalog = discover_session_agents(home=home)
    call = ToolCall(
        "agent-1",
        "agent",
        {
            "prompt": "inspect",
            "description": "reader",
            "preset": "reader",
            "background": False,
        },
    )
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path / "session")
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )

    [event async for event in loop.run_turn("start")]

    child_prompt = backend.calls[1][0]
    assert "Follow the repository reading rules." in str(child_prompt)
    assert {schema["name"] for schema in backend.calls[1][1]} == {"read"}
    schema = next(item for item in backend.calls[0][1] if item["name"] == "agent")
    assert schema["parameters"]["properties"]["preset"]["enum"] == catalog.names()
    await loop.close()


@pytest.mark.asyncio
async def test_custom_agent_model_selects_the_child_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    _write_agent(
        home / "agents" / "modelled.md",
        "modelled",
        "known model",
        "body",
        "model: gpt-5.4\n",
    )
    catalog = discover_session_agents(home=home)
    call = ToolCall(
        "agent-modelled",
        "agent",
        {
            "prompt": "inspect",
            "description": "modelled",
            "preset": "modelled",
            "background": False,
        },
    )
    parent_backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    child_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    selected: list[object] = []

    def select_child(_loop: object, model: object):
        selected.append(model)
        return child_backend, None

    monkeypatch.setattr(agent_runner, "resolve_child_backend", select_child)
    store = ConversationStore(tmp_path / "session")
    loop = AgentLoop(
        parent_backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )

    [event async for event in loop.run_turn("start")]

    assert selected == ["gpt-5.4"]
    assert child_backend.calls
    await loop.close()
