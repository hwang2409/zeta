from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from zeta.agent import runner as agent_runner
from zeta.core.abort import AbortGenerationRegistry
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.media.image_policy import ANTHROPIC_IMAGE_POLICY, CODEX_IMAGE_POLICY
from zeta.protocol.types import TextContent, ToolCall
from zeta.providers.ollama import OllamaBackend
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
    _write_agent(
        home / "agents" / "bad-delegation.md",
        "bad-delegation",
        "invalid",
        "body",
        "allow_delegation: nope\n",
    )

    catalog = discover_session_agents(home=home, project_dir=project)

    assert catalog.find("same").source == "project"
    assert catalog.find("same").prompt_suffix == "project body"
    assert catalog.find("unknown").model is None
    assert catalog.find("plain").model is None
    assert catalog.find("plain").tool_names is None
    assert catalog.find("modelled").model == "gpt-5.4"
    assert "bad-delegation" not in catalog.names()
    assert any(
        "allow_delegation must be a boolean" in notice for notice in catalog.notices
    )
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
    legacy = original.to_snapshot()
    for item in legacy:
        item.pop("allow_delegation")
    assert AgentCatalog.from_snapshot(legacy) == original


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
@pytest.mark.parametrize("environment_wins", [True, False])
async def test_model_selected_child_resolves_ollama_endpoint_centrally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_wins: bool
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (home / "settings.toml").write_text(
        'ollama_base_url = "http://settings.example"\n', encoding="utf-8"
    )
    (project / ".zeta").mkdir(parents=True)
    (project / ".zeta" / "settings.toml").write_text(
        'ollama_base_url = "http://project.example"\n', encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    if environment_wins:
        monkeypatch.setenv("ZETA_OLLAMA_BASE_URL", "http://environment.example")
        expected = "http://environment.example"
    else:
        monkeypatch.delenv("ZETA_OLLAMA_BASE_URL", raising=False)
        expected = "http://settings.example"
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(project / "session"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        token_budget=200_000,
    )
    try:
        backend, error = agent_runner.resolve_child_backend(loop, "qwen3:4b")
        assert error is None
        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == expected
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=(
                    json.dumps({"message": {"content": "ok"}, "done": True}) + "\n"
                ).encode(),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend.client = client
            [event async for event in backend.complete([], [])]
        assert payloads[0]["options"]["num_ctx"] == 40_960
    finally:
        await loop.close()
        loop.store.close()


@pytest.mark.asyncio
async def test_model_selected_ollama_child_uses_one_effective_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    payloads: list[dict] = []

    async def consume_child(child_loop: AgentLoop, prompt: str, **kwargs):
        del prompt, kwargs
        assert child_loop.context_assembler.token_budget == 40_960
        assert isinstance(child_loop.backend, OllamaBackend)

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=(
                    json.dumps({"message": {"content": "ok"}, "done": True})
                    + "\n"
                ).encode(),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            child_loop.backend.client = client
            [event async for event in child_loop.backend.complete([], [])]
        return {
            "content": [{"type": "text", "text": "done"}],
            "isError": False,
            "structuredContent": None,
        }

    monkeypatch.setattr(agent_runner, "consume_child", consume_child)
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "parent"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=discover_packaged_agents(),
        token_budget=200_000,
    )
    call = ToolCall(
        "child-call",
        "agent",
        {
            "prompt": "answer",
            "description": "ollama child",
            "model": "qwen3:4b",
            "background": False,
        },
    )
    try:
        result = await loop._run_agent_tool(
            call,
            call.arguments,
            AbortGenerationRegistry().new_generation(),
            None,
        )
        assert result["isError"] is False
        assert payloads[0]["options"]["num_ctx"] == 40_960
    finally:
        await loop.close()
        loop.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent_policy", "child_provider", "expected_policy"),
    [
        (CODEX_IMAGE_POLICY, "anthropic", ANTHROPIC_IMAGE_POLICY),
        (ANTHROPIC_IMAGE_POLICY, "codex", CODEX_IMAGE_POLICY),
    ],
)
async def test_cross_provider_child_uses_its_own_image_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_policy: object,
    child_provider: str,
    expected_policy: object,
) -> None:
    call = ToolCall(
        "agent-cross-provider",
        "agent",
        {
            "prompt": "inspect",
            "description": "cross provider",
            "background": False,
        },
    )
    child_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    child_backend.provider = child_provider
    child_backend.model = "selected-model"
    monkeypatch.setattr(
        agent_runner,
        "resolve_child_backend",
        lambda _loop, _model: (child_backend, None),
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        ConversationStore(tmp_path),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    loop.tool_registry.image_policy = parent_policy
    captured: list[object] = []
    original_clone = loop.tool_registry.clone_for_session

    def capture_clone(store: ConversationStore, **kwargs: object):
        clone = original_clone(store, **kwargs)
        captured.append(clone.image_policy)
        return clone

    monkeypatch.setattr(loop.tool_registry, "clone_for_session", capture_clone)

    [event async for event in loop.run_turn("start")]

    assert captured == [expected_policy]
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
        "model: gpt-5.4\nallow_delegation: false\n",
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
    assert catalog.find("modelled").allow_delegation is False
    assert (
        AgentCatalog.from_snapshot(catalog.to_snapshot())
        .find("modelled")
        .allow_delegation
        is False
    )
    parent_tools = {schema["name"] for schema in parent_backend.calls[0][1]}
    child_tools = {schema["name"] for schema in child_backend.calls[0][1]}
    assert child_tools == parent_tools - {"agent"}
    await loop.close()
