from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MessageOrigin,
    StreamEvent,
    TextContent,
    ToolCall,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import discover_session_agents
from zeta.tools import ToolRegistry
from zeta.tui import theme
from zeta.tui.render import render_event


def _write_skill(path: Path, name: str, description: str = "focused") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nskill body\n",
        encoding="utf-8",
    )


def _write_agent(
    path: Path, name: str, description: str, body: str = "profile body"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def _agent_enums(schemas: tuple[dict[str, object], ...]) -> tuple[list[str], list[str]]:
    agent = next(schema for schema in schemas if schema["name"] == "agent")
    properties = agent["parameters"]["properties"]
    return properties["agent_type"]["enum"], properties["preset"]["enum"]


def _loop(
    tmp_path: Path,
    backend: FakeBackend | None = None,
    *,
    explicit_schemas: bool = False,
) -> tuple[AgentLoop, Path, Path]:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir(parents=True)
    store = ConversationStore(tmp_path / "session", cwd=project)
    catalog = discover_session_agents(home=home, project_dir=project)
    registry = ToolRegistry(
        project,
        session_store=store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
    )
    schemas = tuple(registry.schemas) if explicit_schemas else None
    parameters = inspect.signature(AgentLoop).parameters
    live_scope = (
        {"live_config_home": home, "live_config_project_dir": project}
        if "live_config_home" in parameters
        else {"skill_home": home, "skill_project_dir": project}
    )
    loop = AgentLoop(
        backend or FakeBackend([]),
        store,
        registry=registry,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=catalog,
        tool_schemas=schemas,
        **live_scope,
    )
    return loop, home, project


async def _refresh(loop: AgentLoop) -> None:
    refresher = getattr(loop, "refresh_live_config", None)
    if refresher is None:
        refresher = loop.refresh_skills
    await refresher()


def _definition(loop: AgentLoop):
    reference = getattr(loop.tool_registry, "definition_reference", None)
    return (
        reference("agent")
        if reference is not None
        else loop.tool_registry._tools.get("agent")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit_schemas", [False, True], ids=["registry", "explicit"]
)
async def test_live_config_adds_profile_to_next_agent_schema_and_spawn(
    tmp_path: Path, explicit_schemas: bool
) -> None:
    call = ToolCall(
        "specialist-1",
        "agent",
        {
            "prompt": "inspect",
            "description": "specialist task",
            "agent_type": "specialist",
        },
    )
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("child done")])]
    )
    loop, home, _project = _loop(tmp_path, backend, explicit_schemas=explicit_schemas)
    _write_agent(
        home / "agents" / "specialist.md",
        "specialist",
        "focused specialist",
        "committed specialist body",
    )

    _ = [event async for event in loop.run_turn("start", origin=MessageOrigin.USER)]

    assert "specialist" in _agent_enums(backend.calls[0][1])[0]
    assert "specialist" in _agent_enums(backend.calls[0][1])[1]
    assert len(backend.calls) >= 2, "live profile failed execution validation"
    child_prompt = backend.calls[1][0][0]
    assert any(
        isinstance(block, TextContent) and "committed specialist body" in block.text
        for block in child_prompt.content
    )
    await loop.close()


@pytest.mark.asyncio
async def test_running_child_refreshes_delegation_catalog_without_changing_selected_profile(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path)
    profile = home / "agents" / "specialist.md"
    _write_agent(profile, "specialist", "focused specialist", "first body")
    await _refresh(loop)
    assert "specialist" in loop.tool_registry.agent_catalog.names(), (
        "new profile was not adopted"
    )
    selected = loop.tool_registry.agent_catalog.find("specialist")
    delegate_call = ToolCall(
        "delegate-1",
        "agent",
        {
            "prompt": "delegate",
            "description": "delegate task",
            "agent_type": "delegate",
        },
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[delegate_call]),
            ScriptedTurn([TextContent("delegate done")]),
        ]
    )
    project = Path(loop.store.cwd)
    child_store = ConversationStore(tmp_path / "child-session", cwd=project)
    child_registry = ToolRegistry(
        project,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=loop.tool_registry.agent_catalog,
    )
    child = AgentLoop(
        backend,
        child_store,
        registry=child_registry,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=loop.tool_registry.agent_catalog,
        live_config_home=home,
        live_config_project_dir=project,
        system_prompt=selected.prompt_suffix,
    )

    _write_agent(profile, "specialist", "focused specialist", "second body")
    _write_agent(home / "agents" / "delegate.md", "delegate", "delegated specialist")
    _ = [
        event
        async for event in child.run_turn("next", origin=MessageOrigin.USER)
    ]

    assert selected.prompt_suffix == "first body"
    child_prompt = child._runtime_prompt.system_prompt
    assert any(
        isinstance(block, TextContent) and block.text == "first body"
        for block in child_prompt.content
    )
    assert "delegate" in _agent_enums(backend.calls[0][1])[0]
    assert len(backend.calls) >= 2, "running child rejected the new delegate profile"
    await child.close()
    await loop.close()


@pytest.mark.asyncio
async def test_live_config_invalid_profile_keeps_last_good_and_commits_independent_profile(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path)
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    profile = home / "agents" / "specialist.md"
    _write_agent(profile, "specialist", "focused specialist", "last good body")
    await _refresh(loop)
    assert "specialist" in loop.tool_registry.agent_catalog.names(), (
        "new profile was not adopted"
    )
    events.clear()

    profile.write_text("not a profile", encoding="utf-8")
    _write_agent(home / "agents" / "independent.md", "independent", "valid profile")
    await _refresh(loop)
    await _refresh(loop)

    assert loop.tool_registry.agent_catalog.find("specialist").prompt_suffix == (
        "last good body"
    )
    assert "independent" in loop.tool_registry.agent_catalog.names()
    assert len(events) == 1
    assert events[0].type.value == "config_changed"
    assert events[0].data["diagnostics"], "profile failure was not visible"
    await loop.close()


@pytest.mark.asyncio
async def test_agent_semantic_edit_emits_one_config_change(tmp_path: Path) -> None:
    loop, home, _project = _loop(tmp_path)
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    profile = home / "agents" / "specialist.md"
    _write_agent(profile, "specialist", "first description")
    await _refresh(loop)
    events.clear()

    _write_agent(profile, "specialist", "second description")
    await _refresh(loop)
    await _refresh(loop)

    assert len(events) == 1
    assert events[0].data["changes"]["agents"]["updated"] == ["specialist"]
    assert "second description" in next(
        schema["description"]
        for schema in loop.tool_registry.schemas
        if schema["name"] == "agent"
    )
    await loop.close()


@pytest.mark.asyncio
async def test_profile_semantic_edit_then_touch_is_noop(tmp_path: Path) -> None:
    loop, home, _project = _loop(tmp_path, explicit_schemas=True)
    profile = home / "agents" / "specialist.md"
    _write_agent(profile, "specialist", "focused specialist")
    await _refresh(loop)
    assert "specialist" in loop.tool_registry.agent_catalog.names(), (
        "new profile was not adopted"
    )
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    initial_schema = json.dumps(loop.tool_schemas, sort_keys=True).encode()
    _write_agent(profile, "specialist", "updated specialist")
    await _refresh(loop)
    assert len(events) == 1
    assert json.dumps(loop.tool_schemas, sort_keys=True).encode() != initial_schema
    events.clear()
    refs = (
        loop.tool_registry.agent_catalog,
        loop.tool_registry.skill_catalog,
        _definition(loop),
        loop.tool_schemas,
        loop._runtime_prompt,
    )
    before = json.dumps(loop.tool_schemas, sort_keys=True).encode()

    stat = profile.stat()
    os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    await _refresh(loop)

    assert refs == (
        loop.tool_registry.agent_catalog,
        loop.tool_registry.skill_catalog,
        _definition(loop),
        loop.tool_schemas,
        loop._runtime_prompt,
    )
    assert all(
        old is new
        for old, new in zip(
            refs,
            (
                loop.tool_registry.agent_catalog,
                loop.tool_registry.skill_catalog,
                _definition(loop),
                loop.tool_schemas,
                loop._runtime_prompt,
            ),
            strict=True,
        )
    )
    assert json.dumps(loop.tool_schemas, sort_keys=True).encode() == before
    assert events == []
    await loop.close()


@pytest.mark.asyncio
async def test_live_config_rejects_candidate_when_prompt_changes_in_flight(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path, explicit_schemas=True)
    profile = home / "agents" / "specialist.md"
    _write_agent(profile, "specialist", "focused specialist")
    entered = threading.Event()
    release = threading.Event()
    watcher = getattr(loop, "_live_config_watcher", None)
    if watcher is not None:
        original = watcher.prepare

        def paused(base, *, force: bool = False):
            candidate = original(base, force=force)
            entered.set()
            release.wait()
            return candidate

        watcher.prepare = paused
    else:
        watcher = loop._skill_watcher
        assert watcher is not None
        original = watcher.scan

        def paused(*, force: bool = False):
            candidate = original(force=force)
            entered.set()
            release.wait()
            return candidate

        watcher.scan = paused
    task = asyncio.create_task(_refresh(loop))
    assert await asyncio.to_thread(entered.wait, 2)
    before = (
        loop.tool_registry.agent_catalog,
        _definition(loop),
        loop.tool_schemas,
        loop._runtime_prompt,
    )
    loop._set_runtime_prompt(loop._runtime_prompt)
    release.set()
    await task

    assert before == (
        loop.tool_registry.agent_catalog,
        _definition(loop),
        loop.tool_schemas,
        loop._runtime_prompt,
    )
    with pytest.raises(ValueError, match="unknown agent preset"):
        loop.tool_registry.agent_catalog.find("specialist")
    await _refresh(loop)
    assert "specialist" in loop.tool_registry.agent_catalog.names(), (
        "stale discard did not retry the complete live candidate"
    )
    await loop.close()


@pytest.mark.asyncio
async def test_live_config_initial_manifest_is_not_read_during_loop_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zeta.config.live as live_module

    event_thread = threading.get_ident()
    observed: list[int] = []
    original = live_module._manifest

    def record_manifest(scope):
        observed.append(threading.get_ident())
        return original(scope)

    # test-contract: seam thread identity is observable only at the manifest seam
    monkeypatch.setattr(live_module, "_manifest", record_manifest)
    loop, _home, _project = _loop(tmp_path)

    assert observed == [], "loop construction read the live-config filesystem"
    await loop.refresh_live_config(force=True)
    assert observed and all(thread != event_thread for thread in observed)
    await loop.close()


@pytest.mark.asyncio
async def test_live_slash_projection_is_prepared_off_loop(tmp_path: Path) -> None:
    loop, home, project = _loop(tmp_path)
    registry = create_slash_registry(
        zeta_home=home,
        project_dir=project,
        skill_catalog=SkillCatalog.empty(),
    )
    event_thread = threading.get_ident()
    prepared_on: list[int] = []
    committed_on: list[int] = []

    class SurfaceListener:
        def __call__(self, catalog, _agents) -> None:
            committed_on.append(threading.get_ident())
            registry.register_skills(catalog)

        def prepare_live_config(self, catalog, agents):
            prepared_on.append(threading.get_ident())
            return registry.prepare_live_config(catalog, agents)

        def commit_live_config(self, projection) -> None:
            committed_on.append(threading.get_ident())
            registry.commit_live_config(projection)

    loop.add_live_config_listener(SurfaceListener())
    _write_skill(home / "skills" / "new-skill.md", "new-skill")

    await loop.refresh_live_config()

    assert prepared_on and prepared_on[0] != event_thread
    assert committed_on == [event_thread]
    assert [skill.name for skill in registry.skill_entries] == ["new-skill"]
    await loop.close()


@pytest.mark.asyncio
async def test_live_config_prepares_schema_and_prompt_off_loop(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path)
    _write_agent(home / "agents" / "specialist.md", "specialist", "focused")
    watcher = getattr(loop, "_live_config_watcher", None)
    event_thread = threading.get_ident()
    prepare_threads: list[int] = []
    commit_threads: list[int] = []
    if watcher is not None:
        original_prepare = watcher.prepare
        original_commit = watcher.commit

        def prepare(base, *, force: bool = False):
            prepare_threads.append(threading.get_ident())
            return original_prepare(base, force=force)

        def commit(candidate):
            commit_threads.append(threading.get_ident())
            return original_commit(candidate)

        watcher.prepare = prepare
        watcher.commit = commit
    else:
        watcher = loop._skill_watcher
        assert watcher is not None
        original_scan = watcher.scan
        original_commit = watcher.commit

        def scan(*, force: bool = False):
            prepare_threads.append(threading.get_ident())
            return original_scan(force=force)

        def commit(candidate):
            commit_threads.append(threading.get_ident())
            return original_commit(candidate)

        watcher.scan = scan
        watcher.commit = commit
    await _refresh(loop)

    assert prepare_threads == [prepare_threads[0]]
    assert prepare_threads[0] != event_thread
    if commit_threads:
        assert commit_threads == [event_thread]
    assert "specialist" in loop.tool_registry.agent_catalog.names(), (
        "off-loop preparation omitted the agent catalog"
    )
    await loop.close()


@pytest.mark.asyncio
async def test_config_changed_event_renders_through_real_render_event(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path)
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    _write_agent(home / "agents" / "specialist.md", "specialist", "focused")
    await _refresh(loop)
    assert len(events) == 1, "integrated refresh did not publish one event"

    rendered = render_event(events[0])
    console = Console(width=80, record=True)
    console.print(rendered)

    assert isinstance(rendered, Text)
    assert rendered.style == theme.NOTICE
    assert console.export_text().strip() == "agent profiles changed: added specialist"
    await loop.close()


@pytest.mark.asyncio
async def test_many_live_changes_stay_bounded_on_tui_and_serve_paths(
    tmp_path: Path,
) -> None:
    from zeta.server.event_contract import config_changed_wire_fields
    from zeta.server.protocol import MAX_FRAME_BYTES, FrameCodec

    loop, home, _project = _loop(tmp_path)
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    for index in range(132):
        name = f"item-{index}-" + "x" * 8_000
        _write_skill(home / "skills" / f"skill-{index}.md", name)
        _write_agent(home / "agents" / f"agent-{index}.md", name, "focused")

    await loop.refresh_live_config()

    assert len(events) == 1
    payload = events[0].data
    skill_changes = payload["changes"]["skills"]
    agent_changes = payload["changes"]["agents"]
    assert len(skill_changes["added"]) == 26, "skill changes were not bounded"
    assert len(agent_changes["added"]) == 26, "agent changes were not bounded"
    assert skill_changes["added_count"] == 132
    assert agent_changes["added_count"] == 132
    assert skill_changes["added"][-1] == "and 107 more"
    assert agent_changes["added"][-1] == "and 107 more"

    rendered = render_event(events[0])
    assert isinstance(rendered, Text)
    assert rendered.plain == payload["message"]
    assert rendered.plain.count("and 107 more") == 2

    encoded = FrameCodec().notification(
        "config_changed", "session-1", **config_changed_wire_fields(events[0])
    )
    frame = json.loads(encoded)
    assert len(encoded) < MAX_FRAME_BYTES
    assert frame["params"]["event"] == "config_changed"
    assert frame["params"]["data"] == payload
    await loop.close()


@pytest.mark.asyncio
async def test_added_profile_is_detected_with_unchanged_parent_stat(
    tmp_path: Path,
) -> None:
    loop, home, _project = _loop(tmp_path)
    root = home / "agents"
    root.mkdir(parents=True)
    stat = root.stat()
    _write_agent(root / "specialist.md", "specialist", "focused specialist")
    os.utime(root, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    await _refresh(loop)

    assert "specialist" in _agent_enums(tuple(loop.tool_registry.schemas))[0]
    await loop.close()


@pytest.mark.asyncio
async def test_config_changed_projection_is_byte_exact(tmp_path: Path) -> None:
    from zeta.server import event_contract
    from zeta.server.protocol import FrameCodec

    loop, home, _project = _loop(tmp_path)
    events: list[StreamEvent] = []
    loop.set_background_event_sink(events.append)
    _write_agent(home / "agents" / "specialist.md", "specialist", "focused")

    await _refresh(loop)

    assert len(events) == 1, "integrated refresh did not publish one event"
    projector = getattr(
        event_contract,
        "config_changed_wire_fields",
        lambda event: {"data": dict(event.data)},
    )
    encoded = FrameCodec().notification(
        "config_changed", "session-1", **projector(events[0])
    )
    expected = (
        '{"jsonrpc":"2.0","method":"event","params":'
        '{"event":"config_changed","session_id":"session-1","data":'
        + json.dumps(events[0].data, separators=(",", ":"))
        + "}}\n"
    ).encode()
    assert encoded == expected
    await loop.close()
