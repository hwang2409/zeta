from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.project_context import ProjectContext, load_project_context
from zeta.core.session import SessionManager
from zeta.core.session_execution import SessionOwnedError
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.project_registry import Project, ProjectRegistry
from zeta.protocol.types import MessageOrigin, TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.runtime.project_registry_state import ProjectRegistryState
from zeta.runtime.prompt_resume import ResumedPrompt, resume_prompt
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.storage_format import advance
from zeta.tools import ToolRegistry

load_project_context = partial(
    load_project_context, registry=ProjectRegistryState(None)
)


class _UnavailableRegistry(ProjectRegistryState):
    def __bool__(self) -> bool:
        return False


def _manager_with_prompt(
    home: Path,
    cwd: Path,
    *,
    system_prompt: str,
    prompt_recipe: str | None,
    project_memory_offset: int | None = None,
    project_memory_length: int | None = None,
    project_memory_digest: str | None = None,
    prompt_components: dict[str, dict[str, int | str]] | None = None,
) -> tuple[SessionManager, str]:
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex",
        model="offline",
        cwd=cwd,
        system_prompt=system_prompt,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        prompt_recipe=prompt_recipe,
        project_memory_offset=project_memory_offset,
        project_memory_length=project_memory_length,
        project_memory_digest=project_memory_digest,
        prompt_components=prompt_components,
        auto_project=False,
    )
    opened.store.close()
    return manager, opened.metadata.session_id


def _write_skill(path: Path, name: str = "new-skill") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: newly installed\n---\n\nskill body\n",
        encoding="utf-8",
    )


def _write_agent(path: Path, name: str = "new-agent") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: newly installed\n---\nagent body\n",
        encoding="utf-8",
    )


def _resume(
    manager: SessionManager,
    session_id: str,
    home: Path,
    cwd: Path,
    context_loader: Callable[..., ProjectContext],
    *,
    live_stores: list[object] | None = None,
) -> ResumedPrompt:
    opened = manager.open(session_id)
    try:
        result = resume_prompt(
            opened.metadata,
            manager=manager,
            store=opened.store,
            home=home,
            repo_root=cwd,
            inbox_enabled=False,
            context_loader=context_loader,
            registry=manager.project_registry_state,
        )
    except BaseException:
        opened.store.close()
        raise
    if live_stores is None:
        opened.store.close()
    else:
        live_stores.append(opened.store)
    return result


def _staggered_resumes(
    manager: SessionManager,
    session_id: str,
    home: Path,
    cwd: Path,
    *,
    context_loader: Callable[..., ProjectContext],
    monkeypatch: pytest.MonkeyPatch,
) -> list[ResumedPrompt]:
    persisted = threading.Event()
    release_first = threading.Event()
    original_adopt = __import__(
        "zeta.runtime.prompt_resume", fromlist=["_adopted_context"]
    )._adopted_context

    def pause_after_persist(
        metadata: object, notices: tuple[str, ...]
    ) -> ProjectContext:
        if threading.current_thread().name == "first-resumer":
            persisted.set()
            assert release_first.wait(timeout=5)
        return original_adopt(metadata, notices)

    monkeypatch.setattr(
        "zeta.runtime.prompt_resume._adopted_context", pause_after_persist
    )
    results: list[ResumedPrompt] = []
    live_stores: list[object] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                _resume(
                    manager,
                    session_id,
                    home,
                    cwd,
                    context_loader,
                    live_stores=live_stores,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - report thread failures
            errors.append(exc)

    first = threading.Thread(target=run, name="first-resumer")
    second = threading.Thread(target=run, name="second-resumer")
    first.start()
    assert persisted.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_first.set()
    first.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(results) == 2
    for store in live_stores:
        store.close()
    return results


def _numbered_default_contexts(
    cwd: Path,
) -> tuple[Callable[..., ProjectContext], list[int]]:
    calls: list[int] = []

    def load(**kwargs: object) -> ProjectContext:
        del kwargs
        calls.append(len(calls) + 1)
        return ProjectContext(
            f"default fallback {calls[-1]}",
            (cwd / "context.md",),
            prompt_recipe="default",
        )

    return load, calls


def _assert_one_persisted_prompt(
    manager: SessionManager,
    session_id: str,
    results: list[ResumedPrompt],
) -> None:
    saved = manager.read_metadata(session_id)
    assert [result.context.system_prompt for result in results] == [
        saved.system_prompt,
        saved.system_prompt,
    ]
    assert "prompt_composition_epoch" not in saved.to_storage_dict()
    assert "prompt_composition_owner_pid" not in saved.to_storage_dict()


@pytest.mark.asyncio
async def test_conservative_resume_discovers_new_skill(tmp_path: Path) -> None:
    home = tmp_path / "home"
    original_prompt = "unknown recipe prompt bytes"
    manager, session_id = await asyncio.to_thread(
        lambda: _manager_with_prompt(
            home,
            tmp_path,
            system_prompt=original_prompt,
            prompt_recipe=None,
        )
    )
    _write_skill(home / "skills" / "new-skill" / "SKILL.md")

    resumed = await asyncio.to_thread(
        lambda: _resume(
            manager,
            session_id,
            home,
            tmp_path,
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError(f"unexpected default rebuild: {kwargs}")
            ),
        )
    )
    registry = ToolRegistry(tmp_path, skill_catalog=resumed.skill_catalog)
    loaded = await registry.execute(
        ToolCall("load-new-skill", "skill", {"name": "new-skill"}),
        _skip_approval=True,
    )

    assert loaded["isError"] is False
    assert loaded["content"][0]["text"].startswith("skill body")
    assert resumed.context.system_prompt == original_prompt
    assert manager.read_metadata(session_id).system_prompt == original_prompt


@pytest.mark.asyncio
async def test_child_of_conservative_resumed_session_sees_new_skill(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    manager, session_id = await asyncio.to_thread(
        lambda: _manager_with_prompt(
            home,
            tmp_path,
            system_prompt="unknown recipe prompt bytes",
            prompt_recipe=None,
        )
    )
    _write_skill(home / "skills" / "new-skill" / "SKILL.md")
    resumed = await asyncio.to_thread(
        lambda: _resume(manager, session_id, home, tmp_path, lambda **kwargs: None)
    )
    parent_registry = ToolRegistry(tmp_path, skill_catalog=resumed.skill_catalog)
    child_store = ConversationStore(tmp_path / "child-session")
    try:
        child_registry = parent_registry.clone_for_session(child_store)
        loaded = await child_registry.execute(
            ToolCall("load-new-skill", "skill", {"name": "new-skill"}),
            _skip_approval=True,
        )
    finally:
        child_store.close()

    assert loaded["isError"] is False
    assert loaded["content"][0]["text"].startswith("skill body")


def test_conservative_resume_discovers_new_agent_profile(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt="unknown recipe prompt bytes",
        prompt_recipe=None,
    )
    _write_agent(home / "agents" / "new-agent.md")

    resumed = _resume(manager, session_id, home, tmp_path, lambda **kwargs: None)

    assert resumed.agent_catalog.find("new-agent").prompt_suffix == "agent body"
    assert manager.read_metadata(session_id).agent_catalog == (
        resumed.agent_catalog.to_snapshot()
    )


@pytest.mark.asyncio
async def test_concurrent_resume_child_uses_one_fresh_profile_generation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    profile = home / "agents" / "specialist.md"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "---\nname: specialist\ndescription: version one specialist\n"
        "tools: [read]\n---\nv1 body\n",
        encoding="utf-8",
    )
    manager, session_id = await asyncio.to_thread(
        _manager_with_prompt,
        home,
        tmp_path,
        system_prompt="unknown recipe prompt bytes",
        prompt_recipe=None,
    )
    first_opened = await asyncio.to_thread(manager.open, session_id)
    second_opened = None
    loop = None
    try:
        first = await asyncio.to_thread(
            resume_prompt,
            first_opened.metadata,
            manager=manager,
            store=first_opened.store,
            home=home,
            repo_root=tmp_path,
            inbox_enabled=False,
            registry=manager.project_registry_state,
        )
        assert first.agent_catalog.find("specialist").prompt_suffix == "v1 body"

        profile.write_text(
            "---\nname: specialist\ndescription: version two specialist\n"
            "tools: [read, bash]\nallow_delegation: false\n---\nv2 body\n",
            encoding="utf-8",
        )
        second_opened = await asyncio.to_thread(manager.open, session_id)
        second = await asyncio.to_thread(
            resume_prompt,
            second_opened.metadata,
            manager=manager,
            store=second_opened.store,
            home=home,
            repo_root=tmp_path,
            inbox_enabled=False,
            registry=manager.project_registry_state,
        )
        profile.unlink()

        backend = FakeBackend(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            "specialist-1",
                            "agent",
                            {
                                "prompt": "inspect the task",
                                "description": "specialist task",
                                "preset": "specialist",
                            },
                        )
                    ]
                ),
                ScriptedTurn([TextContent("child complete")]),
            ]
        )
        registry = ToolRegistry(
            tmp_path,
            session_store=second_opened.store,
            skill_catalog=second.skill_catalog,
            agent_catalog=second.agent_catalog,
        )
        loop = AgentLoop(
            backend,
            second_opened.store,
            registry=registry,
            max_turns=1,
            system_prompt=second.context.system_prompt,
            skill_catalog=second.skill_catalog,
        )
        agent_schema = next(
            schema for schema in registry.schemas if schema["name"] == "agent"
        )
        _ = [event async for event in loop.run_turn("start", origin=MessageOrigin.USER)]

        assert len(backend.calls) == 2, "child did not start from runtime b's catalog"
        preset = second.agent_catalog.find("specialist")
        assert preset.prompt_suffix == "v2 body"
        assert preset.selection_guidance == "version two specialist"
        assert preset.tool_names == frozenset({"read", "bash"})
        assert preset.allow_delegation is False
        assert loop.tool_registry.agent_catalog is second.agent_catalog
        agent_type = agent_schema["parameters"]["properties"]["agent_type"]
        assert agent_type["enum"] == second.agent_catalog.names()
        assert "version two specialist" in agent_type["description"]
        child_system = backend.calls[1][0][0]
        assert any(
            isinstance(block, TextContent) and "v2 body" in block.text
            for block in child_system.content
        )
    finally:
        if loop is not None:
            await loop.close()
        if second_opened is not None:
            second_opened.store.close()
        first_opened.store.close()


@pytest.mark.parametrize("prompt_recipe", [None, "default"])
def test_resume_surfaces_skill_parse_notice(
    tmp_path: Path, prompt_recipe: str | None
) -> None:
    home = tmp_path / "home"
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt="unknown recipe prompt bytes",
        prompt_recipe=prompt_recipe,
    )
    malformed = home / "skills" / "broken" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("not frontmatter", encoding="utf-8")

    resumed = _resume(
        manager,
        session_id,
        home,
        tmp_path,
        lambda **kwargs: ProjectContext("rebuilt", (), prompt_recipe="default"),
    )
    slash_registry = create_slash_registry(skill_catalog=resumed.skill_catalog)

    assert any(
        str(malformed) in notice and "missing YAML frontmatter" in notice
        for notice in slash_registry.notices
    )


def test_staggered_recorded_default_resumes_adopt_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home", tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    loader, calls = _numbered_default_contexts(tmp_path)

    results = _staggered_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=loader,
        monkeypatch=monkeypatch,
    )

    _assert_one_persisted_prompt(manager, session_id, results)
    assert calls == [1]


def test_staggered_recipe_less_empty_resumes_adopt_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home", tmp_path, system_prompt="", prompt_recipe=None
    )
    loader, calls = _numbered_default_contexts(tmp_path)

    results = _staggered_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=loader,
        monkeypatch=monkeypatch,
    )

    _assert_one_persisted_prompt(manager, session_id, results)
    assert calls == [1]
    assert manager.read_metadata(session_id).prompt_recipe == "default"


def test_custom_prompt_without_owned_skill_span_stays_unchanged_and_notices_restart(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    prompt = (
        "custom before\n<zeta-skills>\nAvailable skills:\n- user: keep me\n"
        "</zeta-skills>\ncustom after"
    )
    manager, session_id = _manager_with_prompt(
        home, tmp_path, system_prompt=prompt, prompt_recipe="custom"
    )
    _write_skill(home / "skills" / "new.md")

    resumed = _resume(
        manager,
        session_id,
        home,
        tmp_path,
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"unexpected default rebuild: {kwargs}")
        ),
    )

    assert resumed.context.system_prompt == prompt
    assert (
        resumed.context.notices.count(
            "skills changed, but this session's prompt predates owned skill spans; "
            "restart the session to refresh its skill index"
        )
        == 1
    )


def test_live_runtime_adoption_uses_current_skill_index_in_memory(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    prompt = f"identity\n\n{SkillCatalog.empty().index()}"
    index = SkillCatalog.empty().index()
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt=prompt,
        prompt_recipe="custom",
        prompt_components={
            "skill_index": {
                "offset": prompt.index(index),
                "length": len(index),
                "digest": hashlib.sha256(index.encode()).hexdigest(),
            }
        },
    )
    stores: list[object] = []

    def loader(**kwargs: object) -> ProjectContext:
        raise AssertionError(f"unexpected default rebuild: {kwargs}")

    try:
        first = _resume(
            manager,
            session_id,
            home,
            tmp_path,
            loader,
            live_stores=stores,
        )
        _write_skill(home / "skills" / "new.md")
        second = _resume(
            manager,
            session_id,
            home,
            tmp_path,
            loader,
            live_stores=stores,
        )

        assert "newly installed" not in first.context.system_prompt
        assert "newly installed" in second.context.system_prompt
        assert second.skill_catalog.skills[0].name == "new-skill"
        assert manager.read_metadata(session_id).system_prompt == prompt
    finally:
        for store in stores:
            store.close()


def test_closed_runtime_allows_next_resume_to_rebuild(tmp_path: Path) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home", tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    loader, calls = _numbered_default_contexts(tmp_path)

    first = _resume(manager, session_id, tmp_path / "home", tmp_path, loader)
    second = _resume(manager, session_id, tmp_path / "home", tmp_path, loader)

    assert first.context.system_prompt == "default fallback 1"
    assert second.context.system_prompt == "default fallback 2"
    assert manager.read_metadata(session_id).system_prompt == "default fallback 2"
    assert calls == [1, 2]


def _start_resume_process(
    *,
    home: Path,
    cwd: Path,
    session_id: str,
    ready: Path,
    release: Path,
    close_before_ready: bool,
) -> subprocess.Popen[bytes]:
    close = "opened.store.close()" if close_before_ready else ""
    script = f"""from pathlib import Path
import os
import time
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.runtime.prompt_resume import resume_prompt
home = Path({str(home)!r})
cwd = Path({str(cwd)!r})
manager = SessionManager(home)
opened = manager.open({session_id!r})
resume_prompt(opened.metadata, manager=manager, store=opened.store, home=home, repo_root=cwd, inbox_enabled=False, context_loader=lambda **kwargs: ProjectContext("child runtime", (), prompt_recipe="default"), registry=manager.project_registry_state)
{close}
Path({str(ready)!r}).write_text("ready")
owner_pid = {os.getpid()}
deadline = time.monotonic() + 120
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        break
    time.sleep(0.1)
"""
    return subprocess.Popen([sys.executable, "-c", script], cwd=cwd)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def test_serve_style_switch_releases_runtime_lease_while_process_lives(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    manager, session_id = _manager_with_prompt(
        home, tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    child = _start_resume_process(
        home=home,
        cwd=tmp_path,
        session_id=session_id,
        ready=ready,
        release=release,
        close_before_ready=True,
    )
    try:
        _wait_for(ready)
        loader, calls = _numbered_default_contexts(tmp_path)
        resumed = _resume(manager, session_id, home, tmp_path, loader)
        assert child.poll() is None
        assert resumed.context.system_prompt == "default fallback 1"
        assert calls == [1]
    finally:
        release.write_text("release")
        child.wait(timeout=10)


def test_process_exit_releases_runtime_lease(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager, session_id = _manager_with_prompt(
        home, tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    child = _start_resume_process(
        home=home,
        cwd=tmp_path,
        session_id=session_id,
        ready=ready,
        release=release,
        close_before_ready=False,
    )
    try:
        _wait_for(ready)
        loader, calls = _numbered_default_contexts(tmp_path)
        with pytest.raises(SessionOwnedError, match="active in another zeta process"):
            _resume(manager, session_id, home, tmp_path, loader)
        assert calls == []
        child.kill()
        child.wait(timeout=10)
        rebuilt = _resume(manager, session_id, home, tmp_path, loader)
        assert rebuilt.context.system_prompt == "default fallback 1"
        assert calls == [1]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_transfer_excludes_runtime_lease_and_destination_rebuilds(
    tmp_path: Path,
) -> None:
    from zeta.remote_sync import LocalTransport, pull_session, push_session

    source = tmp_path / "source"
    remote = tmp_path / "remote"
    destination = tmp_path / "destination"
    manager, session_id = _manager_with_prompt(
        source, tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    source_stores: list[object] = []
    source_loader, _ = _numbered_default_contexts(tmp_path)
    _resume(
        manager,
        session_id,
        source,
        tmp_path,
        source_loader,
        live_stores=source_stores,
    )
    try:
        push_session(source, LocalTransport(remote), session_id=session_id)
        assert not (remote / "sessions" / session_id / "runtime.lease").exists()
        pull_session(destination, LocalTransport(remote), session_id=session_id)
        assert not (destination / "sessions" / session_id / "runtime.lease").exists()

        calls: list[int] = []
        resumed = _resume(
            SessionManager(destination),
            session_id,
            destination,
            tmp_path,
            lambda **kwargs: (
                calls.append(1),
                ProjectContext("destination rebuild", (), prompt_recipe="default"),
            )[1],
        )
        assert resumed.context.system_prompt == "destination rebuild"
        assert calls == [1]
        stored = (destination / "sessions" / session_id / "meta.json").read_text()
        assert "prompt_composition_owner" not in stored
        assert "prompt_composition_epoch" not in stored
    finally:
        for store in source_stores:
            store.close()


def _project_session(
    tmp_path: Path, recipe: str, memory: str
) -> tuple[Path, Path, SessionManager, ProjectRegistry, Project, str]:
    """Create a project-bound session whose prompt owns a ``memory`` span."""

    home = tmp_path / "home"
    repository = tmp_path / "repo"
    repository.mkdir()
    advance(home, expected_epoch=0)
    manager = SessionManager(home)
    registry = manager.project_registry
    assert registry is not None
    project = registry.create_project("demo", "scope", repository)
    registry.initialize_memory(project.project_id)
    initial = registry.memory_snapshot(project.project_id)
    registry.compare_and_swap_memory(
        project.project_id,
        expected_digest=initial.digest,
        updates={"brief.md": memory},
    )
    old_projection = registry.project_context_projection(project.project_id)
    old_context = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        session_project=project,
        cwd_project=project,
        registry=registry,
        project_projection=old_projection,
    )
    prompt = old_context.system_prompt
    offset = old_context.memory_offset
    length = old_context.memory_length
    digest = old_context.memory_digest
    assert offset is not None and length is not None and digest is not None
    components = {
        "project_memory": {
            "offset": offset,
            "length": length,
            "digest": digest,
        }
    }
    if recipe == "custom":
        prefix = "unknown-prefix\n"
        suffix = "\nunknown-suffix"
        prompt = prefix + prompt + suffix
        offset += len(prefix)
        components["project_memory"]["offset"] = offset
    opened = manager.create(
        provider="codex",
        model="offline",
        cwd=repository,
        project_id=project.project_id,
        system_prompt=prompt,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        project_memory_offset=offset,
        project_memory_length=length,
        project_memory_digest=digest,
        prompt_recipe=recipe,
        prompt_components=components,
    )
    session_id = opened.metadata.session_id
    opened.store.close()
    return home, repository, manager, registry, project, session_id


@pytest.mark.parametrize("recipe", ["custom", "default"])
def test_resume_uses_project_snapshot_captured_before_execution(
    tmp_path: Path, recipe: str
) -> None:
    home, repository, manager, registry, project, session_id = _project_session(
        tmp_path, recipe, "old memory"
    )
    before_update = registry.memory_snapshot(project.project_id)
    registry.compare_and_swap_memory(
        project.project_id,
        expected_digest=before_update.digest,
        updates={"brief.md": "current memory"},
    )
    projection = registry.project_context_projection(project.project_id)
    expected = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        session_project=project,
        cwd_project=project,
        registry=registry,
        project_projection=projection,
    )

    resume_registry = registry if recipe == "custom" else _UnavailableRegistry(None)
    resumed = manager.open(session_id)
    result: ResumedPrompt | None = None
    failure: BaseException | None = None
    try:
        result = resume_prompt(
            resumed.metadata,
            manager=manager,
            store=resumed.store,
            home=home,
            repo_root=repository,
            inbox_enabled=True,
            session_project=project,
            cwd_project=project,
            registry=resume_registry,
            project_projection=projection,
        )
    except BaseException as exc:  # noqa: BLE001 - asserted below
        failure = exc
    finally:
        resumed.store.close()

    assert failure is None, f"{recipe} resume did not succeed: {failure}"
    assert result is not None
    assert "current memory" in result.context.system_prompt
    assert "old memory" not in result.context.system_prompt
    if recipe == "custom":
        assert result.context.system_prompt.startswith("unknown-prefix\n")
        assert result.context.system_prompt.endswith("\nunknown-suffix")
    else:
        assert result.context.system_prompt == expected.system_prompt


def _set_memory(registry: ProjectRegistry, project_id: str, text: str) -> None:
    snapshot = registry.memory_snapshot(project_id)
    registry.compare_and_swap_memory(
        project_id, expected_digest=snapshot.digest, updates={"brief.md": text}
    )


def _hold_memory(registry: ProjectRegistry, project_id: str):
    """Hold the real project memory authority (the root lock before p06)."""

    if not hasattr(registry, "_project_lock"):
        return registry._locked(write=True)
    return registry._project_lock(project_id, "memory", exclusive=True)


def _start_public_resume(
    manager: SessionManager,
    registry: ProjectRegistry,
    project: Project,
    session_id: str,
    *,
    home: Path,
    repo_root: Path,
) -> tuple[threading.Event, threading.Event, list[object], threading.Thread]:
    """Run the serve/TUI writable resume sequence: open, capture, then resume."""

    captured = threading.Event()
    go = threading.Event()
    outcome: list[object] = []

    def public_resume() -> None:
        try:
            opened = manager.open(session_id)
            try:
                projection = registry.project_context_projection(project.project_id)
                captured.set()
                assert go.wait(10), "test did not release the resume"
                outcome.append(
                    resume_prompt(
                        opened.metadata,
                        manager=manager,
                        store=opened.store,
                        home=home,
                        repo_root=repo_root,
                        inbox_enabled=True,
                        session_project=projection.session_project,
                        cwd_project=projection.session_project,
                        registry=registry,
                        project_projection=projection,
                    ).context.system_prompt
                )
            finally:
                opened.store.close()
        except BaseException as exc:  # noqa: BLE001 - asserted by the test
            outcome.append(exc)

    resume = threading.Thread(target=public_resume)
    resume.start()
    return captured, go, outcome, resume


def _assert_resumed_with(
    manager: SessionManager, session_id: str, outcome: list[object]
) -> None:
    assert len(outcome) == 1 and isinstance(outcome[0], str), outcome
    persisted = manager.read_metadata(session_id).system_prompt
    for prompt in (outcome[0], persisted):
        assert "current memory" in prompt and "old memory" not in prompt, (
            "resume adopted a stale composition after its memory input changed"
        )


def test_p06_7_writable_resume_reads_memory_before_shape_commit(
    tmp_path: Path,
) -> None:
    home, repository, manager, registry, project, session_id = _project_session(
        tmp_path, "default", "original memory"
    )
    _set_memory(registry, project.project_id, "old memory")
    captured, go, outcome, resume = _start_public_resume(
        manager, registry, project, session_id, home=home, repo_root=repository
    )
    try:
        assert captured.wait(10), "resume did not capture project context"
        _set_memory(registry, project.project_id, "current memory")
        with _hold_memory(registry, project.project_id):
            go.set()
            persisted = manager.read_metadata(session_id, lock_timeout=10)
            assert not outcome and "old memory" not in persisted.system_prompt, (
                "resume persisted a stale composition before re-reading memory"
            )
    finally:
        go.set()
        resume.join(10)
    assert not resume.is_alive()
    _assert_resumed_with(manager, session_id, outcome)


_MEMORY_HOLDER = """
import sys
import time
from pathlib import Path
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry

home, project_id, session_id, signals = sys.argv[1:5]
signals = Path(signals)
manager = SessionManager(Path(home))
registry = ProjectRegistry(Path(home) / "projects")
held = (
    registry._project_lock(project_id, "memory", exclusive=True)
    if hasattr(registry, "_project_lock")
    else registry._locked(write=True)
)


def wait(name):
    deadline = time.monotonic() + 30
    while not (signals / name).exists():
        assert time.monotonic() < deadline, name
        time.sleep(0.01)


with held:
    (signals / "held").write_text("held")
    wait("metadata")
    try:
        manager.read_metadata(session_id, lock_timeout=10)
    except Exception as exc:
        (signals / "metadata-done").write_text(f"refused: {exc}")
    else:
        (signals / "metadata-done").write_text("read")
    wait("release")
"""


def test_p06_8_prompt_resume_process_progress(tmp_path: Path) -> None:
    home, repository, manager, registry, project, session_id = _project_session(
        tmp_path, "default", "original memory"
    )
    _set_memory(registry, project.project_id, "old memory")
    signals = tmp_path / "signals"
    signals.mkdir()
    captured, go, outcome, resume = _start_public_resume(
        manager, registry, project, session_id, home=home, repo_root=repository
    )
    holder: subprocess.Popen[bytes] | None = None
    try:
        assert captured.wait(10), "resume did not capture project context"
        _set_memory(registry, project.project_id, "current memory")
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _MEMORY_HOLDER,
                str(home),
                project.project_id,
                session_id,
                str(signals),
            ]
        )
        _wait_for(signals / "held")
        go.set()
        (signals / "metadata").write_text("go")
        _wait_for(signals / "metadata-done")
        assert (signals / "metadata-done").read_text() == "read", (
            "public metadata operation did not finish while memory was held"
        )
        assert not outcome, "resume adopted a composition without the memory authority"
    finally:
        go.set()
        (signals / "release").write_text("release")
        resume.join(10)
        if holder is not None:
            holder.wait(10)
    assert not resume.is_alive()
    assert holder.returncode == 0
    _assert_resumed_with(manager, session_id, outcome)
