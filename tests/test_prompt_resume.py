from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.prompt_resume import ResumedPrompt, resume_prompt
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools import ToolRegistry


def _manager_with_prompt(
    home: Path,
    cwd: Path,
    *,
    system_prompt: str,
    prompt_recipe: str | None,
    project_memory_offset: int | None = None,
    project_memory_length: int | None = None,
    project_memory_digest: str | None = None,
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

    def pause_after_persist(metadata: object, notices: tuple[str, ...]) -> ProjectContext:
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


def _numbered_default_contexts(cwd: Path) -> tuple[Callable[..., ProjectContext], list[int]]:
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
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt=original_prompt,
        prompt_recipe=None,
    )
    _write_skill(home / "skills" / "new-skill" / "SKILL.md")

    resumed = _resume(
        manager,
        session_id,
        home,
        tmp_path,
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"unexpected default rebuild: {kwargs}")
        ),
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
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt="unknown recipe prompt bytes",
        prompt_recipe=None,
    )
    _write_skill(home / "skills" / "new-skill" / "SKILL.md")
    resumed = _resume(manager, session_id, home, tmp_path, lambda **kwargs: None)
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


def test_resume_surfaces_skill_parse_notice(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager, session_id = _manager_with_prompt(
        home,
        tmp_path,
        system_prompt="unknown recipe prompt bytes",
        prompt_recipe=None,
    )
    malformed = home / "skills" / "broken" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("not frontmatter", encoding="utf-8")

    resumed = _resume(manager, session_id, home, tmp_path, lambda **kwargs: None)
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


def test_staggered_memory_span_resumes_adopt_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "prefix "
    memory = "old memory"
    prompt = f"{prefix}{memory} suffix"
    offset = len(prefix)
    manager, session_id = _manager_with_prompt(
        tmp_path / "home",
        tmp_path,
        system_prompt=prompt,
        prompt_recipe=None,
        project_memory_offset=offset,
        project_memory_length=len(memory),
        project_memory_digest=hashlib.sha256(memory.encode()).hexdigest(),
    )
    calls: list[int] = []

    def refresh(stored_prompt: str, **kwargs: object) -> str:
        del stored_prompt, kwargs
        calls.append(len(calls) + 1)
        return f"{prefix}new memory {calls[-1]} suffix"

    monkeypatch.setattr("zeta.runtime.prompt_resume.refresh_project_memory", refresh)
    results = _staggered_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"unexpected default rebuild: {kwargs}")
        ),
        monkeypatch=monkeypatch,
    )

    _assert_one_persisted_prompt(manager, session_id, results)
    assert calls == [1]


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
    script = f'''from pathlib import Path
import os
import time
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.runtime.prompt_resume import resume_prompt
home = Path({str(home)!r})
cwd = Path({str(cwd)!r})
manager = SessionManager(home)
opened = manager.open({session_id!r})
resume_prompt(opened.metadata, manager=manager, store=opened.store, home=home, repo_root=cwd, inbox_enabled=False, context_loader=lambda **kwargs: ProjectContext("child runtime", (), prompt_recipe="default"))
{close}
Path({str(ready)!r}).write_text("ready")
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
'''
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
        adopted = _resume(manager, session_id, home, tmp_path, loader)
        assert adopted.context.system_prompt == "child runtime"
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
