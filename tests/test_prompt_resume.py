from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.runtime.prompt_resume import ResumedPrompt, resume_prompt
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog


@dataclass(frozen=True)
class _Owner:
    pid: int
    started: str


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
        provider="fake",
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


def _resume(
    manager: SessionManager,
    session_id: str,
    home: Path,
    cwd: Path,
    context_loader: Callable[..., ProjectContext],
) -> ResumedPrompt:
    return resume_prompt(
        manager.read_metadata(session_id),
        manager=manager,
        home=home,
        repo_root=cwd,
        inbox_enabled=False,
        context_loader=context_loader,
    )


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
    monkeypatch.setattr(
        "zeta.core.store.prompt_composition.current_process_identity",
        lambda: _Owner(
            101 if threading.current_thread().name == "first-resumer" else 202,
            "first-start" if threading.current_thread().name == "first-resumer" else "second-start",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "zeta.core.store.prompt_composition.process_is_live",
        lambda owner: owner.pid == 101 and owner.started == "first-start",
        raising=False,
    )

    results: list[ResumedPrompt] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                _resume(manager, session_id, home, cwd, context_loader)
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
    assert saved.prompt_composition_epoch == 1
    assert saved.prompt_composition_owner_pid == 101


def test_staggered_recorded_default_resumes_adopt_live_owner(
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


def test_staggered_recipe_less_empty_resumes_adopt_live_owner(
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


def test_staggered_memory_span_resumes_adopt_live_owner(
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


def test_dead_prompt_composition_owner_allows_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home", tmp_path, system_prompt="recorded", prompt_recipe="default"
    )
    loader, calls = _numbered_default_contexts(tmp_path)
    owners = iter(
        [_Owner(101, "first-start"), _Owner(202, "second-start")]
    )
    monkeypatch.setattr(
        "zeta.core.store.prompt_composition.current_process_identity",
        lambda: next(owners),
    )
    monkeypatch.setattr(
        "zeta.core.store.prompt_composition.process_is_live", lambda owner: False
    )

    first = _resume(manager, session_id, tmp_path / "home", tmp_path, loader)
    second = _resume(manager, session_id, tmp_path / "home", tmp_path, loader)

    assert first.context.system_prompt == "default fallback 1"
    assert second.context.system_prompt == "default fallback 2"
    saved = manager.read_metadata(session_id)
    assert saved.system_prompt == "default fallback 2"
    assert saved.prompt_composition_epoch == 2
    assert saved.prompt_composition_owner_pid == 202
    assert calls == [1, 2]
