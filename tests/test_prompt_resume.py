from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager, SessionMetadata
from zeta.runtime.prompt_resume import ResumedPrompt, resume_prompt
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog


def _manager_with_prompt(
    home: Path,
    cwd: Path,
    *,
    system_prompt: str,
    prompt_recipe: str | None,
    prompt_components: dict[str, dict[str, int | str]] | None = None,
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
        prompt_components=prompt_components,
        project_memory_offset=project_memory_offset,
        project_memory_length=project_memory_length,
        project_memory_digest=project_memory_digest,
        auto_project=False,
    )
    opened.store.close()
    return manager, opened.metadata.session_id


def _concurrent_resumes(
    manager: SessionManager,
    session_id: str,
    home: Path,
    cwd: Path,
    *,
    context_loader: Callable[..., ProjectContext],
) -> list[ResumedPrompt]:
    metadata = [manager.read_metadata(session_id), manager.read_metadata(session_id)]
    results: list[ResumedPrompt] = []
    errors: list[BaseException] = []

    def run(item: SessionMetadata) -> None:
        try:
            results.append(
                resume_prompt(
                    item,
                    manager=manager,
                    home=home,
                    repo_root=cwd,
                    inbox_enabled=False,
                    context_loader=context_loader,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report worker failures here
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(item,)) for item in metadata]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    return results


def _different_contexts(cwd: Path) -> Callable[..., ProjectContext]:
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    count = 0

    def load(**kwargs: object) -> ProjectContext:
        del kwargs
        nonlocal count
        with lock:
            count += 1
            prompt = f"default fallback {count}"
        barrier.wait()
        return ProjectContext(
            prompt,
            (cwd / "context.md",),
            prompt_recipe="default",
        )

    return load


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
    assert [result.context.prompt_components for result in results] == [
        saved.prompt_components,
        saved.prompt_components,
    ]


def test_concurrent_legacy_empty_prompt_resumes_adopt_one_default(
    tmp_path: Path,
) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home",
        tmp_path,
        system_prompt="",
        prompt_recipe=None,
    )

    results = _concurrent_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=_different_contexts(tmp_path),
    )

    _assert_one_persisted_prompt(manager, session_id, results)
    assert manager.read_metadata(session_id).prompt_recipe == "default"


def test_concurrent_recorded_default_resumes_adopt_one_composition(
    tmp_path: Path,
) -> None:
    manager, session_id = _manager_with_prompt(
        tmp_path / "home",
        tmp_path,
        system_prompt="recorded default",
        prompt_recipe="default",
    )

    results = _concurrent_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=_different_contexts(tmp_path),
    )

    _assert_one_persisted_prompt(manager, session_id, results)


def test_concurrent_memory_span_resumes_adopt_one_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "prefix "
    memory = "old memory"
    prompt = f"{prefix}{memory} suffix"
    offset = len(prefix)
    digest = hashlib.sha256(memory.encode()).hexdigest()
    manager, session_id = _manager_with_prompt(
        tmp_path / "home",
        tmp_path,
        system_prompt=prompt,
        prompt_recipe="custom",
        prompt_components={
            "project_memory": {
                "offset": offset,
                "length": len(memory),
                "digest": digest,
            }
        },
        project_memory_offset=offset,
        project_memory_length=len(memory),
        project_memory_digest=digest,
    )
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    count = 0

    def refresh(stored_prompt: str, **kwargs: object) -> str:
        del stored_prompt, kwargs
        nonlocal count
        with lock:
            count += 1
            replacement = f"new memory {count}"
        barrier.wait()
        return f"{prefix}{replacement} suffix"

    monkeypatch.setattr("zeta.runtime.prompt_resume.refresh_project_memory", refresh)

    results = _concurrent_resumes(
        manager,
        session_id,
        tmp_path / "home",
        tmp_path,
        context_loader=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"unexpected default rebuild: {kwargs}")
        ),
    )

    _assert_one_persisted_prompt(manager, session_id, results)
