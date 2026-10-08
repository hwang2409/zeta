from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.agent.runner import _child_base_system_prompt
from zeta.core.project_context import load_project_context
from zeta.core.session import SessionManager
from zeta.memory.entry_store import (
    AddOperation,
    MemoryEntry,
    MemoryKind,
    MemorySchema,
    MemorySource,
    ResolveOperation,
)
from zeta.memory.prompt_projection import MEMORY_PROMPT_BYTE_CAP, render_entry_memory
from zeta.project_registry import ProjectRegistry
from zeta.server.runtime import ServerRuntime
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog


def _source(seq: int, *, observed_at: str = "2026-10-08T12:00:00Z") -> tuple[MemorySource, ...]:
    return (
        MemorySource(
            session_id="session-1",
            seq_start=seq,
            seq_end=seq,
            origins=("user",),
            observed_at=observed_at,
            evidence_rank=2,
        ),
    )


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _schema() -> MemorySchema:
    return MemorySchema(
        version=1,
        profile="test",
        kinds=(
            MemoryKind("state", "Current state", "State.", "recent", 80, 100),
            MemoryKind("decisions", "Decisions", "Decisions.", "always", 90, 100),
            MemoryKind("reference", "Reference", "Reference.", "on_demand", 100, 100),
        ),
    )


def _entry_project(tmp_path: Path) -> tuple[Path, Path, ProjectRegistry, str]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    registry._create_entry_memory_for_test(project.project_id, _schema())
    return home, workspace, registry, project.project_id


def _add(registry: ProjectRegistry, project_id: str, operations: tuple[AddOperation, ...]):
    snapshot = registry._entry_memory_state(project_id)
    return registry._compare_and_swap_entries(
        project_id,
        expected_digest=snapshot.digest,
        operations=operations,
        reconciliation_key=_key(str(snapshot.state.generation)),
        now="2026-10-08T12:00:00Z",
    )


def test_prompt_orders_active_entries_by_schema_priority(tmp_path: Path) -> None:
    _home, _workspace, registry, project_id = _entry_project(tmp_path)
    current = _add(
        registry,
        project_id,
        (
            AddOperation("state", "newer state", _source(1, observed_at="2026-10-08T13:00:00Z")),
            AddOperation("decisions", "older decision", _source(2)),
            AddOperation("reference", "hidden reference", _source(3)),
            AddOperation("decisions", "newer decision", _source(4, observed_at="2026-10-08T14:00:00Z")),
        ),
    )
    decision_ids = [
        entry.id
        for entry in current.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.kind == "decisions"
    ]
    accepted_state = next(
        entry
        for entry in current.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.kind == "state"
    )
    entries = dict(current.state.entries)
    entries[accepted_state.id] = dataclasses.replace(
        accepted_state, accepted_at="2026-10-08T15:00:00Z", accepted_by="user"
    )
    projection = render_entry_memory(
        dataclasses.replace(current.state, entries=entries),
        now="2026-10-08T15:00:00Z",
    )

    assert projection.block.index("newer decision") < projection.block.index("older decision")
    assert projection.block.index("older decision") < projection.block.index("newer state")
    assert "hidden reference" not in projection.block
    assert all(entry_id in projection.block for entry_id in decision_ids)
    assert "<zeta-automatic-notes>" in projection.block


def test_resolve_and_expire_leave_next_composed_prompt(tmp_path: Path) -> None:
    _home, _workspace, registry, project_id = _entry_project(tmp_path)
    added = _add(
        registry,
        project_id,
        (
            AddOperation("state", "resolved item", _source(1)),
            AddOperation(
                "state",
                "expired item",
                _source(2),
                expires_at="2026-10-08T12:30:00Z",
            ),
            AddOperation("decisions", "kept item", _source(3)),
        ),
    )
    resolved_id = next(
        entry.id
        for entry in added.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.text == "resolved item"
    )
    updated = registry._compare_and_swap_entries(
        project_id,
        expected_digest=added.digest,
        operations=(ResolveOperation(resolved_id, _source(4)),),
        reconciliation_key=_key("resolve"),
    )
    projection = render_entry_memory(updated.state, now="2026-10-08T13:00:00Z")

    assert "resolved item" not in projection.block
    assert "expired item" not in projection.block
    assert "kept item" in projection.block


def test_prompt_cap_keeps_complete_entries_and_reports_omissions(tmp_path: Path) -> None:
    _home, _workspace, registry, project_id = _entry_project(tmp_path)
    operations = tuple(
        AddOperation("state", f"entry-{index}-" + "x" * 4000, _source(index + 1))
        for index in range(20)
    )
    state = _add(registry, project_id, operations).state
    projection = render_entry_memory(state, now="2026-10-08T13:00:00Z")

    assert len(projection.block.encode()) <= MEMORY_PROMPT_BYTE_CAP
    assert projection.omitted_count > 0
    assert f"omitted entries: {projection.omitted_count}" in projection.block
    assert not projection.block.endswith("x")


def test_format_one_prompt_block_is_byte_identical_golden(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace-v1"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("legacy", "test", workspace)
    registry.initialize_memory(project.project_id)
    registry.update_memory(project.project_id, {"state.md": "# State\n\nA < B.\n"})

    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project.project_id,
    )
    assert context.memory_offset is not None and context.memory_length is not None
    block = context.system_prompt[
        context.memory_offset : context.memory_offset + context.memory_length
    ]
    memory_root = home / "projects" / project.project_id / "memory"
    expected_sections = []
    for name, body in (
        ("brief.md", "# Brief\n"),
        ("state.md", "# State\n\nA &lt; B.\n"),
        ("backlog.md", "# Backlog\n"),
        ("changelog.md", "# Changelog\n"),
        ("decisions.md", "# Decisions\n"),
    ):
        expected_sections.append(
            f'<zeta-project-instructions source="{memory_root / name}">\n'
            f"{body}\n</zeta-project-instructions>"
        )
    assert block == (
        f"<zeta-project-memory>\nproject-id: {project.project_id}\n"
        + "\n\n".join(expected_sections)
        + "\n</zeta-project-memory>"
    )


def test_format_two_outer_budget_keeps_short_complete_entry(tmp_path: Path) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(
        registry,
        project_id,
        (
            AddOperation("decisions", "short retained entry", _source(1)),
            AddOperation("state", "large omitted entry " + "x" * 4_000, _source(2)),
        ),
    )

    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        byte_cap=1_000,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )

    assert "short retained entry" in context.system_prompt
    assert "large omitted entry" not in context.system_prompt
    assert context.memory_offset is not None


def test_active_run_prompt_is_byte_stable_after_memory_commit(tmp_path: Path) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(registry, project_id, (AddOperation("state", "before", _source(1)),))
    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )
    active_prompt = context.system_prompt
    changed_text = "UNIQUE-MEMORY-COMMIT-AFTER-SNAPSHOT"
    _add(registry, project_id, (AddOperation("decisions", changed_text, _source(2)),))

    assert context.system_prompt == active_prompt
    assert changed_text not in active_prompt
    assert changed_text in load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    ).system_prompt


def test_same_cwd_child_inherits_parent_memory_snapshot(tmp_path: Path) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(registry, project_id, (AddOperation("state", "parent snapshot", _source(1)),))
    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )
    parent_prompt = context.system_prompt
    _add(registry, project_id, (AddOperation("state", "new memory", _source(2)),))
    loop = SimpleNamespace(
        context_assembler=SimpleNamespace(system_prompt=parent_prompt),
        store=SimpleNamespace(cwd=str(workspace)),
    )

    assert _child_base_system_prompt(loop, str(workspace)) is parent_prompt
    equivalent_cwd = str(workspace / ".." / workspace.name)
    assert _child_base_system_prompt(loop, equivalent_cwd) is parent_prompt
    symlink_cwd = tmp_path / "workspace-link"
    symlink_cwd.symlink_to(workspace, target_is_directory=True)
    assert _child_base_system_prompt(loop, str(symlink_cwd)) is parent_prompt
    assert "new memory" not in parent_prompt


def test_fresh_cwd_child_loads_current_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(registry, project_id, (AddOperation("state", "current child memory", _source(1)),))
    parent_cwd = tmp_path / "parent"
    parent_cwd.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    loop = SimpleNamespace(
        context_assembler=SimpleNamespace(system_prompt="parent prompt"),
        store=SimpleNamespace(cwd=str(parent_cwd)),
        active_home=str(home),
        tool_registry=SimpleNamespace(skill_catalog=SkillCatalog.empty()),
        root_project_id=project_id,
    )

    child_prompt = _child_base_system_prompt(loop, str(workspace))
    assert isinstance(child_prompt, str)
    assert "current child memory" in child_prompt


@pytest.mark.asyncio
async def test_serve_resume_rebuilds_identity_and_memory_components(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "serve-workspace"
    workspace.mkdir()
    (home / "AGENTS.md").parent.mkdir(parents=True)
    (home / "AGENTS.md").write_text("identity before", encoding="utf-8")
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("serve", "test", workspace)
    registry.initialize_memory(project.project_id)
    registry.update_memory(project.project_id, {"state.md": "# State\n\nmemory before\n"})

    first_runtime = ServerRuntime(home, cwd=workspace, provider="codex")
    metadata = await first_runtime.create_session()
    first_prompt = first_runtime.loop.context_assembler.system_prompt.content[0].text
    assert metadata.prompt_recipe == "default"
    assert metadata.prompt_components

    (home / "AGENTS.md").write_text("identity after", encoding="utf-8")
    registry.update_memory(project.project_id, {"state.md": "# State\n\nmemory after\n"})
    assert (await first_runtime.resume_session(metadata.session_id)).session_id == metadata.session_id
    assert first_runtime.loop.context_assembler.system_prompt.content[0].text == first_prompt
    await first_runtime.close()

    resumed_runtime = ServerRuntime(home, cwd=tmp_path, provider="codex")
    resumed_metadata = await resumed_runtime.resume_session(metadata.session_id)
    resumed_prompt = resumed_runtime.loop.context_assembler.system_prompt.content[0].text
    assert resumed_metadata.prompt_recipe == "default"
    default_component = resumed_metadata.prompt_components["default_context"]
    assert default_component["digest"] == hashlib.sha256(resumed_prompt.encode()).hexdigest()
    assert default_component["length"] == len(resumed_prompt)
    await resumed_runtime.close()

    assert "identity after" in resumed_prompt
    assert "identity before" not in resumed_prompt
    assert "memory after" in resumed_prompt
    assert "memory before" not in resumed_prompt


@pytest.mark.asyncio
async def test_serve_unknown_recipe_preserves_custom_bytes_and_refreshes_owned_memory(
    tmp_path: Path,
) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(registry, project_id, (AddOperation("state", "memory before", _source(1)),))
    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )
    assert context.memory_offset is not None
    assert context.memory_length is not None
    assert context.memory_digest is not None
    custom_prefix = "CUSTOM-LEGACY-PROMPT\n"
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex",
        model="offline",
        cwd=workspace,
        system_prompt=custom_prefix + context.system_prompt,
        context_files=context.files,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        project_id=project_id,
        auto_project=False,
        project_memory_offset=context.memory_offset + len(custom_prefix),
        project_memory_length=context.memory_length,
        project_memory_digest=context.memory_digest,
    )
    session_id = opened.metadata.session_id
    opened.store.close()
    _add(registry, project_id, (AddOperation("state", "memory after", _source(2)),))

    runtime = ServerRuntime(home, cwd=tmp_path, provider="codex")
    metadata = await runtime.resume_session(session_id)
    resumed_prompt = runtime.loop.context_assembler.system_prompt.content[0].text
    await runtime.close()

    assert resumed_prompt.startswith(custom_prefix)
    assert "memory before" in resumed_prompt
    assert "memory after" in resumed_prompt
    assert metadata.prompt_recipe is None


def test_format_two_projection_is_fixture_only(tmp_path: Path) -> None:
    home, workspace, registry, project_id = _entry_project(tmp_path)
    _add(registry, project_id, (AddOperation("state", "fixture entry", _source(1)),))
    context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )
    assert "fixture entry" in context.system_prompt

    production = registry.create_project("production", "test", tmp_path / "production")
    registry.initialize_memory(production.project_id)
    assert registry.memory_state(production.project_id).contents.keys() == {
        "brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"
    }
