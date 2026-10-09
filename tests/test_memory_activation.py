from __future__ import annotations

import argparse
import dataclasses
import io
from pathlib import Path

import pytest

from zeta.cli import project as project_cli
from zeta.memory.entry_store import AddOperation, MemoryEntry, MemorySource
from zeta.memory.profiles import memory_profile
from zeta.project_errors import ProjectRegistryError
from zeta.project_registry import ProjectRegistry


def _source() -> tuple[MemorySource, ...]:
    return (MemorySource("session", 1, 1, ("user",), "2026-10-09T00:00:00Z", 6),)


def _entry_project(tmp_path: Path, profile: str = "zeta") -> tuple[ProjectRegistry, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("demo", "test", workspace)
    registry.activate_entry_memory(project.project_id, profile)
    return registry, project.project_id


def test_zeta_and_messaging_profiles_apply_distinct_defaults(tmp_path: Path) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    first = registry.create_project("zeta", "test", tmp_path / "zeta")
    second = registry.create_project("messages", "test", tmp_path / "messages")
    registry.activate_entry_memory(first.project_id, "zeta")
    registry.activate_entry_memory(second.project_id, "messaging")

    assert registry._entry_memory_state(first.project_id).state.schema == memory_profile("zeta")
    assert registry._entry_memory_state(second.project_id).state.schema == memory_profile("messaging")


def test_profile_change_requires_mapping_for_removed_nonempty_kind(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    before = registry._entry_memory_state(project_id)
    changed = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation("state", "Active state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = changed.receipts[0].result_ids[0]

    with pytest.raises(ProjectRegistryError, match="map-kind"):
        registry.set_memory_profile(project_id, "messaging")
    registry.set_memory_profile(project_id, "messaging", kind_mappings={"state": "threads"})
    entry = registry._entry_memory_state(project_id).state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.kind == "threads"


def test_profile_change_preserves_explicit_expiry(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    before = registry._entry_memory_state(project_id)
    changed = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation("state", "Temporary.", _source(), expires_at="2027-01-01T00:00:00Z"),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = changed.receipts[0].result_ids[0]
    registry.set_memory_profile(project_id, "messaging", kind_mappings={"state": "threads"})
    entry = registry._entry_memory_state(project_id).state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.expires_at == "2027-01-01T00:00:00Z"


def test_activation_refuses_missing_format_two_capability(tmp_path: Path) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("demo", "test", tmp_path / "workspace")
    with pytest.raises(ProjectRegistryError, match="missing format-2 capabilities: sync"):
        registry.activate_entry_memory(
            project.project_id,
            "zeta",
            capabilities=frozenset({"updater", "prompt_projection", "commands_api"}),
        )
    assert registry.memory_format(project.project_id) == 1


def test_rollback_target_stays_protected_past_normal_retention(tmp_path: Path) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("demo", "test", tmp_path / "workspace")
    registry.update_memory(
        project.project_id, {"state.md": "# Current state\nBefore migration.\n"}
    )
    registry.migrate_memory(
        project.project_id, migrated_at="2026-10-09T00:00:00Z"
    )
    for generation in range(130):
        current = registry._entry_memory_state(project.project_id)
        registry._replace_entry_state_for_test(
            project.project_id,
            dataclasses.replace(current.state, generation=current.state.generation + 1),
            expected_digest=current.digest,
        )

    registry.rollback_memory_migration(project.project_id)

    assert registry.memory_format(project.project_id) == 1
    assert dict(registry.load_memory(project.project_id))["state.md"] == (
        "# Current state\nBefore migration.\n"
    )


def _cli_args(project_id: str, action: str) -> argparse.Namespace:
    return argparse.Namespace(
        project_verb="memory", project=project_id, remote=action, action=None,
        detail=None, sync_project=None, remote_home=None, accept=None, set=[],
        from_file=[], json=False, map_kind=[], resolve_kind=[],
    )


def test_migrate_rollback_and_finalize_commands_use_real_project_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "test", tmp_path / "workspace")
    registry.initialize_memory(project.project_id)
    monkeypatch.setenv("ZETA_HOME", str(home))

    assert project_cli.run(_cli_args(project.project_id, "migrate"), stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert registry.memory_format(project.project_id) == 2
    assert project_cli.run(_cli_args(project.project_id, "rollback"), stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert registry.memory_format(project.project_id) == 1

    assert project_cli.run(_cli_args(project.project_id, "migrate"), stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert project_cli.run(_cli_args(project.project_id, "finalize"), stdout=io.StringIO(), stderr=io.StringIO()) == 0
    with pytest.raises(ProjectRegistryError, match="finalized"):
        registry.rollback_memory_migration(project.project_id)


@pytest.mark.asyncio
async def test_project_update_refuses_format_two_project(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from zeta.tools.project import _update_project

    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    projects = ProjectRegistry(home / "projects")
    project = projects.create_project("demo", "test", workspace)
    projects.activate_entry_memory(project.project_id)
    registry = SimpleNamespace(
        project_registry=projects,
        project_id=project.project_id,
        cwd=workspace,
    )

    result = await _update_project(
        registry, {"name": "state.md", "content": "# State\nwrong format"}
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["message"] == (
        "this project uses entry memory; memory is maintained automatically"
    )
