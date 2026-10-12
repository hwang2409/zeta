from __future__ import annotations

import argparse
import dataclasses
import io
import json
from pathlib import Path

import pytest

from zeta.cli import project as project_cli
from zeta.memory.entry_store import (
    AddOperation,
    MemoryEntry,
    MemorySource,
    UpdateOperation,
    canonical_state_bytes,
    state_from_bytes,
)
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


def test_zeta_profile_update_changes_only_new_state_expiry(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    current = registry._entry_memory_state(project_id)
    legacy_kinds = tuple(
        dataclasses.replace(kind, default_expiry_days=30)
        if kind.key == "state"
        else kind
        for kind in current.state.schema.kinds
    )
    legacy_state = dataclasses.replace(
        current.state,
        schema=dataclasses.replace(current.state.schema, version=3, kinds=legacy_kinds),
    )
    current = registry._replace_entry_state_for_test(
        project_id, legacy_state, expected_digest=current.digest
    )
    existing = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(AddOperation("state", "Existing state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    existing_id = existing.receipts[0].result_ids[0]

    updated = registry.set_memory_profile(project_id, "zeta")
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=updated.digest,
        operations=(AddOperation("state", "New state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    new_id = added.receipts[0].result_ids[0]

    assert added.state.schema.version == 4
    assert added.state.entries[existing_id].expires_at == "2026-11-08T00:00:00.000000Z"  # type: ignore[union-attr]
    assert added.state.entries[new_id].expires_at == "2026-10-16T00:00:00.000000Z"  # type: ignore[union-attr]


def test_default_expiry_uses_new_profile_after_later_observation(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    current = registry._entry_memory_state(project_id)
    legacy_kinds = tuple(
        dataclasses.replace(kind, default_expiry_days=30)
        if kind.key == "state"
        else kind
        for kind in current.state.schema.kinds
    )
    legacy_state = dataclasses.replace(
        current.state,
        schema=dataclasses.replace(current.state.schema, version=3, kinds=legacy_kinds),
    )
    current = registry._replace_entry_state_for_test(
        project_id, legacy_state, expected_digest=current.digest
    )
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(AddOperation("state", "Existing state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = added.receipts[0].result_ids[0]
    new_entry = added.state.entries[entry_id]
    assert isinstance(new_entry, MemoryEntry)
    assert new_entry.expiry_is_default is True
    raw = json.loads(canonical_state_bytes(added.state))
    raw["entries"][entry_id].pop("expiry_is_default")
    legacy_state = state_from_bytes(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    )
    registry._replace_entry_state_for_test(
        project_id, legacy_state, expected_digest=added.digest
    )

    changed = registry.set_memory_profile(project_id, "zeta")
    observed = registry._compare_and_swap_entries(
        project_id,
        expected_digest=changed.digest,
        operations=(
            UpdateOperation(
                entry_id,
                (
                    MemorySource(
                        "session", 2, 2, ("user",), "2026-10-20T00:00:00Z", 6
                    ),
                ),
            ),
        ),
        reconciliation_key=None,
        now="2026-10-20T00:00:00Z",
    )

    entry = observed.state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.expires_at == "2026-10-27T00:00:00.000000Z"


def test_explicit_expiry_survives_profile_change_and_later_observation(
    tmp_path: Path,
) -> None:
    registry, project_id = _entry_project(tmp_path)
    current = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(
            AddOperation(
                "state",
                "Explicit expiry.",
                _source(),
                expires_at="2027-01-01T00:00:00Z",
            ),
        ),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = added.receipts[0].result_ids[0]

    changed = registry.set_memory_profile(project_id, "zeta")
    observed = registry._compare_and_swap_entries(
        project_id,
        expected_digest=changed.digest,
        operations=(
            UpdateOperation(
                entry_id,
                (
                    MemorySource(
                        "session", 2, 2, ("user",), "2026-10-20T00:00:00Z", 6
                    ),
                ),
            ),
        ),
        reconciliation_key=None,
        now="2026-10-20T00:00:00Z",
    )

    entry = observed.state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.expires_at == "2027-01-01T00:00:00Z"
    assert entry.expiry_is_default is False


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


def test_profile_change_undo_restores_schema_and_kind_mapping(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    before = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation("state", "Mapped state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = added.receipts[0].result_ids[0]
    changed = registry.set_memory_profile(
        project_id, "messaging", kind_mappings={"state": "threads"}
    )

    undone = registry._undo_entry_transaction(project_id, changed.version)

    assert undone.state.schema == added.state.schema
    assert undone.state.entries == added.state.entries
    entry = undone.state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.kind == "state"


def test_profile_change_undo_restores_resolved_entries(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    before = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation("state", "Resolved during profile change.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = added.receipts[0].result_ids[0]
    changed = registry.set_memory_profile(
        project_id, "messaging", resolve_kinds=frozenset({"state"})
    )
    assert entry_id not in changed.state.entries

    undone = registry._undo_entry_transaction(project_id, changed.version)

    assert undone.state.schema == added.state.schema
    assert undone.state.entries == added.state.entries


def test_profile_change_undo_rejects_dependent_later_entry_change(tmp_path: Path) -> None:
    registry, project_id = _entry_project(tmp_path)
    initial = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Mapped state.", _source()),),
        reconciliation_key=None,
        now="2026-10-09T00:00:00Z",
    )
    entry_id = added.receipts[0].result_ids[0]
    changed = registry.set_memory_profile(
        project_id, "messaging", kind_mappings={"state": "threads"}
    )
    later = registry._compare_and_swap_entries(
        project_id,
        expected_digest=changed.digest,
        operations=(UpdateOperation(entry_id, _source(), text="Later edit."),),
        reconciliation_key=None,
        now="2026-10-09T00:01:00Z",
    )

    with pytest.raises(ProjectRegistryError, match="dependent later operations"):
        registry._undo_entry_transaction(project_id, changed.version)

    assert registry._entry_memory_state(project_id).digest == later.digest


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


@pytest.mark.parametrize("operation", ("activate", "migrate"))
def test_activation_refuses_consumer_without_format_two_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from zeta import project_memory_commands

    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("demo", "test", tmp_path / "workspace")
    monkeypatch.setattr(project_memory_commands, "SUPPORTED_MEMORY_FORMATS", frozenset({1}))

    with pytest.raises(
        ProjectRegistryError, match="missing format-2 capabilities: commands_api"
    ):
        if operation == "activate":
            registry.activate_entry_memory(project.project_id, "zeta")
        else:
            registry.migrate_memory(project.project_id)

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


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("schema", "set-profile", "zeta", "extra"), "memory schema takes no additional entry IDs"),
        (("schema", "set-profile", "zeta", "--reason", "ignored"), "--reason is only valid for memory resolve"),
        (("migrate", "extra"), "memory migrate takes no arguments"),
        (("migrate", "--reason", "ignored"), "--reason is only valid for memory resolve"),
        (("accept", "state.md", "extra"), "memory accept requires exactly one file"),
        (("accept", "state.md", "--reason", "ignored"), "--reason is only valid for memory resolve"),
    ),
)
def test_non_resolve_memory_actions_reject_resolve_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    message: str,
) -> None:
    home = tmp_path / "home"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "test", tmp_path / "workspace")
    registry.initialize_memory(project.project_id)
    monkeypatch.setenv("ZETA_HOME", str(home))
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    project_cli.add_subcommand(commands)
    args = parser.parse_args(("project", "memory", project.project_id, *arguments))
    stderr = io.StringIO()

    assert project_cli.run(args, stdout=io.StringIO(), stderr=stderr) == 1
    assert stderr.getvalue() == f"zeta: {message}\n"


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
