from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from zeta.memory.entry_store import (
    AddOperation,
    ExpireOperation,
    MemoryEntry,
    MemoryKind,
    MemorySchema,
    MemorySource,
    MissingEntry,
    ResolveOperation,
    SupersedeOperation,
    UpdateOperation,
    canonical_state_bytes,
    state_from_bytes,
    state_to_dict,
    validate_state,
)
from zeta.project_errors import ProjectRegistryError
from zeta.project_registry import ProjectRegistry


def _schema() -> MemorySchema:
    return MemorySchema(
        version=1,
        profile="test",
        kinds=(
            MemoryKind(
                key="state",
                name="State",
                description="Current project state.",
                prompt_mode="recent",
                prompt_priority=80,
                prompt_max_entries=100,
                default_expiry_days=30,
            ),
            MemoryKind(
                key="decisions",
                name="Decisions",
                description="Binding project decisions.",
                prompt_mode="always",
                prompt_priority=90,
                prompt_max_entries=100,
            ),
        ),
    )


def _source(seq: int = 1) -> tuple[MemorySource, ...]:
    return (
        MemorySource(
            session_id="session-1",
            seq_start=seq,
            seq_end=seq,
            origins=("user",),
            observed_at="2026-10-08T12:00:00Z",
            evidence_rank=2,
        ),
    )


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _registry(tmp_path: Path) -> tuple[ProjectRegistry, str]:
    root = tmp_path / "projects"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(root)
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    return registry, project.project_id


def _entry_values(registry: ProjectRegistry, project_id: str) -> list[MemoryEntry]:
    state = registry._entry_memory_state(project_id).state
    return [value for value in state.entries.values() if isinstance(value, MemoryEntry)]


def test_entry_operations_are_atomic_and_validate_links(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Alpha", _source()),),
        reconciliation_key=_key("add-alpha"),
    )
    entry_id = next(iter(added.state.entries))

    with pytest.raises(ProjectRegistryError, match="entry"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=added.digest,
            operations=(
                UpdateOperation(entry_id, _source(2), text="Changed"),
                ResolveOperation("m_missing", _source(2)),
            ),
            reconciliation_key=_key("invalid-group"),
        )

    unchanged = registry._entry_memory_state(project_id)
    assert unchanged.digest == added.digest
    entry = unchanged.state.entries[entry_id]
    assert isinstance(entry, MemoryEntry)
    assert entry.text == "Alpha"

    malformed_entries = dict(unchanged.state.entries)
    malformed_entries[entry_id] = dataclasses.replace(
        entry, superseded_by=("m_00000000000000000000000000000000",)
    )
    with pytest.raises(ProjectRegistryError, match="dangling"):
        validate_state(dataclasses.replace(unchanged.state, entries=malformed_entries))


def test_supersede_creates_reciprocal_links(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("decisions", "Use A.", _source()),),
        reconciliation_key=_key("add-a"),
    )
    old_id = next(iter(added.state.entries))

    result = registry._compare_and_swap_entries(
        project_id,
        expected_digest=added.digest,
        operations=(
            SupersedeOperation(
                (old_id,), "decisions", "Use B.", _source(2)
            ),
        ),
        reconciliation_key=_key("replace-a-with-b"),
    )
    entries = _entry_values(registry, project_id)
    old = next(entry for entry in entries if entry.id == old_id)
    replacement = next(entry for entry in entries if entry.id != old_id)
    assert old.status == "superseded"
    assert old.superseded_by == (replacement.id,)
    assert replacement.supersedes == (old.id,)
    assert result.receipts[0].target_ids == (old.id,)
    assert result.receipts[0].result_ids == (replacement.id,)


def test_entry_cas_rejects_stale_base(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "First", _source()),),
        reconciliation_key=_key("first"),
    )

    with pytest.raises(ProjectRegistryError, match="digest mismatch"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=initial.digest,
            operations=(AddOperation("state", "Stale", _source(2)),),
            reconciliation_key=_key("stale"),
        )


def test_inactive_body_ages_out_after_retention_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zeta.project_memory_history as history

    monkeypatch.setattr(history, "MAX_RETAINED_VERSIONS", 4)
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Old body", _source()),),
        reconciliation_key=_key("old"),
    )
    old_id = next(iter(added.state.entries))
    current = registry._compare_and_swap_entries(
        project_id,
        expected_digest=added.digest,
        operations=(SupersedeOperation((old_id,), "state", "New body", _source(2)),),
        reconciliation_key=_key("supersede"),
    )
    replacement_id = next(entry_id for entry_id in current.state.entries if entry_id != old_id)
    assert isinstance(current.state.entries[old_id], MemoryEntry)

    for seq in range(3, 7):
        current = registry._compare_and_swap_entries(
            project_id,
            expected_digest=current.digest,
            operations=(UpdateOperation(replacement_id, _source(seq), text=f"New {seq}"),),
            reconciliation_key=_key(f"update-{seq}"),
        )

    compacted = current.state.entries[old_id]
    assert isinstance(compacted, MissingEntry)
    assert compacted.id == old_id
    assert "Old body" not in json.dumps(state_to_dict(current.state))


def test_memory_accept_and_undo_target_entry_ids(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Automatic", _source()),),
        reconciliation_key=_key("automatic"),
    )
    entry_id = next(iter(added.state.entries))

    accepted = registry._accept_memory_entry(project_id, entry_id)
    accepted_entry = accepted.state.entries[entry_id]
    assert isinstance(accepted_entry, MemoryEntry)
    assert accepted_entry.automatic is True
    assert accepted_entry.accepted_by == "user"
    assert accepted_entry.accepted_at is not None

    undone = registry._undo_memory_entry(project_id, entry_id)
    restored = undone.state.entries[entry_id]
    assert isinstance(restored, MemoryEntry)
    assert restored.accepted_at is None
    assert restored.accepted_by is None
    assert undone.receipts[0].target_ids == (entry_id,)


def test_resolve_and_expire_transition_active_entries(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(
            AddOperation("state", "Open work", _source()),
            AddOperation("state", "Temporary state", _source()),
        ),
        reconciliation_key=_key("two-active-entries"),
    )
    first_id, second_id = added.state.entries

    transitioned = registry._compare_and_swap_entries(
        project_id,
        expected_digest=added.digest,
        operations=(
            ResolveOperation(first_id, _source(2)),
            ExpireOperation(second_id, "validity window ended"),
        ),
        reconciliation_key=_key("two-transitions"),
    )

    first = transitioned.state.entries[first_id]
    second = transitioned.state.entries[second_id]
    assert isinstance(first, MemoryEntry) and first.status == "resolved"
    assert isinstance(second, MemoryEntry) and second.status == "expired"
    assert [receipt.type for receipt in transitioned.receipts] == ["resolve", "expire"]


def test_reconciliation_key_deduplicates_entry_transaction(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    key = _key("same-range-and-fragment")
    first = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Only once", _source()),),
        reconciliation_key=key,
    )
    repeated = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "Must not apply", _source()),),
        reconciliation_key=key,
    )

    assert first.published is True
    assert repeated.published is False
    assert repeated.version == first.version
    assert repeated.state == first.state
    assert len(repeated.state.entries) == 1


def test_entry_state_is_canonical_bounded_and_uses_shared_secret_safety(
    tmp_path: Path,
) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    assert state_from_bytes(canonical_state_bytes(initial.state)) == initial.state

    with pytest.raises(ProjectRegistryError, match="secret"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=initial.digest,
            operations=(
                AddOperation(
                    "state",
                    "api_key=abcdefghijklmnopqrstuvwxyz",
                    _source(),
                ),
            ),
            reconciliation_key=_key("secret"),
        )
    with pytest.raises(ProjectRegistryError, match="entry text"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=initial.digest,
            operations=(AddOperation("state", "x" * 4097, _source()),),
            reconciliation_key=_key("oversized"),
        )
    with pytest.raises(ProjectRegistryError, match="outside supplied evidence"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=initial.digest,
            operations=(AddOperation("state", "Wrong range", _source(2)),),
            reconciliation_key=_key("wrong-evidence"),
            evidence=("session-1", 1, 1),
        )


def test_no_public_path_can_create_or_activate_format_two(tmp_path: Path) -> None:
    registry, project_id = _registry(tmp_path)
    registry.update_memory(project_id, {"state.md": "# Current state\nformat one\n"})
    snapshot = registry.memory_snapshot(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=snapshot.digest,
        updates={"backlog.md": "# Backlog\nstill format one\n"},
        provenance={"session_id": "session-1", "seq_start": 1, "seq_end": 1},
    )
    source_workspace = tmp_path / "source-workspace"
    source_workspace.mkdir()
    source_registry = ProjectRegistry(tmp_path / "source-projects")
    source_project = source_registry.create_project(
        "source", "source", source_workspace
    )
    source_registry.update_memory(
        source_project.project_id, {"brief.md": "# Brief\nimported format one\n"}
    )
    before_import = registry.memory_snapshot(project_id)
    registry.import_memory(
        project_id,
        source_registry.export_memory(source_project.project_id),
        expected_digest=before_import.digest,
    )

    pointer = json.loads(
        (registry.root / project_id / "memory-current.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            registry.root
            / project_id
            / "memory-versions"
            / "versions"
            / f"{pointer['current']}.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest.get("format", 1) == 1
    assert isinstance(manifest["snapshot"], dict)
    assert not any(
        name.startswith(("entry_memory", "create_entry_memory"))
        for name in dir(registry)
    )

    fixture = registry._create_entry_memory_for_test(project_id, _schema())
    assert fixture.state.format == 2


@pytest.mark.parametrize("step", ["snapshot", "manifest", "publish"])
def test_entry_store_crash_publication_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())

    def fail_at(actual: str) -> None:
        if actual == step:
            raise OSError("injected publication failure")

    monkeypatch.setattr(registry, "_memory_transaction_step", fail_at)
    with pytest.raises(OSError, match="injected"):
        registry._compare_and_swap_entries(
            project_id,
            expected_digest=initial.digest,
            operations=(AddOperation("state", "Published", _source()),),
            reconciliation_key=_key(f"crash-{step}"),
        )

    visible = registry._entry_memory_state(project_id).state
    entries = [entry for entry in visible.entries.values() if isinstance(entry, MemoryEntry)]
    assert [entry.text for entry in entries] == (["Published"] if step == "publish" else [])

@pytest.mark.parametrize("invalid_rank", [0, 7, True])
def test_memory_source_evidence_rank_is_canonical_and_bounded(
    tmp_path: Path, invalid_rank: object
) -> None:
    registry, project_id = _registry(tmp_path)
    initial = registry._create_entry_memory_for_test(project_id, _schema())
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("decisions", "Use Postgres.", _source()),),
        reconciliation_key=_key("ranked-source"),
    )
    entry = next(
        value for value in added.state.entries.values() if isinstance(value, MemoryEntry)
    )
    encoded = canonical_state_bytes(added.state)
    assert json.loads(encoded)["entries"][entry.id]["sources"][0]["evidence_rank"] == 2
    assert state_from_bytes(encoded) == added.state

    invalid_source = dataclasses.replace(entry.sources[0], evidence_rank=invalid_rank)
    invalid_entry = dataclasses.replace(entry, sources=(invalid_source,))
    invalid_state = dataclasses.replace(
        added.state, entries={entry.id: invalid_entry}
    )
    with pytest.raises(ProjectRegistryError, match="entry source"):
        canonical_state_bytes(invalid_state)
