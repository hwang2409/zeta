from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pytest

from zeta.memory.entry_reconciler import reconcile_entry_range
from zeta.memory.entry_store import (
    AddOperation,
    ExpireOperation,
    MemoryEntry,
    MemorySource,
    ResolveOperation,
    SupersedeOperation,
    UpdateOperation,
    apply_operations,
    empty_state,
)
from zeta.memory.migration import migrate_format_one, reverse_migration
from zeta.memory.profiles import memory_profile
from zeta.memory.prompt_projection import render_entry_memory
from zeta.memory.reconciler import Transcript
from zeta.memory.user_authorization import MemoryMutationAuthorization
from zeta.project_errors import ProjectRegistryError
from zeta.project_memory_commands import run_memory_command
from zeta.project_memory_history import PROJECT_MEMORY_FILES
from zeta.project_registry import ProjectRegistry
from zeta.remote_sync import LocalTransport, push_project_memory, resolve_project_memory
from zeta.remote_sync.errors import RemoteSyncError
from zeta.remote_sync.ssh import SshTransport


def _source(seq: int = 1) -> tuple[MemorySource, ...]:
    return (
        MemorySource(
            session_id="fixture",
            seq_start=seq,
            seq_end=seq,
            origins=("user",),
            observed_at=f"2026-10-08T00:00:{seq:02d}Z",
            evidence_rank=6,
        ),
    )


def _fixture(home: Path, workspace: Path) -> tuple[ProjectRegistry, str]:
    workspace.mkdir(parents=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    registry._create_entry_memory_for_test(project.project_id, memory_profile("zeta"))
    return registry, project.project_id


def _add(registry: ProjectRegistry, project_id: str, text: str, seq: int = 1) -> str:
    before = registry._entry_memory_state(project_id)
    result = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation("state", text, _source(seq)),),
        reconciliation_key=None,
    )
    return result.receipts[0].result_ids[0]


def _active_texts(registry: ProjectRegistry, project_id: str) -> set[str]:
    state = registry._entry_memory_state(project_id).state
    return {
        entry.text
        for entry in state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.status == "active"
    }


def test_entry_sync_merges_independent_additions(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)

    _add(registry, project_id, "local addition", 1)
    _add(ProjectRegistry(second / "projects"), project_id, "remote addition", 2)

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == ()
    assert _active_texts(registry, project_id) == {"local addition", "remote addition"}
    assert _active_texts(ProjectRegistry(second / "projects"), project_id) == {
        "local addition",
        "remote addition",
    }


def test_entry_sync_conflicts_concurrent_same_id_edits(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    entry_id = _add(registry, project_id, "base", 1)
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_registry = ProjectRegistry(second / "projects")

    for owner, text, seq in ((registry, "local edit", 2), (remote_registry, "remote edit", 3)):
        before = owner._entry_memory_state(project_id)
        owner._compare_and_swap_entries(
            project_id,
            expected_digest=before.digest,
            operations=(UpdateOperation(entry_id, _source(seq), text=text),),
            reconciliation_key=None,
        )

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)
    assert result.conflicts == (entry_id,)
    assert _active_texts(registry, project_id) == {"local edit"}
    assert _active_texts(remote_registry, project_id) == {"remote edit"}

    resolved = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="local"
    )
    assert resolved.conflicts == ()
    assert _active_texts(registry, project_id) == {"local edit"}
    assert _active_texts(remote_registry, project_id) == {"local edit"}


def test_entry_sync_propagates_status_and_conflicts_schema(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    entry_id = _add(registry, project_id, "open", 1)
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_registry = ProjectRegistry(second / "projects")

    before = registry._entry_memory_state(project_id)
    from zeta.memory.entry_store import ResolveOperation

    registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(ResolveOperation(entry_id, _source(2)),),
        reconciliation_key=None,
    )
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_entry = remote_registry._entry_memory_state(project_id).state.entries[entry_id]
    assert isinstance(remote_entry, MemoryEntry)
    assert remote_entry.status == "resolved"

    # A schema is one merge unit. Concurrent revisions must not blend kind meanings.
    local = registry._entry_memory_state(project_id)
    remote = remote_registry._entry_memory_state(project_id)
    registry._replace_entry_state_for_test(
        project_id,
        dataclasses.replace(local.state, schema=dataclasses.replace(local.state.schema, version=4)),
        expected_digest=local.digest,
    )
    remote_registry._replace_entry_state_for_test(
        project_id,
        dataclasses.replace(remote.state, schema=dataclasses.replace(remote.state.schema, version=5)),
        expected_digest=remote.digest,
    )
    result = push_project_memory(first, LocalTransport(second), project_id=project_id)
    assert result.conflicts == ("schema",)


@pytest.mark.asyncio
async def test_format_two_fixture_end_to_end_updater_prompt_commands_and_sync(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    transcript = Transcript(
        "fixture-session",
        (
            {
                "seq": 1,
                "type": "message",
                "data": {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "PR 5 is ready."}],
                        "metadata": {"zeta.origin": "user"},
                    }
                },
            },
        ),
    )

    async def invoke(_prompt: str) -> str:
        return json.dumps(
            {
                "operations": [
                    {
                        "op": "add",
                        "kind": "state",
                        "text": "PR 5 is ready.",
                        "sources": [{"seq_start": 1, "seq_end": 1}],
                        "reason": "direct user state",
                    }
                ]
            }
        )

    result = await reconcile_entry_range(
        registry=registry,
        project_id=project_id,
        transcript=transcript,
        reconciliation_key=hashlib.sha256(b"fixture-range").hexdigest(),
        invoke=invoke,
        cas_retries=3,
        as_of=date(2026, 10, 8),
        now="2026-10-08T12:00:00.000000Z",
    )
    entry_id = result.changed_entry_ids[0]
    projection = render_entry_memory(
        registry._entry_memory_state(project_id).state,
        now="2026-10-08T12:00:00.000000Z",
    )
    assert "PR 5 is ready." in projection.block
    assert run_memory_command(
        registry,
        project_id,
        f"accept {entry_id}",
        authorization=MemoryMutationAuthorization.direct_slash(),
    ) == f"memory accepted: {entry_id}"

    synced = push_project_memory(first, LocalTransport(second), project_id=project_id)
    assert synced.conflicts == ()
    remote_entry = ProjectRegistry(second / "projects")._entry_memory_state(
        project_id
    ).state.entries[entry_id]
    assert isinstance(remote_entry, MemoryEntry)
    assert remote_entry.accepted_by == "user"


def test_messaging_profile_correction_completion_and_expiry_matrix() -> None:
    project_id = "p_" + "1" * 32
    initial = empty_state(project_id, memory_profile("messaging"))
    seeded, seeded_receipts = apply_operations(
        initial,
        (
            AddOperation("preferences", "Prefers amber.", _source(1)),
            AddOperation("commitments", "Send the draft.", _source(2)),
            AddOperation(
                "routines",
                "Temporary morning routine.",
                _source(3),
                expires_at="2026-10-09T00:00:00Z",
            ),
        ),
        reconciliation_key=hashlib.sha256(b"messaging-seed").hexdigest(),
        automatic=True,
        now="2026-10-08T00:00:00Z",
    )
    preference_id, commitment_id, routine_id = (
        receipt.result_ids[0] for receipt in seeded_receipts
    )
    final, _ = apply_operations(
        seeded,
        (
            SupersedeOperation(
                (preference_id,), "preferences", "Prefers cobalt.", _source(4)
            ),
            ResolveOperation(commitment_id, _source(5)),
            ExpireOperation(routine_id, "controlled clock elapsed"),
        ),
        reconciliation_key=hashlib.sha256(b"messaging-finish").hexdigest(),
        automatic=True,
        now="2026-10-10T00:00:00Z",
    )
    active = {
        entry.text
        for entry in final.entries.values()
        if isinstance(entry, MemoryEntry) and entry.status == "active"
    }
    assert active == {"Prefers cobalt."}
    assert final.entries[preference_id].status == "superseded"  # type: ignore[union-attr]
    assert final.entries[commitment_id].status == "resolved"  # type: ignore[union-attr]
    assert final.entries[routine_id].status == "expired"  # type: ignore[union-attr]


def test_entry_sync_transport_cas_rejects_changed_destination(tmp_path: Path) -> None:
    home = tmp_path / "home"
    remote = tmp_path / "remote"
    registry, project_id = _fixture(home, tmp_path / "workspace")
    push_project_memory(home, LocalTransport(remote), project_id=project_id)

    class RacingTransport(LocalTransport):
        def publish_project(self, project_id: str, snapshot: Path, *, expected_digest: str) -> None:
            _add(ProjectRegistry(self.home / "projects"), project_id, "raced", 4)
            super().publish_project(project_id, snapshot, expected_digest=expected_digest)

    _add(registry, project_id, "source", 3)
    with pytest.raises(RemoteSyncError, match="changed"):
        push_project_memory(home, RacingTransport(remote), project_id=project_id)


def test_entry_sync_uses_python_only_ssh_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "ssh"
    shim.write_text(
        "#!/bin/sh\n[ \"$1\" = -- ] && shift\nshift\nexec /bin/sh -c \"$1\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()
    registry, project_id = _fixture(first, tmp_path / "workspace")
    _add(registry, project_id, "through ssh", 1)

    result = push_project_memory(
        first,
        SshTransport("fixture", str(second), name="fixture"),
        project_id=project_id,
    )

    assert result.conflicts == ()
    assert _active_texts(ProjectRegistry(second / "projects"), project_id) == {
        "through ssh"
    }


def test_entry_sync_refuses_mixed_formats_without_mutation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _registry, project_id = _fixture(first, tmp_path / "workspace")
    legacy = ProjectRegistry(second / "projects")
    workspace = tmp_path / "legacy-workspace"
    workspace.mkdir()
    legacy_project = legacy.create_project("test", "test", workspace)
    assert legacy_project.project_id != project_id
    # Copy project metadata under the same fixture ID, but retain format 1 memory.
    source_record = first / "projects" / project_id / "project.json"
    target = second / "projects" / project_id
    target.mkdir(mode=0o700)
    (target / "project.json").write_bytes(source_record.read_bytes())
    (target / "memory").mkdir(mode=0o700)

    before = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(target.rglob("*")) if path.is_file())).hexdigest()
    with pytest.raises(RemoteSyncError, match="mixed.*migration|required",):
        push_project_memory(first, LocalTransport(second), project_id=project_id)
    after = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(target.rglob("*")) if path.is_file())).hexdigest()
    assert after == before


def _legacy_fixture(tmp_path: Path) -> tuple[ProjectRegistry, str, dict[str, str]]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("test", "test", workspace)
    contents = {
        name: f"# {name}\n\nExact {name} body.\n" if index != 2 else ""
        for index, name in enumerate(PROJECT_MEMORY_FILES)
    }
    for name, content in contents.items():
        registry.update_memory(project.project_id, {name: content})
    return registry, project.project_id, contents


def test_legacy_five_file_migration_round_trips_exactly(tmp_path: Path) -> None:
    registry, project_id, contents = _legacy_fixture(tmp_path)
    source = registry.memory_snapshot(project_id)
    plan = migrate_format_one(
        project_id=project_id,
        contents=source.contents,
        source_digest=source.digest,
        source_version=registry.memory_state(project_id).version,
        migrated_at="2026-10-08T12:00:00Z",
    )
    assert plan.rendered_mirrors == contents
    assert reverse_migration(plan) == contents
    assert migrate_format_one(
        project_id=project_id,
        contents=source.contents,
        source_digest=source.digest,
        source_version=registry.memory_state(project_id).version,
        migrated_at="2026-10-08T12:00:00Z",
    ) == plan


def test_migration_rollback_restores_format_one_pointer(tmp_path: Path) -> None:
    registry, project_id, contents = _legacy_fixture(tmp_path)
    before_version = registry.memory_state(project_id).version

    migrated = registry._migrate_memory_for_test(
        project_id, migrated_at="2026-10-08T12:00:00Z"
    )
    assert registry.memory_format(project_id) == 2
    assert migrated.source_version == before_version
    assert registry._migrate_memory_for_test(
        project_id, migrated_at="2026-10-09T12:00:00Z"
    ) == migrated

    registry._rollback_memory_migration_for_test(project_id)
    assert registry.memory_format(project_id) == 1
    assert registry.memory_state(project_id).version == before_version
    assert registry.memory_snapshot(project_id).contents == contents


def test_migration_engine_is_dormant_from_public_paths(tmp_path: Path) -> None:
    registry, project_id, _ = _legacy_fixture(tmp_path)
    assert registry.memory_format(project_id) == 1
    assert not hasattr(registry, "migrate_memory")
    assert not hasattr(registry, "rollback_memory_migration")
    with pytest.raises(ProjectRegistryError):
        registry.import_memory(project_id, object(), expected_digest="0" * 64)  # type: ignore[arg-type]
    assert registry.memory_format(project_id) == 1
