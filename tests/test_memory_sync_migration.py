from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zeta.cli import project as project_cli
from zeta.core.project_context import load_project_context, refresh_project_memory
from zeta.memory.auto import AutoMemoryConfig, AutoMemoryReconciler
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
from zeta.memory.entry_views import render_all_kinds
from zeta.memory.migration import migrate_format_one, reverse_migration
from zeta.memory.profiles import memory_profile
from zeta.memory.user_authorization import MemoryMutationAuthorization
from zeta.project_memory_commands import run_memory_command
from zeta.project_memory_history import PROJECT_MEMORY_FILES
from zeta.project_registry import ProjectRegistry
from zeta.remote_sync import (
    LocalTransport,
    pull_project_memory,
    push_project_memory,
    resolve_project_memory,
)
from zeta.remote_sync.errors import RemoteSyncError
from zeta.remote_sync.memory import project_digest
from zeta.remote_sync.project_publish import ProjectPublicationError, _validate_snapshot
from zeta.remote_sync.ssh import SshTransport
from zeta.server.project_requests import ProjectRequests
from zeta.server.protocol import FrameCodec
from zeta.server.slash_commands import ServerSlashSession
from zeta.skills import SkillCatalog
from zeta.tui.slash_handlers import SlashHandlerMixin


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
    registry.activate_entry_memory(project.project_id, "zeta")
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


def _add_kind(
    registry: ProjectRegistry, project_id: str, kind: str, text: str, seq: int
) -> str:
    before = registry._entry_memory_state(project_id)
    result = registry._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(AddOperation(kind, text, _source(seq)),),
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


def test_entry_sync_concurrent_supersession_is_one_set_conflict(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    predecessor_id = _add(registry, project_id, "base", 1)
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_registry = ProjectRegistry(second / "projects")

    replacement_ids = []
    for owner, text, seq in (
        (registry, "local replacement", 2),
        (remote_registry, "remote replacement", 3),
    ):
        before = owner._entry_memory_state(project_id)
        result = owner._compare_and_swap_entries(
            project_id,
            expected_digest=before.digest,
            operations=(
                SupersedeOperation((predecessor_id,), "state", text, _source(seq)),
            ),
            reconciliation_key=None,
        )
        replacement_ids.append(result.receipts[0].result_ids[0])

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == (predecessor_id,)
    assert _active_texts(registry, project_id) == {"local replacement"}
    assert _active_texts(remote_registry, project_id) == {"remote replacement"}
    sync_records = list((first / "projects" / project_id / "sync").glob("*.json"))
    assert len(sync_records) == 1
    conflicts = json.loads(sync_records[0].read_text())["conflicts"]
    assert len(conflicts) == 1
    assert set(conflicts[predecessor_id]["entry_ids"]) == {
        predecessor_id,
        *replacement_ids,
    }


def test_conflict_resolution_preserves_post_conflict_independent_entries(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    entry_id = _add(registry, project_id, "base", 1)
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_registry = ProjectRegistry(second / "projects")

    for owner, text, seq in (
        (registry, "local edit", 2),
        (remote_registry, "remote edit", 3),
    ):
        before = owner._entry_memory_state(project_id)
        owner._compare_and_swap_entries(
            project_id,
            expected_digest=before.digest,
            operations=(UpdateOperation(entry_id, _source(seq), text=text),),
            reconciliation_key=None,
        )
    conflicted = push_project_memory(
        first, LocalTransport(second), project_id=project_id
    )
    assert conflicted.conflicts == (entry_id,)

    independent_id = _add(remote_registry, project_id, "remote after conflict", 4)
    resolved = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="local"
    )

    assert resolved.conflicts == ()
    assert _active_texts(registry, project_id) == {
        "local edit",
        "remote after conflict",
    }
    assert _active_texts(remote_registry, project_id) == {
        "local edit",
        "remote after conflict",
    }
    assert independent_id in registry._entry_memory_state(project_id).state.entries


def test_imported_entry_can_be_undone(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    registry, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote_registry = ProjectRegistry(second / "projects")
    imported_id = _add(remote_registry, project_id, "remote addition", 1)

    push_project_memory(first, LocalTransport(second), project_id=project_id)
    assert imported_id in registry._entry_memory_state(project_id).state.entries

    result = registry._undo_memory_entry(project_id, imported_id)

    assert result.published
    assert imported_id not in result.state.entries
    assert registry._entry_memory_log(project_id, entry_id=imported_id)[-1]["kind"] == "entry-undo"


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
async def test_public_migration_end_to_end_with_real_content(tmp_path: Path) -> None:
    first = tmp_path / "home"
    second = tmp_path / "second"
    workspace = tmp_path / "workspace"
    registry, project_id, original_contents = _legacy_fixture(tmp_path)
    starting_context = load_project_context(
        cwd=workspace,
        repo_root=workspace,
        zeta_home=first,
        catalog=SkillCatalog.empty(),
        project_id=project_id,
    )
    monkey_home = os.environ.get("ZETA_HOME")
    os.environ["ZETA_HOME"] = str(first)
    try:
        assert project_cli.run(
            _cli_args(project_id, remote="migrate"),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        ) == 0
    finally:
        if monkey_home is None:
            os.environ.pop("ZETA_HOME", None)
        else:
            os.environ["ZETA_HOME"] = monkey_home

    session_dir = first / "sessions" / ("a" * 32)
    session_dir.mkdir(parents=True)
    (session_dir / "conversation.jsonl").write_text(
        json.dumps(
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
            }
        ) + "\n"
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

    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id="a" * 32,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(minimum_interval=0),
    )
    runner.before_eviction(1, 1)
    await runner.drain()
    await runner.close()
    entry_id = next(
        entry.id
        for entry in registry._entry_memory_state(project_id).state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.text == "PR 5 is ready."
    )
    resumed = refresh_project_memory(
        starting_context.system_prompt,
        home=first,
        project_id=project_id,
        memory_offset=starting_context.memory_offset,
        memory_length=starting_context.memory_length,
        memory_digest=starting_context.memory_digest,
    )
    assert "PR 5 is ready." in resumed
    assert entry_id in run_memory_command(registry, project_id, "log")
    assert run_memory_command(
        registry,
        project_id,
        f"accept {entry_id}",
        authorization=MemoryMutationAuthorization.direct_slash(),
    ) == f"memory accepted: {entry_id}"
    assert "undo complete" in run_memory_command(
        registry,
        project_id,
        f"undo {entry_id}",
        authorization=MemoryMutationAuthorization.direct_slash(),
    )

    shown = ProjectRequests(
        home=first,
        runtime=SimpleNamespace(manager=SimpleNamespace(list_sessions_read_only=list)),
        codec=FrameCodec(),
    ).dispatch(
        1,
        "project_show",
        {"project_id": project_id},
        features=frozenset({"projects-memory-v2"}),
    )
    assert shown["memory"]["profile"] == "zeta"
    assert push_project_memory(
        first, LocalTransport(second), project_id=project_id
    ).conflicts == ()
    assert ProjectRegistry(second / "projects").memory_format(project_id) == 2

    registry.rollback_memory_migration(project_id)
    assert registry.memory_format(project_id) == 1
    assert dict(registry.load_memory(project_id)) == original_contents


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

    local_before = _tree_snapshot(first / "projects" / project_id)
    remote_before = _tree_snapshot(target)
    with pytest.raises(RemoteSyncError, match="mixed.*migration|required"):
        push_project_memory(first, LocalTransport(second), project_id=project_id)
    assert _tree_snapshot(first / "projects" / project_id) == local_before
    assert _tree_snapshot(target) == remote_before


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


@pytest.mark.asyncio
async def test_inflight_format_one_reconciliation_during_migration(
    tmp_path: Path,
) -> None:
    registry, project_id, contents = _legacy_fixture(tmp_path)
    home = tmp_path / "home"
    session_dir = home / "sessions" / ("b" * 32)
    session_dir.mkdir(parents=True)
    (session_dir / "conversation.jsonl").write_text(
        json.dumps(
            {
                "seq": 1,
                "type": "message",
                "data": {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Remember in flight."}],
                        "metadata": {"zeta.origin": "user"},
                    }
                },
            }
        ) + "\n"
    )
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    async def invoke(_prompt: str) -> str:
        provider_started.set()
        await release_provider.wait()
        return json.dumps(
            {
                "changes": [
                    {
                        "file": "state.md",
                        "content": "# Current state\n\nin-flight format one write\n",
                        "sources": [
                            {
                                "session_id": "b" * 32,
                                "seq_start": 1,
                                "seq_end": 1,
                            }
                        ],
                    }
                ]
            }
        )

    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id="b" * 32,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(minimum_interval=0),
    )
    runner.before_eviction(1, 1)
    drain = asyncio.create_task(runner.drain())
    await provider_started.wait()
    await asyncio.to_thread(registry.migrate_memory, project_id)
    release_provider.set()
    await drain
    await runner.close()

    assert registry.memory_format(project_id) == 2
    snapshot = registry._entry_memory_state(project_id)
    assert snapshot.state.project_id == project_id
    assert "in-flight format one write" not in {
        entry.text
        for entry in snapshot.state.entries.values()
        if isinstance(entry, MemoryEntry)
    }
    assert {
        f"{kind}.md": content
        for kind, content in render_all_kinds(snapshot.state).items()
    } == contents


@pytest.mark.parametrize("step", ("snapshot", "manifest", "publish"))
def test_migration_publication_failure_at_each_step_leaves_readable_state(
    tmp_path: Path, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, project_id, contents = _legacy_fixture(tmp_path)

    def fail_at(actual: str) -> None:
        if actual == step:
            raise RuntimeError(f"failed at {step}")

    monkeypatch.setattr(registry, "_memory_transaction_step", fail_at)
    with pytest.raises(RuntimeError, match=f"failed at {step}"):
        registry.migrate_memory(project_id)

    fresh = ProjectRegistry(tmp_path / "home" / "projects")
    if fresh.memory_format(project_id) == 1:
        assert dict(fresh.load_memory(project_id)) == contents
    else:
        assert fresh._entry_memory_state(project_id).state.project_id == project_id


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


@pytest.mark.parametrize(
    "tamper",
    ("source-digest", "before-snapshot", "source-version", "migrated-state"),
)
def test_sync_rejects_migration_manifest_without_source_and_plan_integrity(
    tmp_path: Path, tamper: str
) -> None:
    registry, project_id, _ = _legacy_fixture(tmp_path)
    registry.migrate_memory(project_id, migrated_at="2026-10-09T00:00:00Z")
    source = registry.root / project_id
    snapshot = tmp_path / "snapshot" / project_id
    shutil.copytree(source, snapshot)
    pointer = json.loads((snapshot / "memory-current.json").read_text())
    current = pointer["current"]
    manifest_path = snapshot / "memory-versions" / "versions" / f"{current}.json"
    manifest = json.loads(manifest_path.read_text())
    blobs = snapshot / "memory-versions" / "blobs"

    if tamper == "source-digest":
        manifest["source_digest"] = "0" * 64
    elif tamper == "before-snapshot":
        before = manifest["before_snapshot"]
        before["brief.md"] = before["state.md"]
    elif tamper == "source-version":
        manifest["source_version"] = pointer["history"][0]
    else:
        state_digest = manifest["snapshot"]
        state = json.loads((blobs / state_digest).read_text())
        entry = next(iter(state["entries"].values()))
        entry["text"] = "Tampered migrated fact."
        payload = json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        replacement = hashlib.sha256(payload).hexdigest()
        (blobs / replacement).write_bytes(payload)
        (blobs / replacement).chmod(0o600)
        manifest["snapshot"] = replacement
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(ProjectPublicationError, match="migration"):
        _validate_snapshot(snapshot, project_id)


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
    pointer_path = registry.root / project_id / "memory-current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["history"] = [pointer["current"]]
    pointer_path.write_text(json.dumps(pointer))

    registry._rollback_memory_migration_for_test(project_id)
    assert registry.memory_format(project_id) == 1
    assert registry.memory_state(project_id).version == before_version
    assert registry.memory_snapshot(project_id).contents == contents


def _tree_snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (
            path.stat(follow_symlinks=False).st_mode,
            path.read_bytes() if path.is_file() else b"",
        )
        for path in sorted(root.rglob("*"))
    }


def _format_snapshot(home: Path) -> dict[str, int]:
    projects = home / "projects"
    if not projects.exists():
        return {}
    registry = ProjectRegistry(projects)
    return {
        path.name: registry.memory_format(path.name)
        for path in sorted(projects.glob("p_*"))
        if path.is_dir()
    }


def _cli_args(project_id: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "project_verb": "memory",
        "project": project_id,
        "directory": ".",
        "name": None,
        "scope": "",
        "canonical_integration_root": None,
        "remote": None,
        "action": None,
        "sync_project": None,
        "remote_home": None,
        "accept": None,
        "set": [],
        "from_file": [],
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry_point",
    (
        "cli-read",
        "cli-set",
        "cli-from-file",
        "cli-accept",
        "tui-log",
        "tui-undo",
        "tui-accept",
        "serve-log",
        "serve-undo",
        "serve-accept",
        "serve-project-request",
        "sync-push",
        "sync-pull",
        "project-create",
        "project-init",
        "automatic-updater",
    ),
)
async def test_dormancy_public_entry_points_leave_format_one_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    project_id = project.project_id
    remote = tmp_path / "remote"
    before = {
        "local": _format_snapshot(home),
        "remote": _format_snapshot(remote),
    }
    monkeypatch.setenv("ZETA_HOME", str(home))

    if entry_point.startswith("cli-"):
        args = _cli_args(project_id)
        if entry_point == "cli-set":
            args.set = [["state.md", "# Current state\n\nset\n"]]
        elif entry_point == "cli-from-file":
            source = tmp_path / "state.md"
            source.write_text("# Current state\n\nfrom file\n")
            args.from_file = [["state.md", str(source)]]
        elif entry_point == "cli-accept":
            current = registry.memory_snapshot(project_id)
            registry.compare_and_swap_memory(
                project_id,
                expected_digest=current.digest,
                updates={"state.md": "# Current state\n\nautomatic\n"},
                provenance={"session_id": "fixture", "seq_start": 1, "seq_end": 1},
            )
            args = _cli_args(
                "accept", remote="state.md", directory=str(workspace)
            )
            monkeypatch.chdir(workspace)
        with patch.object(
            MemoryMutationAuthorization, "authorize", return_value=None
        ):
            assert project_cli.run(
                args, stdout=io.StringIO(), stderr=io.StringIO()
            ) == 0
    elif entry_point.startswith(("tui-", "serve-")):
        action = entry_point.split("-", 1)[1]
        if action == "undo":
            registry.update_memory(
                project_id, {"state.md": "# Current state\n\nundo target\n"}
            )
        elif action == "accept":
            current = registry.memory_snapshot(project_id)
            registry.compare_and_swap_memory(
                project_id,
                expected_digest=current.digest,
                updates={"state.md": "# Current state\n\nautomatic\n"},
                provenance={"session_id": "fixture", "seq_start": 1, "seq_end": 1},
            )
            action = "accept state.md"
        if entry_point.startswith("tui-"):
            tui = SimpleNamespace(
                loop=SimpleNamespace(
                    project_registry=registry,
                    session_metadata=SimpleNamespace(project_id=project_id),
                    memory_reconciler=None,
                )
            )
            SlashHandlerMixin.slash_memory(tui, action)
        else:
            runtime = SimpleNamespace(
                metadata=SimpleNamespace(project_id=project_id),
                manager=SimpleNamespace(project_registry=registry),
                loop=None,
            )
            ServerSlashSession(runtime).slash_memory(action)
    elif entry_point == "serve-project-request":
        runtime = SimpleNamespace(
            manager=SimpleNamespace(list_sessions_read_only=list)
        )
        ProjectRequests(home=home, runtime=runtime, codec=FrameCodec()).dispatch(
            1, "project_show", {"project_id": project_id}
        )
    elif entry_point == "sync-push":
        push_project_memory(home, LocalTransport(remote), project_id=project_id)
    elif entry_point == "sync-pull":
        push_project_memory(home, LocalTransport(remote), project_id=project_id)
        ProjectRegistry(remote / "projects").update_memory(
            project_id, {"state.md": "# Current state\n\nremote\n"}
        )
        pull_project_memory(home, LocalTransport(remote), project_id=project_id)
    elif entry_point == "project-create":
        args = _cli_args(
            "unused",
            project_verb="create",
            name="created",
            scope="scope",
            canonical_integration_root=str(tmp_path / "created"),
        )
        assert project_cli.run(
            args, stdout=io.StringIO(), stderr=io.StringIO()
        ) == 0
    elif entry_point == "project-init":
        new_workspace = tmp_path / "initialized"
        new_workspace.mkdir()
        args = _cli_args(
            "unused",
            project_verb="init",
            directory=str(new_workspace),
            name="initialized",
            scope="scope",
        )
        assert project_cli.run(
            args, stdout=io.StringIO(), stderr=io.StringIO()
        ) == 0
    else:
        session_dir = home / "sessions" / ("a" * 32)
        session_dir.mkdir(parents=True)
        (session_dir / "conversation.jsonl").write_text(
            json.dumps(
                {
                    "seq": 1,
                    "id": "message",
                    "parent_id": None,
                    "type": "message",
                    "data": {
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "remember this"}],
                            "metadata": {"zeta.origin": "user"},
                        }
                    },
                }
            )
            + "\n"
        )

        async def invoke(prompt: str) -> str:
            expected = json.loads(
                prompt.split("Current digest: ", 1)[1].splitlines()[0]
            )
            return json.dumps(
                {
                    "base_digest": expected,
                    "changes": [],
                }
            )

        runner = AutoMemoryReconciler(
            registry=registry,
            project_id=project_id,
            session_id="a" * 32,
            session_dir=session_dir,
            invoke=invoke,
            config=AutoMemoryConfig(minimum_interval=0),
        )
        runner.before_eviction(1, 1)
        await runner.drain()
        await runner.close()

    after = {
        "local": _format_snapshot(home),
        "remote": _format_snapshot(remote),
    }
    assert set(before["local"]).issubset(after["local"])
    assert after["local"][project_id] == 1
    if entry_point in {"project-create", "project-init"}:
        assert list(after["local"].values()).count(2) == 1
    else:
        assert all(
            memory_format == 1
            for stores in after.values()
            for memory_format in stores.values()
        )


def test_migration_engine_activates_only_the_explicit_project(tmp_path: Path) -> None:
    registry, project_id, _ = _legacy_fixture(tmp_path)
    other = registry.create_project("other", "test", tmp_path / "other")

    registry.migrate_memory(project_id, migrated_at="2026-10-09T00:00:00Z")

    assert registry.memory_format(project_id) == 2
    assert registry.memory_format(other.project_id) == 1


def _replace_schema(registry: ProjectRegistry, project_id: str, schema: object) -> None:
    current = registry._entry_memory_state(project_id)
    registry._replace_entry_state_for_test(
        project_id,
        dataclasses.replace(current.state, schema=schema),
        expected_digest=current.digest,
    )


def _supersede(
    registry: ProjectRegistry, project_id: str, entry_id: str, text: str, seq: int
) -> str:
    current = registry._entry_memory_state(project_id)
    result = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(SupersedeOperation((entry_id,), "state", text, _source(seq)),),
        reconciliation_key=None,
    )
    return result.receipts[0].result_ids[0]


def _create_entry_conflict(
    first: Path, second: Path
) -> tuple[ProjectRegistry, ProjectRegistry, str, str]:
    local, project_id = _fixture(first, first.parent / "workspace")
    entry_id = _add(local, project_id, "base", 1)
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote = ProjectRegistry(second / "projects")
    for owner, text, seq in ((local, "local edit", 2), (remote, "remote edit", 3)):
        current = owner._entry_memory_state(project_id)
        owner._compare_and_swap_entries(
            project_id,
            expected_digest=current.digest,
            operations=(UpdateOperation(entry_id, _source(seq), text=text),),
            reconciliation_key=None,
        )
    assert push_project_memory(
        first, LocalTransport(second), project_id=project_id
    ).conflicts == (entry_id,)
    return local, remote, project_id, entry_id


def test_resolve_remote_after_post_conflict_supersession(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, entry_id = _create_entry_conflict(first, second)
    replacement_id = _supersede(remote, project_id, entry_id, "remote replacement", 4)

    result = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="remote"
    )

    assert result.conflicts == ()
    assert _active_texts(local, project_id) == {"remote replacement"}
    assert replacement_id in local._entry_memory_state(project_id).state.entries


def test_resolve_local_after_post_conflict_supersession(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, entry_id = _create_entry_conflict(first, second)
    replacement_id = _supersede(local, project_id, entry_id, "local replacement", 4)

    result = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="local"
    )

    assert result.conflicts == ()
    assert _active_texts(remote, project_id) == {"local replacement"}
    assert replacement_id in remote._entry_memory_state(project_id).state.entries


def test_conflict_record_expands_to_current_closure(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _, remote, project_id, entry_id = _create_entry_conflict(first, second)
    replacement_id = _supersede(remote, project_id, entry_id, "remote replacement", 4)

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == (min(entry_id, replacement_id),)
    state_path = next((first / "projects" / project_id / "sync").glob("*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    conflict = next(iter(state["conflicts"].values()))
    assert set(conflict["entry_ids"]) == {entry_id, replacement_id}


def _create_schema_conflict_with_incompatible_remote_entry(
    first: Path, second: Path
) -> tuple[ProjectRegistry, ProjectRegistry, str, str]:
    local, project_id = _fixture(first, first.parent / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    remote = ProjectRegistry(second / "projects")
    _replace_schema(local, project_id, memory_profile("messaging"))
    remote_state = remote._entry_memory_state(project_id)
    _replace_schema(
        remote,
        project_id,
        dataclasses.replace(remote_state.state.schema, version=4),
    )
    remote_entry_id = _add(remote, project_id, "remote zeta state", 5)
    return local, remote, project_id, remote_entry_id


def test_concurrent_schema_change_with_incompatible_entry_records_schema_conflict(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, remote_entry_id = (
        _create_schema_conflict_with_incompatible_remote_entry(first, second)
    )

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == ("schema",)
    assert local._entry_memory_state(project_id).state.schema.profile == "messaging"
    assert remote._entry_memory_state(project_id).state.schema.profile == "zeta"
    assert remote_entry_id not in local._entry_memory_state(project_id).state.entries
    assert remote_entry_id in remote._entry_memory_state(project_id).state.entries


def test_schema_conflict_resolution_handles_incompatible_kinds(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, remote_entry_id = (
        _create_schema_conflict_with_incompatible_remote_entry(first, second)
    )
    assert push_project_memory(
        first, LocalTransport(second), project_id=project_id
    ).conflicts == ("schema",)

    result = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="remote"
    )

    assert result.conflicts == ()
    for registry in (local, remote):
        state = registry._entry_memory_state(project_id).state
        assert state.schema.profile == "zeta"
        assert remote_entry_id in state.entries


def test_schema_conflict_defers_post_conflict_incompatible_additions(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, remote_entry_id = (
        _create_schema_conflict_with_incompatible_remote_entry(first, second)
    )
    assert push_project_memory(
        first, LocalTransport(second), project_id=project_id
    ).conflicts == ("schema",)
    local_entry_id = _add_kind(local, project_id, "people", "local person", 6)
    second_remote_id = _add(remote, project_id, "later remote state", 7)

    unresolved = push_project_memory(
        first, LocalTransport(second), project_id=project_id
    )

    assert unresolved.conflicts == ("schema",)
    assert set(local._entry_memory_state(project_id).state.entries) == {local_entry_id}
    assert set(remote._entry_memory_state(project_id).state.entries) == {
        remote_entry_id,
        second_remote_id,
    }
    resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="local"
    )
    assert set(local._entry_memory_state(project_id).state.entries) == {local_entry_id}
    assert set(remote._entry_memory_state(project_id).state.entries) == {local_entry_id}


class _FailingPublishTransport(LocalTransport):
    def __init__(self, home: Path, *, after_publish: bool) -> None:
        super().__init__(home)
        self.after_publish = after_publish
        self.failed = False

    def publish_project(
        self, project_id: str, snapshot: Path, *, expected_digest: str
    ) -> None:
        if not self.failed and not self.after_publish:
            self.failed = True
            raise RemoteSyncError("injected failure before remote publish")
        super().publish_project(project_id, snapshot, expected_digest=expected_digest)
        if not self.failed:
            self.failed = True
            raise RemoteSyncError("injected failure after remote publish")


def test_sync_recovers_after_failure_between_remote_and_local_publish(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    entry_id = _add(local, project_id, "must survive", 1)
    transport = _FailingPublishTransport(second, after_publish=True)
    with pytest.raises(RemoteSyncError, match="after remote publish"):
        push_project_memory(first, transport, project_id=project_id)
    newer_entry_id = _add(local, project_id, "newer local change", 2)

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == ()
    for registry in (local, ProjectRegistry(second / "projects")):
        entries = registry._entry_memory_state(project_id).state.entries
        assert entry_id in entries
        assert newer_entry_id in entries


def test_initial_sync_recovers_after_remote_only_publish(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    entry_id = _add(local, project_id, "initial entry", 1)
    with pytest.raises(RemoteSyncError, match="after remote publish"):
        push_project_memory(
            first,
            _FailingPublishTransport(second, after_publish=True),
            project_id=project_id,
        )

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == ()
    for registry in (local, ProjectRegistry(second / "projects")):
        assert entry_id in registry._entry_memory_state(project_id).state.entries


def test_resolution_recovers_after_failure_between_remote_and_local_publish(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, entry_id = _create_entry_conflict(first, second)
    transport = _FailingPublishTransport(second, after_publish=True)
    with pytest.raises(RemoteSyncError, match="after remote publish"):
        resolve_project_memory(first, transport, project_id=project_id, accept="remote")

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == ()
    assert _active_texts(local, project_id) == {"remote edit"}
    assert _active_texts(remote, project_id) == {"remote edit"}
    assert entry_id in local._entry_memory_state(project_id).state.entries


def test_recovery_does_not_reapply_stale_resolution_over_newer_edit(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, entry_id = _create_entry_conflict(first, second)
    with pytest.raises(RemoteSyncError, match="after remote publish"):
        resolve_project_memory(
            first,
            _FailingPublishTransport(second, after_publish=True),
            project_id=project_id,
            accept="local",
        )
    before = remote._entry_memory_state(project_id)
    remote._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(
            UpdateOperation(entry_id, _source(4), text="remote post-publish choice"),
        ),
        reconciliation_key=None,
    )

    result = push_project_memory(first, LocalTransport(second), project_id=project_id)

    assert result.conflicts == (entry_id,)
    assert _active_texts(local, project_id) == {"local edit"}
    assert _active_texts(remote, project_id) == {"remote post-publish choice"}


def test_new_explicit_resolution_supersedes_pending_choice_or_fails_clearly(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, remote, project_id, entry_id = _create_entry_conflict(first, second)
    with pytest.raises(RemoteSyncError, match="after remote publish"):
        resolve_project_memory(
            first,
            _FailingPublishTransport(second, after_publish=True),
            project_id=project_id,
            accept="local",
        )
    before = remote._entry_memory_state(project_id)
    remote._compare_and_swap_entries(
        project_id,
        expected_digest=before.digest,
        operations=(
            UpdateOperation(entry_id, _source(4), text="remote post-publish choice"),
        ),
        reconciliation_key=None,
    )

    result = resolve_project_memory(
        first, LocalTransport(second), project_id=project_id, accept="remote"
    )

    assert result.conflicts == ()
    assert _active_texts(local, project_id) == {"remote post-publish choice"}
    assert _active_texts(remote, project_id) == {"remote post-publish choice"}


def test_sync_failure_before_remote_publish_is_no_op(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    _add(local, project_id, "pending local", 1)
    before_local = project_digest(first / "projects" / project_id)
    before_remote = project_digest(second / "projects" / project_id)

    with pytest.raises(RemoteSyncError, match="before remote publish"):
        push_project_memory(
            first,
            _FailingPublishTransport(second, after_publish=False),
            project_id=project_id,
        )

    assert project_digest(first / "projects" / project_id) == before_local
    assert project_digest(second / "projects" / project_id) == before_remote


def _run_killed_publication(
    first: Path, second: Path, project_id: str, boundary: str
) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        from zeta.remote_sync import LocalTransport, push_project_memory
        import zeta.remote_sync.project_publish as publication

        first, second, project_id, boundary = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

        def killing_step(step):
            if boundary in {"blob", "orphan"} and step == "snapshot":
                os._exit(91)
            if boundary == "manifest" and step == "manifest":
                os._exit(92)
            if boundary == "pointer" and step == "publish":
                os._exit(93)

        publication.transaction_step = killing_step
        push_project_memory(first, LocalTransport(second), project_id=project_id)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(first), str(second), project_id, boundary],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("boundary", ("blob", "manifest", "pointer"))
def test_registry_reads_never_see_missing_project_during_memory_sync(
    tmp_path: Path, boundary: str
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    _add(local, project_id, "pending publication", 1)

    killed = _run_killed_publication(first, second, project_id, boundary)

    assert killed.returncode in {91, 92, 93, 95}, killed.stderr
    fresh = ProjectRegistry(second / "projects")
    assert project_id in {project.project_id for project in fresh.list_projects()}
    assert fresh.show_project(project_id).project_id == project_id
    assert fresh._entry_memory_state(project_id).state.project_id == project_id


def test_memory_sync_does_not_replace_project_directory(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    _add(local, project_id, "pending publication", 1)
    project = first / "projects" / project_id
    inode = project.stat().st_ino
    concurrent = project / "inbox" / "during-sync.json"

    class ConcurrentWriter(LocalTransport):
        def publish_project(
            self, project_id: str, snapshot: Path, *, expected_digest: str
        ) -> None:
            super().publish_project(project_id, snapshot, expected_digest=expected_digest)
            concurrent.parent.mkdir()
            concurrent.write_text('{"kept": true}\n', encoding="utf-8")

    push_project_memory(first, ConcurrentWriter(second), project_id=project_id)

    assert project.stat().st_ino == inode
    assert concurrent.read_text(encoding="utf-8") == '{"kept": true}\n'


def test_interrupted_sync_leaves_no_orphan_artifacts(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    _add(local, project_id, "pending publication", 1)

    killed = _run_killed_publication(first, second, project_id, "orphan")
    assert killed.returncode in {91, 94}, killed.stderr
    push_project_memory(first, LocalTransport(second), project_id=project_id)

    for home in (first, second):
        projects = home / "projects"
        assert not list(projects.glob(f".{project_id}.install-*"))
        assert not list(projects.glob(f".{project_id}.backup-*"))
        assert not list(projects.glob(f".{project_id}.replace-*"))
        assert not list(projects.glob(f".{project_id}.incoming-*"))
        project = projects / project_id
        pointer = json.loads((project / "memory-current.json").read_text())
        referenced: set[str] = set()
        for version in pointer["history"]:
            manifest = json.loads(
                (project / "memory-versions" / "versions" / f"{version}.json").read_text()
            )
            for field in ("snapshot", "before_snapshot"):
                value = manifest[field]
                referenced.update(value.values() if isinstance(value, dict) else (value,))
        blobs = {path.name for path in (project / "memory-versions" / "blobs").iterdir()}
        assert blobs <= referenced

def test_sync_state_size_bounded_over_add_sync_compact_cycles(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    local, project_id = _fixture(first, tmp_path / "workspace")
    push_project_memory(first, LocalTransport(second), project_id=project_id)
    for seq in range(1, 41):
        _add(local, project_id, f"temporary {seq}", seq)
        push_project_memory(first, LocalTransport(second), project_id=project_id)
        for registry in (local, ProjectRegistry(second / "projects")):
            current = registry._entry_memory_state(project_id)
            registry._replace_entry_state_for_test(
                project_id,
                dataclasses.replace(current.state, entries={}),
                expected_digest=current.digest,
            )
        push_project_memory(first, LocalTransport(second), project_id=project_id)

    state_path = next((first / "projects" / project_id / "sync").glob("*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["entries"] == {}
    assert state_path.stat().st_size < 4096
