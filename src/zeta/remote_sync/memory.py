"""Transactional project-memory synchronization.

This module owns the complete memory-sync transaction: local snapshot locking,
remote snapshot acquisition, baseline and conflict interpretation, and CAS
publication. Transports only fetch and publish validated project snapshots.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from ..memory.entry_store import state_digest
from ..memory.entry_sync import (
    EntryMemoryExport,
    ResolutionCandidate,
    merge_entry_states,
    merge_version_receipts,
    recoverable_resolutions,
)
from ..project_errors import UnsupportedMemoryFormatError
from ..project_memory_history import MAX_MEMORY_MIRROR_FILE_SIZE, MemoryExport
from ..project_registry import MAX_RECORD_SIZE, ProjectRegistry, ProjectRegistryError
from .errors import RemoteSyncError

MEMORY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
_EXCLUDED_NAMES = frozenset({".lock", ".spill.lock"})
_MISSING = "missing"
_MACHINE_ID = ".machine-id"
_MAX_STATE_BYTES = 512 * 1024
MemorySyncExport = MemoryExport | EntryMemoryExport


@dataclass(frozen=True, slots=True)
class MemoryTransferResult:
    project_id: str
    updated: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SharedSyncState:
    state: dict[str, object]
    previous_digest: str
    transition_id: str | None = None
    resolutions: dict[str, Literal["local", "remote"]] | None = None
    candidates: dict[str, ResolutionCandidate] | None = None


class _Digest(Protocol):
    def update(self, data: bytes) -> object: ...


class ProjectSnapshotTransport(Protocol):
    """Transport seam for validated project snapshots and CAS publication."""

    name: str | None
    machine_id: str

    def fetch_project(self, project_id: str, destination: Path) -> str: ...

    def publish_project(
        self, project_id: str, snapshot: Path, *, expected_digest: str
    ) -> None: ...


def sync_project_memory(
    home: Path,
    transport: ProjectSnapshotTransport,
    *,
    project_id: str,
    direction: Literal["push", "pull"],
) -> MemoryTransferResult:
    """Synchronize one project while holding its local registry lease."""

    local_id = _machine_id(home)
    peer = _peer_machine_id(transport)
    _reject_same_machine(local_id, peer)
    state_key = _state_key(local_id, peer)
    local_project = home / "projects" / project_id
    with _registry_lock(home / "projects"):
        _recover_interrupted_replacement(local_project)
        local_expected = project_digest(local_project)
        if local_expected == _MISSING and direction == "push":
            raise RemoteSyncError(f"project {project_id} was not found")
        with tempfile.TemporaryDirectory(prefix="zeta-memory-sync-") as temporary:
            root = Path(temporary)
            local_root = root / "local"
            remote_root = root / "remote"
            local_root.mkdir(mode=0o700)
            remote_root.mkdir(mode=0o700)
            local = local_root / project_id
            remote = remote_root / project_id
            local_export: MemorySyncExport | None = None
            if local_expected != _MISSING:
                copy_project_snapshot(local_project, local)
                _validate_project_snapshot(local, project_id)
                local_export = _materialize_memory_export(local, project_id)
            remote_expected = transport.fetch_project(project_id, remote)
            if remote_expected == _MISSING:
                if direction == "pull":
                    raise RemoteSyncError(f"remote project {project_id} was not found")
                copy_project_snapshot(local, remote)
                remote_export = local_export
            else:
                _validate_project_snapshot(remote, project_id)
                remote_export = _materialize_memory_export(remote, project_id)
            if local_expected == _MISSING:
                copy_project_snapshot(remote, local)
                local_export = remote_export
            if local_export is None or remote_export is None:
                raise RemoteSyncError("project memory snapshot is missing")
            source, destination = (local, remote) if direction == "push" else (remote, local)
            source_export, destination_export = (
                (local_export, remote_export)
                if direction == "push"
                else (remote_export, local_export)
            )
            if type(source_export) is not type(destination_export):
                raise RemoteSyncError(
                    "mixed project memory formats cannot synchronize; migration is required"
                )
            memory_format = 2 if isinstance(source_export, EntryMemoryExport) else 1
            shared = _shared_state(
                source, destination, state_key, project_id, memory_format=memory_format
            )
            state = shared.state
            if isinstance(local_export, EntryMemoryExport) and isinstance(
                remote_export, EntryMemoryExport
            ):
                recovery_resolutions = _stored_recovery_resolutions(
                    local_export, remote_export, shared
                )
                (
                    result,
                    local_merged,
                    remote_merged,
                    local_changed,
                    remote_changed,
                    transition_candidates,
                ) = _merge_entries(
                    local_export,
                    remote_export,
                    state=state,
                    project_id=project_id,
                    resolutions=recovery_resolutions,
                )
                _finish_entry_transition(
                    state,
                    shared,
                    recovery_resolutions,
                    transition_candidates,
                )
                merged = remote_merged
                source_merged = local_merged
                source = local
                destination = remote
            else:
                result, merged, source_merged = _merge(
                    source,
                    destination,
                    source_export=source_export,
                    destination_export=destination_export,
                    state=state,
                    project_id=project_id,
                    source_label="local" if direction == "push" else peer,
                )
            _write_state(local, state_key, state)
            _write_state(remote, state_key, state)
            _import_merged_memory(
                destination,
                project_id,
                merged,
                changed_entry_ids=(
                    remote_changed
                    if isinstance(merged, EntryMemoryExport)
                    else ()
                ),
                provenance={"source": "remote_sync", "peer": peer},
            )
            if source_merged is not None:
                _import_merged_memory(
                    source,
                    project_id,
                    source_merged,
                    changed_entry_ids=local_changed,
                    provenance={"source": "remote_sync", "peer": peer},
                )
            transport.publish_project(
                project_id, remote, expected_digest=remote_expected
            )
            if project_digest(local_project) != local_expected:
                raise RemoteSyncError("local project changed during memory sync; retry")
            _atomic_replace_directory(local, local_project)
            return result


def resolve_project_memory(
    home: Path,
    transport: ProjectSnapshotTransport,
    *,
    project_id: str,
    accept: Literal["local", "remote"],
) -> MemoryTransferResult:
    """Resolve all recorded conflicts by explicitly accepting one side."""

    if accept not in {"local", "remote"}:
        raise RemoteSyncError("accept must be local or remote")
    local_id = _machine_id(home)
    peer = _peer_machine_id(transport)
    _reject_same_machine(local_id, peer)
    state_key = _state_key(local_id, peer)
    local_project = home / "projects" / project_id
    with _registry_lock(home / "projects"):
        _recover_interrupted_replacement(local_project)
        if not local_project.is_dir():
            raise RemoteSyncError(f"project {project_id} was not found")
        local_expected = project_digest(local_project)
        with tempfile.TemporaryDirectory(prefix="zeta-memory-resolve-") as temporary:
            root = Path(temporary)
            local_root = root / "local"
            remote_root = root / "remote"
            local_root.mkdir(mode=0o700)
            remote_root.mkdir(mode=0o700)
            local = local_root / project_id
            remote = remote_root / project_id
            copy_project_snapshot(local_project, local)
            _validate_project_snapshot(local, project_id)
            local_export = _materialize_memory_export(local, project_id)
            remote_expected = transport.fetch_project(project_id, remote)
            if remote_expected == _MISSING:
                raise RemoteSyncError(f"remote project {project_id} was not found")
            _validate_project_snapshot(remote, project_id)
            remote_export = _materialize_memory_export(remote, project_id)
            shared = _shared_state(local, remote, state_key, project_id)
            state = shared.state
            conflicts = state["conflicts"]
            if not conflicts:
                raise RemoteSyncError("project memory has no unresolved conflicts")
            chosen = local if accept == "local" else remote
            other = remote if accept == "local" else local
            chosen_export = local_export if accept == "local" else remote_export
            other_export = remote_export if accept == "local" else local_export
            if type(chosen_export) is not type(other_export):
                raise RemoteSyncError(
                    "mixed project memory formats cannot synchronize; migration is required"
                )
            if isinstance(local_export, EntryMemoryExport) and isinstance(
                remote_export, EntryMemoryExport
            ):
                resolution_choices = _resolution_choices(
                    local_export, remote_export, shared, conflicts, accept
                )
                (
                    merged_result,
                    local_merged,
                    remote_merged,
                    local_changed,
                    remote_changed,
                    transition_candidates,
                ) = _merge_entries(
                    local_export,
                    remote_export,
                    state=state,
                    project_id=project_id,
                    resolutions=resolution_choices,
                )
                _finish_entry_transition(
                    state,
                    shared,
                    resolution_choices,
                    transition_candidates,
                )
                _write_state(local, state_key, state)
                _write_state(remote, state_key, state)
                _import_merged_memory(
                    local,
                    project_id,
                    local_merged,
                    changed_entry_ids=local_changed,
                    provenance={"source": "remote_sync", "peer": peer},
                )
                _import_merged_memory(
                    remote,
                    project_id,
                    remote_merged,
                    changed_entry_ids=remote_changed,
                    provenance={"source": "remote_sync", "peer": peer},
                )
                transport.publish_project(
                    project_id, remote, expected_digest=remote_expected
                )
                if project_digest(local_project) != local_expected:
                    raise RemoteSyncError(
                        "local project changed during memory resolution; retry"
                    )
                _atomic_replace_directory(local, local_project)
                return merged_result
            assert isinstance(chosen_export, MemoryExport)
            assert isinstance(other_export, MemoryExport)
            contents = dict(other_export.contents)
            automatic = set(other_export.automatic_files)
            updated: list[str] = []
            files = state["files"]
            for name in sorted(conflicts):
                _copy_memory_file(chosen / "memory" / name, other / "memory" / name)
                contents[name] = chosen_export.contents[name]
                if name in chosen_export.automatic_files:
                    automatic.add(name)
                else:
                    automatic.discard(name)
                files[name] = _file_digest(chosen / "memory" / name)
                updated.append(name)
            state["conflicts"] = {}
            _write_state(local, state_key, state)
            _write_state(remote, state_key, state)
            merged = MemoryExport(
                contents,
                ProjectRegistry._memory_digest_value(contents),
                _merge_versions(local_export.versions, remote_export.versions),
                tuple(sorted(automatic)),
            )
            _import_merged_memory(
                other,
                project_id,
                merged,
                provenance={"source": "remote_sync", "peer": peer},
            )
            transport.publish_project(
                project_id, remote, expected_digest=remote_expected
            )
            if project_digest(local_project) != local_expected:
                raise RemoteSyncError("local project changed during memory resolution; retry")
            _atomic_replace_directory(local, local_project)
            return MemoryTransferResult(project_id, tuple(updated), ())


def fetch_local_project(
    home: Path, project_id: str, destination: Path, *, peer: str
) -> str:
    """Fetch one local-adapter snapshot under a non-blocking remote lease."""

    project = home / "projects" / project_id
    with _registry_lock(home / "projects", blocking=False, peer=peer):
        _recover_interrupted_replacement(project)
        if not project.is_dir():
            return _MISSING
        for name in MEMORY_FILES:
            path = project / "memory" / name
            if path.is_file() and path.stat().st_size > MAX_MEMORY_MIRROR_FILE_SIZE:
                raise RemoteSyncError(f"memory file {name} is too large")
        digest = project_digest(project)
        copy_project_snapshot(project, destination)
        return digest


def publish_local_project(
    home: Path,
    project_id: str,
    snapshot: Path,
    *,
    expected_digest: str,
) -> None:
    """CAS-publish one local-adapter snapshot under the registry lease."""

    _validate_project_snapshot(snapshot, project_id)
    project = home / "projects" / project_id
    with _registry_lock(home / "projects"):
        _recover_interrupted_replacement(project)
        if project_digest(project) != expected_digest:
            raise RemoteSyncError("remote changed during transfer; retry after inspection")
        project.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = project.parent / f".{project_id}.incoming-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        copy_project_snapshot(snapshot, staging)
        _atomic_replace_directory(staging, project)


def copy_project_snapshot(source: Path, destination: Path) -> None:
    """Copy only project identity, standard memory, history, and sync state."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = source / "project.json"
    if not record.is_file() or record.is_symlink():
        raise RemoteSyncError("project record is missing or unsafe")
    _atomic_copy_file(record, destination / "project.json")
    source_memory = source / "memory"
    destination_memory = destination / "memory"
    destination_memory.mkdir(mode=0o700)
    try:
        memory_info = source_memory.lstat()
    except FileNotFoundError:
        memory_info = None
    if memory_info is not None:
        if not stat.S_ISDIR(memory_info.st_mode) or stat.S_ISLNK(memory_info.st_mode):
            raise RemoteSyncError("project memory directory is unsafe")
        for path in sorted(source_memory.iterdir()):
            if path.name in MEMORY_FILES or any(
                path.name.startswith(f"{name}.conflict-") for name in MEMORY_FILES
            ):
                if not path.is_file() or path.is_symlink():
                    raise RemoteSyncError("project memory file is unsafe")
                _atomic_copy_file(path, destination_memory / path.name)
    for relative in (
        Path("memory/history"),
        Path("memory-versions"),
        Path("memory-current.json"),
        Path("history"),
        Path("sync"),
    ):
        path = source / relative
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise RemoteSyncError(f"project snapshot member is unsafe: {relative}")
        if stat.S_ISDIR(info.st_mode):
            _copy_tree(path, destination / relative)
        elif stat.S_ISREG(info.st_mode):
            _atomic_copy_file(path, destination / relative)
        else:
            raise RemoteSyncError(f"project snapshot member is unsafe: {relative}")


def _import_merged_memory(
    target: Path,
    project_id: str,
    merged: MemorySyncExport,
    *,
    changed_entry_ids: tuple[str, ...] = (),
    provenance: dict[str, str],
) -> None:
    registry = ProjectRegistry(target.parent)
    if isinstance(merged, EntryMemoryExport):
        current = registry._export_entry_memory(project_id)
        registry._import_entry_memory(
            project_id,
            merged,
            expected_digest=current.digest,
            changed_entry_ids=changed_entry_ids,
            provenance=provenance,
        )
    else:
        current = registry.export_memory(project_id)
        registry.import_memory(
            project_id,
            merged,
            expected_digest=current.digest,
            provenance=provenance,
        )


def _materialize_memory_export(snapshot: Path, project_id: str) -> MemorySyncExport:
    try:
        registry = ProjectRegistry(snapshot.parent)
        if registry.memory_format(project_id) == 2:
            return registry._export_entry_memory(project_id)
        exported = registry.export_memory(project_id)
    except ProjectRegistryError as exc:
        raise RemoteSyncError("invalid project memory store") from exc
    for name, content in exported.contents.items():
        _atomic_write_file(content.encode("utf-8"), snapshot / "memory" / name)
    return exported


def project_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return _MISSING
    has_version_store = (root / "memory-current.json").is_file()
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name in _EXCLUDED_NAMES
            or (
                has_version_store
                and path.parent == root / "memory"
                and path.name in MEMORY_FILES
            )
        ):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        _update_digest_from_file(digest, path)
    return digest.hexdigest()


def _validate_project_snapshot(snapshot: Path, project_id: str) -> None:
    if snapshot.name != project_id:
        raise RemoteSyncError("project snapshot path does not match its project ID")
    try:
        registry = ProjectRegistry(snapshot.parent)
        registry.show_project(project_id)
        for name in MEMORY_FILES:
            path = snapshot / "memory" / name
            if path.is_file() and path.stat().st_size > MAX_MEMORY_MIRROR_FILE_SIZE:
                raise ProjectRegistryError(f"memory file {name} is too large")
        if registry.memory_format(project_id) == 2:
            registry._entry_memory_state(project_id)
        else:
            registry.load_memory(project_id, byte_cap=MAX_RECORD_SIZE)
    except UnsupportedMemoryFormatError:
        raise
    except ProjectRegistryError as exc:
        raise RemoteSyncError(f"invalid project snapshot: {exc}") from exc


def _merge(
    source: Path,
    destination: Path,
    *,
    source_export: MemorySyncExport,
    destination_export: MemorySyncExport,
    state: dict[str, object],
    project_id: str,
    source_label: str,
) -> tuple[MemoryTransferResult, MemorySyncExport, EntryMemoryExport | None]:
    if not isinstance(source_export, MemoryExport) or not isinstance(
        destination_export, MemoryExport
    ):
        raise RemoteSyncError(
            "mixed project memory formats cannot synchronize; migration is required"
        )
    files = state["files"]
    unresolved = state["conflicts"]
    updated: list[str] = []
    conflicts: list[str] = []
    contents = dict(destination_export.contents)
    automatic = set(destination_export.automatic_files)
    for name in MEMORY_FILES:
        source_path = source / "memory" / name
        destination_path = destination / "memory" / name
        source_digest = _file_digest(source_path)
        destination_digest = _file_digest(destination_path)
        if name in unresolved:
            conflicts.append(name)
            continue
        if source_digest == destination_digest:
            files[name] = source_digest
            if name in source_export.automatic_files:
                automatic.add(name)
            continue
        old = files.get(name)
        source_changed = old is None or source_digest != old
        destination_changed = old is None or destination_digest != old
        if destination_changed or (source_changed and destination_changed):
            _keep_conflict(source_path, destination / "memory", name, peer=source_label)
            unresolved[name] = {"digests": sorted({source_digest, destination_digest})}
            conflicts.append(name)
            continue
        _copy_memory_file(source_path, destination_path)
        contents[name] = source_export.contents[name]
        if name in source_export.automatic_files:
            automatic.add(name)
        else:
            automatic.discard(name)
        files[name] = source_digest
        updated.append(name)
    merged = MemoryExport(
        contents,
        ProjectRegistry._memory_digest_value(contents),
        _merge_versions(destination_export.versions, source_export.versions),
        tuple(sorted(automatic)),
    )
    return (
        MemoryTransferResult(project_id, tuple(updated), tuple(conflicts)),
        merged,
        None,
    )


def _merge_entries(
    local: EntryMemoryExport,
    remote: EntryMemoryExport,
    *,
    state: dict[str, object],
    project_id: str,
    resolutions: dict[str, Literal["local", "remote"]] | None = None,
) -> tuple[
    MemoryTransferResult,
    EntryMemoryExport,
    EntryMemoryExport,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, ResolutionCandidate],
]:
    baseline = state.get("entries")
    schema_baseline = state.get("schema_digest")
    conflicts = state.get("conflicts")
    if (
        not isinstance(baseline, dict)
        or not isinstance(conflicts, dict)
        or schema_baseline is not None
        and not _valid_digest(schema_baseline)
    ):
        raise RemoteSyncError("memory synchronization state is invalid")
    try:
        result = merge_entry_states(
            local.state,
            remote.state,
            entry_baseline=baseline,
            schema_baseline=schema_baseline,
            conflicts=conflicts,
            resolutions=resolutions,
        )
    except ProjectRegistryError as exc:
        raise RemoteSyncError(f"memory synchronization state is invalid: {exc}") from exc
    state["entries"] = result.entry_baseline
    state["schema_digest"] = result.schema_baseline
    state["conflicts"] = {
        key: conflict.to_dict() for key, conflict in result.conflicts.items()
    }
    versions = merge_version_receipts(local.versions, remote.versions)
    local_export = EntryMemoryExport(
        result.local,
        state_digest(result.local),
        local.version,
        versions,
    )
    remote_export = EntryMemoryExport(
        result.remote,
        state_digest(result.remote),
        remote.version,
        versions,
    )
    return (
        MemoryTransferResult(project_id, result.updated, result.conflict_keys),
        local_export,
        remote_export,
        result.local_changed,
        result.remote_changed,
        result.resolved_candidates,
    )


def _merge_versions(
    first: tuple[dict[str, object], ...], second: tuple[dict[str, object], ...]
) -> tuple[dict[str, object], ...]:
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in (*first, *second):
        version = record.get("version")
        if isinstance(version, str):
            if version in seen:
                continue
            seen.add(version)
        merged.append(record)
    return tuple(merged)


def _shared_state(
    first: Path,
    second: Path,
    peer: str,
    project_id: str,
    *,
    memory_format: int = 1,
) -> _SharedSyncState:
    first_state = _read_state(first, peer, project_id, required=False)
    second_state = _read_state(second, peer, project_id, required=False)
    if first_state is not None and second_state is not None:
        if first_state == second_state:
            return _shared_sync_state(first_state)
        recovered = _recover_partial_entry_transition(first_state, second_state)
        if recovered is not None:
            return recovered
        raise RemoteSyncError("memory synchronization state differs between peers")
    if first_state is not None:
        return _recover_missing_entry_predecessor(
            first_state, project_id
        ) or _shared_sync_state(first_state)
    if second_state is not None:
        return _recover_missing_entry_predecessor(
            second_state, project_id
        ) or _shared_sync_state(second_state)
    if memory_format == 2:
        return _shared_sync_state(
            {
                "schema": 3,
                "project_id": project_id,
                "format": 2,
                "entries": {},
                "schema_digest": None,
                "conflicts": {},
            }
        )
    return _shared_sync_state(
        {"schema": 2, "project_id": project_id, "files": {}, "conflicts": {}}
    )


def _shared_sync_state(state: dict[str, object]) -> _SharedSyncState:
    copied = copy.deepcopy(state)
    return _SharedSyncState(copied, _sync_state_digest(copied))


def _recover_partial_entry_transition(
    first: dict[str, object], second: dict[str, object]
) -> _SharedSyncState | None:
    for newer, older in ((first, second), (second, first)):
        transition = newer.get("transition")
        if (
            newer.get("schema") != 3
            or not isinstance(transition, dict)
            or transition.get("previous_state") != _sync_state_digest(older)
        ):
            continue
        transition_id = transition.get("id")
        resolutions = transition.get("resolutions")
        candidates = _parse_transition_candidates(transition.get("candidates"))
        if (
            not isinstance(transition_id, str)
            or not isinstance(resolutions, dict)
            or candidates is None
        ):
            continue
        return _SharedSyncState(
            copy.deepcopy(older),
            _sync_state_digest(older),
            transition_id=transition_id,
            resolutions=dict(resolutions),  # validated by _read_state
            candidates=candidates,
        )
    return None


def _recover_missing_entry_predecessor(
    newer: dict[str, object], project_id: str
) -> _SharedSyncState | None:
    transition = newer.get("transition")
    if newer.get("schema") != 3 or not isinstance(transition, dict):
        return None
    predecessor = {
        "schema": 3,
        "project_id": project_id,
        "format": 2,
        "entries": {},
        "schema_digest": None,
        "conflicts": {},
    }
    previous_digest = _sync_state_digest(predecessor)
    if transition.get("previous_state") != previous_digest:
        return None
    transition_id = transition.get("id")
    resolutions = transition.get("resolutions")
    candidates = _parse_transition_candidates(transition.get("candidates"))
    if (
        not isinstance(transition_id, str)
        or not isinstance(resolutions, dict)
        or candidates is None
    ):
        return None
    return _SharedSyncState(
        predecessor,
        previous_digest,
        transition_id=transition_id,
        resolutions=dict(resolutions),  # validated by _read_state
        candidates=candidates,
    )


def _stored_recovery_resolutions(
    local: EntryMemoryExport,
    remote: EntryMemoryExport,
    shared: _SharedSyncState,
) -> dict[str, Literal["local", "remote"]]:
    if shared.resolutions is None or shared.candidates is None:
        return {}
    return recoverable_resolutions(
        local.state,
        remote.state,
        candidates=shared.candidates,
        resolutions=shared.resolutions,
    )


def _resolution_choices(
    local: EntryMemoryExport,
    remote: EntryMemoryExport,
    shared: _SharedSyncState,
    conflicts: object,
    accept: Literal["local", "remote"],
) -> dict[str, Literal["local", "remote"]]:
    if not isinstance(conflicts, dict):
        raise RemoteSyncError("memory synchronization state is invalid")
    explicit = {key: accept for key in conflicts}
    if shared.resolutions and any(
        choice != accept for choice in shared.resolutions.values()
    ):
        return explicit
    if shared.resolutions:
        return _stored_recovery_resolutions(local, remote, shared)
    return explicit


def _finish_entry_transition(
    state: dict[str, object],
    shared: _SharedSyncState,
    resolutions: dict[str, Literal["local", "remote"]],
    candidates: dict[str, ResolutionCandidate],
) -> None:
    if state.get("schema") != 3:
        return
    state["transition"] = {
        "id": shared.transition_id or uuid.uuid4().hex,
        "previous_state": shared.previous_digest,
        "resolutions": dict(sorted(resolutions.items())),
        "candidates": {
            key: candidate.to_dict() for key, candidate in sorted(candidates.items())
        },
    }


def _sync_state_digest(state: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()

def _read_state(
    project: Path,
    peer: str,
    project_id: str,
    *,
    required: bool = True,
) -> dict[str, object] | None:
    path = project / "sync" / f"{peer}.json"
    if not path.exists():
        if required:
            raise RemoteSyncError("memory synchronization state is missing")
        return None
    try:
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_STATE_BYTES + 1)
            if len(encoded) > _MAX_STATE_BYTES:
                raise RemoteSyncError("memory synchronization state is invalid")
        value = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteSyncError("memory synchronization state is invalid") from exc
    if not _valid_state(value, project_id):
        raise RemoteSyncError("memory synchronization state is invalid")
    return value


def _valid_state(value: object, project_id: str) -> bool:
    if isinstance(value, dict) and value.get("schema") == 3:
        return _valid_entry_sync_state(value, project_id)
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "project_id",
        "files",
        "conflicts",
    }:
        return False
    files = value.get("files")
    conflicts = value.get("conflicts")
    if (
        value.get("schema") != 2
        or value.get("project_id") != project_id
        or not isinstance(files, dict)
        or not isinstance(conflicts, dict)
        or not set(files).issubset(MEMORY_FILES)
        or not set(conflicts).issubset(MEMORY_FILES)
    ):
        return False
    if any(not _valid_digest(digest) for digest in files.values()):
        return False
    for conflict in conflicts.values():
        if not isinstance(conflict, dict) or set(conflict) != {"digests"}:
            return False
        digests = conflict.get("digests")
        if (
            not isinstance(digests, list)
            or len(digests) != 2
            or digests != sorted(set(digests))
            or any(not _valid_digest(digest) for digest in digests)
        ):
            return False
    return True


def _valid_entry_sync_state(value: dict[str, object], project_id: str) -> bool:
    required = {
        "schema",
        "project_id",
        "format",
        "entries",
        "schema_digest",
        "conflicts",
    }
    if set(value) not in {frozenset(required), frozenset((*required, "transition"))}:
        return False
    transition = value.get("transition")
    if transition is not None and not _valid_entry_transition(transition):
        return False
    entries = value.get("entries")
    conflicts = value.get("conflicts")
    schema_value = value.get("schema_digest")
    if (
        value.get("project_id") != project_id
        or value.get("format") != 2
        or not isinstance(entries, dict)
        or not isinstance(conflicts, dict)
        or schema_value is not None
        and not _valid_digest(schema_value)
        or any(
            not isinstance(entry_id, str)
            or not entry_id.startswith("m_")
            or not _valid_digest(digest)
            for entry_id, digest in entries.items()
        )
        or any(
            key != "schema" and (not isinstance(key, str) or not key.startswith("m_"))
            for key in conflicts
        )
    ):
        return False
    occupied: set[str] = set()
    for key, conflict in conflicts.items():
        if not isinstance(key, str) or not isinstance(conflict, dict):
            return False
        kind = conflict.get("kind")
        digests = conflict.get("digests")
        if (
            kind not in {"entries", "schema"}
            or not isinstance(digests, list)
            or len(digests) not in {1, 2}
            or digests != sorted(set(digests))
            or any(not _valid_digest(digest) for digest in digests)
        ):
            return False
        if kind == "schema":
            if key != "schema" or set(conflict) != {"kind", "digests"}:
                return False
            continue
        entry_ids = conflict.get("entry_ids")
        if (
            set(conflict) != {"kind", "entry_ids", "digests"}
            or not isinstance(entry_ids, list)
            or not entry_ids
            or entry_ids != sorted(set(entry_ids))
            or key != min(entry_ids)
            or occupied.intersection(entry_ids)
            or any(
                not isinstance(entry_id, str) or not entry_id.startswith("m_")
                for entry_id in entry_ids
            )
        ):
            return False
        occupied.update(entry_ids)
    return True



def _parse_transition_candidates(
    value: object,
) -> dict[str, ResolutionCandidate] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, ResolutionCandidate] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            return None
        kind = raw.get("kind")
        expected = {"kind", "local", "remote", "prepared"}
        entry_ids: tuple[str, ...] = ()
        if kind == "entries":
            expected.add("entry_ids")
            raw_ids = raw.get("entry_ids")
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or raw_ids != sorted(set(raw_ids))
                or any(
                    not isinstance(entry_id, str) or not entry_id.startswith("m_")
                    for entry_id in raw_ids
                )
            ):
                return None
            entry_ids = tuple(raw_ids)
        elif kind != "schema":
            return None
        if set(raw) != expected or any(
            not _valid_digest(raw.get(name))
            for name in ("local", "remote", "prepared")
        ):
            return None
        parsed[key] = ResolutionCandidate(
            kind,
            entry_ids,
            str(raw["local"]),
            str(raw["remote"]),
            str(raw["prepared"]),
        )
    return parsed


def _valid_entry_transition(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "id",
        "previous_state",
        "resolutions",
        "candidates",
    }:
        return False
    transition_id = value.get("id")
    resolutions = value.get("resolutions")
    candidates = _parse_transition_candidates(value.get("candidates"))
    return (
        isinstance(transition_id, str)
        and len(transition_id) == 32
        and all(character in "0123456789abcdef" for character in transition_id)
        and _valid_digest(value.get("previous_state"))
        and isinstance(resolutions, dict)
        and candidates is not None
        and set(candidates) == set(resolutions)
        and all(
            isinstance(key, str)
            and (key == "schema" or key.startswith("m_"))
            and choice in {"local", "remote"}
            for key, choice in resolutions.items()
        )
    )

def _valid_digest(value: object) -> bool:
    return value == _MISSING or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_state(project: Path, peer: str, state: object) -> None:
    path = project / "sync" / f"{peer}.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _keep_conflict(source: Path, destination: Path, name: str, *, peer: str) -> None:
    if not source.exists():
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    _atomic_copy_file(source, destination / f"{name}.conflict-{peer}-{stamp}")


def _copy_memory_file(source: Path, destination: Path) -> None:
    if source.exists():
        _atomic_copy_file(source, destination)
    elif destination.exists():
        destination.unlink()


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return _MISSING
    digest = hashlib.sha256()
    _update_digest_from_file(digest, path)
    return digest.hexdigest()


def _update_digest_from_file(digest: _Digest, path: Path) -> None:
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


@contextmanager
def _registry_lock(
    projects: Path, *, blocking: bool = True, peer: str | None = None
) -> Iterator[None]:
    projects.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = projects / ".lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RemoteSyncError("project registry lock is unsafe")
        os.fchmod(fd, 0o600)
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError as exc:
            raise RemoteSyncError(f"project registry busy on {peer}; retry") from exc
        yield
    finally:
        os.close(fd)


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    token = uuid.uuid4().hex
    parent = destination.parent
    install = parent / f".{destination.name}.install-{token}"
    backup = parent / f".{destination.name}.backup-{token}"
    journal = parent / f".{destination.name}.replace-{token}.json"
    os.replace(staging, install)
    _fsync_directory(parent)
    _atomic_write_file(
        json.dumps(
            {"project_id": destination.name, "install": install.name, "backup": backup.name},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        journal,
    )
    _fsync_directory(parent)
    try:
        if destination.exists():
            os.replace(destination, backup)
            _fsync_directory(parent)
        os.replace(install, destination)
        _fsync_directory(parent)
    except BaseException:
        _recover_interrupted_replacement(destination)
        raise
    _finish_directory_replacement(destination, install, backup, journal)


def _recover_interrupted_replacement(destination: Path) -> None:
    parent = destination.parent
    for journal in sorted(parent.glob(f".{destination.name}.replace-*.json")):
        try:
            value = json.loads(journal.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RemoteSyncError("project replacement journal is invalid") from exc
        token = journal.name.removeprefix(f".{destination.name}.replace-").removesuffix(
            ".json"
        )
        expected = {
            "project_id": destination.name,
            "install": f".{destination.name}.install-{token}",
            "backup": f".{destination.name}.backup-{token}",
        }
        if value != expected or len(token) != 32 or any(
            character not in "0123456789abcdef" for character in token
        ):
            raise RemoteSyncError("project replacement journal is invalid")
        install = parent / expected["install"]
        backup = parent / expected["backup"]
        if destination.exists():
            try:
                _validate_recovery_snapshot(destination, destination.name)
            except RemoteSyncError:
                if not backup.is_dir():
                    raise
                shutil.rmtree(destination)
                os.replace(backup, destination)
                _fsync_directory(parent)
            _finish_directory_replacement(destination, install, backup, journal)
            continue
        candidate = install if install.is_dir() else backup
        if not candidate.is_dir():
            raise RemoteSyncError("project replacement journal has no recoverable snapshot")
        try:
            _validate_recovery_snapshot(candidate, destination.name)
        except RemoteSyncError:
            if candidate == backup or not backup.is_dir():
                raise
            candidate = backup
            _validate_recovery_snapshot(candidate, destination.name)
        os.replace(candidate, destination)
        _fsync_directory(parent)
        _finish_directory_replacement(destination, install, backup, journal)


def _validate_recovery_snapshot(candidate: Path, project_id: str) -> None:
    with tempfile.TemporaryDirectory(prefix="zeta-memory-recovery-") as temporary:
        copy = Path(temporary) / project_id
        copy_project_snapshot(candidate, copy)
        _validate_project_snapshot(copy, project_id)


def _finish_directory_replacement(
    destination: Path, install: Path, backup: Path, journal: Path
) -> None:
    for path in (install, backup):
        if path.exists():
            shutil.rmtree(path)
    _fsync_directory(destination.parent)
    journal.unlink(missing_ok=True)
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_file(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode):
            raise RemoteSyncError("project snapshot file is unsafe")
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_stream,
            temporary.open("wb") as destination_stream,
        ):
            shutil.copyfileobj(source_stream, destination_stream)
        temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        os.close(source_fd)
        temporary.unlink(missing_ok=True)


def _copy_tree(source: Path, destination: Path) -> None:
    try:
        root_info = source.lstat()
    except FileNotFoundError as exc:
        raise RemoteSyncError("project snapshot directory is unsafe") from exc
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise RemoteSyncError("project snapshot directory is unsafe")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise RemoteSyncError(f"project snapshot contains a symlink: {relative}")
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif path.is_file():
            _atomic_copy_file(path, target)
        else:
            raise RemoteSyncError(f"project snapshot contains an unsupported file: {relative}")


def _machine_id(home: Path) -> str:
    path = home / _MACHINE_ID
    lock_path = home / f"{_MACHINE_ID}.lock"
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path.touch(mode=0o600, exist_ok=True)
        lock_path.chmod(0o600)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = path.read_text(encoding="ascii").strip() if path.exists() else ""
            if _valid_machine_id(value):
                if stat.S_IMODE(path.stat().st_mode) != 0o600:
                    path.chmod(0o600)
                return value
            value = uuid.uuid4().hex
            temporary_path: Path | None = None
            try:
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.", dir=path.parent
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(fd, "w", encoding="ascii") as temporary:
                    temporary.write(value + "\n")
                temporary_path.chmod(0o600)
                current = path.read_text(encoding="ascii").strip() if path.exists() else ""
                if _valid_machine_id(current):
                    return current
                os.replace(temporary_path, path)
                return path.read_text(encoding="ascii").strip()
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    except (OSError, UnicodeError) as exc:
        raise RemoteSyncError("cannot read machine identity") from exc


def _peer_machine_id(transport: ProjectSnapshotTransport) -> str:
    value = transport.machine_id
    if not isinstance(value, str) or not _valid_machine_id(value):
        raise RemoteSyncError("remote machine identity is invalid")
    return value


def _reject_same_machine(local: str, peer: str) -> None:
    if local == peer:
        raise RemoteSyncError(
            "local and remote machine identities are equal; regenerate one by deleting "
            "~/.zeta/.machine-id on that machine and retry"
        )


def _state_key(first: str, second: str) -> str:
    return f"{min(first, second)}--{max(first, second)}"


def _valid_machine_id(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).parts != (value,) or "\x00" in value:
        raise RemoteSyncError(f"{label} must be one safe path component")
    return value


__all__ = [
    "MEMORY_FILES",
    "MemoryTransferResult",
    "ProjectSnapshotTransport",
    "copy_project_snapshot",
    "fetch_local_project",
    "project_digest",
    "publish_local_project",
    "resolve_project_memory",
    "sync_project_memory",
]
