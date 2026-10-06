"""Transactional project-memory synchronization.

This module owns the complete memory-sync transaction: local snapshot locking,
remote snapshot acquisition, baseline and conflict interpretation, and CAS
publication. Transports only fetch and publish validated project snapshots.
"""

from __future__ import annotations

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

from ..project_registry import MAX_RECORD_SIZE, ProjectRegistry, ProjectRegistryError
from .errors import RemoteSyncError

MEMORY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
_EXCLUDED_NAMES = frozenset({".lock", ".spill.lock"})
_MISSING = "missing"
_MACHINE_ID = ".machine-id"
_MAX_STATE_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class MemoryTransferResult:
    project_id: str
    updated: tuple[str, ...]
    conflicts: tuple[str, ...]


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
            if local_expected != _MISSING:
                copy_project_snapshot(local_project, local)
                _validate_project_snapshot(local, project_id)
            remote_expected = transport.fetch_project(project_id, remote)
            if remote_expected == _MISSING:
                if direction == "pull":
                    raise RemoteSyncError(f"remote project {project_id} was not found")
                copy_project_snapshot(local, remote)
            else:
                _validate_project_snapshot(remote, project_id)
            if local_expected == _MISSING:
                copy_project_snapshot(remote, local)
            source, destination = (local, remote) if direction == "push" else (remote, local)
            state = _shared_state(source, destination, state_key, project_id)
            result = _merge(
                source,
                destination,
                state=state,
                project_id=project_id,
                source_label="local" if direction == "push" else peer,
            )
            _write_state(local, state_key, state)
            _write_state(remote, state_key, state)
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
            remote_expected = transport.fetch_project(project_id, remote)
            if remote_expected == _MISSING:
                raise RemoteSyncError(f"remote project {project_id} was not found")
            _validate_project_snapshot(remote, project_id)
            state = _shared_state(local, remote, state_key, project_id)
            conflicts = state["conflicts"]
            if not conflicts:
                raise RemoteSyncError("project memory has no unresolved conflicts")
            chosen = local if accept == "local" else remote
            other = remote if accept == "local" else local
            updated: list[str] = []
            files = state["files"]
            for name in sorted(conflicts):
                _copy_memory_file(chosen / "memory" / name, other / "memory" / name)
                files[name] = _file_digest(chosen / "memory" / name)
                updated.append(name)
            state["conflicts"] = {}
            _write_state(local, state_key, state)
            _write_state(remote, state_key, state)
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
        if not project.is_dir():
            return _MISSING
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
    if source_memory.is_dir():
        for path in sorted(source_memory.iterdir()):
            if path.name in MEMORY_FILES or any(
                path.name.startswith(f"{name}.conflict-") for name in MEMORY_FILES
            ):
                if not path.is_file() or path.is_symlink():
                    raise RemoteSyncError("project memory file is unsafe")
                _atomic_copy_file(path, destination_memory / path.name)
    for relative in (Path("memory/history"), Path("history"), Path("sync")):
        path = source / relative
        if path.exists():
            _copy_tree(path, destination / relative)


def project_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return _MISSING
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in _EXCLUDED_NAMES:
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
        registry.load_memory(project_id, byte_cap=MAX_RECORD_SIZE)
    except ProjectRegistryError as exc:
        raise RemoteSyncError(f"invalid project snapshot: {exc}") from exc


def _merge(
    source: Path,
    destination: Path,
    *,
    state: dict[str, object],
    project_id: str,
    source_label: str,
) -> MemoryTransferResult:
    files = state["files"]
    unresolved = state["conflicts"]
    updated: list[str] = []
    conflicts: list[str] = []
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
        files[name] = source_digest
        updated.append(name)
    return MemoryTransferResult(project_id, tuple(updated), tuple(conflicts))


def _shared_state(
    first: Path, second: Path, peer: str, project_id: str
) -> dict[str, dict[str, object] | object]:
    first_state = _read_state(first, peer, project_id, required=False)
    second_state = _read_state(second, peer, project_id, required=False)
    if first_state is not None and second_state is not None and first_state != second_state:
        raise RemoteSyncError("memory synchronization state differs between peers")
    if first_state is not None:
        return first_state
    if second_state is not None:
        return second_state
    return {"schema": 2, "project_id": project_id, "files": {}, "conflicts": {}}


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
    backup = destination.parent / f".{destination.name}.replaced-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, destination)


def _copy_tree(source: Path, destination: Path) -> None:
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
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="ascii", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as temporary:
                temporary.write(value + "\n")
                temporary_path = Path(temporary.name)
            temporary_path.chmod(0o600)
            try:
                if path.exists():
                    path.unlink()
                os.link(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return path.read_text(encoding="ascii").strip()
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
