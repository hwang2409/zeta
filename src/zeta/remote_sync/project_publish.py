"""Destination-side project snapshot publication, safe to ship over SSH."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path

from .. import project_schema

MEMORY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
_EXCLUDED_NAMES = frozenset({".lock", ".spill.lock"})
_MISSING = "missing"
_MAX_HISTORY = 128
_MAX_FORMAT_TWO_STATE_SIZE = 8 * 1024 * 1024


class ProjectPublicationError(Exception):
    """The staged project cannot be published without losing concurrent work."""


def publish_local_project(
    home: Path, project_id: str, snapshot: Path, *, expected_digest: str
) -> None:
    """Validate and CAS-publish a staged project without replacing an existing one."""

    _validate_snapshot(snapshot, project_id)
    projects = home / "projects"
    projects.mkdir(parents=True, exist_ok=True, mode=0o700)
    project = projects / project_id
    lock_fd = os.open(projects / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(lock_fd, 0o600)
    with os.fdopen(lock_fd, "r+b") as registry_lock:
        fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
        _remove_incoming(projects, project_id)
        if project_digest(project) != expected_digest:
            raise ProjectPublicationError(
                "remote changed during transfer; retry after inspection"
            )
        if expected_digest == _MISSING:
            incoming = projects / f".{project_id}.incoming-{os.getpid()}"
            try:
                shutil.copytree(snapshot, incoming, copy_function=_copy_file)
                os.rename(incoming, project)
                _fsync_directory(projects)
            finally:
                if incoming.exists():
                    shutil.rmtree(incoming)
            return
        if not project.is_dir() or project.is_symlink():
            raise ProjectPublicationError(f"project {project_id} was not found")
        publish_version(project, snapshot)
        _publish_side_files(project, snapshot)


def publish_version(project: Path, snapshot: Path) -> None:
    """Publish the staged current memory state through one atomic pointer swap."""

    source_pointer = _read_pointer(snapshot)
    destination_pointer = _read_pointer(project)
    source_manifest = _read_manifest(snapshot, source_pointer["current"])
    destination_manifest = _read_manifest(project, destination_pointer["current"])
    source_payloads, scalar = _manifest_payloads(snapshot, source_manifest, "snapshot")
    before_payloads, destination_scalar = _manifest_payloads(
        project, destination_manifest, "snapshot"
    )
    if scalar != destination_scalar:
        raise ProjectPublicationError("project memory format changed during transfer")

    version = uuid.uuid4().hex
    old_history = list(destination_pointer["history"])
    retained = [*old_history, version][-_MAX_HISTORY:]
    root = project / "memory-versions"
    blobs = root / "blobs"
    versions = root / "versions"
    blobs.mkdir(parents=True, exist_ok=True, mode=0o700)
    versions.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot_value = _publish_payloads(blobs, source_payloads, scalar)
    before_value = _publish_payloads(blobs, before_payloads, scalar)
    _fsync_directory(blobs)
    transaction_step("snapshot")

    fields = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"version", "snapshot", "before_snapshot"}
    }
    if not scalar:
        fields["before_automatic_files"] = destination_manifest.get(
            "automatic_files", []
        )
    manifest = {
        "version": version,
        **fields,
        "snapshot": snapshot_value,
        "before_snapshot": before_value,
    }
    atomic_publish_file(
        versions, f"{version}.json", json.dumps(manifest, sort_keys=True).encode()
    )
    _fsync_directory(versions)
    transaction_step("manifest")
    atomic_publish_file(
        project,
        "memory-current.json",
        json.dumps({"current": version, "history": retained}, sort_keys=True).encode(),
        sync_directory=True,
    )
    transaction_step("publish")
    _prune_versions(root, set(retained))


def transaction_step(step: str) -> None:
    """Expose durable publication boundaries for crash testing."""


def atomic_publish_file(
    directory: Path, name: str, data: bytes, *, sync_directory: bool = False
) -> None:
    temporary = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / name)
        if sync_directory:
            _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def transfer_digest(root: Path) -> str:
    """Hash every file in one exact transfer tree."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProjectPublicationError("project snapshot contains a symlink")
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


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
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_snapshot(snapshot: Path, project_id: str) -> None:
    if snapshot.name != project_id or not snapshot.is_dir() or snapshot.is_symlink():
        raise ProjectPublicationError(
            "project snapshot path does not match its project ID"
        )
    try:
        project_schema.validate_project_record(_read_record(snapshot / "project.json"), project_id)
    except ValueError as exc:
        raise ProjectPublicationError(str(exc)) from exc
    pointer_path = snapshot / "memory-current.json"
    if pointer_path.exists():
        pointer = _read_pointer(snapshot)
        for version in pointer["history"]:
            manifest = _read_manifest(snapshot, version)
            _manifest_payloads(snapshot, manifest, "snapshot")
            _manifest_payloads(snapshot, manifest, "before_snapshot")
    else:
        if (snapshot / "memory-versions").exists():
            raise ProjectPublicationError("project memory pointer is missing")
        _validate_legacy_memory(snapshot)
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise ProjectPublicationError("project snapshot contains a symlink")


def _read_record(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size > project_schema.MAX_RECORD_SIZE
        ):
            raise ProjectPublicationError("invalid project snapshot file: project.json")
        payload = path.read_bytes()
        if len(payload) > project_schema.MAX_RECORD_SIZE:
            raise ProjectPublicationError("invalid project snapshot file: project.json")
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except ProjectPublicationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProjectPublicationError("invalid project snapshot file: project.json") from exc
    if not isinstance(value, dict):
        raise ProjectPublicationError("invalid project snapshot file: project.json")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProjectPublicationError(
            f"invalid project snapshot file: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ProjectPublicationError(f"invalid project snapshot file: {path.name}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ProjectPublicationError("duplicate JSON object key")
        value[key] = item
    return value


def _validate_legacy_memory(snapshot: Path) -> None:
    memory = snapshot / "memory"
    try:
        info = memory.lstat()
    except OSError as exc:
        raise ProjectPublicationError("project memory directory is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise ProjectPublicationError("project memory directory is unsafe")
    for name in MEMORY_FILES:
        path = memory / name
        try:
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ProjectPublicationError(
                    f"project memory file {name} is unsafe"
                )
            payload = path.read_bytes()
            project_schema.decode_legacy_memory(payload, name)
        except ProjectPublicationError:
            raise
        except (OSError, ValueError) as exc:
            raise ProjectPublicationError(str(exc)) from exc


def _read_pointer(root: Path) -> dict[str, object]:
    pointer = _read_json(root / "memory-current.json")
    current, history = pointer.get("current"), pointer.get("history")
    if (
        not isinstance(current, str)
        or len(current) != 32
        or not isinstance(history, list)
        or not history
        or history[-1] != current
        or any(not isinstance(item, str) or len(item) != 32 for item in history)
    ):
        raise ProjectPublicationError("invalid project memory pointer")
    return pointer


def _read_manifest(root: Path, version: object) -> dict[str, object]:
    if (
        not isinstance(version, str)
        or len(version) != 32
        or any(character not in "0123456789abcdef" for character in version)
    ):
        raise ProjectPublicationError("invalid project memory version")
    manifest = _read_json(root / "memory-versions" / "versions" / f"{version}.json")
    if manifest.get("version") != version:
        raise ProjectPublicationError("invalid project memory manifest")
    return manifest


def _manifest_payloads(
    root: Path, manifest: dict[str, object], key: str
) -> tuple[dict[str, bytes], bool]:
    value = manifest.get(key)
    memory_format = manifest.get("format", 1)
    if memory_format == 2 and isinstance(value, str):
        values = {"state": value}
        scalar = True
    elif isinstance(value, dict) and (
        (memory_format == 2 and set(value) == {"state"})
        or (memory_format == 1 and set(value) == set(MEMORY_FILES))
    ):
        values = value
        scalar = False
    else:
        raise ProjectPublicationError("invalid project memory manifest payload")
    if not all(isinstance(digest, str) for digest in values.values()):
        raise ProjectPublicationError("invalid project memory manifest payload")
    payloads: dict[str, bytes] = {}
    for name, digest in values.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ProjectPublicationError("invalid project memory blob digest")
        path = root / "memory-versions" / "blobs" / digest
        try:
            info = path.lstat()
            payload = path.read_bytes()
        except OSError as exc:
            raise ProjectPublicationError("project memory blob is missing") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise ProjectPublicationError("project memory blob is invalid")
        if memory_format == 1:
            try:
                project_schema.decode_legacy_memory(payload, name)
            except ValueError as exc:
                raise ProjectPublicationError(str(exc)) from exc
        else:
            _validate_format_two_payload(payload)
        payloads[name] = payload
    return payloads, scalar


def _validate_format_two_payload(payload: bytes) -> None:
    if len(payload) > _MAX_FORMAT_TWO_STATE_SIZE:
        raise ProjectPublicationError("format-2 memory state is too large")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProjectPublicationError("format-2 memory state is malformed") from exc
    fields = {
        "format",
        "project_id",
        "generation",
        "schema",
        "entries",
        "compacted_through_version",
    }
    if not isinstance(value, dict) or set(value) != fields or value.get("format") != 2:
        raise ProjectPublicationError("invalid format-2 memory state")
    if not isinstance(value.get("schema"), dict) or not isinstance(
        value.get("entries"), dict
    ):
        raise ProjectPublicationError("invalid format-2 memory state")


def _publish_payloads(blobs: Path, payloads: dict[str, bytes], scalar: bool) -> object:
    digests: dict[str, str] = {}
    for name, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        digests[name] = digest
        destination = blobs / digest
        if not destination.exists():
            atomic_publish_file(blobs, digest, payload)
    return next(iter(digests.values())) if scalar else digests


def _copy_file(source: str, destination: str) -> str:
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)
    return destination


def _publish_side_files(project: Path, snapshot: Path) -> None:
    atomic_publish_file(
        project, "project.json", (snapshot / "project.json").read_bytes()
    )
    for directory in ("memory", "sync"):
        source = snapshot / directory
        if not source.is_dir():
            continue
        destination = project / directory
        destination.mkdir(mode=0o700, exist_ok=True)
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            if directory == "sync" and path.suffix != ".json":
                continue
            if directory == "memory" and not (
                path.name in MEMORY_FILES
                or any(
                    path.name.startswith(f"{name}.conflict-") for name in MEMORY_FILES
                )
            ):
                continue
            atomic_publish_file(destination, path.name, path.read_bytes())


def _prune_versions(root: Path, retained: set[str]) -> None:
    versions, blobs = root / "versions", root / "blobs"
    referenced: set[str] = set()
    for version in retained:
        manifest = _read_manifest(root.parent, version)
        for key in ("snapshot", "before_snapshot"):
            value = manifest[key]
            referenced.update(value.values() if isinstance(value, dict) else (value,))
    for path in versions.glob("*.json"):
        if path.stem not in retained:
            path.unlink()
    for path in blobs.iterdir():
        if path.name not in referenced:
            path.unlink()


def _remove_incoming(projects: Path, project_id: str) -> None:
    for path in projects.glob(f".{project_id}.incoming-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
