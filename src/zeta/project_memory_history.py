"""Transactional, retained project-memory versions."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .memory.entry_history import EntryMemoryHistoryMixin
from .memory.entry_views import GENERATED_MIRROR_HEADER
from .memory.version_store import (
    PreparedVersion,
    PublicationContext,
    publish_version,
    require_memory_format,
)
from .memory_migration_plan import reachable_version_pruning_plan
from .project_errors import ProjectRegistryError
from .project_schema import MAX_MEMORY_FILE_SIZE

MAX_RETAINED_VERSIONS = 128
PROJECT_MEMORY_FILES = (
    "brief.md",
    "state.md",
    "backlog.md",
    "changelog.md",
    "decisions.md",
)
_VERSION_ROOT = "memory-versions"
_CURRENT = "memory-current.json"
_MEMORY_MIRROR_HEADER = GENERATED_MIRROR_HEADER
MAX_MEMORY_MIRROR_FILE_SIZE = MAX_MEMORY_FILE_SIZE + len(
    _MEMORY_MIRROR_HEADER.encode("utf-8")
)
_LOG = logging.getLogger(__name__)


def _now() -> str:
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Complete memory contents and the authoritative CAS digest."""

    contents: dict[str, str]
    digest: str


@dataclass(frozen=True, slots=True)
class MemoryCASResult:
    """The authoritative snapshot and whether this call published it."""

    contents: list[tuple[str, str]]
    published: bool
    version: str


@dataclass(frozen=True, slots=True)
class MemoryContextEntry:
    """One current memory file and whether automatic reconciliation wrote it."""

    name: str
    content: str
    automatic: bool


@dataclass(frozen=True, slots=True)
class MemoryExport:
    """Storage-independent current memory and retained version provenance."""

    contents: dict[str, str]
    digest: str
    versions: tuple[dict[str, object], ...]
    automatic_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryState:
    """Current authoritative memory and its version metadata."""

    contents: dict[str, str]
    digest: str
    version: str | None
    automatic_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryVersionFile:
    """One retained version's file content and its parent content."""

    record: dict[str, object]
    content: str
    parent_content: str


@dataclass(frozen=True, slots=True)
class _FormatOnePayloadAdapter:
    """Prepare the legacy five-file payload for the shared publisher."""

    owner: Any
    contents: Mapping[str, str]
    before: Mapping[str, str]
    kind: str
    provenance: Mapping[str, object] | None
    files: list[str] | None
    target_version: str | None
    source_digest: str | None
    source_history: list[dict[str, object]] | None
    source_automatic_files: set[str] | None

    def prepare(self, context: PublicationContext) -> PreparedVersion[None]:
        old_records = list(context.old_manifests)
        before_automatic = self.owner._automatic_files_from_records(old_records)
        automatic = set(before_automatic)
        changed = set(self.files or ())
        if self.kind == "update" and self.provenance is not None:
            automatic.update(changed)
        elif self.kind == "accept":
            automatic.difference_update(changed)
        elif self.kind == "import":
            automatic = set(self.source_automatic_files or ())
        elif self.kind == "undo" and self.target_version is not None:
            target = next(
                (
                    record
                    for record in old_records
                    if record.get("version") == self.target_version
                ),
                None,
            )
            restored = (
                self.owner._automatic_file_set(target.get("before_automatic_files"))
                if target is not None
                else None
            )
            if restored is None:
                target_index = next(
                    (
                        index
                        for index, record in enumerate(old_records)
                        if record.get("version") == self.target_version
                    ),
                    0,
                )
                restored = self.owner._automatic_files_from_records(
                    old_records[:target_index]
                )
            automatic = restored
        fields: dict[str, object] = {
            "kind": self.kind,
            "created_at": _now(),
            "automatic_files": sorted(automatic),
            "before_automatic_files": sorted(before_automatic),
        }
        if self.provenance is not None:
            fields["provenance"] = dict(self.provenance)
        if self.files is not None:
            fields["files"] = sorted(self.files)
        if self.target_version is not None:
            fields["target_version"] = self.target_version
        if self.source_digest is not None:
            fields["source_digest"] = self.source_digest
        if self.source_history is not None:
            fields["source_history"] = self.source_history
        return PreparedVersion(
            {name: self.contents.get(name, "").encode() for name in PROJECT_MEMORY_FILES},
            {name: self.before.get(name, "").encode() for name in PROJECT_MEMORY_FILES},
            fields,
            None,
        )


class ProjectMemoryHistoryMixin(EntryMemoryHistoryMixin):
    """Own atomic snapshots, CAS, provenance, dedupe, undo, and retention."""

    @staticmethod
    def _entry_retention_limit() -> int:
        return MAX_RETAINED_VERSIONS

    def _require_format_one(self, directory_fd: int) -> None:
        require_memory_format(
            directory_fd,
            1,
            pointer_reader=self._pointer,
            version_handles=lambda fd, create: self._version_handles(fd, create=create),
            manifest_reader=self._manifest,
        )

    @staticmethod
    def _memory_digest_value(memory: Mapping[str, str]) -> str:
        digest = hashlib.sha256()
        for name in PROJECT_MEMORY_FILES:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(memory.get(name, "").encode())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _memory_transaction_step(step: str) -> None:
        """Fault-injection seam used to verify crash publication semantics."""

    @staticmethod
    def _open_directory(parent_fd: int, name: str, *, create: bool = False) -> int:
        created = False
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
        try:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ProjectRegistryError("project memory version store is unavailable") from exc
        info = os.fstat(fd)
        if stat.S_IMODE(info.st_mode) & 0o077:
            os.close(fd)
            raise ProjectRegistryError("project memory version store is unsafe")
        if created:
            os.fsync(fd)
            os.fsync(parent_fd)
        return fd

    @staticmethod
    def _read_json(directory_fd: int, name: str) -> dict[str, object] | None:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ProjectRegistryError("project memory metadata is unsafe")
            payload = os.read(fd, info.st_size + 1)
            value = json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectRegistryError("project memory metadata is malformed") from exc
        finally:
            os.close(fd)
        if not isinstance(value, dict):
            raise ProjectRegistryError("project memory metadata is malformed")
        return value

    def _legacy_entries(self, directory_fd: int) -> dict[str, str]:
        memory_fd = self._memory_fd(directory_fd)
        try:
            result: dict[str, str] = {}
            for name in PROJECT_MEMORY_FILES:
                try:
                    content = self._read_memory_file(memory_fd, name)
                except FileNotFoundError:
                    continue
                if content.startswith(_MEMORY_MIRROR_HEADER):
                    raise ProjectRegistryError(
                        "project memory pointer is missing; generated memory mirror "
                        "cannot be used as authoritative memory"
                    )
                result[name] = content
            return result
        finally:
            os.close(memory_fd)

    def _legacy_contents(self, directory_fd: int) -> dict[str, str]:
        entries = self._legacy_entries(directory_fd)
        return {name: entries.get(name, "") for name in PROJECT_MEMORY_FILES}

    def _version_handles(self, directory_fd: int, *, create: bool) -> tuple[int, int, int]:
        root = self._open_directory(directory_fd, _VERSION_ROOT, create=create)
        try:
            blobs = self._open_directory(root, "blobs", create=create)
            versions = self._open_directory(root, "versions", create=create)
        except Exception:
            os.close(root)
            raise
        return root, blobs, versions

    def _pointer(self, directory_fd: int) -> dict[str, object] | None:
        pointer = self._read_json(directory_fd, _CURRENT)
        if pointer is None:
            return None
        current, history = pointer.get("current"), pointer.get("history")
        if not isinstance(current, str) or not isinstance(history, list):
            raise ProjectRegistryError("project memory pointer is malformed")
        if any(not isinstance(item, str) for item in history) or current not in history:
            raise ProjectRegistryError("project memory pointer is malformed")
        return pointer

    def _manifest(self, versions_fd: int, version: str) -> dict[str, object]:
        if not re.fullmatch(r"[0-9a-f]{32}", version):
            raise ProjectRegistryError("project memory version is malformed")
        value = self._read_json(versions_fd, f"{version}.json")
        if value is None:
            raise ProjectRegistryError("project memory version is missing")
        return value

    def _contents_from_manifest(
        self, blobs_fd: int, manifest: Mapping[str, object], key: str = "snapshot"
    ) -> dict[str, str]:
        files = manifest.get(key)
        if not isinstance(files, dict) or set(files) != set(PROJECT_MEMORY_FILES):
            raise ProjectRegistryError("project memory version is malformed")
        result: dict[str, str] = {}
        for name in PROJECT_MEMORY_FILES:
            digest = files.get(name)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ProjectRegistryError("project memory version is malformed")
            try:
                fd = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=blobs_fd)
            except OSError as exc:
                raise ProjectRegistryError("project memory snapshot is incomplete") from exc
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MEMORY_FILE_SIZE:
                    raise ProjectRegistryError("project memory snapshot is unsafe")
                payload = os.read(fd, info.st_size + 1)
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ProjectRegistryError("project memory snapshot is corrupt")
                result[name] = payload.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ProjectRegistryError("project memory snapshot is unreadable") from exc
            finally:
                os.close(fd)
        return result

    @staticmethod
    def _mirror_file_matches(memory_fd: int, name: str, expected: bytes) -> bool:
        try:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=memory_fd,
            )
        except OSError:
            return False
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o400
                or info.st_size != len(expected)
            ):
                return False
            payload = bytearray()
            while len(payload) < len(expected):
                chunk = os.read(fd, len(expected) - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            return bytes(payload) == expected
        except OSError:
            return False
        finally:
            os.close(fd)

    @staticmethod
    def _publish_mirror_file(memory_fd: int, name: str, payload: bytes) -> None:
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=memory_fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fchmod(stream.fileno(), 0o400)
                os.fsync(stream.fileno())
            os.replace(
                temporary,
                name,
                src_dir_fd=memory_fd,
                dst_dir_fd=memory_fd,
            )
        finally:
            try:
                os.unlink(temporary, dir_fd=memory_fd)
            except FileNotFoundError:
                pass

    def _refresh_memory_mirror(
        self,
        directory_fd: int,
        contents: Mapping[str, str],
        *,
        mirror_path: os.PathLike[str],
    ) -> None:
        """Best-effort refresh of the derived Markdown view."""
        self._require_format_one(directory_fd)
        try:
            memory_fd = self._memory_fd(directory_fd)
            try:
                changed = False
                for name in PROJECT_MEMORY_FILES:
                    payload = (_MEMORY_MIRROR_HEADER + contents.get(name, "")).encode()
                    if self._mirror_file_matches(memory_fd, name, payload):
                        continue
                    self._publish_mirror_file(memory_fd, name, payload)
                    changed = True
                if changed:
                    os.fsync(memory_fd)
            finally:
                os.close(memory_fd)
        except (OSError, ProjectRegistryError) as exc:
            _LOG.warning("could not refresh project memory mirror at %s: %s", mirror_path, exc)

    def _snapshot_locked(self, directory_fd: int) -> MemorySnapshot:
        self._require_format_one(directory_fd)
        pointer = self._pointer(directory_fd)
        if pointer is None:
            contents = self._legacy_contents(directory_fd)
        else:
            root, blobs, versions = self._version_handles(directory_fd, create=False)
            try:
                manifest = self._manifest(versions, str(pointer["current"]))
                contents = self._contents_from_manifest(blobs, manifest)
            finally:
                os.close(versions)
                os.close(blobs)
                os.close(root)
        return MemorySnapshot(contents, self._memory_digest_value(contents))

    def ensure_memory_supported(self, project_id: str) -> None:
        """Validate that the project uses a memory format supported by this build."""
        self.memory_format(project_id)

    def memory_snapshot(self, project_id: str) -> MemorySnapshot:
        """Read the complete bounded memory set and digest atomically."""
        def read(root_fd: int) -> MemorySnapshot:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._snapshot_locked(directory_fd)
            finally:
                os.close(directory_fd)

        return self._read(read)

    def memory_digest(self, project_id: str) -> str:
        return self.memory_snapshot(project_id).digest

    def _load_memory_view(self, project_id: str, byte_cap: int) -> list[tuple[str, str]]:
        def read(root_fd: int) -> list[tuple[str, str]]:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    return list(self._legacy_entries(directory_fd).items())
                return list(self._snapshot_locked(directory_fd).contents.items())
            finally:
                os.close(directory_fd)

        candidates = self._read(read)
        result: list[tuple[str, str]] = []
        remaining = byte_cap
        for name, content in candidates:
            size = len(content.encode())
            if size <= remaining:
                result.append((name, content))
                remaining -= size
        return result

    def _records_locked(self, directory_fd: int) -> list[dict[str, object]]:
        self._require_format_one(directory_fd)
        pointer = self._pointer(directory_fd)
        if pointer is None:
            return []
        root, blobs, versions = self._version_handles(directory_fd, create=False)
        try:
            return [self._manifest(versions, item) for item in pointer["history"]]
        finally:
            os.close(versions)
            os.close(blobs)
            os.close(root)

    @staticmethod
    def _automatic_file_set(value: object) -> set[str] | None:
        if not isinstance(value, list) or any(
            not isinstance(name, str) or name not in PROJECT_MEMORY_FILES
            for name in value
        ):
            return None
        return set(value)

    @classmethod
    def _automatic_files_from_records(
        cls, records: list[dict[str, object]]
    ) -> set[str]:
        """Replay retained provenance into the origin of each current file."""
        automatic: set[str] = set()
        before_versions: dict[str, set[str]] = {}
        for record in records:
            version = record.get("version")
            if isinstance(version, str):
                before_versions[version] = set(automatic)
            recorded = cls._automatic_file_set(record.get("automatic_files"))
            if recorded is not None:
                automatic = recorded
                continue
            kind = record.get("kind")
            files = record.get("files")
            changed = (
                {
                    name
                    for name in files
                    if isinstance(name, str) and name in PROJECT_MEMORY_FILES
                }
                if isinstance(files, list)
                else set()
            )
            if kind == "update" and isinstance(record.get("provenance"), dict):
                automatic.update(changed)
            elif kind == "accept":
                automatic.difference_update(changed)
            elif kind == "import":
                source = record.get("source_history")
                automatic = (
                    cls._automatic_files_from_records(source)
                    if isinstance(source, list)
                    and all(isinstance(item, dict) for item in source)
                    else set()
                )
            elif kind == "undo":
                target = record.get("target_version")
                automatic = set(before_versions.get(str(target), set()))
        return automatic

    def load_memory_for_context(
        self, project_id: str, *, byte_cap: int = 64 * 1024
    ) -> list[MemoryContextEntry]:
        """Load current memory with automatic provenance for safe prompt rendering."""
        if type(byte_cap) is not int or byte_cap < 0 or byte_cap > 1024 * 1024:
            raise ProjectRegistryError("invalid memory byte cap")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    candidates = list(self._legacy_entries(directory_fd).items())
                    automatic: set[str] = set()
                else:
                    snapshot = self._snapshot_locked(directory_fd)
                    self._refresh_memory_mirror(
                        directory_fd,
                        snapshot.contents,
                        mirror_path=self.root / project_id / "memory",
                    )
                    candidates = list(snapshot.contents.items())
                    automatic = self._automatic_files_from_records(
                        self._records_locked(directory_fd)
                    )
            finally:
                os.close(directory_fd)
        result: list[MemoryContextEntry] = []
        remaining = byte_cap
        for name, content in candidates:
            size = len(content.encode())
            if size <= remaining:
                result.append(MemoryContextEntry(name, content, name in automatic))
                remaining -= size
        return result

    @staticmethod
    def _validate_updates(updates: Mapping[str, str]) -> None:
        if not updates or set(updates) - set(PROJECT_MEMORY_FILES):
            raise ProjectRegistryError("memory updates must name standard files")
        for name, content in updates.items():
            if (
                not isinstance(content, str)
                or len(content.encode()) > MAX_MEMORY_FILE_SIZE
                or "\x00" in content
            ):
                raise ProjectRegistryError(
                    f"memory file {name} is too large or not valid text"
                )

    @staticmethod
    def _validate_provenance(provenance: Mapping[str, object] | None) -> dict[str, object]:
        value = dict(provenance or {})
        if provenance is None:
            return value
        if set(value) - {
            "session_id",
            "seq_start",
            "seq_end",
            "fragment_start",
            "fragment_end",
            "model",
            "usage",
        }:
            raise ProjectRegistryError("invalid memory provenance")
        session_id, start, end = (
            value.get("session_id"),
            value.get("seq_start"),
            value.get("seq_end"),
        )
        fragment_start = value.get("fragment_start")
        fragment_end = value.get("fragment_end")
        usage, model = value.get("usage", {}), value.get("model")
        if (
            not isinstance(session_id, str)
            or not session_id
            or type(start) is not int
            or type(end) is not int
            or start < 1
            or start > end
            or (fragment_start is None) != (fragment_end is None)
            or (
                fragment_start is not None
                and (
                    type(fragment_start) is not int
                    or type(fragment_end) is not int
                    or fragment_start < 0
                    or fragment_start >= fragment_end
                )
            )
            or (model is not None and (not isinstance(model, str) or not model))
            or not isinstance(usage, dict)
            or any(
                not isinstance(key, str) or type(item) is not int or item < 0
                for key, item in usage.items()
            )
        ):
            raise ProjectRegistryError("invalid memory provenance")
        return value

    def _publish_version(
        self,
        directory_fd: int,
        *,
        project_id: str,
        contents: Mapping[str, str],
        before: Mapping[str, str],
        kind: str,
        provenance: Mapping[str, object] | None = None,
        files: list[str] | None = None,
        target_version: str | None = None,
        source_digest: str | None = None,
        source_history: list[dict[str, object]] | None = None,
        source_automatic_files: set[str] | None = None,
    ) -> str:
        self._require_format_one(directory_fd)
        adapter = _FormatOnePayloadAdapter(
            self,
            contents=contents,
            before=before,
            kind=kind,
            provenance=provenance,
            files=files,
            target_version=target_version,
            source_digest=source_digest,
            source_history=source_history,
            source_automatic_files=source_automatic_files,
        )
        published = publish_version(
            directory_fd,
            adapter=adapter,
            retention_limit=MAX_RETAINED_VERSIONS,
            reset_history=False,
            pointer_reader=self._pointer,
            version_handles=lambda fd, create: self._version_handles(fd, create=create),
            manifest_reader=self._manifest,
            transaction_step=self._memory_transaction_step,
            prune_versions=self._prune_versions,
        )
        self._refresh_memory_mirror(
            directory_fd,
            contents,
            mirror_path=self.root / project_id / "memory",
        )
        return published.version

    def _prune_versions(self, blobs_fd: int, versions_fd: int, retained: set[str]) -> None:
        plan = reachable_version_pruning_plan(
            retained, lambda version: self._manifest(versions_fd, version)
        )
        for name in os.listdir(versions_fd):
            if name.endswith(".json") and name[:-5] not in plan.versions:
                os.unlink(name, dir_fd=versions_fd)
        for name in os.listdir(blobs_fd):
            if name not in plan.blobs:
                os.unlink(name, dir_fd=blobs_fd)
        os.fsync(versions_fd)
        os.fsync(blobs_fd)

    @staticmethod
    def _logical_history(
        records: list[dict[str, object]],
    ) -> tuple[dict[str, object], ...]:
        logical: list[dict[str, object]] = []
        seen: set[str] = set()
        fields = (
            "version",
            "kind",
            "created_at",
            "files",
            "target_version",
            "provenance",
            "source_digest",
            "automatic_files",
            "before_automatic_files",
        )
        for record in records:
            imported = record.get("source_history")
            candidates = [*imported, record] if isinstance(imported, list) else [record]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                version = candidate.get("version")
                if isinstance(version, str) and version in seen:
                    continue
                summary = {key: candidate[key] for key in fields if key in candidate}
                if isinstance(version, str):
                    seen.add(version)
                logical.append(summary)
        return tuple(logical)

    def export_memory(self, project_id: str) -> MemoryExport:
        """Export logical memory without exposing the version-store layout."""
        def read(root_fd: int) -> MemoryExport:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._snapshot_locked(directory_fd)
                records = self._records_locked(directory_fd)
                return MemoryExport(
                    dict(snapshot.contents),
                    snapshot.digest,
                    self._logical_history(records),
                    tuple(sorted(self._automatic_files_from_records(records))),
                )
            finally:
                os.close(directory_fd)

        return self._read(read)

    def import_memory(
        self,
        project_id: str,
        exported: MemoryExport,
        *,
        expected_digest: str,
        provenance: Mapping[str, object] | None = None,
    ) -> MemoryCASResult:
        """CAS-import one logical snapshot as a version with its source history."""
        if not isinstance(exported, MemoryExport):
            raise ProjectRegistryError("invalid memory export")
        self._validate_updates(exported.contents)
        if (
            set(exported.contents) != set(PROJECT_MEMORY_FILES)
            or exported.digest != self._memory_digest_value(exported.contents)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        ):
            raise ProjectRegistryError("invalid memory export")
        try:
            history_payload = json.dumps(exported.versions, sort_keys=True).encode()
        except (TypeError, ValueError) as exc:
            raise ProjectRegistryError("invalid memory export") from exc
        if len(history_payload) > 512 * 1024 or any(
            not isinstance(item, dict) for item in exported.versions
        ):
            raise ProjectRegistryError("memory export history is too large")
        automatic_files = self._automatic_file_set(list(exported.automatic_files))
        if automatic_files is None or tuple(sorted(automatic_files)) != exported.automatic_files:
            raise ProjectRegistryError("invalid memory export")
        source_history = [dict(item) for item in exported.versions]
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                before = self._snapshot_locked(directory_fd)
                if before.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                records = self._records_locked(directory_fd)
                if (
                    before.contents == exported.contents
                    and self._automatic_files_from_records(records) == automatic_files
                ):
                    pointer = self._pointer(directory_fd)
                    version = "" if pointer is None else str(pointer["current"])
                    if pointer is not None:
                        self._refresh_memory_mirror(
                            directory_fd,
                            before.contents,
                            mirror_path=self.root / project_id / "memory",
                        )
                    return MemoryCASResult(
                        list(exported.contents.items()), False, version
                    )
                version = self._publish_version(
                    directory_fd,
                    project_id=project_id,
                    contents=exported.contents,
                    before=before.contents,
                    kind="import",
                    files=[
                        name
                        for name in PROJECT_MEMORY_FILES
                        if before.contents.get(name) != exported.contents[name]
                    ],
                    provenance=provenance,
                    source_digest=exported.digest,
                    source_history=source_history,
                    source_automatic_files=automatic_files,
                )
                return MemoryCASResult(
                    list(exported.contents.items()), True, version
                )
            finally:
                os.close(directory_fd)

    def memory_state(self, project_id: str) -> MemoryState:
        """Read the current authoritative snapshot and origin flags atomically."""
        def read(root_fd: int) -> MemoryState:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._snapshot_locked(directory_fd)
                records = self._records_locked(directory_fd)
                pointer = self._pointer(directory_fd)
                return MemoryState(
                    dict(snapshot.contents),
                    snapshot.digest,
                    None if pointer is None else str(pointer["current"]),
                    tuple(sorted(self._automatic_files_from_records(records))),
                )
            finally:
                os.close(directory_fd)

        return self._read(read)

    def memory_version_file(
        self, project_id: str, version: str, name: str
    ) -> MemoryVersionFile:
        """Read one retained file and its parent snapshot for comparison."""
        if name not in PROJECT_MEMORY_FILES:
            raise ProjectRegistryError("invalid memory file name")
        def read(root_fd: int) -> MemoryVersionFile:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                self._require_format_one(directory_fd)
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    manifest = self._manifest(versions_fd, version)
                    content = self._contents_from_manifest(blobs_fd, manifest)[name]
                    parent = self._contents_from_manifest(
                        blobs_fd, manifest, "before_snapshot"
                    )[name]
                    return MemoryVersionFile(dict(manifest), content, parent)
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
            finally:
                os.close(directory_fd)

        return self._read(read)

    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]:
        if type(limit) is not int or limit < 1 or limit > 10_000:
            raise ProjectRegistryError("invalid memory history limit")
        def read(root_fd: int) -> list[dict[str, object]]:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._records_locked(directory_fd)
                return [
                    item
                    for item in records
                    if item.get("kind") in {"update", "import", "accept", "undo"}
                ][-limit:]
            finally:
                os.close(directory_fd)

        return self._read(read)

    def compare_and_swap_memory(
        self,
        project_id: str,
        *,
        expected_digest: str,
        updates: Mapping[str, str],
        provenance: Mapping[str, object] | None = None,
    ) -> MemoryCASResult:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ProjectRegistryError("invalid memory digest")
        self._validate_updates(updates)
        provenance_value = self._validate_provenance(provenance)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._records_locked(directory_fd)
                duplicate = next(
                    (
                        item
                        for item in records
                        if provenance is not None
                        and item.get("kind") == "update"
                        and isinstance(item.get("provenance"), dict)
                        and all(
                            item["provenance"].get(key) == provenance_value.get(key)
                            for key in (
                                "session_id",
                                "seq_start",
                                "seq_end",
                                "fragment_start",
                                "fragment_end",
                            )
                        )
                    ),
                    None,
                )
                if duplicate is not None:
                    current = self._snapshot_locked(directory_fd).contents
                    return MemoryCASResult(
                        list(current.items()), False, str(duplicate["version"])
                    )
                snapshot = self._snapshot_locked(directory_fd)
                if snapshot.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                contents = dict(snapshot.contents)
                contents.update(updates)
                version = self._publish_version(
                    directory_fd,
                    project_id=project_id,
                    contents=contents,
                    before=snapshot.contents,
                    kind="update" if provenance is not None else "manual",
                    provenance=provenance_value if provenance is not None else None,
                    files=list(updates),
                )
                return MemoryCASResult(list(contents.items()), True, version)
            finally:
                os.close(directory_fd)

    def _replace_memory(self, project_id: str, updates: Mapping[str, str]) -> list[tuple[str, str]]:
        self._validate_updates(updates)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._snapshot_locked(directory_fd)
                contents = dict(snapshot.contents)
                contents.update(updates)
                self._publish_version(
                    directory_fd,
                    project_id=project_id,
                    contents=contents,
                    before=snapshot.contents,
                    kind="manual",
                    files=list(updates),
                )
                return list(contents.items())
            finally:
                os.close(directory_fd)

    def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]:
        """Mark one automatic memory file as trusted by explicit user action."""
        if name not in PROJECT_MEMORY_FILES:
            raise ProjectRegistryError("invalid memory file name")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._snapshot_locked(directory_fd)
                self._publish_version(
                    directory_fd,
                    project_id=project_id,
                    contents=snapshot.contents,
                    before=snapshot.contents,
                    kind="accept",
                    provenance={"accepted_by": "user"},
                    files=[name],
                )
                return list(snapshot.contents.items())
            finally:
                os.close(directory_fd)

    def undo_memory(self, project_id: str) -> list[tuple[str, str]]:
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._records_locked(directory_fd)
                undone = {item.get("target_version") for item in records if item.get("kind") == "undo"}
                target = next(
                    (
                        item
                        for item in reversed(records)
                        if item.get("kind") in {"update", "import"} and item.get("version") not in undone
                    ),
                    None,
                )
                if target is None:
                    raise ProjectRegistryError("project memory history is empty")
                root, blobs_fd, versions_fd = self._version_handles(directory_fd, create=False)
                try:
                    restored = self._contents_from_manifest(
                        blobs_fd, target, "before_snapshot"
                    )
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                current = self._snapshot_locked(directory_fd).contents
                self._publish_version(
                    directory_fd,
                    project_id=project_id,
                    contents=restored,
                    before=current,
                    kind="undo",
                    target_version=str(target["version"]),
                )
                return list(restored.items())
            finally:
                os.close(directory_fd)
