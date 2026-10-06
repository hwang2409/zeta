"""Transactional, retained project-memory versions."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from .core.session_files import atomic_publish_file
from .project_errors import ProjectRegistryError

MAX_MEMORY_FILE_SIZE = 128 * 1024
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


class ProjectMemoryHistoryMixin:
    """Own atomic snapshots, CAS, provenance, dedupe, undo, and retention."""

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
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
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

    def _legacy_contents(self, directory_fd: int) -> dict[str, str]:
        memory_fd = self._memory_fd(directory_fd)
        try:
            result: dict[str, str] = {}
            for name in PROJECT_MEMORY_FILES:
                try:
                    result[name] = self._read_memory_file(memory_fd, name)
                except FileNotFoundError:
                    result[name] = ""
            return result
        finally:
            os.close(memory_fd)

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

    def _snapshot_locked(self, directory_fd: int) -> MemorySnapshot:
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

    def memory_snapshot(self, project_id: str) -> MemorySnapshot:
        """Read the complete bounded memory set and digest under one lock."""
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._snapshot_locked(directory_fd)
            finally:
                os.close(directory_fd)

    def memory_digest(self, project_id: str) -> str:
        return self.memory_snapshot(project_id).digest

    def _load_memory_view(self, project_id: str, byte_cap: int) -> list[tuple[str, str]]:
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    memory_fd = self._memory_fd(directory_fd)
                    try:
                        candidates = []
                        for name in PROJECT_MEMORY_FILES:
                            try:
                                candidates.append((name, self._read_memory_file(memory_fd, name)))
                            except FileNotFoundError:
                                pass
                    finally:
                        os.close(memory_fd)
                else:
                    candidates = list(self._snapshot_locked(directory_fd).contents.items())
            finally:
                os.close(directory_fd)
        result: list[tuple[str, str]] = []
        remaining = byte_cap
        for name, content in candidates:
            size = len(content.encode())
            if size <= remaining:
                result.append((name, content))
                remaining -= size
        return result

    def _records_locked(self, directory_fd: int) -> list[dict[str, object]]:
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
        contents: Mapping[str, str],
        before: Mapping[str, str],
        kind: str,
        provenance: Mapping[str, object] | None = None,
        files: list[str] | None = None,
        target_version: str | None = None,
    ) -> None:
        root, blobs_fd, versions_fd = self._version_handles(directory_fd, create=True)
        try:
            def blobs_for(values: Mapping[str, str]) -> dict[str, str]:
                result: dict[str, str] = {}
                for name in PROJECT_MEMORY_FILES:
                    payload = values.get(name, "").encode()
                    digest = hashlib.sha256(payload).hexdigest()
                    result[name] = digest
                    try:
                        os.stat(digest, dir_fd=blobs_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        atomic_publish_file(blobs_fd, digest, payload)
                return result

            file_blobs = blobs_for(contents)
            before_blobs = blobs_for(before)
            self._memory_transaction_step("snapshot")
            pointer = self._pointer(directory_fd)
            old_history = [] if pointer is None else list(pointer["history"])
            version = uuid.uuid4().hex
            manifest: dict[str, object] = {
                "version": version,
                "kind": kind,
                "created_at": _now(),
                "snapshot": file_blobs,
                "before_snapshot": before_blobs,
            }
            if provenance is not None:
                manifest["provenance"] = dict(provenance)
            if files is not None:
                manifest["files"] = sorted(files)
            if target_version is not None:
                manifest["target_version"] = target_version
            atomic_publish_file(
                versions_fd,
                f"{version}.json",
                json.dumps(manifest, sort_keys=True).encode(),
            )
            self._memory_transaction_step("manifest")
            history = [*old_history, version][-MAX_RETAINED_VERSIONS:]
            atomic_publish_file(
                directory_fd,
                _CURRENT,
                json.dumps({"current": version, "history": history}, sort_keys=True).encode(),
            )
            self._memory_transaction_step("publish")
            self._prune_versions(blobs_fd, versions_fd, set(history))
        finally:
            os.close(versions_fd)
            os.close(blobs_fd)
            os.close(root)

    def _prune_versions(self, blobs_fd: int, versions_fd: int, retained: set[str]) -> None:
        referenced: set[str] = set()
        for version in retained:
            manifest = self._manifest(versions_fd, version)
            for key in ("snapshot", "before_snapshot"):
                values = manifest.get(key)
                if isinstance(values, dict):
                    referenced.update(item for item in values.values() if isinstance(item, str))
        for name in os.listdir(versions_fd):
            if name.endswith(".json") and name[:-5] not in retained:
                os.unlink(name, dir_fd=versions_fd)
        for name in os.listdir(blobs_fd):
            if name not in referenced:
                os.unlink(name, dir_fd=blobs_fd)
        os.fsync(versions_fd)
        os.fsync(blobs_fd)

    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]:
        if type(limit) is not int or limit < 1 or limit > 10_000:
            raise ProjectRegistryError("invalid memory history limit")
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._records_locked(directory_fd)
                return [item for item in records if item.get("kind") in {"update", "undo"}][-limit:]
            finally:
                os.close(directory_fd)

    def compare_and_swap_memory(
        self,
        project_id: str,
        *,
        expected_digest: str,
        updates: Mapping[str, str],
        provenance: Mapping[str, object] | None = None,
    ) -> list[tuple[str, str]]:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ProjectRegistryError("invalid memory digest")
        self._validate_updates(updates)
        provenance_value = self._validate_provenance(provenance)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._records_locked(directory_fd)
                if provenance is not None and any(
                    item.get("kind") == "update"
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
                    for item in records
                ):
                    current = self._snapshot_locked(directory_fd).contents
                    return list(current.items())
                snapshot = self._snapshot_locked(directory_fd)
                if snapshot.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                contents = dict(snapshot.contents)
                contents.update(updates)
                self._publish_version(
                    directory_fd,
                    contents=contents,
                    before=snapshot.contents,
                    kind="update" if provenance is not None else "manual",
                    provenance=provenance_value if provenance is not None else None,
                    files=list(updates),
                )
                return list(contents.items())
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
                    contents=contents,
                    before=snapshot.contents,
                    kind="manual",
                    files=list(updates),
                )
                return list(contents.items())
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
                        if item.get("kind") == "update" and item.get("version") not in undone
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
                    contents=restored,
                    before=current,
                    kind="undo",
                    target_version=str(target["version"]),
                )
                return list(restored.items())
            finally:
                os.close(directory_fd)
