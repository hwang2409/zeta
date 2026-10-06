"""Versioned, compare-and-swap project-memory storage."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping

from .core.session_files import atomic_publish_file
from .project_errors import ProjectRegistryError

MAX_MEMORY_FILE_SIZE = 128 * 1024
PROJECT_MEMORY_FILES = (
    "brief.md",
    "state.md",
    "backlog.md",
    "changelog.md",
    "decisions.md",
)
MEMORY_HISTORY_FILE = "memory-history.jsonl"


def _now() -> str:
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class ProjectMemoryHistoryMixin:
    """Grouped memory changes and their append-only provenance history."""

    @staticmethod
    def _memory_digest_value(memory: Mapping[str, str]) -> str:
        digest = hashlib.sha256()
        for name in PROJECT_MEMORY_FILES:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(memory.get(name, "").encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def memory_digest(self, project_id: str) -> str:
        """Return the stable digest used by grouped memory compare-and-swap."""
        return self._memory_digest_value(dict(self.load_memory(project_id)))

    @staticmethod
    def _history_records(directory_fd: int) -> list[dict[str, object]]:
        try:
            fd = os.open(
                MEMORY_HISTORY_FILE,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ProjectRegistryError("project memory history is unreadable") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ProjectRegistryError("project memory history is unsafe")
            if info.st_size > 100 * 1024 * 1024:
                raise ProjectRegistryError("project memory history exceeds the limit")
            payload = os.read(fd, info.st_size + 1)
        finally:
            os.close(fd)
        records: list[dict[str, object]] = []
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not payload.endswith(b"\n"):
                    break
                raise ProjectRegistryError("project memory history is malformed")
            if not isinstance(value, dict):
                raise ProjectRegistryError("project memory history is malformed")
            records.append(value)
        return records

    @staticmethod
    def _append_history(directory_fd: int, record: Mapping[str, object]) -> None:
        payload = (json.dumps(record, sort_keys=True) + "\n").encode()
        if len(payload) > 2 * 1024 * 1024:
            raise ProjectRegistryError("project memory history record is too large")
        try:
            fd = os.open(
                MEMORY_HISTORY_FILE,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ProjectRegistryError("project memory history is unsafe")
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
        except OSError as exc:
            raise ProjectRegistryError("project memory history is unwritable") from exc
        finally:
            if "fd" in locals():
                os.close(fd)

    def memory_log(
        self, project_id: str, *, limit: int = 100
    ) -> list[dict[str, object]]:
        """Return recent versioned automatic memory changes."""
        if type(limit) is not int or limit < 1 or limit > 10_000:
            raise ProjectRegistryError("invalid memory history limit")
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._history_records(directory_fd)[-limit:]
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
        """Apply a grouped replacement against one base and version auto changes."""
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ProjectRegistryError("invalid memory digest")
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
        provenance_value = dict(provenance or {})
        if provenance is not None:
            session_id = provenance_value.get("session_id")
            seq_start = provenance_value.get("seq_start")
            seq_end = provenance_value.get("seq_end")
            if (
                not isinstance(session_id, str)
                or not session_id
                or type(seq_start) is not int
                or type(seq_end) is not int
                or seq_start < 1
                or seq_start > seq_end
            ):
                raise ProjectRegistryError("invalid memory provenance")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                history = self._history_records(directory_fd)
                if provenance is not None and any(
                    item.get("kind") == "update"
                    and item.get("provenance") == provenance_value
                    for item in history
                ):
                    memory_fd = self._memory_fd(directory_fd)
                    try:
                        return [
                            (name, self._read_memory_file(memory_fd, name))
                            for name in PROJECT_MEMORY_FILES
                        ]
                    finally:
                        os.close(memory_fd)
                memory_fd = self._memory_fd(directory_fd)
                try:
                    current: dict[str, str] = {}
                    for name in PROJECT_MEMORY_FILES:
                        try:
                            current[name] = self._read_memory_file(memory_fd, name)
                        except FileNotFoundError:
                            current[name] = ""
                    if self._memory_digest_value(current) != expected_digest:
                        raise ProjectRegistryError("project memory digest mismatch")
                    for name, content in updates.items():
                        atomic_publish_file(
                            memory_fd,
                            name,
                            content.encode(),
                            sync_directory=False,
                        )
                    os.fsync(memory_fd)
                finally:
                    os.close(memory_fd)
                if provenance is not None:
                    self._append_history(
                        directory_fd,
                        {
                            "version": uuid.uuid4().hex,
                            "kind": "update",
                            "created_at": _now(),
                            "provenance": provenance_value,
                            "files": sorted(updates),
                            "before": {name: current[name] for name in updates},
                            "after": dict(updates),
                        },
                    )
            finally:
                os.close(directory_fd)
        return self.load_memory(project_id)

    def undo_memory(self, project_id: str) -> list[tuple[str, str]]:
        """Restore the files from the newest automatic memory update."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                history = self._history_records(directory_fd)
                target = next(
                    (item for item in reversed(history) if item.get("kind") == "update"),
                    None,
                )
                if target is None or not isinstance(target.get("before"), dict):
                    raise ProjectRegistryError("project memory history is empty")
                target_version = target.get("version")
                if any(
                    item.get("kind") == "undo"
                    and item.get("target_version") == target_version
                    for item in history
                ):
                    raise ProjectRegistryError("latest memory update is already undone")
                before = target["before"]
                updates = {
                    name: content
                    for name, content in before.items()
                    if name in PROJECT_MEMORY_FILES and isinstance(content, str)
                }
                memory_fd = self._memory_fd(directory_fd)
                try:
                    current = {
                        name: self._read_memory_file(memory_fd, name)
                        for name in updates
                    }
                    for name, content in updates.items():
                        atomic_publish_file(
                            memory_fd, name, content.encode(), sync_directory=False
                        )
                    os.fsync(memory_fd)
                finally:
                    os.close(memory_fd)
                self._append_history(
                    directory_fd,
                    {
                        "version": uuid.uuid4().hex,
                        "kind": "undo",
                        "created_at": _now(),
                        "target_version": target_version,
                        "files": sorted(updates),
                        "before": current,
                        "after": updates,
                    },
                )
            finally:
                os.close(directory_fd)
        return self.load_memory(project_id)

