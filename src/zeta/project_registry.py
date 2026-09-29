"""Hardened, identity-only project and lane registry.

This module deliberately contains no session or execution concepts.  Records are
small JSON documents stored below ``~/.zeta/projects`` and are published only
while holding the registry lock.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import errno
import fcntl
import json
import os
import re
import secrets
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .core.session_files import atomic_publish_file

SCHEMA_VERSION = 1
ID_PREFIX = "p_"
LANE_ID_PREFIX = "l_"
ID_HEX_LENGTH = 32
MAX_NAME_LENGTH = 128
MAX_SCOPE_LENGTH = 4096
MAX_PROJECTS = 10_000
MAX_LANES = 1_000
MAX_RECORD_SIZE = 10_000_000
MAX_MEMORY_FILE_SIZE = 128 * 1024
MAX_SESSION_REFERENCE_SIZE = 4096
MAX_SESSION_REFERENCES = 10_000
MAX_CREATE_RETRIES = 32
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")
_LANE_ID = re.compile(r"l_[0-9a-f]{32}\Z")
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_SESSION_ROLES = {"session", "orchestrator", "worker"}


class ProjectRegistryError(ValueError):
    """A registry operation was rejected or stored state is unsafe."""


@dataclass(frozen=True)
class Lane:
    lane_id: str
    project_id: str
    name: str
    scope: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "project_id": self.project_id,
            "name": self.name,
            "scope": self.scope,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    scope: str
    created_at: str
    updated_at: str
    canonical_integration_root: str | None
    lanes: tuple[Lane, ...] = ()

    def to_dict(self, *, include_lanes: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "project_id": self.project_id,
            "name": self.name,
            "scope": self.scope,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "canonical_integration_root": self.canonical_integration_root,
        }
        if include_lanes:
            value["lanes"] = [lane.to_dict() for lane in self.lanes]
        return value


def _now() -> str:
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_text(
    value: object, field: str, maximum: int, *, nonempty: bool = True
) -> str:
    if not isinstance(value, str) or (nonempty and not value) or len(value) > maximum:
        raise ProjectRegistryError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    if "\x00" in value or any(ord(char) < 32 for char in value):
        raise ProjectRegistryError(f"{field} contains a control character")
    return value


def _validate_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProjectRegistryError(f"invalid {field}")
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProjectRegistryError(f"invalid {field}") from exc
    if (
        parsed.tzinfo != _dt.UTC
        or parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") != value
    ):
        raise ProjectRegistryError(f"invalid {field}")
    return value


def _validate_id(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ProjectRegistryError(f"invalid {field}")
    return value


def _validate_root(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    root = _validate_text(value, "canonical_integration_root", 4096)
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(root) != root:
        raise ProjectRegistryError(
            "canonical_integration_root must be an absolute normalized path"
        )
    # Directory identity is canonical, not textual; this also avoids /var vs
    # /private/var mismatches on macOS.
    return str(path.expanduser().resolve(strict=False))


def _new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(ID_HEX_LENGTH // 2)


def _rename_without_replacement(root_fd: int, staging: str, final: str) -> None:
    """Atomically rename a directory only if its destination is absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameatx_np"):
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            root_fd,
            os.fsencode(staging),
            root_fd,
            os.fsencode(final),
            0x00000004,  # RENAME_EXCL on Darwin
        )
    elif hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            root_fd,
            os.fsencode(staging),
            root_fd,
            os.fsencode(final),
            0x1,  # RENAME_NOREPLACE on Linux
        )
    else:
        try:
            os.stat(final, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return os.rename(staging, final, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        raise FileExistsError(final)
    if result != 0:
        error = ctypes.get_errno()
        if error == 0:
            error = errno.EIO
        raise OSError(error, os.strerror(error), final)


class ProjectRegistry:
    """A local registry whose root can be overridden for tests."""

    def __init__(self, root: Path | str | None = None):
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / ".zeta" / "projects"
        )

    @contextmanager
    def _locked(self, *, write: bool) -> Iterator[int]:
        self._ensure_root()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(self.root, flags)
        except OSError as exc:
            raise ProjectRegistryError(
                "projects registry root is not a safe directory"
            ) from exc
        try:
            info = os.fstat(root_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1:
                raise ProjectRegistryError(
                    "projects registry root has unsafe permissions or type"
                )
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise ProjectRegistryError(
                    "projects registry root has unsafe permissions"
                )
            os.fchmod(root_fd, 0o700)
            for attempt in range(10):
                try:
                    lock_fd = os.open(
                        ".lock",
                        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=root_fd,
                    )
                    break
                except FileNotFoundError as exc:
                    if attempt == 9:
                        raise ProjectRegistryError("cannot open registry lock") from exc
                    os.close(root_fd)
                    time.sleep(0.001)
                    root_fd = os.open(self.root, flags)
                    info = os.fstat(root_fd)
                    if (
                        not stat.S_ISDIR(info.st_mode)
                        or info.st_nlink < 1
                        or stat.S_IMODE(info.st_mode) & 0o077
                    ):
                        raise ProjectRegistryError(
                            "projects registry root changed unsafely"
                        )
                    os.fchmod(root_fd, 0o700)
                except OSError as exc:
                    raise ProjectRegistryError("cannot open registry lock") from exc
            try:
                lock_info = os.fstat(lock_fd)
                if (
                    not stat.S_ISREG(lock_info.st_mode)
                    or lock_info.st_nlink != 1
                    or stat.S_IMODE(lock_info.st_mode) & 0o077
                ):
                    raise ProjectRegistryError(
                        "registry lock is not a private regular unshared file"
                    )
                os.fchmod(lock_fd, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
                yield root_fd
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        finally:
            os.close(root_fd)

    def _ensure_root(self) -> None:
        created = False
        try:
            existing = self.root.lstat()
        except FileNotFoundError:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            existing = self.root.lstat()
            created = True
        if not stat.S_ISDIR(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
            raise ProjectRegistryError("projects registry root is not a directory")
        if not created and stat.S_IMODE(existing.st_mode) & 0o077:
            raise ProjectRegistryError("projects registry root has unsafe permissions")

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ProjectRegistryError("duplicate JSON object key")
            value[key] = item
        return value

    @staticmethod
    def _read_fd(directory_fd: int, name: str) -> dict[str, object]:
        try:
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
        except OSError as exc:
            raise ProjectRegistryError(f"cannot read project record {name}") from exc
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ProjectRegistryError(
                    f"project record {name} is not a private regular file"
                )
            try:
                with os.fdopen(os.dup(fd), "rb") as stream:
                    data = stream.read(MAX_RECORD_SIZE + 1)
                    if len(data) > MAX_RECORD_SIZE or stream.read(1):
                        raise ProjectRegistryError(
                            f"project record {name} is too large"
                        )
                    decoder = json.JSONDecoder(
                        object_pairs_hook=lambda pairs: ProjectRegistry._unique_object(
                            pairs
                        )
                    )
                    text = data.decode("utf-8")
                    value, end = decoder.raw_decode(text)
                    if text[end:].strip():
                        raise ProjectRegistryError(f"malformed project record {name}")
            except ProjectRegistryError:
                raise
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ) as exc:
                raise ProjectRegistryError(f"malformed project record {name}") from exc
            if not isinstance(value, dict):
                raise ProjectRegistryError(f"malformed project record {name}")
            return value
        finally:
            os.close(fd)

    @staticmethod
    def _publish(directory_fd: int, name: str, value: dict[str, object]) -> None:
        data = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        atomic_publish_file(directory_fd, name, data, sync_directory=True)

    @staticmethod
    def _publish_directory(root_fd: int, staging: str, final: str) -> None:
        _rename_without_replacement(root_fd, staging, final)
        os.fsync(root_fd)

    def _project_dir(
        self, root_fd: int, project_id: str, *, create: bool = False
    ) -> int:
        _validate_id(project_id, _PROJECT_ID, "project_id")
        if create:
            try:
                os.mkdir(project_id, 0o700, dir_fd=root_fd)
            except FileExistsError as exc:
                raise ProjectRegistryError(
                    f"project {project_id} already exists"
                ) from exc
        try:
            fd = os.open(
                project_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
            )
        except OSError as exc:
            raise ProjectRegistryError(f"project {project_id} not found") from exc
        info = os.fstat(fd)
        if stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink < 1:
            os.close(fd)
            raise ProjectRegistryError("project directory has unsafe permissions")
        return fd

    @staticmethod
    def _decode(value: dict[str, object], expected_id: str | None = None) -> Project:
        allowed = {
            "schema_version",
            "project_id",
            "name",
            "scope",
            "created_at",
            "updated_at",
            "canonical_integration_root",
            "lanes",
        }
        if set(value) != allowed or value.get("schema_version") != SCHEMA_VERSION:
            raise ProjectRegistryError("unknown or invalid project schema")
        project_id = _validate_id(value["project_id"], _PROJECT_ID, "project_id")
        if expected_id is not None and project_id != expected_id:
            raise ProjectRegistryError("project ID does not match its path")
        name = _validate_text(value["name"], "name", MAX_NAME_LENGTH)
        scope = _validate_text(value["scope"], "scope", MAX_SCOPE_LENGTH)
        created = _validate_timestamp(value["created_at"], "created_at")
        updated = _validate_timestamp(value["updated_at"], "updated_at")
        root = _validate_root(value["canonical_integration_root"])
        raw_lanes = value["lanes"]
        if not isinstance(raw_lanes, list) or len(raw_lanes) > MAX_LANES:
            raise ProjectRegistryError("invalid lanes")
        lanes: list[Lane] = []
        ids: set[str] = set()
        names: set[str] = set()
        for raw in raw_lanes:
            if not isinstance(raw, dict) or set(raw) != {
                "lane_id",
                "project_id",
                "name",
                "scope",
                "created_at",
                "updated_at",
            }:
                raise ProjectRegistryError("unknown or invalid lane schema")
            lane_id = _validate_id(raw["lane_id"], _LANE_ID, "lane_id")
            if lane_id in ids:
                raise ProjectRegistryError("duplicate lane ID")
            if raw["project_id"] != project_id:
                raise ProjectRegistryError("lane belongs to another project")
            lane_name = _validate_text(raw["name"], "lane name", MAX_NAME_LENGTH)
            if lane_name in names:
                raise ProjectRegistryError("duplicate lane name")
            ids.add(lane_id)
            names.add(lane_name)
            lanes.append(
                Lane(
                    lane_id,
                    project_id,
                    lane_name,
                    _validate_text(raw["scope"], "lane scope", MAX_SCOPE_LENGTH),
                    _validate_timestamp(raw["created_at"], "lane created_at"),
                    _validate_timestamp(raw["updated_at"], "lane updated_at"),
                )
            )
        lanes.sort(key=lambda lane: (lane.name, lane.lane_id))
        return Project(project_id, name, scope, created, updated, root, tuple(lanes))

    @staticmethod
    def _encode(project: Project) -> dict[str, object]:
        result = project.to_dict()
        result["schema_version"] = SCHEMA_VERSION
        return result

    def create_project(
        self, name: str, scope: str, canonical_integration_root: str | None = None
    ) -> Project:
        name = _validate_text(name, "name", MAX_NAME_LENGTH)
        scope = _validate_text(scope, "scope", MAX_SCOPE_LENGTH)
        canonical_integration_root = _validate_root(canonical_integration_root)
        with self._locked(write=True) as root_fd:
            projects = self._list_locked(root_fd)
            if len(projects) >= MAX_PROJECTS:
                raise ProjectRegistryError("project limit exceeded")
            if any(project.name == name for project in projects):
                raise ProjectRegistryError("project name already exists")
            if canonical_integration_root is not None and any(
                project.canonical_integration_root == canonical_integration_root
                for project in projects
            ):
                raise ProjectRegistryError("canonical integration root already exists")
            now = _now()
            for _ in range(MAX_CREATE_RETRIES):
                project = Project(
                    _new_id(ID_PREFIX),
                    name,
                    scope,
                    now,
                    now,
                    canonical_integration_root,
                )
                staging = f".staging-{project.project_id}-{uuid.uuid4().hex}"
                try:
                    os.mkdir(staging, 0o700, dir_fd=root_fd)
                except FileExistsError:
                    continue
                directory_fd = os.open(
                    staging,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                try:
                    self._publish(directory_fd, "project.json", self._encode(project))
                    os.mkdir("memory", 0o700, dir_fd=directory_fd)
                    memory_fd = os.open(
                        "memory",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        for filename, heading in (
                            ("brief.md", "# Brief\n"),
                            ("state.md", "# Current state\n"),
                            ("backlog.md", "# Backlog\n"),
                            ("changelog.md", "# Changelog\n"),
                            ("decisions.md", "# Decisions\n"),
                        ):
                            fd = os.open(
                                filename,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600,
                                dir_fd=memory_fd,
                            )
                            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                                stream.write(heading)
                                stream.flush()
                                os.fsync(stream.fileno())
                        os.fsync(memory_fd)
                    finally:
                        os.close(memory_fd)
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                try:
                    self._publish_directory(root_fd, staging, project.project_id)
                except FileExistsError:
                    try:
                        cleanup_fd = os.open(
                            staging,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=root_fd,
                        )
                        try:
                            os.unlink("project.json", dir_fd=cleanup_fd)
                        finally:
                            os.close(cleanup_fd)
                    except OSError:
                        pass
                    try:
                        os.rmdir(staging, dir_fd=root_fd)
                    except OSError:
                        pass
                    continue
                return project
            raise ProjectRegistryError("could not allocate a unique project ID")

    def _list_locked(self, root_fd: int) -> list[Project]:
        project_names = [
            name for name in os.listdir(root_fd) if _PROJECT_ID.fullmatch(name)
        ]
        if len(project_names) > MAX_PROJECTS:
            raise ProjectRegistryError("project limit exceeded")
        projects = []
        for name in project_names:
            directory_fd = self._project_dir(root_fd, name)
            try:
                try:
                    record = self._read_fd(directory_fd, "project.json")
                except ProjectRegistryError as exc:
                    if isinstance(exc.__cause__, FileNotFoundError):
                        # A directory visible after a crash is not a published record.
                        continue
                    raise
                projects.append(self._decode(record, name))
            finally:
                os.close(directory_fd)
        return sorted(projects, key=lambda project: (project.name, project.project_id))

    def list_projects(self) -> list[Project]:
        with self._locked(write=False) as root_fd:
            return self._list_locked(root_fd)

    def show_project(
        self, project_id: str | None = None, *, name: str | None = None
    ) -> Project:
        with self._locked(write=False) as root_fd:
            projects = self._list_locked(root_fd)
            matches = [
                p
                for p in projects
                if (project_id is not None and p.project_id == project_id)
                or (name is not None and p.name == name)
            ]
            if len(matches) != 1:
                raise ProjectRegistryError("project not found or ambiguous")
            return matches[0]

    def add_lane(self, project_id: str, name: str, scope: str) -> Lane:
        _validate_text(name, "lane name", MAX_NAME_LENGTH)
        _validate_text(scope, "lane scope", MAX_SCOPE_LENGTH)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                project = self._decode(
                    self._read_fd(directory_fd, "project.json"), project_id
                )
                if any(lane.name == name for lane in project.lanes):
                    raise ProjectRegistryError("lane name already exists")
                if len(project.lanes) >= MAX_LANES:
                    raise ProjectRegistryError("lane limit exceeded")
                now = _now()
                existing_ids = {item.lane_id for item in project.lanes}
                for _ in range(MAX_CREATE_RETRIES):
                    lane = Lane(
                        _new_id(LANE_ID_PREFIX), project_id, name, scope, now, now
                    )
                    if lane.lane_id in existing_ids:
                        continue
                    updated = Project(
                        project.project_id,
                        project.name,
                        project.scope,
                        project.created_at,
                        now,
                        project.canonical_integration_root,
                        (*project.lanes, lane),
                    )
                    self._publish(directory_fd, "project.json", self._encode(updated))
                    return lane
                raise ProjectRegistryError("could not allocate a unique lane ID")
            finally:
                os.close(directory_fd)

    def list_lanes(self, project_id: str) -> list[Lane]:
        return list(self.show_project(project_id).lanes)

    def show_lane(self, project_id: str, lane_id: str) -> Lane:
        _validate_id(lane_id, _LANE_ID, "lane_id")
        lanes = self.list_lanes(project_id)
        for lane in lanes:
            if lane.lane_id == lane_id:
                return lane
        raise ProjectRegistryError("lane not found")

    def find_for_directory(self, directory: str | Path) -> Project | None:
        """Return the most-specific project whose canonical root contains directory."""
        path = Path(directory).expanduser().resolve()
        matches = []
        for project in self.list_projects():
            if project.canonical_integration_root is None:
                continue
            try:
                path.relative_to(Path(project.canonical_integration_root))
            except ValueError:
                continue
            matches.append(project)
        return max(
            matches,
            key=lambda item: len(item.canonical_integration_root or ""),
            default=None,
        )

    def _project_path(self, project_id: str, name: str) -> Path:
        _validate_id(project_id, _PROJECT_ID, "project_id")
        path = self.root / project_id / name
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ProjectRegistryError(f"project file {name} not found") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ProjectRegistryError(f"project file {name} is unsafe")
        return path

    def _memory_fd(self, directory_fd: int) -> int:
        try:
            memory_fd = os.open(
                "memory",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ProjectRegistryError(
                "project memory directory is unavailable"
            ) from exc
        info = os.fstat(memory_fd)
        if stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink < 1:
            os.close(memory_fd)
            raise ProjectRegistryError("project memory directory is unsafe")
        return memory_fd

    @staticmethod
    def _read_memory_file(memory_fd: int, name: str) -> str:
        try:
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=memory_fd
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ProjectRegistryError(
                f"project memory file {name} is unreadable"
            ) from exc
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ProjectRegistryError(f"project memory file {name} is unsafe")
            if info.st_size > MAX_MEMORY_FILE_SIZE:
                raise ProjectRegistryError(f"project memory file {name} is too large")
            data = os.read(fd, MAX_MEMORY_FILE_SIZE + 1)
            if len(data) > MAX_MEMORY_FILE_SIZE:
                raise ProjectRegistryError(f"project memory file {name} is too large")
            return data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            if isinstance(exc, ProjectRegistryError):
                raise
            raise ProjectRegistryError(
                f"project memory file {name} is unreadable"
            ) from exc
        finally:
            os.close(fd)

    def load_memory(
        self, project_id: str, *, byte_cap: int = 64 * 1024
    ) -> list[tuple[str, str]]:
        """Load bounded, human-editable memory; malformed files are rejected."""
        if type(byte_cap) is not int or byte_cap < 0 or byte_cap > MAX_RECORD_SIZE:
            raise ProjectRegistryError("invalid memory byte cap")
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                try:
                    memory_fd = self._memory_fd(directory_fd)
                except ProjectRegistryError as exc:
                    if isinstance(exc.__cause__, FileNotFoundError):
                        return []
                    raise
                try:
                    result = []
                    remaining = byte_cap
                    for name in (
                        "brief.md",
                        "state.md",
                        "backlog.md",
                        "changelog.md",
                        "decisions.md",
                    ):
                        try:
                            content = self._read_memory_file(memory_fd, name)
                        except FileNotFoundError:
                            continue
                        size = len(content.encode("utf-8"))
                        if size <= remaining:
                            result.append((name, content))
                            remaining -= size
                    return result
                finally:
                    os.close(memory_fd)
            finally:
                os.close(directory_fd)

    def update_memory(
        self, project_id: str, updates: dict[str, str]
    ) -> list[tuple[str, str]]:
        """Replace exactly one standard memory file under the registry lock."""
        allowed = {"brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"}
        if len(updates) != 1 or set(updates) - allowed:
            raise ProjectRegistryError(
                "each memory update must replace exactly one standard memory file"
            )
        for name, content in updates.items():
            if (
                not isinstance(content, str)
                or len(content.encode("utf-8")) > MAX_MEMORY_FILE_SIZE
            ):
                raise ProjectRegistryError(
                    f"memory file {name} is too large or not text"
                )
            if "\x00" in content:
                raise ProjectRegistryError(
                    f"memory file {name} contains a NUL character"
                )
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                memory_fd = self._memory_fd(directory_fd)
                try:
                    for name, content in updates.items():
                        atomic_publish_file(
                            memory_fd,
                            name,
                            content.encode("utf-8"),
                            sync_directory=False,
                        )
                    os.fsync(memory_fd)
                finally:
                    os.close(memory_fd)
            finally:
                os.close(directory_fd)
        return self.load_memory(project_id)

    @staticmethod
    def _decode_session_link(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "session_id",
            "transcript_path",
            "role",
            "parent_session_id",
            "recorded_at",
        }:
            raise ProjectRegistryError("malformed project session reference")
        _validate_text(value["session_id"], "session_id", 128)
        path = _validate_text(
            value["transcript_path"], "transcript_path", MAX_SCOPE_LENGTH
        )
        if len(path.encode("utf-8")) > MAX_SESSION_REFERENCE_SIZE:
            raise ProjectRegistryError("project session reference is too large")
        role = _validate_text(value["role"], "role", 64)
        if role not in _SESSION_ROLES:
            raise ProjectRegistryError("invalid project session role")
        parent = value["parent_session_id"]
        if parent is not None:
            _validate_text(parent, "parent_session_id", 128)
        recorded = _validate_timestamp(value["recorded_at"], "recorded_at")
        return {
            "session_id": value["session_id"],
            "transcript_path": path,
            "role": role,
            "parent_session_id": parent,
            "recorded_at": recorded,
        }

    @classmethod
    def _read_session_records(
        cls, directory_fd: int, *, repair_torn_final: bool
    ) -> list[dict[str, object]]:
        """Read the bounded JSONL reference file from one pinned regular fd.

        This is intentionally the one reader used by listing and append.  It
        never opens a FIFO/block device, never performs an unbounded read while
        holding the registry lock, and can discard only a torn final line.
        """
        maximum = MAX_SESSION_REFERENCES * MAX_SESSION_REFERENCE_SIZE
        try:
            fd = os.open(
                "sessions.jsonl",
                (os.O_RDWR if repair_torn_final else os.O_RDONLY)
                | os.O_NOFOLLOW
                | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return []
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ProjectRegistryError("project session references are unsafe")
            if info.st_size > maximum:
                raise ProjectRegistryError(
                    "project session references exceed the limit"
                )
            # Parse incrementally.  In particular, never materialize the
            # complete file and a split-lines copy (the old implementation
            # briefly used roughly 2x the configured maximum).
            records: list[dict[str, object]] = []
            pending = bytearray()
            total = 0
            line_count = 0

            def consume(line: bytes) -> None:
                nonlocal line_count
                if len(line) > MAX_SESSION_REFERENCE_SIZE:
                    raise ProjectRegistryError("project session reference is too large")
                try:
                    value = cls._decode_session_link(json.loads(line.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                    raise ProjectRegistryError(
                        "malformed project session reference"
                    ) from exc
                line_count += 1
                if line_count > MAX_SESSION_REFERENCES:
                    raise ProjectRegistryError(
                        "project session references exceed the limit"
                    )
                records.append(value)

            while total <= maximum:
                data = os.read(fd, min(64 * 1024, maximum + 1 - total))
                if not data:
                    break
                total += len(data)
                pending.extend(data)
                while True:
                    try:
                        position = pending.index(10)
                    except ValueError:
                        if len(pending) > MAX_SESSION_REFERENCE_SIZE:
                            raise ProjectRegistryError(
                                "project session reference is too large"
                            )
                        break
                    consume(bytes(pending[:position]))
                    del pending[: position + 1]
            if total > maximum:
                raise ProjectRegistryError(
                    "project session references exceed the limit"
                )
            if pending:
                if not repair_torn_final:
                    raise ProjectRegistryError("torn project session reference")
                complete = total - len(pending)
                os.ftruncate(fd, complete)
            return records
        except OSError as exc:
            raise ProjectRegistryError(
                "project session references are unreadable"
            ) from exc
        finally:
            os.close(fd)

    def list_session_links(
        self, project_id: str, *, limit: int = 100
    ) -> list[dict[str, object]]:
        """Return recent bounded session references without reading transcripts."""
        if type(limit) is not int or limit < 1 or limit > MAX_SESSION_REFERENCES:
            raise ProjectRegistryError("invalid session link limit")
        with self._locked(write=False) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                records = self._read_session_records(
                    directory_fd, repair_torn_final=False
                )
            finally:
                os.close(directory_fd)
        return records[-limit:]

    def record_session(
        self,
        project_id: str,
        *,
        session_id: str,
        transcript_path: str,
        role: str = "session",
        parent_session_id: str | None = None,
    ) -> None:
        """Append a transcript reference while holding the registry lock."""
        _validate_id(project_id, _PROJECT_ID, "project_id")
        _validate_text(session_id, "session_id", 128)
        _validate_text(transcript_path, "transcript_path", MAX_SCOPE_LENGTH)
        _validate_text(role, "role", 64)
        if role not in _SESSION_ROLES:
            raise ProjectRegistryError("invalid project session role")
        _validate_text(session_id, "session_id", 128)
        if parent_session_id is not None:
            _validate_text(parent_session_id, "parent_session_id", 128)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                record = {
                    "session_id": session_id,
                    "transcript_path": transcript_path,
                    "role": role,
                    "parent_session_id": parent_session_id,
                    "recorded_at": _now(),
                }
                payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                if len(payload) > MAX_SESSION_REFERENCE_SIZE:
                    raise ProjectRegistryError("project session reference is too large")
                existing = self._read_session_records(
                    directory_fd, repair_torn_final=True
                )
                if any(item["session_id"] == session_id for item in existing):
                    return
                if len(existing) >= MAX_SESSION_REFERENCES:
                    raise ProjectRegistryError(
                        "project session reference limit exceeded"
                    )
                try:
                    try:
                        os.stat(
                            "sessions.jsonl", dir_fd=directory_fd, follow_symlinks=False
                        )
                        created = False
                    except FileNotFoundError:
                        created = True
                    fd = os.open(
                        "sessions.jsonl",
                        os.O_WRONLY
                        | os.O_APPEND
                        | os.O_CREAT
                        | os.O_NOFOLLOW
                        | os.O_NONBLOCK,
                        0o600,
                        dir_fd=directory_fd,
                    )
                    info = os.fstat(fd)
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or stat.S_IMODE(info.st_mode) & 0o077
                    ):
                        os.close(fd)
                        raise ProjectRegistryError(
                            "project session references are unsafe"
                        )
                    if (
                        info.st_size + len(payload)
                        > MAX_SESSION_REFERENCES * MAX_SESSION_REFERENCE_SIZE
                    ):
                        os.close(fd)
                        raise ProjectRegistryError(
                            "project session reference limit exceeded"
                        )
                    with os.fdopen(fd, "ab") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if created:
                        os.fsync(directory_fd)
                except ProjectRegistryError:
                    raise
                except OSError as exc:
                    raise ProjectRegistryError("cannot record project session") from exc
            finally:
                os.close(directory_fd)

    def initialize_memory(self, project_id: str) -> None:
        """Create the standard memory files without overwriting human edits."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                try:
                    os.mkdir("memory", 0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                memory_fd = os.open(
                    "memory",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    for name, heading in (
                        ("brief.md", "# Brief\n"),
                        ("state.md", "# Current state\n"),
                        ("backlog.md", "# Backlog\n"),
                        ("changelog.md", "# Changelog\n"),
                        ("decisions.md", "# Decisions\n"),
                    ):
                        try:
                            fd = os.open(
                                name,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600,
                                dir_fd=memory_fd,
                            )
                        except FileExistsError:
                            continue
                        with os.fdopen(fd, "w", encoding="utf-8") as stream:
                            stream.write(heading)
                            stream.flush()
                            os.fsync(stream.fileno())
                    os.fsync(memory_fd)
                finally:
                    os.close(memory_fd)
            finally:
                os.close(directory_fd)
