"""Hardened, identity-only project and lane registry.

This module deliberately contains no session or execution concepts.  Records are
small JSON documents stored below ``~/.zeta/projects`` and are published only
while holding the registry lock.
"""

from __future__ import annotations

import datetime as _dt
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

SCHEMA_VERSION = 1
ID_PREFIX = "p_"
LANE_ID_PREFIX = "l_"
ID_HEX_LENGTH = 32
MAX_NAME_LENGTH = 128
MAX_SCOPE_LENGTH = 4096
MAX_PROJECTS = 10_000
MAX_LANES = 1_000
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")
_LANE_ID = re.compile(r"l_[0-9a-f]{32}\Z")


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
    root = _validate_text(value, "canonical_integration_root", 4096)
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(root) != root:
        raise ProjectRegistryError(
            "canonical_integration_root must be an absolute normalized path"
        )
    return root


def _new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(ID_HEX_LENGTH // 2)


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
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_nlink < 1
                or (stat.S_IMODE(info.st_mode) & 0o077)
            ):
                raise ProjectRegistryError(
                    "projects registry root has unsafe permissions or type"
                )
            # Several processes may initialize a registry at once. Re-open
            # after a transient first-use mkdir/open race on strict platforms.
            for attempt in range(10):
                try:
                    lock_fd = os.open(
                        ".lock",
                        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=root_fd,
                    )
                    break
                except FileNotFoundError:
                    if attempt == 9:
                        raise
                    os.close(root_fd)
                    time.sleep(0.001)
                    root_fd = os.open(self.root, flags)
            try:
                lock_info = os.fstat(lock_fd)
                if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1:
                    raise ProjectRegistryError(
                        "registry lock is not a regular unshared file"
                    )
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
        if created:
            os.chmod(self.root, 0o700)
        elif stat.S_IMODE(existing.st_mode) & 0o077:
            raise ProjectRegistryError("projects registry root has unsafe permissions")

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
                    value = json.loads(stream.read(10_000_001))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProjectRegistryError(f"malformed project record {name}") from exc
            if not isinstance(value, dict):
                raise ProjectRegistryError(f"malformed project record {name}")
            return value
        finally:
            os.close(fd)

    @staticmethod
    def _publish(directory_fd: int, name: str, value: dict[str, object]) -> None:
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        data = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd
            )
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass

    def _project_dir(
        self, root_fd: int, project_id: str, *, create: bool = False
    ) -> int:
        _validate_id(project_id, _PROJECT_ID, "project_id")
        if create:
            try:
                os.mkdir(project_id, 0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
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
            now = _now()
            project = Project(
                _new_id(ID_PREFIX), name, scope, now, now, canonical_integration_root
            )
            directory_fd = self._project_dir(root_fd, project.project_id, create=True)
            try:
                self._publish(directory_fd, "project.json", self._encode(project))
            finally:
                os.close(directory_fd)
            return project

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
                projects.append(
                    self._decode(self._read_fd(directory_fd, "project.json"), name)
                )
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
                lane = Lane(_new_id(LANE_ID_PREFIX), project_id, name, scope, now, now)
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
