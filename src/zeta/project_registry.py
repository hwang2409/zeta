"""Hardened, identity-only project registry.

This module deliberately contains no session or execution concepts.  Records are
small JSON documents stored below ``~/.zeta/projects`` and are published only
while holding the registry lock.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .core.session_files import atomic_publish_file
from .project_errors import ProjectNotFoundError, ProjectRegistryError
from .project_memory_history import ProjectMemoryHistoryMixin

SCHEMA_VERSION = 1
ID_PREFIX = "p_"
ID_HEX_LENGTH = 32
MAX_NAME_LENGTH = 128
MAX_SCOPE_LENGTH = 4096
MAX_PROJECTS = 10_000
MAX_RECORD_SIZE = 10_000_000
MAX_MEMORY_FILE_SIZE = 128 * 1024
MAX_SESSION_REFERENCE_SIZE = 4096
MAX_SESSION_REFERENCES = 10_000
MAX_CREATE_RETRIES = 32
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_SESSION_ROLES = {"session", "orchestrator", "worker"}
_READ_RETRIES = 10
_T = TypeVar("_T")


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    scope: str
    created_at: str
    updated_at: str
    canonical_integration_root: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "scope": self.scope,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "canonical_integration_root": self.canonical_integration_root,
        }


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


class ProjectRegistry(ProjectMemoryHistoryMixin):
    """A local registry whose root can be overridden for tests."""

    def __init__(self, root: Path | str | None = None):
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / ".zeta" / "projects"
        )

    def _open_root(self) -> tuple[int, os.stat_result]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(self.root, flags)
        except FileNotFoundError as exc:
            raise ProjectNotFoundError("projects registry does not exist") from exc
        except OSError as exc:
            raise ProjectRegistryError(
                "projects registry root is not a safe directory"
            ) from exc
        info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_nlink < 1
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            os.close(root_fd)
            raise ProjectRegistryError(
                "projects registry root has unsafe permissions or type"
            )
        return root_fd, info

    @staticmethod
    def _open_lock(root_fd: int, *, create: bool) -> int | None:
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
        try:
            lock_fd = os.open(
                ".lock", flags | os.O_NOFOLLOW, 0o600, dir_fd=root_fd
            )
        except FileNotFoundError:
            if not create:
                return None
            raise
        except OSError as exc:
            raise ProjectRegistryError("cannot open registry lock") from exc
        lock_info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or stat.S_IMODE(lock_info.st_mode) & 0o077
        ):
            os.close(lock_fd)
            raise ProjectRegistryError(
                "registry lock is not a private regular unshared file"
            )
        return lock_fd

    def _root_unchanged_without_lock(
        self, root_fd: int, original: os.stat_result
    ) -> bool:
        try:
            current = os.stat(self.root, follow_symlinks=False)
            lock_fd = self._open_lock(root_fd, create=False)
        except (FileNotFoundError, ProjectNotFoundError):
            return False
        if lock_fd is not None:
            os.close(lock_fd)
            return False
        return (
            stat.S_ISDIR(current.st_mode)
            and not stat.S_ISLNK(current.st_mode)
            and current.st_dev == original.st_dev
            and current.st_ino == original.st_ino
        )

    def _read(self, operation: Callable[[int], _T]) -> _T:
        """Read one consistent registry snapshot without creating its lock."""
        for attempt in range(_READ_RETRIES):
            root_fd, root_info = self._open_root()
            try:
                lock_fd = self._open_lock(root_fd, create=False)
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_SH)
                        return operation(root_fd)
                    finally:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
                try:
                    result = operation(root_fd)
                except Exception:
                    if self._root_unchanged_without_lock(root_fd, root_info):
                        raise
                else:
                    if self._root_unchanged_without_lock(root_fd, root_info):
                        return result
            finally:
                os.close(root_fd)
            if attempt + 1 < _READ_RETRIES:
                time.sleep(0.001)
        raise ProjectRegistryError("projects registry changed during read")

    @contextmanager
    def _locked(self, *, write: bool) -> Iterator[int]:
        if not write:
            raise AssertionError("read operations must use _read")
        self._ensure_root()
        for attempt in range(_READ_RETRIES):
            root_fd, _ = self._open_root()
            try:
                os.fchmod(root_fd, 0o700)
                try:
                    lock_fd = self._open_lock(root_fd, create=True)
                except FileNotFoundError as exc:
                    if attempt + 1 == _READ_RETRIES:
                        raise ProjectRegistryError("cannot open registry lock") from exc
                else:
                    assert lock_fd is not None
                    try:
                        os.fchmod(lock_fd, 0o600)
                        fcntl.flock(lock_fd, fcntl.LOCK_EX)
                        yield root_fd
                        return
                    finally:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
            finally:
                os.close(root_fd)
            time.sleep(0.001)
        raise ProjectRegistryError("cannot open registry lock")

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
        # Imported lazily because remote_sync imports ProjectRegistry during startup.
        from .remote_sync.project_publish import validate_project_record

        try:
            fields = validate_project_record(value, expected_id)
        except ValueError as exc:
            raise ProjectRegistryError(str(exc)) from exc
        return Project(*fields)

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
        try:
            return self._read(self._list_locked)
        except ProjectNotFoundError:
            return []

    def show_project(
        self, project_id: str | None = None, *, name: str | None = None
    ) -> Project:
        def read(root_fd: int) -> Project:
            projects = self._list_locked(root_fd)
            matches = [
                p
                for p in projects
                if (project_id is not None and p.project_id == project_id)
                or (name is not None and p.name == name)
            ]
            if len(matches) != 1:
                raise ProjectNotFoundError("project not found or ambiguous")
            return matches[0]

        return self._read(read)

    def find_or_create_for_directory(
        self, directory: str | Path, *, name: str | None = None, scope: str = "git"
    ) -> Project:
        """Find the project for a repository root, creating it safely if absent.

        Creation races are resolved by re-reading the registry after another
        process publishes the same canonical root.
        """
        path = Path(directory).expanduser().resolve()
        existing = self.find_for_directory(path)
        if existing is not None:
            return existing
        base_name = name or path.name
        try:
            return self.create_project(base_name, scope, path)
        except ProjectRegistryError:
            existing = self.find_for_directory(path)
            if existing is not None:
                return existing
            # Name allocation is retried under create_project's registry lock;
            # the canonical root check still makes concurrent callers converge.
            parent_name = path.parent.name or "repo"
            candidates = [f"{base_name}-{parent_name}",
                          f"{base_name}-{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"]
            for candidate in candidates:
                try:
                    return self.create_project(candidate, scope, path)
                except ProjectRegistryError:
                    existing = self.find_for_directory(path)
                    if existing is not None:
                        return existing
            raise

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
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, min(16 * 1024, MAX_MEMORY_FILE_SIZE - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MEMORY_FILE_SIZE:
                    raise ProjectRegistryError(
                        f"project memory file {name} is too large"
                    )
                chunks.append(chunk)
            from .remote_sync.project_publish import decode_legacy_memory

            try:
                return decode_legacy_memory(b"".join(chunks), name)
            except ValueError as exc:
                raise ProjectRegistryError(str(exc)) from exc
        except OSError as exc:
            raise ProjectRegistryError(
                f"project memory file {name} is unreadable"
            ) from exc
        finally:
            os.close(fd)

    def load_memory(
        self, project_id: str, *, byte_cap: int = 64 * 1024
    ) -> list[tuple[str, str]]:
        """Load bounded project memory; malformed legacy files are rejected."""
        if type(byte_cap) is not int or byte_cap < 0 or byte_cap > MAX_RECORD_SIZE:
            raise ProjectRegistryError("invalid memory byte cap")
        return self._load_memory_view(project_id, byte_cap)

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
        return self._replace_memory(project_id, updates)

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
        except OSError as exc:
            # O_NOFOLLOW turns a symlinked reference into ELOOP; keep the
            # public contract that unsafe references raise a registry error
            # rather than leaking a raw OSError to callers.
            raise ProjectRegistryError(
                "project session references are unreadable"
            ) from exc
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
            if pending and repair_torn_final:
                # A crash can leave one unterminated final record. Readers must
                # ignore that bounded tail without requiring a later append;
                # writers may additionally repair it while holding the lock.
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
        def read(root_fd: int) -> list[dict[str, object]]:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._read_session_records(
                    directory_fd, repair_torn_final=False
                )[-limit:]
            finally:
                os.close(directory_fd)

        return self._read(read)

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
                self._append_reference_blob(directory_fd, payload)
            finally:
                os.close(directory_fd)

    def record_sessions(
        self, project_id: str, records: list[Mapping[str, object]]
    ) -> None:
        """Append several transcript references under a single read and write.

        Child-lineage reconciliation collects many durable intents at once.
        Recording them one-by-one re-reads the whole JSONL registry per intent
        (quadratic in a large tree); this validates every candidate, reads the
        existing references exactly once, dedupes against them and within the
        batch, and appends only the missing references while holding the lock.
        """
        _validate_id(project_id, _PROJECT_ID, "project_id")
        if not records:
            return
        payloads: list[tuple[str, bytes]] = []
        batch_ids: set[str] = set()
        for record in records:
            session_id = record["session_id"]
            transcript_path = record["transcript_path"]
            role = record.get("role", "session")
            parent_session_id = record.get("parent_session_id")
            _validate_text(session_id, "session_id", 128)
            _validate_text(transcript_path, "transcript_path", MAX_SCOPE_LENGTH)
            _validate_text(role, "role", 64)
            if role not in _SESSION_ROLES:
                raise ProjectRegistryError("invalid project session role")
            if parent_session_id is not None:
                _validate_text(parent_session_id, "parent_session_id", 128)
            if session_id in batch_ids:
                continue
            batch_ids.add(session_id)
            payload_record = {
                "session_id": session_id,
                "transcript_path": transcript_path,
                "role": role,
                "parent_session_id": parent_session_id,
                "recorded_at": _now(),
            }
            payload = (json.dumps(payload_record, sort_keys=True) + "\n").encode("utf-8")
            if len(payload) > MAX_SESSION_REFERENCE_SIZE:
                raise ProjectRegistryError("project session reference is too large")
            payloads.append((session_id, payload))
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                existing = self._read_session_records(
                    directory_fd, repair_torn_final=True
                )
                existing_ids = {item["session_id"] for item in existing}
                blob = b"".join(
                    payload
                    for session_id, payload in payloads
                    if session_id not in existing_ids
                )
                if not blob:
                    return
                added = sum(
                    1 for session_id, _ in payloads if session_id not in existing_ids
                )
                if len(existing) + added > MAX_SESSION_REFERENCES:
                    raise ProjectRegistryError(
                        "project session reference limit exceeded"
                    )
                self._append_reference_blob(directory_fd, blob)
            finally:
                os.close(directory_fd)

    @staticmethod
    def _append_reference_blob(directory_fd: int, blob: bytes) -> None:
        """Append pre-encoded reference bytes to the pinned JSONL file safely."""
        try:
            try:
                os.stat("sessions.jsonl", dir_fd=directory_fd, follow_symlinks=False)
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
                raise ProjectRegistryError("project session references are unsafe")
            if (
                info.st_size + len(blob)
                > MAX_SESSION_REFERENCES * MAX_SESSION_REFERENCE_SIZE
            ):
                os.close(fd)
                raise ProjectRegistryError("project session reference limit exceeded")
            with os.fdopen(fd, "ab") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
            if created:
                os.fsync(directory_fd)
        except ProjectRegistryError:
            raise
        except OSError as exc:
            raise ProjectRegistryError("cannot record project session") from exc

    def initialize_memory(self, project_id: str) -> None:
        """Create the standard memory files without overwriting human edits."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                self._require_format_one(directory_fd)
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
