"""Secure, bounded storage for complete tool outputs."""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..core.session_files import SessionError, child_directory, open_session_file

SPILL_DIRECTORY = "spill"
SPILL_MAX_BYTES = 100 * 1024 * 1024
SPILL_SESSION_MAX_BYTES = 256 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_LOCK_FILE = ".spill.lock"


@dataclass(frozen=True)
class SpillArtifact:
    path: Path
    byte_size: int


class SpillFullError(RuntimeError):
    """The root session has no spill capacity left."""


@dataclass(frozen=True)
class _SpillDirectory:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _RetainedArtifact:
    directory: _SpillDirectory
    name: str
    byte_size: int
    group: object | None


class SpillByteBudget:
    """Bound and reclaim spill bytes for one registry tree."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = SPILL_SESSION_MAX_BYTES if max_bytes is None else max_bytes
        self._used_bytes = 0
        self._retained: dict[tuple[_SpillDirectory, str], _RetainedArtifact] = {}
        self._reclaiming: set[tuple[_SpillDirectory, str]] = set()
        self._protected_groups: set[object] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def reserve(
        self,
        byte_count: int,
        *,
        directory: _SpillDirectory | None = None,
        directory_fd: int | None = None,
    ) -> None:
        while True:
            artifact = self._reserve_or_select_oldest(
                byte_count, directory=directory
            )
            if artifact is None:
                return
            self._reclaim(artifact, directory=directory, directory_fd=directory_fd)

    def release(self, byte_count: int) -> None:
        with self._lock:
            if byte_count > self._used_bytes:
                raise ValueError("spill release exceeds its reservation")
            self._used_bytes -= byte_count

    def retain(
        self,
        directory: _SpillDirectory,
        artifacts: Mapping[str, int],
        *,
        group: object | None = None,
    ) -> None:
        with self._lock:
            if group is not None:
                self._protected_groups.add(group)
            for name, byte_size in artifacts.items():
                key = (directory, name)
                if key in self._retained:
                    raise ValueError("spill artifact is already retained")
                self._retained[key] = _RetainedArtifact(
                    directory, name, byte_size, group
                )

    def unprotect(self, group: object) -> None:
        with self._lock:
            self._protected_groups.discard(group)

    def forget(self, directory: _SpillDirectory, name: str) -> None:
        key = (directory, name)
        with self._lock:
            artifact = self._retained.pop(key, None)
            self._reclaiming.discard(key)
            if artifact is not None:
                self._used_bytes -= artifact.byte_size
            self._condition.notify_all()

    def _reserve_or_select_oldest(
        self, byte_count: int, *, directory: _SpillDirectory | None
    ) -> _RetainedArtifact | None:
        with self._condition:
            while True:
                if self._used_bytes + byte_count <= self.max_bytes:
                    self._used_bytes += byte_count
                    return None
                for key, artifact in self._retained.items():
                    if (
                        key not in self._reclaiming
                        and artifact.group not in self._protected_groups
                    ):
                        self._reclaiming.add(key)
                        return artifact
                if self._reclaiming:
                    for key in self._reclaiming:
                        artifact = self._retained.get(key)
                        if artifact is not None and artifact.directory == directory:
                            return artifact
                    self._condition.wait()
                    continue
                raise SpillFullError("spill budget is full")

    def _reclaim(
        self,
        artifact: _RetainedArtifact,
        *,
        directory: _SpillDirectory | None,
        directory_fd: int | None,
    ) -> None:
        if directory == artifact.directory and directory_fd is not None:
            self._unlink_if_safe(directory_fd, artifact.name)
        else:
            self._reopen_and_unlink(artifact)
        self.forget(artifact.directory, artifact.name)

    def _reopen_and_unlink(self, artifact: _RetainedArtifact) -> None:
        try:
            directory_fd = os.open(
                artifact.directory.path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError:
            return
        try:
            info = os.fstat(directory_fd)
            if (info.st_dev, info.st_ino) != (
                artifact.directory.device,
                artifact.directory.inode,
            ):
                return
            with _spill_advisory_lock(directory_fd):
                self._unlink_if_safe(directory_fd, artifact.name)
        except (OSError, SessionError):
            return
        finally:
            os.close(directory_fd)

    @staticmethod
    def _unlink_if_safe(directory_fd: int, name: str) -> None:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                return
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            return


class SpillStore:
    """Own complete tool outputs behind one small, session-scoped interface."""

    def __init__(
        self,
        *,
        session_dir: Path | None = None,
        directory_fd: int | None = None,
        max_bytes: int = SPILL_MAX_BYTES,
        byte_budget: SpillByteBudget | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self._byte_budget = byte_budget or SpillByteBudget()
        self._lock = threading.Lock()
        self._transient_lock = threading.Lock()
        self._directory_fd: int | None = None
        self._directory_identity: _SpillDirectory | None = None
        self._parent_fd: int | None = None
        self._temporary_root: Path | None = None
        if session_dir is None:
            root = Path(tempfile.gettempdir()) / f"zeta-tool-spill-{uuid.uuid4().hex}"
            self.root = root.absolute()
            self._temporary_root = self.root
        else:
            if directory_fd is None:
                raise ValueError("session spill storage requires a directory descriptor")
            # Borrow the store's pinned descriptor. Duplicating it would retain
            # the store's session lease after construction or shutdown failures.
            self._parent_fd = directory_fd
            self.root = Path(session_dir).absolute() / SPILL_DIRECTORY
        self._closed = False
        self._retained: dict[str, int] = {}
        self._transient_bytes = 0
        self._transient_max_bytes = 64 * 1024 * 1024

    def configure_transient_limit(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("transient spill limit must be positive")
        with self._transient_lock:
            self._transient_max_bytes = max_bytes

    def reserve_transient(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte_count must not be negative")
        with self._transient_lock:
            if self._transient_bytes + byte_count > self._transient_max_bytes:
                raise RuntimeError("transient spill budget is full")
            self._byte_budget.reserve(byte_count)
            self._transient_bytes += byte_count

    def release_transient(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("transient spill release must not be negative")
        with self._transient_lock:
            if byte_count > self._transient_bytes:
                raise ValueError("transient spill release exceeds its reservation")
            self._transient_bytes -= byte_count
            self._byte_budget.release(byte_count)

    async def awrite_text_group(
        self,
        tool: str,
        call_id: str,
        named_strings: Mapping[str, str],
    ) -> dict[str, SpillArtifact]:
        """Encode and publish one text result group in a worker thread."""

        worker = asyncio.create_task(
            asyncio.to_thread(self.write_text_group, tool, call_id, named_strings)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise

    async def awrite_bytes(
        self, tool: str, call_id: str, index: int, data: bytes
    ) -> Path:
        """Persist complete bytes without blocking the calling event loop."""

        return await self.awrite_parts(tool, call_id, index, [data])

    async def awrite_parts(
        self,
        tool: str,
        call_id: str,
        index: int,
        parts: Iterable[bytes | BinaryIO],
    ) -> Path:
        """Copy, publish, and evict one artifact in a worker thread."""

        paths = await self.awrite_group(tool, call_id, {str(index): parts})
        return paths[str(index)]

    async def awrite_group(
        self,
        tool: str,
        call_id: str,
        artifacts: Mapping[str, Iterable[bytes | BinaryIO]],
    ) -> dict[str, Path]:
        """Publish one complete result group without blocking the event loop.

        Cancellation stops only the await. The worker keeps the advisory lock
        until every artifact is published and eviction is complete, then
        releases it normally.
        """

        worker = asyncio.create_task(
            asyncio.to_thread(self.write_group, tool, call_id, artifacts)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Keep caller-owned source files alive and keep the publication
            # transaction indivisible before cancellation leaves this method.
            await worker
            raise

    def write_text(self, tool: str, call_id: str, index: int, text: str) -> Path:
        artifacts = self.write_text_group(tool, call_id, {str(index): text})
        return artifacts[str(index)].path

    def write_text_group(
        self,
        tool: str,
        call_id: str,
        named_strings: Mapping[str, str],
    ) -> dict[str, SpillArtifact]:
        """Encode, publish, account for, and evict one text result group."""

        encoded = {name: text.encode("utf-8") for name, text in named_strings.items()}
        paths = self.write_group(
            tool,
            call_id,
            {name: [data] for name, data in encoded.items()},
        )
        return {
            name: SpillArtifact(path=paths[name], byte_size=len(data))
            for name, data in encoded.items()
        }

    def write_bytes(self, tool: str, call_id: str, index: int, data: bytes) -> Path:
        """Persist all bytes, then evict older files without rejecting this write."""

        return self.write_parts(tool, call_id, index, [data])

    def write_parts(
        self,
        tool: str,
        call_id: str,
        index: int,
        parts: Iterable[bytes | BinaryIO],
    ) -> Path:
        """Persist one artifact without loading seekable files into memory."""

        paths = self.write_group(tool, call_id, {str(index): parts})
        return paths[str(index)]

    def write_group(
        self,
        tool: str,
        call_id: str,
        artifacts: Mapping[str, Iterable[bytes | BinaryIO]],
    ) -> dict[str, Path]:
        """Publish and retain all named artifacts as one result group."""

        if not artifacts:
            raise ValueError("spill group must contain at least one artifact")
        publications = {
            artifact: (
                self._name(tool, call_id, artifact),
                f".spill-{uuid.uuid4().hex}.tmp",
                parts,
            )
            for artifact, parts in artifacts.items()
        }
        with self._lock:
            directory_fd = self._ensure_directory()
            directory = self._directory()
            with self._advisory_lock(directory_fd):
                reserved: dict[str, int] = {}
                published: list[str] = []
                group = object()
                protected = False
                try:
                    self._write_staged(
                        directory_fd, directory, publications, reserved
                    )
                    self._publish_staged(directory_fd, publications, published)
                    retained = {
                        name: reserved[temporary]
                        for name, temporary, _parts in publications.values()
                    }
                    self._byte_budget.retain(directory, retained, group=group)
                    protected = True
                    self._retained.update(retained)
                except BaseException:
                    self._rollback_group(
                        directory_fd, directory, publications, published, reserved
                    )
                    raise
                try:
                    self._evict_before(directory_fd, set(published))
                finally:
                    if protected:
                        self._byte_budget.unprotect(group)
        return {
            artifact: (self.root / name).absolute()
            for artifact, (name, _temporary, _parts) in publications.items()
        }

    def _write_staged(
        self,
        directory_fd: int,
        directory: _SpillDirectory,
        publications: Mapping[str, tuple[str, str, Iterable[bytes | BinaryIO]]],
        reserved: dict[str, int],
    ) -> None:
        for name, temporary, parts in publications.values():
            fd = open_session_file(
                directory_fd,
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            reserved[temporary] = 0
            with os.fdopen(fd, "wb") as destination:
                for part in parts:
                    chunks = self._chunks(part)
                    for chunk in chunks:
                        self._byte_budget.reserve(
                            len(chunk),
                            directory=directory,
                            directory_fd=directory_fd,
                        )
                        try:
                            destination.write(chunk)
                        except BaseException:
                            self._byte_budget.release(len(chunk))
                            raise
                        reserved[temporary] += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())

    @staticmethod
    def _chunks(part: bytes | BinaryIO) -> Iterable[bytes]:
        if isinstance(part, bytes):
            return (part,)
        part.seek(0)
        return iter(lambda: part.read(1024 * 1024), b"")

    @staticmethod
    def _publish_staged(
        directory_fd: int,
        publications: Mapping[str, tuple[str, str, Iterable[bytes | BinaryIO]]],
        published: list[str],
    ) -> None:
        for name, temporary, _parts in publications.values():
            os.replace(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            published.append(name)
        os.fsync(directory_fd)

    def _rollback_group(
        self,
        directory_fd: int,
        directory: _SpillDirectory,
        publications: Mapping[str, tuple[str, str, Iterable[bytes | BinaryIO]]],
        published: list[str],
        reserved: dict[str, int],
    ) -> None:
        temporary_by_name = {
            name: temporary for name, temporary, _parts in publications.values()
        }
        retained: dict[str, int] = {}
        for name in published:
            temporary = temporary_by_name[name]
            if not self._try_unlink(directory_fd, name):
                retained[name] = reserved.get(temporary, 0)
        for _name, temporary, _parts in publications.values():
            if not self._try_unlink(directory_fd, temporary):
                retained[temporary] = reserved.get(temporary, 0)
        if retained:
            self._byte_budget.retain(directory, retained)
            self._retained.update(retained)
        self._byte_budget.release(sum(reserved.values()) - sum(retained.values()))

    @staticmethod
    def _try_unlink(directory_fd: int, name: str) -> bool:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    @contextmanager
    def temporary_file(self) -> Iterator[BinaryIO]:
        """Yield an unlinked private file in spill storage."""

        with self._lock:
            directory_fd = self._ensure_directory()
            name = f".private-{uuid.uuid4().hex}.tmp"
            fd = open_session_file(
                directory_fd,
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
            )
            os.unlink(name, dir_fd=directory_fd)
        with os.fdopen(fd, "w+b") as handle:
            yield handle

    def contains(self, path: Path) -> bool:
        try:
            relative = path.absolute().relative_to(self.root)
        except ValueError:
            return False
        return len(relative.parts) == 1

    def open_read(self, path: Path) -> int:
        if not self.contains(path):
            raise ValueError("path is not in this session's spill directory")
        with self._lock:
            return open_session_file(
                self._ensure_directory(), path.name, os.O_RDONLY
            )

    def _name(self, tool: str, call_id: str, artifact: str) -> str:
        if self._closed:
            raise RuntimeError("spill store is closed")
        tool_name = _safe_component(tool, "tool")
        call_name = _safe_component(call_id, "call")
        artifact_name = _safe_component(artifact, "artifact")
        return f"{tool_name}-{call_name}-{artifact_name}-{uuid.uuid4().hex}.txt"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None
            self._parent_fd = None
            if self._temporary_root is not None:
                shutil.rmtree(self._temporary_root, ignore_errors=True)
                self._forget_missing_retained()
        with self._transient_lock:
            self._byte_budget.release(self._transient_bytes)
            self._transient_bytes = 0

    def _forget_missing_retained(self) -> None:
        if self._directory_identity is None:
            return
        for name in list(self._retained):
            if not (self.root / name).exists():
                self._byte_budget.forget(self._directory_identity, name)
                self._retained.pop(name, None)

    def _ensure_directory(self) -> int:
        if self._closed:
            raise RuntimeError("spill store is closed")
        if self._directory_fd is not None:
            return self._directory_fd
        if self._parent_fd is not None:
            with child_directory(
                self._parent_fd, SPILL_DIRECTORY, create=True
            ) as spill_fd:
                self._directory_fd = os.dup(spill_fd)
        else:
            os.mkdir(self.root, mode=0o700)
            os.chmod(self.root, 0o700)
            self._directory_fd = os.open(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        if self._directory_identity is None:
            info = os.fstat(self._directory_fd)
            self._directory_identity = _SpillDirectory(
                self.root, info.st_dev, info.st_ino
            )
        return self._directory_fd

    def _directory(self) -> _SpillDirectory:
        if self._directory_identity is None:
            raise RuntimeError("spill directory is not open")
        return self._directory_identity

    @contextmanager
    def _advisory_lock(self, directory_fd: int) -> Iterator[None]:
        with _spill_advisory_lock(directory_fd):
            yield

    def _evict_before(self, directory_fd: int, newest: set[str]) -> None:
        """Delete oldest files while retaining every file in the newest group."""

        try:
            entries: list[tuple[int, str, int]] = []
            total = 0
            for name in os.listdir(directory_fd):
                if name.startswith("."):
                    continue
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    continue
                total += info.st_size
                entries.append((info.st_mtime_ns, name, info.st_size))
            for _mtime, name, size in sorted(entries):
                if total <= self.max_bytes or name in newest:
                    continue
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    continue
                total -= size
                if self._retained.pop(name, None) is not None:
                    self._byte_budget.forget(self._directory(), name)
        except OSError:
            return


@contextmanager
def _spill_advisory_lock(directory_fd: int) -> Iterator[None]:
    lock_fd = open_session_file(
        directory_fd,
        _LOCK_FILE,
        os.O_RDWR | os.O_CREAT,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _safe_component(value: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip(".-")
    return (cleaned or fallback)[:80]
