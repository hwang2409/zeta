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

from ..core.session_files import child_directory, open_session_file

SPILL_DIRECTORY = "spill"
SPILL_MAX_BYTES = 100 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_LOCK_FILE = ".spill.lock"


@dataclass(frozen=True)
class SpillArtifact:
    path: Path
    byte_size: int


class SpillStore:
    """Own complete tool outputs behind one small, session-scoped interface."""

    def __init__(
        self,
        *,
        session_dir: Path | None = None,
        directory_fd: int | None = None,
        max_bytes: int = SPILL_MAX_BYTES,
    ) -> None:
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._directory_fd: int | None = None
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
            with self._advisory_lock(directory_fd):
                published: list[str] = []
                try:
                    for name, temporary, parts in publications.values():
                        fd = open_session_file(
                            directory_fd,
                            temporary,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        )
                        with os.fdopen(fd, "wb") as destination:
                            for part in parts:
                                if isinstance(part, bytes):
                                    destination.write(part)
                                    continue
                                part.seek(0)
                                shutil.copyfileobj(part, destination)
                            destination.flush()
                            os.fsync(destination.fileno())
                    for name, temporary, _parts in publications.values():
                        os.replace(
                            temporary,
                            name,
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                        )
                        published.append(name)
                    os.fsync(directory_fd)
                except BaseException:
                    for name in published:
                        try:
                            os.unlink(name, dir_fd=directory_fd)
                        except OSError:
                            pass
                    raise
                finally:
                    for _name, temporary, _parts in publications.values():
                        try:
                            os.unlink(temporary, dir_fd=directory_fd)
                        except FileNotFoundError:
                            pass
                self._evict_before(directory_fd, set(published))
        return {
            artifact: (self.root / name).absolute()
            for artifact, (name, _temporary, _parts) in publications.items()
        }

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
        return self._directory_fd

    @contextmanager
    def _advisory_lock(self, directory_fd: int) -> Iterator[None]:
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
        except OSError:
            return


def _safe_component(value: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip(".-")
    return (cleaned or fallback)[:80]
