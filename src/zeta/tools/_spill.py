"""Secure, bounded storage for complete tool outputs."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from ..core.session_files import (
    child_directory,
    open_session_file,
    write_session_file,
)

SPILL_DIRECTORY = "spill"
SPILL_MAX_BYTES = 100 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


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

    def write_text(self, tool: str, call_id: str, index: int, text: str) -> Path:
        return self.write_bytes(tool, call_id, index, text.encode("utf-8"))

    def write_bytes(self, tool: str, call_id: str, index: int, data: bytes) -> Path:
        """Persist all bytes, then evict older files without rejecting this write."""

        name = self._name(tool, call_id, index)
        with self._lock:
            directory_fd = self._ensure_directory()
            write_session_file(directory_fd, name, data)
            self._evict_before(directory_fd, name)
        return (self.root / name).absolute()

    def write_parts(
        self,
        tool: str,
        call_id: str,
        index: int,
        parts: Iterable[bytes | BinaryIO],
    ) -> Path:
        """Persist byte strings and seekable files without loading them into memory."""

        name = self._name(tool, call_id, index)
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        with self._lock:
            directory_fd = self._ensure_directory()
            fd = open_session_file(
                directory_fd,
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            try:
                with os.fdopen(fd, "wb") as destination:
                    for part in parts:
                        if isinstance(part, bytes):
                            destination.write(part)
                            continue
                        part.seek(0)
                        shutil.copyfileobj(part, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            self._evict_before(directory_fd, name)
        return (self.root / name).absolute()

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

    def _name(self, tool: str, call_id: str, index: int) -> str:
        if self._closed:
            raise RuntimeError("spill store is closed")
        tool_name = _safe_component(tool, "tool")
        call_name = _safe_component(call_id, "call")
        return f"{tool_name}-{call_name}-{index}-{uuid.uuid4().hex}.txt"

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

    def _evict_before(self, directory_fd: int, newest: str) -> None:
        entries: list[tuple[int, str, int]] = []
        total = 0
        for name in os.listdir(directory_fd):
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                continue
            total += info.st_size
            entries.append((info.st_mtime_ns, name, info.st_size))
        for _mtime, name, size in sorted(entries):
            if total <= self.max_bytes or name == newest:
                continue
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            total -= size


def _safe_component(value: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip(".-")
    return (cleaned or fallback)[:80]
