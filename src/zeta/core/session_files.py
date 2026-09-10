"""Pinned session directories and cooperative lifetime leases."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path


class SessionError(ValueError):
    """Raised when a session cannot be accessed safely."""


class SessionInUseError(SessionError):
    """Raised when an open store or metadata operation prevents deletion."""


@contextmanager
def session_directory(root: Path, session_id: str, *, exclusive: bool = False):
    """Never create directories; lease the pinned directory inode itself.

    Shared leases cover store lifetimes and metadata mutations. Deletion needs
    an exclusive lease. Closing the descriptors (including on process death)
    releases the lease without leaving a stale lock file.
    """
    with ExitStack() as cleanup:
        try:
            _component(session_id)
            root_fd = cleanup.enter_context(session_root(root))
            session_fd = cleanup.enter_context(child_directory(root_fd, session_id))
            try:
                fcntl.flock(
                    session_fd,
                    (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise SessionInUseError("session is currently open or in use") from exc
            # A waiter must not operate on an inode that deletion already removed.
            if os.fstat(session_fd).st_nlink == 0:
                raise SessionError(f"session {session_id} was not found")
        except FileNotFoundError as exc:
            raise SessionError(f"session {session_id} was not found") from exc
        except OSError as exc:
            raise SessionError(f"session {session_id} could not be accessed: {exc.strerror}") from exc
        # Translate acquisition errors only; preserve the caller's domain errors.
        yield root_fd, session_fd


def _component(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise SessionError("session filename must be one path component")


@contextmanager
def child_directory(parent_fd: int, name: str, *, create: bool = False):
    """Pin a child without following symbolic links, including during creation."""
    _component(name)
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def session_root(path: Path, *, create: bool = False):
    """Resolve nested child-store roots from the first sessions directory."""
    path = path.absolute()
    parts = path.parts
    # The home above sessions is trusted; every component beneath it is not.
    split = parts.index("sessions") if "sessions" in parts else len(parts) - 1
    anchor = Path(*parts[:split])
    if create:
        anchor.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY)
        cleanup.callback(os.close, fd)
        for name in parts[split:]:
            fd = cleanup.enter_context(child_directory(fd, name, create=create))
        yield fd


def read_session_file(directory_fd: int, name: str) -> bytes:
    with os.fdopen(open_session_file(directory_fd, name, os.O_RDONLY), "rb") as handle:
        return handle.read()


def write_session_file(directory_fd: int, name: str, data: bytes) -> None:
    """Publish bytes atomically within a pinned directory."""
    _component(name)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def write_session_json(directory_fd: int, name: str, value: object) -> None:
    write_session_file(directory_fd, name, (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode())


def open_session_file(directory_fd: int, name: str, flags: int) -> int:
    """Open one regular, unshared file without following a link or blocking on a FIFO."""
    _component(name)
    fd = os.open(
        name, (flags & ~os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory_fd,
    )
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SessionError(f"session file must be a regular, unshared file: {name}")
        if flags & os.O_TRUNC:
            os.ftruncate(fd, 0)
        return fd
    except BaseException:
        os.close(fd)
        raise


def copy_session_tree(source_fd: int, destination_fd: int) -> None:
    """Copy a tree through pinned descriptors; never follow links in either tree.

    Git runs in private scratch space because it has no no-follow file API.
    Its persistent tree crosses this boundary only as directories and regular,
    unshared files. Remove stale entries without following nested links.
    """
    import shutil

    names = set(os.listdir(source_fd))
    for name in set(os.listdir(destination_fd)) - names:
        info = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(name, dir_fd=destination_fd)
        else:
            os.unlink(name, dir_fd=destination_fd)
    for name in names:
        info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            with child_directory(source_fd, name) as source_child, child_directory(destination_fd, name, create=True) as destination_child:
                copy_session_tree(source_child, destination_child)
        else:
            write_session_file(destination_fd, name, read_session_file(source_fd, name))
