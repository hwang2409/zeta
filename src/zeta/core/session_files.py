"""Pinned session directories and cooperative lifetime leases."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import os
import stat
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, fields
from pathlib import Path

from ..locking import RETENTION_HOME_ID, domain_authority, execution_lock, shape_lock
from ..storage_format import (
    StorageFeature,
    WriterLease,
    drain_releases,
    feature_enabled,
)


class SessionError(ValueError):
    """Raised when a session cannot be accessed safely."""


class SessionNotFoundError(SessionError):
    """Raised when the requested session does not exist."""


class SessionIdAmbiguousError(SessionError):
    """Raised when a session id prefix matches more than one session."""

    def __init__(self, message: str, candidates: tuple[str, ...]) -> None:
        super().__init__(message)
        self.candidates = candidates


class SessionFileReplaced(SessionError):
    """Raised when read retries keep opening an unlinked session file."""


class SessionInUseError(SessionError):
    """Raised when an open store or metadata operation prevents deletion."""


class SessionReferenceUnavailable(SessionError):
    """Raised when retention is removing a referenced session."""


_READ_RETRIES = 3
SESSION_REFERENCE_GENERATION = "session-reference-generation"
REFERENCES_DOMAIN = "references"


def reservation_path(home: Path, session_id: str) -> Path:
    """Durable reservation marker a deleter writes before removing a session."""
    return home / "session-trash" / "reservations" / f"{session_id}.json"


def quarantine_path(home: Path, session_id: str) -> Path:
    """Directory a deleter moves a session into while removing it."""
    return home / "session-trash" / session_id


@dataclass(frozen=True)
class SessionReferenceAuthority:
    """Proof that one outer publication owns reference admission."""

    gate: SessionReferenceGate
    session_ids: frozenset[str]
    def validate(self, session_ids: tuple[str, ...] | list[str]) -> None:
        self.gate.validate(session_ids)


class SessionReferenceGate:
    """Coordinate all durable session references with retention."""

    def __init__(self, home: Path, lease: WriterLease) -> None:
        self.home = Path(home)
        self.lease = lease

    def _enabled(self) -> bool:
        return feature_enabled(self.lease.observed_epoch, StorageFeature.SESSION_RETENTION)

    @contextmanager
    def _locked(self, *, exclusive: bool, unavailable_id: str | None = None):
        if not self._enabled():
            yield self
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop = False
        else:
            on_loop = True
        if exclusive and on_loop:
            raise SessionError(
                "exclusive session reference authority requested on an asyncio event-loop thread"
            )
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        with ExitStack() as authority:
            try:
                authority.enter_context(
                    domain_authority(
                        self.home,
                        RETENTION_HOME_ID,
                        REFERENCES_DOMAIN,
                        exclusive=exclusive,
                        nonblocking=on_loop,
                    )
                )
            except BlockingIOError as exc:
                session_id = unavailable_id or "unknown"
                raise SessionReferenceUnavailable(
                    f"session reference unavailable: {session_id} is being deleted"
                ) from exc
            except OSError as exc:
                raise SessionError(
                    "session reference authority is unsafe or unavailable"
                ) from exc
            yield self

    @contextmanager
    def _shared(
        self,
        *,
        installed_ids: tuple[str, ...],
        referenced_ids: tuple[str, ...],
    ):
        ids = (*installed_ids, *referenced_ids)
        with self._locked(exclusive=False, unavailable_id=ids[0] if ids else None):
            self.validate(ids)
            if installed_ids:
                self.bump_generation()
            yield SessionReferenceAuthority(self, frozenset(ids))

    def exclusive(self, *, session_id: str | None = None):
        return self._locked(exclusive=True, unavailable_id=session_id)

    def validate(self, session_ids: tuple[str, ...] | list[str]) -> None:
        if not self._enabled():
            return
        for session_id in session_ids:
            _component(session_id)
            reservation = reservation_path(self.home, session_id)
            quarantined = quarantine_path(self.home, session_id)
            if reservation.exists() or quarantined.exists():
                raise SessionReferenceUnavailable(
                    f"session reference unavailable: {session_id} is being deleted"
                )

    def bump_generation(self) -> int:
        if not self._enabled():
            return 0
        path = self.home / SESSION_REFERENCE_GENERATION
        try:
            raw = path.read_text(encoding="ascii")
        except FileNotFoundError:
            value = 0
        else:
            if not raw.endswith("\n") or not raw[:-1].isdigit():
                raise SessionError("session reference generation is malformed")
            value = int(raw[:-1])
            if value > 2**63 - 2:
                raise SessionError("session reference generation is out of range")
        value += 1
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, f"{value}\n".encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
        directory_fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return value


def read_reference_generation(home: Path) -> int:
    """Return the reference generation without touching its admission gate."""
    try:
        fd = os.open(
            Path(home) / SESSION_REFERENCE_GENERATION,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise SessionError("session reference generation is unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SessionError("session reference generation is unsafe")
        raw = os.read(fd, 32).decode("ascii", "replace")
    finally:
        os.close(fd)
    if not raw.endswith("\n") or not raw[:-1].isdigit():
        raise SessionError("session reference generation is malformed")
    return int(raw[:-1])


@contextmanager
def reference_publication(
    home: Path,
    lease: WriterLease,
    *,
    installed_ids: tuple[str, ...] = (),
    referenced_ids: tuple[str, ...] = (),
):
    """Enter reference admission once for one complete public transaction."""
    gate = SessionReferenceGate(home, lease)
    with gate._shared(
        installed_ids=installed_ids,
        referenced_ids=referenced_ids,
    ) as authority:
        yield authority


def retained_reference_publication(
    home: Path,
    lease: WriterLease,
    stack: ExitStack | None,
    authorities: list[SessionReferenceAuthority] | None,
    **publication_ids,
):
    """Keep execution and reference admission alive for a larger transaction."""
    if stack is None:
        return execution_lock(), reference_publication(home, lease, **publication_ids)
    stack.enter_context(execution_lock())
    authority = stack.enter_context(
        reference_publication(home, lease, **publication_ids)
    )
    if authorities is not None:
        authorities.append(authority)
    return nullcontext(), nullcontext(authority)


def copy_session_metadata(target, source) -> None:
    for metadata_field in fields(source):
        if metadata_field.name != "_execution_lease":
            setattr(
                target,
                metadata_field.name,
                copy.deepcopy(getattr(source, metadata_field.name)),
            )


def mutate_session_metadata(
    manager,
    session_id: str,
    update,
    *,
    lock_timeout: float | None,
    execution_lease,
    authority,
    before_release,
):
    updated = None

    def release() -> None:
        if before_release is not None:
            assert updated is not None
            before_release(updated)

    with (
        manager._execution_owner(session_id, execution_lease),
        manager._metadata_lock(
            session_id,
            timeout=lock_timeout,
            authority=authority,
            after_shape=release,
        ) as directory_fd,
    ):
        current = manager._read(session_id, directory_fd=directory_fd)
        updated = update(current)
        manager._write_unlocked(updated, directory_fd=directory_fd)
    assert updated is not None
    return updated


@contextmanager
def session_metadata_lock(
    sessions_dir: Path,
    session_id: str,
    *,
    blocking: bool,
    timeout: float | None,
    authority,
    after_shape: Callable[[], None] | None,
    loop_error: type[SessionError],
):
    if blocking:
        require_off_loop_lock("session metadata", loop_error)
    if timeout is not None:
        deadline = time.monotonic() + timeout
        while True:
            with session_metadata_lock(
                sessions_dir,
                session_id,
                blocking=False,
                timeout=None,
                authority=authority,
                after_shape=after_shape,
                loop_error=loop_error,
            ) as directory_fd:
                if directory_fd is not None:
                    yield directory_fd
                    return
            if time.monotonic() >= deadline:
                raise SessionError("session metadata lock acquisition timed out")
            time.sleep(min(0.05, deadline - time.monotonic()))
    with (
        authority or nullcontext(),
        session_root(sessions_dir) as root_fd,
        child_directory(root_fd, session_id) as directory_fd,
    ):
        try:
            lock_fd = open_session_file(
                directory_fd, ".meta.lock", os.O_RDWR | os.O_CREAT
            )
        except OSError as exc:
            raise SessionError("session metadata lock could not be opened") from exc
        try:
            try:
                fcntl.flock(
                    lock_fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                )
            except BlockingIOError:
                yield None
                return
            with shape_lock():
                yield directory_fd
            if after_shape is not None:
                after_shape()
        finally:
            os.close(lock_fd)

def require_off_loop_lock(
    kind: str, error: Callable[[str], BaseException] = RuntimeError
) -> None:
    """Reject file-lock acquisition on an asyncio event-loop thread."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise error(f"{kind} lock requested on an asyncio event-loop thread")


@contextmanager
def session_directory(
    root: Path,
    session_id: str,
    *,
    exclusive: bool = False,
    lifetime: bool = False,
):
    """Never create directories; lease the pinned directory inode itself.

    Shared leases cover store lifetimes and metadata mutations. Deletion needs
    an exclusive lease. Closing the descriptors (including on process death)
    releases the lease without leaving a stale lock file.
    """
    with ExitStack() as cleanup:
        cleanup.enter_context(nullcontext() if lifetime else shape_lock())
        try:
            _component(session_id)
            root_fd = cleanup.enter_context(session_root(root))
            session_fd = cleanup.enter_context(child_directory(root_fd, session_id))
            if exclusive:
                drain_releases()  # a store this process dropped must not refuse deletion
            try:
                fcntl.flock(
                    session_fd,
                    (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise SessionInUseError("session is currently open or in use") from exc
            # A waiter must not operate on an inode that deletion already removed.
            if os.fstat(session_fd).st_nlink == 0:
                raise SessionNotFoundError(f"session {session_id} was not found")
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"session {session_id} was not found") from exc
        except OSError as exc:
            raise SessionError(
                f"session {session_id} could not be accessed: {exc.strerror}"
            ) from exc
        # Translate acquisition errors only; preserve the caller's domain errors.
        yield root_fd, session_fd


def _component(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise SessionError("session filename must be one path component")


@contextmanager
def descend_directory(parent_fd: int, components: tuple[str, ...]):
    with ExitStack() as cleanup:
        directory_fd = parent_fd
        for component in components:
            directory_fd = cleanup.enter_context(child_directory(directory_fd, component))
        yield directory_fd


def _open_child_directory(parent_fd: int, name: str, *, create: bool = False) -> int:
    """Open one validated child without following symbolic links."""
    _component(name)
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)


@contextmanager
def child_directory(parent_fd: int, name: str, *, create: bool = False):
    """Pin a child without following symbolic links, including during creation."""
    fd = _open_child_directory(parent_fd, name, create=create)
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
    fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in parts[split:]:
            child_fd = _open_child_directory(fd, name, create=create)
            parent_fd = fd
            fd = child_fd
            try:
                os.close(parent_fd)
            except OSError:
                try:
                    os.close(parent_fd)
                finally:
                    raise
        yield fd
    finally:
        os.close(fd)


def read_session_file(directory_fd: int, name: str) -> bytes:
    with os.fdopen(open_session_file(directory_fd, name, os.O_RDONLY), "rb") as handle:
        return handle.read()


def write_session_file(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    sync: bool = True,
    sync_directory: bool | None = None,
) -> None:
    """Publish bytes atomically within a pinned directory."""
    if sync_directory is None:
        sync_directory = sync
    _component(name)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(
        directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        try:
            stream = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with stream as handle:
            handle.write(data)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        if sync_directory:
            os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def write_session_file_if_absent(
    directory_fd: int, name: str, data: bytes
) -> bool:
    """Atomically publish bytes only when the destination does not exist."""
    _component(name)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(
        directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        try:
            stream = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with stream as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        os.fsync(directory_fd)
        return True
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def write_json_file(
    directory_fd: int, name: str, value: object, *, sync: bool = True
) -> None:
    write_session_file(
        directory_fd,
        name,
        (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        sync=sync,
    )


def unlink_session_file(directory_fd: int, name: str) -> None:
    """Unlink one session file and durably publish the directory change."""
    _component(name)
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    os.fsync(directory_fd)


# Registry records use the same pinned atomic primitive as session metadata.
def atomic_publish_file(
    directory_fd: int, name: str, data: bytes, *, sync_directory: bool = True
) -> None:
    write_session_file(
        directory_fd, name, data, sync_directory=sync_directory
    )


def open_session_file(directory_fd: int, name: str, flags: int) -> int:
    """Open one regular, unshared file without following a link or blocking on a FIFO."""
    _component(name)
    for attempt in range(_READ_RETRIES):
        fd = os.open(
            name,
            (flags & ~os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                raise SessionError(
                    f"session file must be a regular, unshared file: {name}"
                )
            if info.st_nlink == 0:
                read_only = flags & os.O_ACCMODE == os.O_RDONLY
                if read_only and attempt + 1 < _READ_RETRIES:
                    os.close(fd)
                    continue
                error = f"session file was replaced while opening: {name}"
                if read_only:
                    raise SessionFileReplaced(error)
                raise SessionError(error)
            if flags & os.O_TRUNC:
                os.ftruncate(fd, 0)
            return fd
        except BaseException:
            os.close(fd)
            raise
    raise AssertionError("unreachable")


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
            with (
                child_directory(source_fd, name) as source_child,
                child_directory(destination_fd, name, create=True) as destination_child,
            ):
                copy_session_tree(source_child, destination_child)
        else:
            write_session_file(destination_fd, name, read_session_file(source_fd, name))
