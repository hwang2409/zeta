"""Exclusive, epoch-bound ownership for mutable session runtimes."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Literal

from ..storage_format import WriterLease
from .checkpoints import ConversationIntegrityError
from .session_files import SessionInUseError, open_session_file

RUNTIME_LEASE_NAME = "runtime.lease"
OwnerSurface = Literal["tui", "headless", "serve", "child", "recovery"]


class StaleSessionOwnerError(ConversationIntegrityError):
    """Raised when a released session capability attempts a mutation."""


class SessionOwnedError(SessionInUseError):
    """Raised when another runtime exclusively owns a session."""

    def __init__(
        self,
        session_id: str,
        *,
        owner_surface: str | None = None,
        owner_pid: int | None = None,
        owner_epoch: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.owner_surface = owner_surface
        self.owner_pid = owner_pid
        self.owner_epoch = owner_epoch
        detail = ""
        if owner_surface is not None and owner_pid is not None:
            detail = f" ({owner_surface} pid {owner_pid})"
        elif owner_surface is not None:
            detail = f" ({owner_surface})"
        elif owner_pid is not None:
            detail = f" (pid {owner_pid})"
        super().__init__(
            f"session {session_id} is active in another zeta process{detail}"
        )


@dataclass(frozen=True, slots=True)
class OwnerHint:
    surface: OwnerSurface
    pid: int = field(default_factory=os.getpid)


@dataclass(slots=True)
class _OwnedLease:
    fd: int
    identity: tuple[int, int]
    epoch: str
    refs: int
    state: Literal["owned", "recovery", "releasing"]


_REGISTRY: dict[tuple[int, int], _OwnedLease] = {}
_REGISTRY_LOCK = threading.Lock()


def _read_owner_hint(fd: int) -> tuple[str | None, int | None, str | None]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4096)
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None
    if type(value) is not dict:
        return None, None, None
    surface = value.get("surface")
    pid = value.get("pid")
    epoch = value.get("execution_epoch")
    if surface not in {"tui", "headless", "serve", "child", "recovery"}:
        surface = None
    if type(pid) is not int or pid <= 0:
        pid = None
    if type(epoch) is not str or len(epoch) != 32:
        epoch = None
    return surface, pid, epoch


def _publish_owner(fd: int, owner: OwnerHint, epoch: str) -> None:
    record = json.dumps(
        {
            "execution_epoch": epoch,
            "pid": owner.pid,
            "process_start": None,
            "surface": owner.surface,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, record)


class SessionExecutionLease:
    """One capability for a process's exclusive session execution lock."""

    def __init__(self, owned: _OwnedLease, epoch: str) -> None:
        self._owned = owned
        self.epoch = epoch
        self._closed = False

    @classmethod
    def acquire(
        cls,
        directory_fd: int,
        *,
        session_id: str,
        owner: OwnerHint,
        recovery: bool = False,
    ) -> SessionExecutionLease:
        """Acquire immediately or raise a typed refusal without mutating the session."""

        # the loop may do non-waiting local file ops; anything that can wait or fsyncs goes off-loop
        stat = os.fstat(directory_fd)
        identity = (stat.st_dev, stat.st_ino)
        with _REGISTRY_LOCK:
            local = _REGISTRY.get(identity)
            if local is not None:
                raise SessionOwnedError(session_id)
            fd = open_session_file(
                directory_fd,
                RUNTIME_LEASE_NAME,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            )
            os.set_inheritable(fd, False)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                surface, pid, epoch = _read_owner_hint(fd)
                os.close(fd)
                raise SessionOwnedError(
                    session_id,
                    owner_surface=surface,
                    owner_pid=pid,
                    owner_epoch=epoch,
                ) from exc
            except BaseException:
                os.close(fd)
                raise
            epoch = uuid.uuid4().hex
            try:
                _publish_owner(fd, owner, epoch)
            except BaseException:
                os.close(fd)
                raise
            owned = _OwnedLease(
                fd=fd,
                identity=identity,
                epoch=epoch,
                refs=1,
                state="recovery" if recovery else "owned",
            )
            _REGISTRY[identity] = owned
            return cls(owned, epoch)

    @classmethod
    def retain_resident(cls, directory_fd: int) -> SessionExecutionLease | None:
        """Retain this process's resident owner, if one exists."""

        stat = os.fstat(directory_fd)
        identity = (stat.st_dev, stat.st_ino)
        with _REGISTRY_LOCK:
            owned = _REGISTRY.get(identity)
            if owned is None or owned.state not in {"owned", "recovery"}:
                return None
            owned.refs += 1
            return cls(owned, owned.epoch)

    @classmethod
    @contextmanager
    def recover(
        cls, directory_fd: int, *, session_id: str, owner: OwnerHint
    ) -> Iterator[SessionExecutionLease | None]:
        """Try to acquire recovery ownership, returning none for a live owner."""

        try:
            lease = cls.acquire(
                directory_fd,
                session_id=session_id,
                owner=owner,
                recovery=True,
            )
        except SessionOwnedError:
            yield None
            return
        try:
            yield lease
        finally:
            lease.close()

    def retain(self) -> SessionExecutionLease:
        """Return another capability retaining this exact kernel lock and epoch."""

        self.assert_owned(self.epoch)
        with _REGISTRY_LOCK:
            self._owned.refs += 1
        return type(self)(self._owned, self.epoch)

    def assert_owned(self, epoch: str | None = None) -> None:
        """Fence mutations made by closed, releasing, or replaced capabilities."""

        owned = self._owned
        expected = self.epoch if epoch is None else epoch
        with _REGISTRY_LOCK:
            if (
                self._closed
                or owned.state not in {"owned", "recovery"}
                or expected != owned.epoch
                or _REGISTRY.get(owned.identity) is not owned
            ):
                raise StaleSessionOwnerError("session execution capability is stale")

    def close(self) -> None:
        """Release this capability and close the lock after its last retained owner."""

        with _REGISTRY_LOCK:
            if self._closed:
                return
            self._closed = True
            owned = self._owned
            owned.refs -= 1
            if owned.refs:
                return
            owned.state = "releasing"
            if _REGISTRY.get(owned.identity) is owned:
                del _REGISTRY[owned.identity]
            fd, owned.fd = owned.fd, -1
        os.close(fd)


class StoreRuntimeLeaseMixin:
    """Expose the store's execution capability to mutation and recovery seams."""

    _runtime_lease: SessionExecutionLease | None

    def _open_execution_lease(
        self,
        directory_fd: int,
        session_id: str,
        execution_lease: SessionExecutionLease | None,
        owner_surface: OwnerSurface,
        recovery_owner: bool,
    ) -> SessionExecutionLease:
        if execution_lease is not None:
            return execution_lease.retain()
        return SessionExecutionLease.acquire(
            directory_fd,
            session_id=session_id,
            owner=OwnerHint(owner_surface),
            recovery=recovery_owner,
        )

    @property
    def execution_lease(self) -> SessionExecutionLease:
        if self._runtime_lease is None:
            raise ConversationIntegrityError("read-only store has no execution lease")
        return self._runtime_lease

    def assert_execution_owner(self) -> None:
        self.execution_lease.assert_owned()

    @contextmanager
    def orphan_recovery_lease(self) -> Iterator[bool]:
        if self._runtime_lease is None:
            raise ConversationIntegrityError("read-only store has no execution lease")
        self._runtime_lease.assert_owned()
        yield self._runtime_lease._owned.state == "recovery"


def close_store_leases(
    runtime_lease: SessionExecutionLease | None,
    directory_lease: ExitStack,
    storage_lease: WriterLease,
) -> None:
    if runtime_lease is not None:
        runtime_lease.close()
    directory_lease.close()
    storage_lease.close()
