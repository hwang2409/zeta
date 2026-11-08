"""Epoch-selected project identity and domain lock protocol."""

from __future__ import annotations

import fcntl
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import TypeVar

from ..core.session_files import require_off_loop_lock
from ..locking import domain_lock, identity_lock
from ..project_errors import (
    ProjectNotFoundError,
    ProjectRegistryBusyError,
    ProjectRegistryError,
)
from ..storage_format import P06_FEATURE_EPOCH, WriterLease

_READ_RETRIES = 10
_PROJECT_REGISTRY_LOCK_TIMEOUT_SECONDS = 1.0
_PROJECT_REGISTRY_LOCK_POLL_SECONDS = 0.05
_T = TypeVar("_T")
_ACTIVE_WRITER_LEASE: ContextVar[WriterLease | None] = ContextVar(
    "project_writer_lease", default=None
)


class ProjectLockingMixin:
    """Own project lock selection, validation, and rank enforcement."""

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
    def _open_lock(root_fd: int) -> int:
        try:
            lock_fd = os.open(
                ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=root_fd
            )
        except FileNotFoundError:
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

    @staticmethod
    def _acquire_lock(lock_fd: int, mode: int) -> None:
        deadline = time.monotonic() + _PROJECT_REGISTRY_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(lock_fd, mode | fcntl.LOCK_NB)
                return
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise ProjectRegistryBusyError() from exc
                time.sleep(_PROJECT_REGISTRY_LOCK_POLL_SECONDS)

    @staticmethod
    def _open_named_lock(directory_fd: int, name: str) -> int:
        try:
            lock_fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ProjectRegistryError(f"cannot open project {name} lock") from exc
        info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            os.close(lock_fd)
            raise ProjectRegistryError("project lock is not a private regular unshared file")
        return lock_fd

    def _validated_project_fd(self, root_fd: int, project_id: str) -> int:
        project_fd = self._project_dir(root_fd, project_id)
        try:
            project = self._decode(
                self._read_fd(project_fd, "project.json"), project_id
            )
            if project.project_id != project_id:
                raise ProjectRegistryError("project identity changed")
            return project_fd
        except BaseException:
            os.close(project_fd)
            raise

    @staticmethod
    def _writer_lease() -> WriterLease:
        lease = _ACTIVE_WRITER_LEASE.get()
        if lease is None:
            raise RuntimeError("project operation is outside its storage lease")
        return lease

    @staticmethod
    @contextmanager
    def _lease_context(lease: WriterLease) -> Iterator[None]:
        token = _ACTIVE_WRITER_LEASE.set(lease)
        try:
            yield
        finally:
            _ACTIVE_WRITER_LEASE.reset(token)

    @contextmanager
    def _identity_authority(self, *, exclusive: bool) -> Iterator[int]:
        require_off_loop_lock("project identity authority", ProjectRegistryError)
        lease = self._storage_lease
        with self._lease_context(lease):
            if exclusive or lease.observed_epoch >= P06_FEATURE_EPOCH:
                self._ensure_root()
            if lease.observed_epoch < P06_FEATURE_EPOCH:
                with identity_lock(), self._locked_home_mode(
                    exclusive=exclusive
                ) as root_fd:
                    yield root_fd
                return
            with identity_lock(), self._locked_home_mode(
                exclusive=exclusive
            ) as root_fd:
                yield root_fd

    @staticmethod
    def _release_lock(lock_fd: int) -> None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    def _named_lock_fd(self, project_fd: int, domain: str) -> int:
        if domain != "inbox":
            return self._open_named_lock(project_fd, f".{domain}.lock")
        try:
            os.mkdir("inbox", mode=0o700, dir_fd=project_fd)
        except FileExistsError:
            pass
        try:
            inbox_fd = os.open(
                "inbox",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=project_fd,
            )
        except OSError as exc:
            raise ProjectRegistryError("project inbox is unavailable") from exc
        try:
            if stat.S_IMODE(os.fstat(inbox_fd).st_mode) & 0o077:
                raise ProjectRegistryError("project inbox is not private")
            return self._open_named_lock(inbox_fd, ".lock")
        finally:
            os.close(inbox_fd)

    @contextmanager
    def _legacy_project_locks(
        self,
        domains: tuple[tuple[str, str], ...],
        *,
        exclusive: bool,
    ) -> Iterator[tuple[int, ...]]:
        with ExitStack() as stack:
            stack.enter_context(identity_lock())
            root_fd, _ = self._open_root()
            stack.callback(os.close, root_fd)
            lock_fd = self._open_lock(root_fd)
            self._acquire_lock(
                lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            )
            stack.callback(self._release_lock, lock_fd)
            project_fds = tuple(
                self._validated_project_fd(root_fd, project_id)
                for project_id, _domain in domains
            )
            for project_fd in project_fds:
                stack.callback(os.close, project_fd)
            yield project_fds

    @contextmanager
    def _named_project_locks(
        self,
        domains: tuple[tuple[str, str], ...],
        *,
        exclusive: bool,
    ) -> Iterator[tuple[int, ...]]:
        with ExitStack() as stack:
            root_fd, _ = self._open_root()
            stack.callback(os.close, root_fd)
            project_fds: list[int] = []
            for project_id, domain in domains:
                stack.enter_context(domain_lock(project_id, domain))
                project_fd = self._validated_project_fd(root_fd, project_id)
                project_fds.append(project_fd)
                stack.callback(os.close, project_fd)
                lock_fd = self._named_lock_fd(project_fd, domain)
                self._acquire_lock(
                    lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                )
                stack.callback(self._release_lock, lock_fd)
            yield tuple(project_fds)

    @contextmanager
    def _project_locks(
        self,
        domains: tuple[tuple[str, str], ...],
        *,
        exclusive: bool,
    ) -> Iterator[tuple[int, ...]]:
        require_off_loop_lock("project domain authority", ProjectRegistryError)
        if domains != tuple(sorted(set(domains))):
            raise AssertionError("project domains must be unique and sorted")
        lease = self._storage_lease
        with self._lease_context(lease):
            if exclusive or lease.observed_epoch >= P06_FEATURE_EPOCH:
                self._ensure_root()
            locks = (
                self._legacy_project_locks
                if lease.observed_epoch < P06_FEATURE_EPOCH
                else self._named_project_locks
            )
            with locks(domains, exclusive=exclusive) as project_fds:
                yield project_fds

    def _identity_read(self, operation: Callable[[int], _T]) -> _T:
        with self._identity_authority(exclusive=False) as root_fd:
            return operation(root_fd)

    @contextmanager
    def _identity_write(self) -> Iterator[int]:
        with self._identity_authority(exclusive=True) as root_fd:
            yield root_fd

    @contextmanager
    def _project_lock(
        self, project_id: str, domain: str, *, exclusive: bool
    ) -> Iterator[int]:
        with self._project_locks(
            ((project_id, domain),), exclusive=exclusive
        ) as project_fds:
            yield project_fds[0]

    @contextmanager
    def _locked_home_mode(self, *, exclusive: bool) -> Iterator[int]:
        for attempt in range(_READ_RETRIES):
            root_fd, _ = self._open_root()
            try:
                os.fchmod(root_fd, 0o700)
                try:
                    lock_fd = self._open_lock(root_fd)
                except FileNotFoundError as exc:
                    if attempt + 1 == _READ_RETRIES:
                        raise ProjectRegistryError("cannot open registry lock") from exc
                else:
                    try:
                        os.fchmod(lock_fd, 0o600)
                        self._acquire_lock(
                            lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                        )
                        yield root_fd
                        return
                    finally:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
            finally:
                os.close(root_fd)
            time.sleep(0.001)
        raise ProjectRegistryError("cannot open registry lock")
