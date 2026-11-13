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
from ..storage_format import StorageFeature, WriterLease, feature_enabled

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
    def _open_lock(root_fd: int, *, create: bool) -> int | None:
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
        try:
            lock_fd = os.open(".lock", flags | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
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

    @staticmethod
    def _acquire_lock(lock_fd: int, mode: int, *, wait: bool = True) -> None:
        deadline = (
            time.monotonic() + _PROJECT_REGISTRY_LOCK_TIMEOUT_SECONDS
            if wait
            else time.monotonic()
        )
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
        wait: bool = True,
    ) -> Iterator[tuple[int, ...]]:
        with ExitStack() as stack:
            stack.enter_context(identity_lock())
            root_fd, _ = self._open_root()
            stack.callback(os.close, root_fd)
            lock_fd = self._open_lock(root_fd, create=True)
            assert lock_fd is not None
            self._acquire_lock(
                lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH, wait=wait
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
        wait: bool = True,
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
                    lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH, wait=wait
                )
                stack.callback(self._release_lock, lock_fd)
            yield tuple(project_fds)

    @contextmanager
    def _project_locks(
        self,
        domains: tuple[tuple[str, str], ...],
        *,
        exclusive: bool,
        wait: bool = True,
    ) -> Iterator[tuple[int, ...]]:
        require_off_loop_lock("project domain authority", ProjectRegistryError)
        if domains != tuple(sorted(set(domains))):
            raise AssertionError("project domains must be unique and sorted")
        lease = self._storage_lease
        with self._lease_context(lease):
            if exclusive:
                self._ensure_root()
            locks = (
                self._legacy_project_locks
                if not feature_enabled(lease.observed_epoch, StorageFeature.PROJECT_LOCKS)
                else self._named_project_locks
            )
            with locks(domains, exclusive=exclusive, wait=wait) as project_fds:
                yield project_fds

    def _identity_read(self, operation: Callable[[int], _T]) -> _T:
        """Read one identity snapshot without creating the root or its lock."""
        require_off_loop_lock("project identity authority", ProjectRegistryError)
        with self._lease_context(self._storage_lease), identity_lock():
            for attempt in range(_READ_RETRIES):
                root_fd, root_info = self._open_root()
                try:
                    lock_fd = self._open_lock(root_fd, create=False)
                    if lock_fd is not None:
                        try:
                            self._acquire_lock(lock_fd, fcntl.LOCK_SH)
                            return operation(root_fd)
                        finally:
                            self._release_lock(lock_fd)
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
                time.sleep(0.001)
        raise ProjectRegistryError("projects registry changed during read")

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

    @contextmanager
    def _identity_write(self) -> Iterator[int]:
        require_off_loop_lock("project identity authority", ProjectRegistryError)
        with self._lease_context(self._storage_lease):
            self._ensure_root()
            with identity_lock(), self._locked_home() as root_fd:
                yield root_fd

    @contextmanager
    def _project_lock(
        self, project_id: str, domain: str, *, exclusive: bool, wait: bool = True
    ) -> Iterator[int]:
        with self._project_locks(
            ((project_id, domain),), exclusive=exclusive, wait=wait
        ) as project_fds:
            yield project_fds[0]

    @contextmanager
    def _locked_home(self) -> Iterator[int]:
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
                        self._acquire_lock(lock_fd, fcntl.LOCK_EX)
                        yield root_fd
                        return
                    finally:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
            finally:
                os.close(root_fd)
            time.sleep(0.001)
        raise ProjectRegistryError("cannot open registry lock")
