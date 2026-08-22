"""Race-resistant, descriptor-anchored access for file tools."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_MAX_ANCESTRY_DEPTH = 256
_Identity = tuple[int, int]


def _normalized_components(parts: tuple[str, ...]) -> list[str]:
    components: list[str] = []
    for part in parts:
        if part in {"", ".", os.sep}:
            continue
        if part == "..":
            if not components:
                raise ValueError("path escaped sandbox")
            components.pop()
            continue
        components.append(part)
    return components


def _anchored_components(registry: ToolRegistry, raw_path: str) -> list[str]:
    candidate = Path(raw_path)
    components = _normalized_components(candidate.parts)
    if candidate.is_absolute():
        cwd_components = _normalized_components(registry.cwd.parts)
        if components[: len(cwd_components)] != cwd_components:
            raise ValueError("path escaped sandbox")
        components = components[len(cwd_components) :]
    if not components:
        raise ValueError("path must name a file")
    return components


def _path_open_error(path: Path | str, error: OSError) -> ValueError:
    if error.errno in {errno.ELOOP, errno.EPERM}:
        return ValueError("path escaped sandbox")
    if error.errno == errno.ENOTDIR:
        return ValueError("path escaped sandbox")
    return ValueError(f"could not open path: {path}: {error}")


if sys.platform == "darwin" and hasattr(fcntl, "F_GETPATH"):

    def _path_from_fd(file_descriptor: int) -> str:
        """Return the kernel path with macOS F_GETPATH."""

        return bytes(
            fcntl.fcntl(file_descriptor, fcntl.F_GETPATH, b"\0" * 1024)
        ).split(b"\0", 1)[0].decode("utf-8")

elif sys.platform.startswith("linux") and os.path.isdir("/proc/self/fd"):

    def _path_from_fd(file_descriptor: int) -> str:
        """Return the kernel path with Linux procfs."""

        return os.readlink(f"/proc/self/fd/{file_descriptor}")

else:

    def _path_from_fd(file_descriptor: int) -> str:
        """Fail clearly when the host has no supported fd path API."""

        del file_descriptor
        raise OSError("no supported file-descriptor path API on this platform")


def _fd_identity(file_descriptor: int) -> _Identity:
    file_stat = os.fstat(file_descriptor)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise ValueError("path escaped sandbox")
    return file_stat.st_dev, file_stat.st_ino


def _verify_ancestry(file_descriptor: int, sandbox_identity: _Identity) -> None:
    """Verify that a directory descriptor has the sandbox as an ancestor."""

    current_fd = os.dup(file_descriptor)
    try:
        for _ in range(_MAX_ANCESTRY_DEPTH):
            if _fd_identity(current_fd) == sandbox_identity:
                return
            try:
                parent_fd = os.open("..", _DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                raise ValueError("path escaped sandbox") from exc
            os.close(current_fd)
            current_fd = parent_fd
        raise ValueError("path escaped sandbox")
    finally:
        os.close(current_fd)


@contextmanager
def open_parent(
    registry: ToolRegistry,
    raw_path: str,
    *,
    create_parents: bool,
) -> Iterator[tuple[list[str], int, Path]]:
    """Open and verify the target's parent directory descriptor."""

    components = _anchored_components(registry, raw_path)
    fallback_path = registry.cwd.joinpath(*components)
    sandbox_fd = registry._open_cwd()
    parent_fd = sandbox_fd
    try:
        for index, component in enumerate(components[:-1]):
            component_path = registry.cwd.joinpath(*components[: index + 1])
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                if not create_parents:
                    raise ValueError(
                        f"parent directory does not exist: {component_path}"
                    ) from exc
                try:
                    os.mkdir(component, 0o755, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                except OSError as mkdir_error:
                    raise _path_open_error(component_path, mkdir_error) from mkdir_error
                try:
                    child_fd = os.open(
                        component, _DIRECTORY_FLAGS, dir_fd=parent_fd
                    )
                except OSError as open_error:
                    raise _path_open_error(component_path, open_error) from open_error
            except OSError as open_error:
                raise _path_open_error(component_path, open_error) from open_error

            try:
                _verify_ancestry(child_fd, registry._cwd_identity)
            except (OSError, ValueError):
                os.close(child_fd)
                raise
            if parent_fd != sandbox_fd:
                os.close(parent_fd)
            parent_fd = child_fd

        _verify_ancestry(parent_fd, registry._cwd_identity)
        yield components, parent_fd, fallback_path
    finally:
        if parent_fd != sandbox_fd:
            os.close(parent_fd)
        os.close(sandbox_fd)


@contextmanager
def open_target(
    registry: ToolRegistry,
    raw_path: str,
    *,
    flags: int,
    mode: int = 0o644,
    create_parents: bool = False,
) -> Iterator[tuple[int, Path]]:
    """Yield a verified target descriptor and its kernel-resolved path."""

    with open_parent(
        registry, raw_path, create_parents=create_parents
    ) as (components, parent_fd, fallback_path):
        parent_identity = _fd_identity(parent_fd)
        try:
            target_fd = os.open(
                components[-1], flags, mode, dir_fd=parent_fd
            )
        except FileExistsError:
            raise
        except OSError as exc:
            raise _path_open_error(fallback_path, exc) from exc

        try:
            if _fd_identity(parent_fd) != parent_identity:
                raise ValueError("path escaped sandbox")
            _verify_ancestry(parent_fd, registry._cwd_identity)
            target_stat = os.fstat(target_fd)
            if not stat.S_ISREG(target_stat.st_mode):
                raise ValueError(f"not a file: {fallback_path}")
            if flags & (os.O_WRONLY | os.O_RDWR) and target_stat.st_nlink > 1:
                raise ValueError(
                    "target has multiple hard links; refuse for sandbox integrity"
                )
            try:
                resolved_path = Path(_path_from_fd(target_fd))
            except OSError as exc:
                raise ValueError(
                    f"could not resolve path: {fallback_path}: {exc}"
                ) from exc
            yield target_fd, resolved_path
        finally:
            try:
                os.close(target_fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
