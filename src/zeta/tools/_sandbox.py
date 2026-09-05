"""One access policy shared across every file tool.

The policy has one owner: :class:`SandboxPolicy`. Each file tool calls
``policy.resolve`` to expand ``~`` and classify the target, then routes
through :func:`open_target`, which picks the descriptor-anchored hardened
walk for inside-cwd targets and a direct open for outside targets. The
outside path matches bash's effective scope so the model does not have to
recover through heredoc-bash after a refusal.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_MAX_ANCESTRY_DEPTH = 256
_Identity = tuple[int, int]


def expand_user_path(raw_path: str) -> str:
    """Expand a leading ``~`` or ``~user``; raise if the user is unknown."""

    if not raw_path.startswith("~"):
        return raw_path
    expanded = os.path.expanduser(raw_path)
    if expanded == raw_path or expanded.startswith("~"):
        raise ValueError(
            f"could not expand user in path {raw_path!r}; use an absolute path"
        )
    return expanded


@dataclass(frozen=True, slots=True)
class ResolvedPath:
    """A file-tool path resolved by the shared sandbox policy."""

    absolute: Path
    in_cwd: bool


class SandboxPolicy:
    """Single owner of the file-tool access decision.

    Every file tool consults one policy so no per-tool divergence remains.
    ``resolve`` expands ``~`` per call and classifies the target as inside
    or outside the session cwd. Inside targets go through the hardened
    walk in :func:`open_target`; outside targets use a direct open.
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def resolve(self, raw_path: object) -> ResolvedPath:
        if type(raw_path) is not str or not raw_path:
            raise ValueError("path must be a nonempty string")
        expanded = expand_user_path(raw_path)
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        absolute = Path(os.path.abspath(candidate))
        try:
            absolute.relative_to(self.cwd)
        except ValueError:
            return ResolvedPath(absolute=absolute, in_cwd=False)
        return ResolvedPath(absolute=absolute, in_cwd=True)

    def describe_roots(self) -> str:
        """Return the allowed workspace root the model should re-target to."""

        return f"session cwd {self.cwd}"


def _escape_error(policy: SandboxPolicy) -> ValueError:
    return ValueError(
        f"path escaped {policy.describe_roots()}; retry with an absolute path"
    )


def _cwd_relative_components(cwd: Path, absolute: Path) -> list[str]:
    relative = absolute.relative_to(cwd)
    components = [part for part in relative.parts if part not in {"", ".", os.sep}]
    if not components:
        raise ValueError("path must name a file")
    return components


def _path_open_error(
    path: Path | str, error: OSError, policy: SandboxPolicy
) -> ValueError:
    if error.errno in {errno.ELOOP, errno.EPERM, errno.ENOTDIR}:
        return _escape_error(policy)
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


def _fd_identity(file_descriptor: int, policy: SandboxPolicy) -> _Identity:
    file_stat = os.fstat(file_descriptor)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise _escape_error(policy)
    return file_stat.st_dev, file_stat.st_ino


def _verify_ancestry(
    file_descriptor: int,
    sandbox_identity: _Identity,
    policy: SandboxPolicy,
) -> None:
    """Verify that a directory descriptor has the sandbox as an ancestor."""

    current_fd = os.dup(file_descriptor)
    try:
        for _ in range(_MAX_ANCESTRY_DEPTH):
            if _fd_identity(current_fd, policy) == sandbox_identity:
                return
            try:
                parent_fd = os.open("..", _DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                raise _escape_error(policy) from exc
            os.close(current_fd)
            current_fd = parent_fd
        raise _escape_error(policy)
    finally:
        os.close(current_fd)


@contextmanager
def _open_parent_in_cwd(
    registry: ToolRegistry,
    components: list[str],
    absolute: Path,
    *,
    create_parents: bool,
) -> Iterator[tuple[int, Path]]:
    policy = registry.policy
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
                    raise _path_open_error(
                        component_path, mkdir_error, policy
                    ) from mkdir_error
                try:
                    child_fd = os.open(
                        component, _DIRECTORY_FLAGS, dir_fd=parent_fd
                    )
                except OSError as open_error:
                    raise _path_open_error(
                        component_path, open_error, policy
                    ) from open_error
            except OSError as open_error:
                raise _path_open_error(
                    component_path, open_error, policy
                ) from open_error

            try:
                _verify_ancestry(child_fd, registry._cwd_identity, policy)
            except (OSError, ValueError):
                os.close(child_fd)
                raise
            if parent_fd != sandbox_fd:
                os.close(parent_fd)
            parent_fd = child_fd

        _verify_ancestry(parent_fd, registry._cwd_identity, policy)
        yield parent_fd, absolute
    finally:
        if parent_fd != sandbox_fd:
            os.close(parent_fd)
        os.close(sandbox_fd)


@contextmanager
def open_parent(
    registry: ToolRegistry,
    raw_path: str,
    *,
    create_parents: bool,
) -> Iterator[tuple[list[str], int, Path]]:
    """Open the target's parent directory descriptor for an inside-cwd path."""

    resolved = registry.policy.resolve(raw_path)
    if not resolved.in_cwd:
        raise _escape_error(registry.policy)
    components = _cwd_relative_components(registry.cwd, resolved.absolute)
    with _open_parent_in_cwd(
        registry,
        components,
        resolved.absolute,
        create_parents=create_parents,
    ) as (parent_fd, absolute):
        yield components, parent_fd, absolute


@contextmanager
def _open_target_in_cwd(
    registry: ToolRegistry,
    resolved: ResolvedPath,
    *,
    flags: int,
    mode: int,
    create_parents: bool,
) -> Iterator[tuple[int, Path]]:
    policy = registry.policy
    components = _cwd_relative_components(registry.cwd, resolved.absolute)
    with _open_parent_in_cwd(
        registry,
        components,
        resolved.absolute,
        create_parents=create_parents,
    ) as (parent_fd, fallback_path):
        parent_identity = _fd_identity(parent_fd, policy)
        try:
            target_fd = os.open(components[-1], flags, mode, dir_fd=parent_fd)
        except FileExistsError:
            raise
        except OSError as exc:
            raise _path_open_error(fallback_path, exc, policy) from exc

        try:
            if _fd_identity(parent_fd, policy) != parent_identity:
                raise _escape_error(policy)
            _verify_ancestry(parent_fd, registry._cwd_identity, policy)
            target_stat = os.fstat(target_fd)
            if not stat.S_ISREG(target_stat.st_mode):
                if stat.S_ISDIR(target_stat.st_mode):
                    raise ValueError(
                        f"{fallback_path} is a directory; "
                        "use bash (e.g. `ls`) to list its contents"
                    )
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


@contextmanager
def _open_target_outside(
    absolute: Path,
    *,
    flags: int,
    mode: int,
    create_parents: bool,
) -> Iterator[tuple[int, Path]]:
    if create_parents and (flags & os.O_CREAT):
        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"could not create parent directory: {absolute.parent}: {exc}"
            ) from exc
    open_flags = flags & ~os.O_NOFOLLOW
    try:
        target_fd = os.open(absolute, open_flags, mode)
    except FileExistsError:
        raise
    except FileNotFoundError as exc:
        if flags & os.O_CREAT:
            raise ValueError(
                f"parent directory does not exist: {absolute.parent}"
            ) from exc
        raise ValueError(f"could not open path: {absolute}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not open path: {absolute}: {exc}") from exc

    try:
        target_stat = os.fstat(target_fd)
        if not stat.S_ISREG(target_stat.st_mode):
            if stat.S_ISDIR(target_stat.st_mode):
                raise ValueError(
                    f"{absolute} is a directory; "
                    "use bash (e.g. `ls`) to list its contents"
                )
            raise ValueError(f"not a file: {absolute}")
        if flags & (os.O_WRONLY | os.O_RDWR) and target_stat.st_nlink > 1:
            raise ValueError(
                "target has multiple hard links; refuse for sandbox integrity"
            )
        yield target_fd, absolute
    finally:
        try:
            os.close(target_fd)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise


@contextmanager
def open_target(
    registry: ToolRegistry,
    raw_path: str,
    *,
    flags: int,
    mode: int = 0o644,
    create_parents: bool = False,
) -> Iterator[tuple[int, Path]]:
    """Yield a verified target descriptor and its kernel-resolved path.

    Inside-cwd targets go through the descriptor-anchored hardened walk;
    outside targets open directly, matching bash's effective scope. In
    both cases the yielded descriptor is the only race-safe handle;
    callers must not open the target by path a second time.
    """

    resolved = registry.policy.resolve(raw_path)
    if resolved.in_cwd:
        with _open_target_in_cwd(
            registry,
            resolved,
            flags=flags,
            mode=mode,
            create_parents=create_parents,
        ) as target_info:
            yield target_info
    else:
        with _open_target_outside(
            resolved.absolute,
            flags=flags,
            mode=mode,
            create_parents=create_parents,
        ) as target_info:
            yield target_info
