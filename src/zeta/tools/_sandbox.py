"""Best-effort anchored file access shared by file tools."""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


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


def _path_for_components(registry: ToolRegistry, components: list[str]) -> Path:
    return registry.cwd.joinpath(*components)


def _path_open_error(path: Path, error: OSError) -> ValueError:
    if error.errno in {errno.ELOOP, errno.EPERM}:
        return ValueError("path escaped sandbox")
    return ValueError(f"could not open path: {path}: {error}")


def _path_from_fd(file_descriptor: int) -> str:
    encoded_path = fcntl.fcntl(file_descriptor, fcntl.F_GETPATH, b"\0" * 1024)
    return bytes(encoded_path).split(b"\0", 1)[0].decode("utf-8")


def open_anchored(
    registry: ToolRegistry,
    raw_path: str,
    *,
    create_parents: bool,
    open_flags: int,
    create_file: bool,
    truncate_existing: bool,
    path_from_fd: Callable[[int], str] = _path_from_fd,
) -> tuple[int, bool, str]:
    components = _anchored_components(registry, raw_path)
    fallback_path = _path_for_components(registry, components)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    sandbox_fd = registry._open_cwd()

    parent_fd = sandbox_fd
    file_descriptor: int | None = None
    try:
        for index, component in enumerate(components[:-1]):
            component_path = _path_for_components(
                registry, components[: index + 1]
            )
            try:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
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
                    child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                except OSError as open_error:
                    raise _path_open_error(component_path, open_error) from open_error
            except OSError as open_error:
                raise _path_open_error(component_path, open_error) from open_error

            if parent_fd != sandbox_fd:
                os.close(parent_fd)
            parent_fd = child_fd

        basename = components[-1]
        try:
            if create_file:
                file_descriptor = os.open(
                    basename,
                    open_flags | os.O_CREAT | os.O_EXCL,
                    0o666,
                    dir_fd=parent_fd,
                )
                was_created = True
            else:
                file_descriptor = os.open(
                    basename,
                    open_flags,
                    dir_fd=parent_fd,
                )
                was_created = False
        except FileExistsError:
            try:
                file_descriptor = os.open(
                    basename,
                    open_flags,
                    dir_fd=parent_fd,
                )
            except OSError as open_error:
                raise _path_open_error(fallback_path, open_error) from open_error
            was_created = False
        except OSError as open_error:
            raise _path_open_error(fallback_path, open_error) from open_error

        actual_path = path_from_fd(file_descriptor)
        if truncate_existing and not was_created:
            os.ftruncate(file_descriptor, 0)
        result = file_descriptor, was_created, actual_path
        file_descriptor = None
        return result
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_fd != sandbox_fd:
            os.close(parent_fd)
        os.close(sandbox_fd)
