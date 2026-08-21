"""The built-in file writing tool."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
from pathlib import Path
from typing import NotRequired, TypedDict

from ..types import StructuredToolResult
from .registry import AbortSignal, ToolRegistry, _success_result, text_block


class WriteArguments(TypedDict):
    path: str
    content: str
    create_parents: NotRequired[bool]


class WriteStructuredContent(TypedDict):
    path: str
    bytes_written: int
    sha256: str
    was_created: bool
    was_overwritten: bool


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


def _open_anchored(
    registry: ToolRegistry,
    raw_path: str,
    create_parents: bool,
) -> tuple[int, bool, str]:
    components = _anchored_components(registry, raw_path)
    fallback_path = _path_for_components(registry, components)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    sandbox_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    try:
        sandbox_fd = os.open(registry.cwd, sandbox_flags)
    except OSError as exc:
        raise ValueError(f"could not open session cwd: {registry.cwd}: {exc}") from exc

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
            file_descriptor = os.open(
                basename,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o666,
                dir_fd=parent_fd,
            )
            was_created = True
        except FileExistsError:
            try:
                file_descriptor = os.open(
                    basename,
                    os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
            except OSError as open_error:
                raise _path_open_error(fallback_path, open_error) from open_error
            was_created = False
        except OSError as open_error:
            raise _path_open_error(fallback_path, open_error) from open_error

        actual_path = _path_from_fd(file_descriptor)
        result = file_descriptor, was_created, actual_path
        file_descriptor = None
        return result
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_fd != sandbox_fd:
            os.close(parent_fd)
        os.close(sandbox_fd)


async def _write(
    registry: ToolRegistry,
    arguments: WriteArguments,
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    try:
        encoded_content = arguments["content"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("content must be valid UTF-8") from exc

    file_descriptor, was_created, path = _open_anchored(
        registry,
        arguments["path"],
        arguments.get("create_parents", False),
    )

    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded_content)
    except OSError as exc:
        raise ValueError(f"could not write file: {path}: {exc}") from exc

    structured_content: WriteStructuredContent = {
        "path": str(path),
        "bytes_written": len(encoded_content),
        "sha256": hashlib.sha256(encoded_content).hexdigest(),
        "was_created": was_created,
        "was_overwritten": not was_created,
    }
    message = f"wrote {len(encoded_content)} bytes to {path}"
    return _success_result(
        text_block(message),
        structured_content=structured_content,
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        "write",
        lambda arguments, abort_signal: _write(registry, arguments, abort_signal),
        description="Write UTF-8 text to a file. Relative paths use the session cwd.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
                "create_parents": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )
