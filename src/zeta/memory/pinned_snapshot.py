"""Pin a project-memory generation under its lock and read it after release."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from typing import Any

from ..project_errors import ProjectRegistryError
from ..project_schema import MAX_MEMORY_FILE_SIZE
from ..storage_format import reject_unknown_keys
from .entry_views import GENERATED_MIRROR_HEADER


@dataclass(frozen=True, slots=True)
class PinnedMemoryFile:
    name: str
    fd: int | None
    digest: str | None


def pin_snapshot_files(
    registry: Any, directory_fd: int, names: tuple[str, ...]
) -> list[PinnedMemoryFile]:
    """Open one memory generation's files while the memory lock is held."""
    registry._require_format_one(directory_fd)
    pointer = registry._pointer(directory_fd)
    if pointer is None:
        memory_fd = registry._memory_fd(directory_fd)
        try:
            pinned: list[PinnedMemoryFile] = []
            for name in names:
                try:
                    fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=memory_fd,
                    )
                except FileNotFoundError:
                    fd = None
                pinned.append(PinnedMemoryFile(name, fd, None))
            return pinned
        except BaseException:
            for item in pinned:
                if item.fd is not None:
                    os.close(item.fd)
            raise
        finally:
            os.close(memory_fd)

    root, blobs, versions = registry._version_handles(directory_fd, create=False)
    pinned = []
    try:
        manifest = registry._manifest(versions, str(pointer["current"]))
        files = manifest.get("snapshot")
        if not isinstance(files, dict):
            raise ProjectRegistryError("project memory version is malformed")
        try:
            reject_unknown_keys(files, "legacy_memory_snapshot")
        except ValueError as exc:
            raise ProjectRegistryError(
                "project memory version is malformed"
            ) from exc
        if set(files) != set(names):
            raise ProjectRegistryError("project memory version is malformed")
        for name in names:
            digest = files.get(name)
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", digest
            ):
                raise ProjectRegistryError("project memory version is malformed")
            try:
                fd = os.open(
                    digest,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=blobs,
                )
            except OSError as exc:
                raise ProjectRegistryError(
                    "project memory snapshot is incomplete"
                ) from exc
            pinned.append(PinnedMemoryFile(name, fd, digest))
        return pinned
    except BaseException:
        for item in pinned:
            if item.fd is not None:
                os.close(item.fd)
        raise
    finally:
        os.close(versions)
        os.close(blobs)
        os.close(root)


def read_pinned_snapshot(files: list[PinnedMemoryFile]) -> dict[str, str]:
    """Read and close pinned files without the memory lock."""
    contents: dict[str, str] = {}
    try:
        for item in files:
            if item.fd is None:
                contents[item.name] = ""
                continue
            info = os.fstat(item.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size > MAX_MEMORY_FILE_SIZE
            ):
                raise ProjectRegistryError("project memory snapshot is unsafe")
            payload = bytearray()
            while len(payload) <= info.st_size:
                chunk = os.read(
                    item.fd,
                    min(64 * 1024, info.st_size + 1 - len(payload)),
                )
                if not chunk:
                    break
                payload.extend(chunk)
            raw = bytes(payload)
            if len(raw) != info.st_size:
                raise ProjectRegistryError("project memory snapshot is unreadable")
            if item.digest is not None:
                if hashlib.sha256(raw).hexdigest() != item.digest:
                    raise ProjectRegistryError("project memory snapshot is corrupt")
            elif raw.startswith(GENERATED_MIRROR_HEADER.encode()):
                raise ProjectRegistryError(
                    "project memory pointer is missing; generated memory mirror "
                    "cannot be used as authoritative memory"
                )
            try:
                contents[item.name] = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProjectRegistryError(
                    "project memory snapshot is unreadable"
                ) from exc
    except OSError as exc:
        raise ProjectRegistryError("project memory snapshot is unreadable") from exc
    finally:
        for item in files:
            if item.fd is not None:
                os.close(item.fd)
    return contents
