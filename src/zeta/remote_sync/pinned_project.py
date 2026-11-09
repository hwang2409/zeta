"""Pin a local project payload under its memory lock and copy it after release."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from ..project_memory_history import MAX_MEMORY_MIRROR_FILE_SIZE, PROJECT_MEMORY_FILES
from .errors import RemoteSyncError
from .project_publish import ProjectPublicationError, synchronized_payload_paths


def pin_project_snapshot(source: Path) -> list[tuple[Path, int]]:
    try:
        payload = synchronized_payload_paths(source)
    except ProjectPublicationError as exc:
        raise RemoteSyncError(str(exc)) from exc
    if source / "project.json" not in payload:
        raise RemoteSyncError("project record is missing or unsafe")
    pinned: list[tuple[Path, int]] = []
    try:
        for path in payload:
            relative = path.relative_to(source)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(fd)
                raise RemoteSyncError("project snapshot file is unsafe")
            if (
                relative.parent == Path("memory")
                and relative.name in PROJECT_MEMORY_FILES
                and info.st_size > MAX_MEMORY_MIRROR_FILE_SIZE
            ):
                os.close(fd)
                raise RemoteSyncError(f"memory file {relative.name} is too large")
            pinned.append((relative, fd))
        return pinned
    except BaseException:
        for _relative, fd in pinned:
            os.close(fd)
        raise


def copy_pinned_project_snapshot(
    pinned: list[tuple[Path, int]], destination: Path
) -> None:
    try:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        directories = {
            parent
            for relative, _fd in pinned
            for parent in relative.parents
            if parent != Path(".")
        }
        for relative in sorted(
            directories, key=lambda item: (len(item.parts), item.parts)
        ):
            directory = destination / relative
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
        memory = destination / "memory"
        memory.mkdir(mode=0o700, exist_ok=True)
        memory.chmod(0o700)
        for relative, fd in pinned:
            target = destination / relative
            with os.fdopen(os.dup(fd), "rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o600)
    finally:
        for _relative, fd in pinned:
            os.close(fd)
