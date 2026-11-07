"""Lossless publication for account-owned JSON mappings."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from typing import Any

from ..core.session_files import open_session_file
from ..storage_format import mapping_mutations, open_writer


def atomic_publish(directory_fd: int, name: str, data: bytes) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(
        directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        published = open_session_file(directory_fd, name, os.O_RDONLY)
        try:
            os.fchmod(published, 0o600)
        finally:
            os.close(published)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def publish_mapping(
    home: str | PathLike[str],
    directory_fd: int,
    name: str,
    value: Mapping[str, Any],
    publish: Callable[[int, str, bytes], None],
    *,
    delete: Iterable[str] = (),
) -> None:
    """Patch owned fields while retaining fields written by newer releases."""

    with open_writer(home) as lease:
        lease.publish_json(
            directory_fd,
            name,
            mapping_mutations(value, delete=delete),
            publish,
        )
