"""Durable retention-owned session sidecars."""

from __future__ import annotations

import json
import os
import stat

from .core.session_files import (
    SessionError,
    atomic_publish_file,
    open_session_file,
    unlink_session_file,
)

RETENTION_PIN_FILENAME = ".retention-pin.json"
_MAX_PIN_BYTES = 256
_PIN_DATA = b'{"pinned":true,"schema_version":1}\n'


def is_retention_pinned(directory_fd: int) -> bool:
    """Return pin state, treating every unsafe present sidecar as pinned."""
    try:
        fd = open_session_file(directory_fd, RETENTION_PIN_FILENAME, os.O_RDONLY)
    except FileNotFoundError:
        try:
            return any(
                name.startswith(f".{RETENTION_PIN_FILENAME}.")
                and name.endswith(".tmp")
                for name in os.listdir(directory_fd)
            )
        except OSError:
            return True
    except (OSError, SessionError):
        return True
    try:
        info = os.fstat(fd)
        if (
            info.st_size > _MAX_PIN_BYTES
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            return True
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            try:
                json.loads(stream.read(_MAX_PIN_BYTES + 1))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                return True
        return True
    except (OSError, ValueError):
        return True
    finally:
        if fd != -1:
            os.close(fd)


def write_retention_pin(directory_fd: int) -> None:
    """Atomically publish the v1 pin sidecar."""
    atomic_publish_file(directory_fd, RETENTION_PIN_FILENAME, _PIN_DATA)


def remove_retention_pin(directory_fd: int) -> None:
    """Remove the pin sidecar; an absent sidecar is already unpinned."""
    unlink_session_file(directory_fd, RETENTION_PIN_FILENAME)
