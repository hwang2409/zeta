"""Durable retention pin sidecars and their reference transaction."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ..core.session_files import (
    SessionError,
    atomic_publish_file,
    open_session_file,
    reference_publication,
    session_directory,
    unlink_session_file,
)
from ..storage_format import StorageFeature, WriterLease, feature_enabled

RETENTION_PIN_FILENAME = ".retention-pin.json"
_MAX_PIN_BYTES = 256
_PIN_DATA = b'{"pinned":true,"schema_version":1}\n'


def is_retention_pinned(directory_fd: int, *, scan=None) -> bool:
    """Return pin state, treating every unsafe present sidecar as pinned."""
    try:
        fd = open_session_file(directory_fd, RETENTION_PIN_FILENAME, os.O_RDONLY)
    except FileNotFoundError:
        try:
            charge = scan.charge_entry if scan is not None else (lambda _size: None)
            with _PinEntryScan(directory_fd, charge) as entries:
                return any(
                    entry.name.startswith(f".{RETENTION_PIN_FILENAME}.")
                    and entry.name.endswith(".tmp")
                    for entry in entries
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


def set_retention_pin(
    home: str | Path, session_id: str, lease: WriterLease, *, pinned: bool
) -> str:
    """Pin or unpin one root in one off-loop reference transaction."""
    if not feature_enabled(lease.observed_epoch, StorageFeature.SESSION_RETENTION):
        return "retention pins unavailable: upgrade the home before pinning"
    with reference_publication(
        Path(home), lease, installed_ids=(session_id,)
    ), session_directory(
        Path(home) / "sessions", session_id, lifetime=True
    ) as (_root_fd, session_fd):
        if pinned:
            write_retention_pin(session_fd)
            return "session pinned"
        remove_retention_pin(session_fd)
        return "session unpinned"


class _PinEntryScan:
    """Enumerate pin sidecars through a no-follow directory descriptor."""

    def __init__(self, directory_fd: int, charge):
        self._fd = os.open(
            ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
        )
        self._entries = os.scandir(self._fd)
        self._charge = charge
        self._pending = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._pending is None:
            entry = next(self._entries)
            info = entry.stat(follow_symlinks=False)
            self._pending = (entry, info.st_size if stat.S_ISREG(info.st_mode) else 0)
        entry, size = self._pending
        self._charge(size)
        self._pending = None
        return entry

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._entries.close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
