"""Read-only liveness checks for session directory leases."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


def session_is_live(session_dir: Path) -> bool:
    """Return whether another process holds the cooperative session lease."""
    try:
        fd = os.open(session_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
