"""Read-only liveness checks for session and process leases."""

from __future__ import annotations

import fcntl
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """A process ID paired with its kernel-reported start time."""

    pid: int
    started: str


def _process_started(pid: int) -> str | None:
    """Return a stable process start value on supported POSIX systems."""

    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    started = completed.stdout.strip()
    return started if completed.returncode == 0 and started else None


_CURRENT_PROCESS = ProcessIdentity(
    os.getpid(), _process_started(os.getpid()) or "unavailable"
)


def current_process_identity() -> ProcessIdentity:
    """Return this process's stable identity, including after a fork."""

    global _CURRENT_PROCESS
    pid = os.getpid()
    if _CURRENT_PROCESS.pid != pid:
        _CURRENT_PROCESS = ProcessIdentity(pid, _process_started(pid) or "unavailable")
    return _CURRENT_PROCESS


def process_is_live(identity: ProcessIdentity) -> bool:
    """Return whether the exact process, excluding PID reuse, is alive."""

    if identity == current_process_identity():
        return True
    return bool(identity.started) and _process_started(identity.pid) == identity.started


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
