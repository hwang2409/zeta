"""Find and stop a tool process tree that outlived a killed zeta.

Before a tool command spawns, zeta takes an exclusive ``flock`` on a record file
and passes that fd to the command. The parent closes its copy after spawn, so
the lock stays held exactly while some process of the tool tree is alive, also
after ``kill -9`` of zeta. Recovery probes the lock: free means nothing is left;
busy means the fd holders are our orphans, found by the file's inode.

Records live under ``$ZETA_HOME/run/tool-processes``, outside the session tree.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...core.session import env_home

_STOP_TIMEOUT_SECONDS = 5.0
# Above the shell's own redirections (the bash tool binds fd 3).
_MIN_RECORD_FD = 10


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _session_records(session_dir: Path) -> Path:
    return env_home() / "run" / "tool-processes" / _digest(str(session_dir))


def _record_path(session_dir: Path, call_id: str) -> Path:
    return _session_records(session_dir) / f"{_digest(call_id)}.lock"


def _write(fd: int, value: dict[str, object]) -> None:
    data = json.dumps(value).encode()
    os.ftruncate(fd, 0)
    os.pwrite(fd, data, 0)


class ToolProcessRecord:
    """The parent's handle on one record; pass ``fd`` to the spawned command."""

    def __init__(self, path: Path, fd: int, call_id: str) -> None:
        self.path = path
        self.fd = fd
        self._call_id = call_id

    def set_pgid(self, pgid: int) -> None:
        _write(self.fd, {"call_id": self._call_id, "pgid": pgid, "stopping": False})

    def close_parent_fd(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def remove(self) -> None:
        """Remove the record once the command was reaped or never spawned."""
        self.close_parent_fd()
        self.path.unlink(missing_ok=True)
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


def open_record(session_dir: Path, call_id: str) -> ToolProcessRecord:
    path = _record_path(session_dir, call_id)
    for attempt in range(3):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            opened = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            break
        except FileNotFoundError:
            # Another call's unlink removed the empty directory between mkdir and open.
            if attempt == 2:
                raise
    try:
        fd = fcntl.fcntl(opened, fcntl.F_DUPFD_CLOEXEC, _MIN_RECORD_FD)
    finally:
        os.close(opened)
    try:
        # Never wait: a busy lock means a live process from an earlier use of this call id.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write(fd, {"call_id": call_id, "pgid": None, "stopping": False})
    except BaseException:
        os.close(fd)
        raise
    return ToolProcessRecord(path, fd, call_id)


@dataclass(frozen=True, slots=True)
class Outcome:
    kind: Literal["none", "exited", "stopped", "unstoppable"]
    groups: tuple[int, ...] = ()
    path: Path | None = None


def _holders(fd: int, path: Path) -> list[int]:
    """Return other processes that hold an fd on the record open as ``fd``."""
    own = os.getpid()
    if os.path.isdir("/proc/self/fd"):
        target = os.fstat(fd)
        found = set()
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                names = os.listdir(f"/proc/{pid}/fd")
            except OSError:
                continue
            for name in names:
                try:
                    info = os.stat(f"/proc/{pid}/fd/{name}")
                except OSError:
                    continue
                if (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino):
                    found.add(int(pid))
                    break
        return sorted(found - {own})
    try:
        listed = subprocess.run(
            ["lsof", "-t", "--", str(path)], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({int(pid) for pid in listed.stdout.split() if pid.isdigit()} - {own})


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _signal_groups(groups: set[int], signum: int) -> None:
    for group in groups:
        try:
            os.killpg(group, signum)
        except OSError:
            pass


def stop_orphan(session_dir: Path, call_id: str) -> Outcome:
    """Stop what is left of a recorded tool call; this blocks, so run it off the event loop."""
    path = _record_path(session_dir, call_id)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return Outcome("none")
    try:
        try:
            saved = json.loads(os.pread(fd, 4096, 0))
            pgid = saved["pgid"] if isinstance(saved["pgid"], int) else None
            stopping = saved["stopping"] is True
        except (OSError, ValueError, TypeError, KeyError):
            pgid, stopping = None, False
        free = _try_lock(fd)
        # A held fd also counts, because flock on some network homes is not inherited.
        holders = _holders(fd, path)
        if free and not holders:
            return Outcome("stopped" if stopping else "exited", (pgid,) if pgid else (), path)
        own_group = os.getpgrp()
        groups = set()
        for holder in holders:
            try:
                groups.add(os.getpgid(holder))
            except OSError:
                pass
        groups.discard(own_group)
        shown = tuple(sorted(groups)) or ((pgid,) if pgid else ())
        if not groups:
            return Outcome("unstoppable", shown, path)
        _write(fd, {"call_id": call_id, "pgid": pgid, "stopping": True})
        _signal_groups(groups, signal.SIGTERM)
        time.sleep(0.1)
        _signal_groups(groups, signal.SIGKILL)
        deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS
        while True:
            free = free or _try_lock(fd)
            if free and not _holders(fd, path):
                return Outcome("stopped", shown, path)
            if time.monotonic() >= deadline:
                return Outcome("unstoppable", shown, path)
            time.sleep(0.05)
    finally:
        os.close(fd)


def sweep(session_dir: Path) -> None:
    """Remove this session's records that no live process holds any more."""
    directory = _session_records(session_dir)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return
    for name in names:
        path = directory / name
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            if _try_lock(fd):
                path.unlink(missing_ok=True)
        finally:
            os.close(fd)
    try:
        directory.rmdir()
    except OSError:
        pass
