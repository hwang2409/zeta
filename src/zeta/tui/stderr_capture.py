"""Capture incidental TUI stderr without writing it to the terminal."""

from __future__ import annotations

import asyncio
import fcntl
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout

from ..core.session import env_home

_TUI_STDERR_MAX_BYTES = 50 * 1024 * 1024
_SETUP = False
_LOCK_FD: int | None = None
_TERMINAL_FD: int | None = None


class CaptureInstallError(RuntimeError):
    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


def _log_path(directory: Path) -> Path:
    return directory / f"{time.time_ns()}-{os.getpid()}.log"


def _retain_logs(directory: Path) -> None:
    try:
        files = sorted(directory.glob("*.log"), key=lambda path: path.stat().st_mtime_ns)
    except OSError:
        return
    retained: list[tuple[Path, int]] = []
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                size = os.fstat(fd).st_size
                retained.append((path, size))
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except OSError:
            continue
    total = sum(size for _, size in retained)
    for path, size in retained:
        if total <= _TUI_STDERR_MAX_BYTES:
            break
        try:
            path.unlink()
            total -= size
        except OSError:
            pass


def _open_capture(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(
        temp_path,
        os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.rename(temp_path, path)
        _retain_logs(path.parent)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        os.close(fd)
        raise
    return fd


def _restore() -> None:
    global _SETUP, _LOCK_FD
    if _TERMINAL_FD is not None:
        try:
            os.dup2(_TERMINAL_FD, 2)
        except OSError:
            pass
    if _LOCK_FD is not None:
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            os.close(_LOCK_FD)
        except OSError:
            pass
        _LOCK_FD = None
    _SETUP = False


def write_terminal(message: str) -> None:
    fd = _TERMINAL_FD
    if fd is None:
        print(message, file=sys.stderr, flush=True)
        return
    try:
        os.write(fd, (message + "\n").encode())
    except OSError:
        pass


def restore_stderr_for_cli() -> None:
    _restore()


def run_tui_session(app: Any, args: Any, run_tui: Any) -> tuple[bool, bool]:
    with patch_stdout(raw=True):
        # prompt_toolkit temporarily proxies stderr to the terminal; incidental
        # diagnostics must keep using the already redirected descriptor.
        sys.stderr = sys.__stderr__
        return asyncio.run(run_tui(app, args))


def run_tui_lifecycle(args: Any, parser: Any, lifecycle: Any) -> tuple[int, str | None]:
    if sys_isatty():
        try:
            _setup()
        except CaptureInstallError as exc:
            print(
                f"zeta: could not start TUI; stderr log unavailable: {exc.path}",
                file=sys.stderr,
            )
            return 1, None
        return lifecycle(args, parser)
    return lifecycle(args, parser)


def sys_isatty() -> bool:
    return os.isatty(0) and os.isatty(1)


def _setup() -> None:
    global _SETUP, _TERMINAL_FD, _LOCK_FD
    if _SETUP:
        return
    primary = env_home() / "logs" / "tui-stderr"
    path = _log_path(primary)
    terminal_fd: int | None = None
    log_fd: int | None = None
    try:
        terminal_fd = os.dup(2)
        try:
            log_fd = _open_capture(path)
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "zeta-tui-stderr"
            path = _log_path(fallback)
            log_fd = _open_capture(path)
        os.dup2(log_fd, 2)
        _TERMINAL_FD = terminal_fd
        _LOCK_FD = log_fd
        _SETUP = True
    except BaseException as exc:
        if log_fd is not None:
            os.close(log_fd)
        if terminal_fd is not None:
            os.close(terminal_fd)
        raise CaptureInstallError(path) from exc


@contextmanager
def stderr_to_log() -> Iterator[None]:
    _setup()
    yield
