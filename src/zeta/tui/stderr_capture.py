"""Capture incidental TUI stderr without writing it to the terminal."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout

from ..core.session import env_home

_TUI_STDERR_LOG = "tui-stderr.log"
_TUI_STDERR_MAX_BYTES = 5 * 1024 * 1024
_FLUSH_TIMEOUT = 2.0
_QUEUE: queue.Queue[bytes] = queue.Queue(maxsize=32)
_LOCK = threading.Lock()
_SETUP = False
_TERMINAL_FD: int | None = None
_PREVIOUS_STDERR: Any = None
_WRITER_STARTED = False
_PREVIOUS_LAST_RESORT: Any = None
_PREVIOUS_WARNING = None
_PREVIOUS_THREAD_HOOK = None
_PREVIOUS_UNRAISABLE_HOOK = None
_CAPTURE_STREAM: Any = None
_READER: threading.Thread | None = None
_DROPPED_ON_SHUTDOWN = 0
_PERSIST_PROCESS = "pytest" not in sys.modules

def _open_log(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() and path.stat().st_size > _TUI_STDERR_MAX_BYTES:
        rotated = path.with_name(path.name + ".1")
        rotated.unlink(missing_ok=True)
        path.replace(rotated)
        os.chmod(rotated, 0o600)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd

def _writer() -> None:
    path = env_home() / "logs" / _TUI_STDERR_LOG
    while True:
        data = _QUEUE.get()
        try:
            fd = _open_log(path)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001, S110 - diagnostics must never affect the TUI
            pass
        finally:
            _QUEUE.task_done()

def _reader(read_fd: int) -> None:
    while True:
        try:
            data = os.read(read_fd, 65536)
        except OSError:
            return
        if not data:
            return
        try:
            _QUEUE.put_nowait(data)
        except queue.Full:
            pass

class _CaptureStream:
    def write(self, text: str) -> int:
        try:
            os.write(2, text.encode())
            return len(text)
        except OSError:
            return len(text)

    def flush(self) -> None:
        return

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return 2


def _restore() -> None:
    if _TERMINAL_FD is not None:
        try:
            os.dup2(_TERMINAL_FD, 2)
        except OSError:
            pass
    if _READER is not None:
        _READER.join(timeout=2)
    if _PREVIOUS_STDERR is not None:
        sys.stderr = _PREVIOUS_STDERR
        sys.__stderr__ = _PREVIOUS_STDERR
    if _PREVIOUS_LAST_RESORT is not None:
        logging.lastResort = _PREVIOUS_LAST_RESORT
    if _PREVIOUS_WARNING is not None:
        warnings.showwarning = _PREVIOUS_WARNING
    if _PREVIOUS_THREAD_HOOK is not None:
        threading.excepthook = _PREVIOUS_THREAD_HOOK
    if _PREVIOUS_UNRAISABLE_HOOK is not None:
        sys.unraisablehook = _PREVIOUS_UNRAISABLE_HOOK

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
        if _CAPTURE_STREAM is not None:
            sys.stderr = _CAPTURE_STREAM
        return asyncio.run(run_tui(app, args))


def run_tui_lifecycle(args: Any, parser: Any, lifecycle: Any) -> tuple[int, str | None]:
    if sys.stdin.isatty() and sys.stdout.isatty():
        with stderr_to_log():
            return lifecycle(args, parser)
    return lifecycle(args, parser)

def _setup() -> None:
    global _SETUP, _TERMINAL_FD, _PREVIOUS_STDERR, _WRITER_STARTED, _READER, _PREVIOUS_LAST_RESORT, _PREVIOUS_WARNING, _PREVIOUS_THREAD_HOOK, _PREVIOUS_UNRAISABLE_HOOK, _CAPTURE_STREAM
    with _LOCK:
        if _SETUP:
            return
        _PREVIOUS_STDERR = sys.stderr
        _TERMINAL_FD = os.dup(2)
        pipe_read, pipe_write = os.pipe()
        os.dup2(pipe_write, 2)
        try:
            os.dup2(pipe_write, _PREVIOUS_STDERR.fileno())
        except (AttributeError, OSError):
            pass
        os.close(pipe_write)
        if not _WRITER_STARTED:
            threading.Thread(target=_writer, name="zeta-tui-stderr", daemon=True).start()
            _WRITER_STARTED = True
        _READER = threading.Thread(target=_reader, args=(pipe_read,), name="zeta-tui-stderr-reader", daemon=True)
        _READER.start()
        _CAPTURE_STREAM = _CaptureStream()
        sys.stderr = _CAPTURE_STREAM
        sys.__stderr__ = sys.stderr
        _PREVIOUS_LAST_RESORT = logging.lastResort
        logging.lastResort = logging.StreamHandler(sys.stderr)
        _PREVIOUS_WARNING = warnings.showwarning
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
        _PREVIOUS_THREAD_HOOK = threading.excepthook
        threading.excepthook = lambda args: sys.stderr.write("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        _PREVIOUS_UNRAISABLE_HOOK = sys.unraisablehook
        sys.unraisablehook = lambda hook: sys.stderr.write(f"{hook.exc_type.__name__}: {hook.exc_value}\\n")
        _SETUP = True
        atexit.register(_flush)

def _flush() -> None:
    global _DROPPED_ON_SHUTDOWN
    deadline = time.monotonic() + _FLUSH_TIMEOUT
    while _QUEUE.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    _DROPPED_ON_SHUTDOWN += _QUEUE.unfinished_tasks
    while True:
        try:
            _QUEUE.get_nowait()
        except queue.Empty:
            break
        else:
            _QUEUE.task_done()


@contextmanager
def stderr_to_log() -> Iterator[None]:
    global _SETUP
    _setup()
    try:
        yield
    finally:
        if not _PERSIST_PROCESS:
            _restore()
            _flush()
            _SETUP = False
