"""Hang watchdog for test teardown and process exit.

pytest's faulthandler timer covers a test's setup and call, but pytest cancels
it as soon as a test fails, and so a hung teardown (for example pytest-asyncio's
loop close waiting on a task that ignores cancellation) prints nothing until the
CI job times out. This watchdog arms during every teardown and from session
finish on: it first prints thread names and asyncio task stacks, then
faulthandler dumps every thread and ends the process, so xdist reports a
crashed worker with a stack instead of a silent hang.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import io
import os
import sys
import threading
from collections.abc import Generator

import pytest

# Every teardown in CI takes well under 15 s (none appears in the 25 slowest
# durations of any CI job), so this only fires on a real hang.
EXIT_WATCHDOG_SECONDS = 120

_stderr_fd: int | None = None
_seconds = float(EXIT_WATCHDOG_SECONDS)

# Bound at import: tests patch these globals (for example `time.monotonic` with a
# fake clock), and the watchdog thread runs during their setup and teardown.
_os_write = os.write
_enumerate_threads = threading.enumerate
_current_frames = sys._current_frames
_all_tasks = asyncio.all_tasks


def _write(text: str) -> None:
    assert _stderr_fd is not None
    _os_write(_stderr_fd, text.encode(errors="replace"))


class _TaskDumper:
    """One daemon thread that prints thread names and task stacks at a deadline.

    It runs before the faulthandler deadline because faulthandler shows neither
    thread names on every Python version nor asyncio tasks. faulthandler stays
    the backstop: it still fires when a C call holds the GIL and this thread
    cannot run.

    The thread reads no clock: `Event.wait(timeout)` times the deadline, and a
    generation number tells a timeout apart from a later arm or disarm.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._generation = 0
        self._seconds: float | None = None
        self._thread: threading.Thread | None = None
        self._reported_failure = False

    def arm(self, seconds: float) -> None:
        with self._lock:
            self._generation += 1
            self._seconds = seconds
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="zeta-test-watchdog", daemon=True
                )
                self._thread.start()
        self._changed.set()

    def disarm(self) -> None:
        with self._lock:
            self._generation += 1
            self._seconds = None
        self._changed.set()

    def _run(self) -> None:
        while True:
            try:
                self._wait_and_dump()
            # Never raise into pytest, and never die silently.
            except Exception as error:  # noqa: BLE001
                if not self._reported_failure:
                    self._reported_failure = True
                    with contextlib.suppress(Exception):
                        _write(f"\nzeta test watchdog: broken: {error!r}\n")

    def _wait_and_dump(self) -> None:
        self._changed.wait()
        with self._lock:
            self._changed.clear()
            generation, seconds = self._generation, self._seconds
        if seconds is None or self._changed.wait(seconds):
            return
        with self._lock:
            if generation != self._generation:
                return
            self._seconds = None
        _write(_describe_threads_and_tasks())


def _describe_threads_and_tasks() -> str:
    out = io.StringIO()
    out.write("\nzeta test watchdog: hang, threads (faulthandler stack follows):\n")
    for thread in _enumerate_threads():
        out.write(f"  0x{thread.ident or 0:016x} {thread.name} daemon={thread.daemon}\n")
    for loop in _running_loops():
        try:
            tasks = _all_tasks(loop)
        except RuntimeError as error:  # the loop thread changed the task set
            out.write(f"tasks of {loop!r} unavailable: {error}\n")
            continue
        out.write(f"{len(tasks)} pending asyncio tasks of {loop!r}:\n")
        for task in tasks:
            task.print_stack(file=out)
    return out.getvalue()


def _running_loops() -> list[asyncio.AbstractEventLoop]:
    loops: list[asyncio.AbstractEventLoop] = []
    for frame in _current_frames().values():
        while frame is not None:
            if frame.f_code.co_name == "run_forever":
                loop = frame.f_locals.get("self")
                if isinstance(loop, asyncio.AbstractEventLoop) and loop not in loops:
                    loops.append(loop)
            frame = frame.f_back
    return loops


_task_dumper = _TaskDumper()


def _arm() -> None:
    # Print threads and tasks shortly before faulthandler dumps and exits.
    _task_dumper.arm(max(_seconds - 10, _seconds / 2))
    faulthandler.dump_traceback_later(_seconds, exit=True, file=_stderr_fd)


def pytest_configure() -> None:
    global _seconds, _stderr_fd
    # Read once: the isolation fixture removes ZETA_* variables during tests.
    _seconds = float(
        os.environ.get("ZETA_TEST_EXIT_WATCHDOG_SECONDS", EXIT_WATCHDOG_SECONDS)
    )
    # A dup of the real stderr: during teardown fd 2 is captured into a temp
    # file that is lost when faulthandler ends the process. Kept open for the
    # life of the process, so the exit watchdog can still write after pytest
    # unconfigures.
    if _stderr_fd is None:
        _stderr_fd = os.dup(sys.__stderr__.fileno())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown() -> Generator[None]:
    # Replaces pytest's per-test timer, which a failing test already cancelled.
    _arm()
    try:
        return (yield)
    finally:
        _task_dumper.disarm()
        faulthandler.cancel_dump_traceback_later()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish() -> None:
    # Never cancelled: a normal exit ends the process before it fires.
    _arm()
