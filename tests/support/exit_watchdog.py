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
import faulthandler
import io
import os
import sys
import threading
import time
from collections.abc import Generator

import pytest

# Every teardown in CI takes well under 15 s (none appears in the 25 slowest
# durations of any CI job), so this only fires on a real hang.
EXIT_WATCHDOG_SECONDS = 120

_stderr_fd: int | None = None
_seconds = float(EXIT_WATCHDOG_SECONDS)


def _write(text: str) -> None:
    assert _stderr_fd is not None
    os.write(_stderr_fd, text.encode(errors="replace"))


class _TaskDumper:
    """One daemon thread that prints thread names and task stacks at a deadline.

    It runs before the faulthandler deadline because faulthandler shows neither
    thread names on every Python version nor asyncio tasks. faulthandler stays
    the backstop: it still fires when a C call holds the GIL and this thread
    cannot run.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None

    def arm(self, seconds: float) -> None:
        with self._changed:
            self._deadline = time.monotonic() + seconds
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="zeta-test-watchdog", daemon=True
                )
                self._thread.start()
            self._changed.notify()

    def disarm(self) -> None:
        with self._changed:
            self._deadline = None
            self._changed.notify()

    def _run(self) -> None:
        with self._changed:
            while True:
                if self._deadline is None:
                    self._changed.wait()
                    continue
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._changed.wait(remaining)
                    continue
                self._deadline = None
                _write(_describe_threads_and_tasks())


def _describe_threads_and_tasks() -> str:
    out = io.StringIO()
    out.write("\nzeta test watchdog: hang, threads (faulthandler stack follows):\n")
    for thread in threading.enumerate():
        out.write(f"  0x{thread.ident or 0:016x} {thread.name} daemon={thread.daemon}\n")
    for loop in _running_loops():
        try:
            tasks = asyncio.all_tasks(loop)
        except RuntimeError as error:  # the loop thread changed the task set
            out.write(f"tasks of {loop!r} unavailable: {error}\n")
            continue
        out.write(f"{len(tasks)} pending asyncio tasks of {loop!r}:\n")
        for task in tasks:
            task.print_stack(file=out)
    return out.getvalue()


def _running_loops() -> list[asyncio.AbstractEventLoop]:
    loops: list[asyncio.AbstractEventLoop] = []
    for frame in sys._current_frames().values():
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
