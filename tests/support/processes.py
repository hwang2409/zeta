from __future__ import annotations

import asyncio
import os
import select
import selectors
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_READY_TOKEN = b"ZETA_TEST_READY\n"


class ChildHandshake:
    def __init__(self, parent_fd: int, child_fd: int) -> None:
        self._parent_fd = parent_fd
        self.child_fd = child_fd
        self._waited = False

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self.child_fd,)

    def child_env(self, name: str = "ZETA_TEST_READY_FD") -> dict[str, str]:
        return {name: str(self.child_fd)}

    async def wait(
        self, *, process: subprocess.Popen[object], timeout: float = 30
    ) -> None:
        await asyncio.to_thread(self.wait_sync, process=process, timeout=timeout)

    def wait_sync(
        self, *, process: subprocess.Popen[object], timeout: float = 30
    ) -> None:
        if self._waited:
            raise AssertionError("child handshake can only be awaited once")
        self._waited = True
        self._close_child_fd()

        with selectors.DefaultSelector() as selector:
            selector.register(self._parent_fd, selectors.EVENT_READ)
            ready = selector.select(timeout)
            return_code = process.poll()
            child_status = (
                "child still running"
                if return_code is None
                else f"child exit code: {return_code}"
            )
            if not ready:
                raise AssertionError(
                    f"child handshake timed out after {timeout}s waiting for ready token; "
                    f"{child_status}"
                )
            token = os.read(self._parent_fd, 4096)

        if not token:
            raise AssertionError(
                f"child handshake reached eof before ready token; {child_status}"
            )
        if token != _READY_TOKEN:
            raise AssertionError(
                f"child handshake received wrong token: {token!r}; {child_status}"
            )

    def _close_child_fd(self) -> None:
        if self.child_fd >= 0:
            os.close(self.child_fd)
            self.child_fd = -1

    def _close(self) -> None:
        self._close_child_fd()
        if self._parent_fd >= 0:
            os.close(self._parent_fd)
            self._parent_fd = -1


@contextmanager
def child_handshake() -> Iterator[ChildHandshake]:
    parent_fd, child_fd = os.pipe()
    os.set_inheritable(child_fd, True)
    handshake = ChildHandshake(parent_fd, child_fd)
    try:
        yield handshake
    finally:
        handshake._close()


def read_pty_until(
    master_fd: int,
    marker: bytes,
    *,
    process: subprocess.Popen[object],
    output: bytearray,
    start: int = 0,
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(master_fd, selectors.EVENT_READ)
        while marker not in output[start:]:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise AssertionError(
                    f"timed out waiting for PTY marker {marker!r}; "
                    f"return code: {process.poll()!r}; captured: {bytes(output[start:])!r}"
                )
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                chunk = b""
            if not chunk:
                raise AssertionError(
                    f"process exited before PTY marker {marker!r}; "
                    f"return code: {process.poll()!r}; captured: {bytes(output[start:])!r}"
                )
            output.extend(chunk)


def reap_process(process: subprocess.Popen[object], *, timeout: float = 30) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


async def wait_pid_exited(pid: int, timeout: float = 10) -> None:
    await asyncio.to_thread(_wait_pid_exited_sync, pid, timeout)


async def assert_pid_gone(pid: int, timeout: float = 10) -> None:
    """Assert that a process exits, including while it awaits reaping."""
    await wait_pid_exited(pid, timeout)


def _wait_pid_exited_sync(pid: int, timeout: float) -> None:
    if hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(pidfd, selectors.EVENT_READ)
                if selector.select(timeout):
                    return
        finally:
            os.close(pidfd)
    elif hasattr(select, "kqueue"):
        queue = select.kqueue()
        event = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:
            if queue.control([event], 1, timeout):
                return
        except ProcessLookupError:
            return
        finally:
            queue.close()
    else:
        # Poll only on platforms without a process-exit event API.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = _pid_state(pid)
            if state is None or state.startswith("Z"):
                return
            # test-contract: latency compatibility fallback must rate-limit ps polling
            time.sleep(0.05)

    state = _pid_state(pid)
    if state is None or state.startswith("Z"):
        return
    raise AssertionError(
        f"pid {pid} did not exit within {timeout}s; last ps state: {state!r}"
    )


def _pid_state(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def wait_for_path_command(
    path: Path,
    *,
    owner_pid: int | None = None,
    timeout: float = 120,
) -> str:
    """Build a marker wait that stops when its test owner exits."""

    owner_pid = os.getpid() if owner_pid is None else owner_pid
    source = f"""
import os
import pathlib
import time

path = pathlib.Path({str(path)!r})
owner_pid = {owner_pid}
deadline = time.monotonic() + {timeout!r}
while not path.exists():
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        raise SystemExit(1)
    if time.monotonic() >= deadline:
        raise SystemExit(1)
    time.sleep(0.1)
""".strip()
    return shlex.join((sys.executable, "-c", source))


async def read_pid_file(path: Path, *, timeout: float = 10.0) -> int:
    """Wait until a child has written its pid, not just created the file.

    A child that runs ``Path(p).write_text(pid)`` creates the file before it
    writes, so ``path.exists()`` alone can see an empty file.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            text = path.read_text().split()
            if text:
                return int(text[0])
        except (FileNotFoundError, ValueError):
            pass
        if loop.time() >= deadline:
            raise AssertionError(f"no pid written to {path} within {timeout}s")
        await asyncio.sleep(0.01)
