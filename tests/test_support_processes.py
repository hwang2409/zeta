from __future__ import annotations

import asyncio
import os
import pty
import shlex
import subprocess
import sys
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from tests.support.processes import (
    ChildHandshake,
    child_handshake,
    owned_child_process,
    read_pty_until,
    reap_process,
    wait_for_path_command,
    wait_pid_exited,
)

_READY_WRITE = "os.write(int(os.environ['ZETA_TEST_READY_FD']), b'ZETA_TEST_READY\\n')"


def _fd_count() -> int:
    for directory in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(directory):
            return len(os.listdir(directory))
    pytest.skip("file descriptor directory is unavailable")


def _spawn_with_handshake(
    source: str, handshake: ChildHandshake
) -> subprocess.Popen[object]:
    return subprocess.Popen(
        [sys.executable, "-c", source],
        env={**os.environ, **handshake.child_env()},
        pass_fds=handshake.pass_fds,
    )


def test_child_handshake_success_restores_file_descriptors() -> None:
    baseline = _fd_count()

    with child_handshake() as handshake:
        process = _spawn_with_handshake(f"import os; {_READY_WRITE}", handshake)
        handshake.wait_sync(process=process)
        assert reap_process(process) == 0

    assert _fd_count() == baseline


@pytest.mark.asyncio
async def test_child_handshake_outlives_early_legacy_marker(tmp_path: Path) -> None:
    baseline = _fd_count()
    marker = tmp_path / "legacy-ready"
    marker_read, marker_write = os.pipe()
    release_read, release_write = os.pipe()

    try:
        with child_handshake() as handshake:
            source = f"""
import os
from pathlib import Path
Path({str(marker)!r}).touch()
os.write({marker_write}, b"marker-created")
os.read({release_read}, 1)
{_READY_WRITE}
"""
            process = await asyncio.to_thread(
                subprocess.Popen,
                [sys.executable, "-c", source],
                env={**os.environ, **handshake.child_env()},
                pass_fds=(*handshake.pass_fds, marker_write, release_read),
            )
            os.close(marker_write)
            marker_write = -1
            os.close(release_read)
            release_read = -1

            assert (
                await asyncio.to_thread(os.read, marker_read, 64) == b"marker-created"
            )
            assert marker.exists()  # The legacy existence assertion can run too early.
            ready = asyncio.create_task(handshake.wait(process=process))
            await asyncio.sleep(0)
            assert not ready.done()

            os.write(release_write, b"1")
            await ready
            assert reap_process(process) == 0
    finally:
        for fd in (marker_read, marker_write, release_read, release_write):
            if fd >= 0:
                os.close(fd)

    assert _fd_count() == baseline


def test_child_handshake_eof_names_child_status_and_restores_fds() -> None:
    baseline = _fd_count()

    with child_handshake() as handshake:
        process = _spawn_with_handshake("raise SystemExit(7)", handshake)
        assert process.wait() == 7
        with pytest.raises(AssertionError, match="eof before ready token") as error:
            handshake.wait_sync(process=process)
        assert str(error.value).endswith("child exit code: 7")

    assert _fd_count() == baseline


def test_child_handshake_rejects_a_wrong_token() -> None:
    with child_handshake() as handshake:
        process = _spawn_with_handshake(
            "import os; os.write(int(os.environ['ZETA_TEST_READY_FD']), b'wrong')",
            handshake,
        )
        assert process.wait() == 0
        with pytest.raises(AssertionError, match="wrong token: b'wrong'") as error:
            handshake.wait_sync(process=process)
        assert str(error.value).endswith("child exit code: 0")


def test_child_handshake_timeout_restores_file_descriptors() -> None:
    baseline = _fd_count()

    with child_handshake() as handshake:
        process = _spawn_with_handshake("import time; time.sleep(120)", handshake)
        with pytest.raises(AssertionError, match="timed out") as error:
            handshake.wait_sync(process=process, timeout=0)
        assert str(error.value).endswith("child still running")
        assert reap_process(process, timeout=0) < 0

    assert _fd_count() == baseline


def test_child_handshake_exception_restores_file_descriptors() -> None:
    baseline = _fd_count()

    with pytest.raises(RuntimeError, match="injected failure"), child_handshake():
        raise RuntimeError("injected failure")

    assert _fd_count() == baseline


def test_read_pty_until_accumulates_through_marker() -> None:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", "print('before READY', flush=True)"],
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    output = bytearray()
    try:
        read_pty_until(master, b"READY", process=process, output=output)
        assert b"before READY" in output
        assert reap_process(process) == 0
    finally:
        if process.poll() is None:
            reap_process(process, timeout=0)
        os.close(master)


def test_read_pty_until_names_early_exit_and_output() -> None:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", "print('partial', flush=True); raise SystemExit(4)"],
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    output = bytearray()
    try:
        with pytest.raises(AssertionError, match="partial"):
            read_pty_until(master, b"missing", process=process, output=output)
        assert reap_process(process) == 4
    finally:
        if process.poll() is None:
            reap_process(process, timeout=0)
        os.close(master)


def test_reap_process_kills_a_process_that_does_not_exit() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])

    assert reap_process(process, timeout=0) < 0
    assert process.poll() is not None


def test_wait_for_path_command_exits_when_owner_is_killed(tmp_path) -> None:
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    waiter = subprocess.Popen(
        shlex.split(
            wait_for_path_command(tmp_path / "never-created", owner_pid=owner.pid)
        )
    )

    try:
        owner.kill()
        assert reap_process(owner, timeout=5) < 0
        assert reap_process(waiter, timeout=5) == 1
    finally:
        if owner.poll() is None:
            reap_process(owner, timeout=0)
        if waiter.poll() is None:
            reap_process(waiter, timeout=0)


class _InjectedError(RuntimeError):
    pass


class _OwnedEndpoint:
    def __init__(self, connection: Connection, failures: set[str]) -> None:
        self._connection = connection
        self._failures = failures

    @property
    def closed(self) -> bool:
        return self._connection.closed

    def send(self, message: object) -> None:
        self._connection.send(message)
        if "send" in self._failures:
            raise _InjectedError("send")

    def close(self) -> None:
        self._connection.close()
        if "endpoint.close" in self._failures:
            raise _InjectedError("endpoint.close")


class _GatedClose:
    def __init__(self, failures: set[str], *, awaits: int = 0) -> None:
        self.entered = [asyncio.Event() for _ in range(awaits)]
        self.release = [asyncio.Event() for _ in range(awaits)]
        self.finished = False
        self._read_fd, self._write_fd = os.pipe()
        self._failures = failures

    async def __call__(self) -> None:
        if self.finished:
            return
        for entered, release in zip(self.entered, self.release, strict=True):
            entered.set()
            await asyncio.wait_for(release.wait(), 10)
        os.close(self._read_fd)
        os.close(self._write_fd)
        self.finished = True
        if "async.close" in self._failures:
            raise _InjectedError("async.close")

    def assert_closed(self) -> None:
        for fd in (self._read_fd, self._write_fd):
            with pytest.raises(OSError):
                os.fstat(fd)


def _receive_stop(connection: Connection) -> None:
    try:
        connection.recv()
    finally:
        connection.close()


def _real_owned_process(
    *, failures: set[str] | None = None, close_awaits: int = 0
) -> tuple[Process, _OwnedEndpoint, _GatedClose, int]:
    receive, send = Pipe(duplex=False)
    process = Process(target=_receive_stop, args=(receive,))
    process.start()
    receive.close()
    assert process.pid is not None
    failures = failures or set()
    return (
        process,
        _OwnedEndpoint(send, failures),
        _GatedClose(failures, awaits=close_awaits),
        process.pid,
    )


_CLEANUP_STEPS = ["async.close", "send", "endpoint.close"]


@pytest.mark.parametrize(
    ("case", "failures", "cancel_at", "expected_error"),
    [
        pytest.param("success", set(), None, None, id="success"),
        pytest.param("duplicate", set(), None, None, id="duplicate-close"),
        *[
            pytest.param(
                "cancel",
                set(),
                await_index,
                None,
                id=f"cancel-close-await-{await_index}",
            )
            for await_index in range(3)
        ],
        *[
            pytest.param("failure", {step}, None, step, id=f"failure-{step}")
            for step in _CLEANUP_STEPS
        ],
        pytest.param(
            "failure",
            {"async.close", "endpoint.close"},
            None,
            "async.close",
            id="first-error-kept",
        ),
    ],
)
@pytest.mark.asyncio
async def test_owned_child_process_cleanup_table(
    case: str,
    failures: set[str],
    cancel_at: int | None,
    expected_error: str | None,
) -> None:
    process, endpoint, close, pid = _real_owned_process(
        failures=failures, close_awaits=3 if cancel_at is not None else 0
    )

    async def use_process() -> None:
        async with owned_child_process(
            process,
            endpoints=(endpoint,),  # type: ignore[arg-type]
            close=close,
            stop=(endpoint, "stop"),  # type: ignore[arg-type]
        ) as owned:
            assert owned is process

    if case == "cancel":
        assert cancel_at is not None
        for release in close.release[:cancel_at]:
            release.set()
        owner = asyncio.create_task(use_process())
        try:
            await asyncio.wait_for(close.entered[cancel_at].wait(), 10)
            owner.cancel()
            for release in close.release[cancel_at:]:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(owner, 10)
        finally:
            for release in close.release:
                release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 10)
    elif case == "duplicate":
        await use_process()
        await use_process()
    elif expected_error is None:
        await use_process()
    else:
        with pytest.raises(_InjectedError, match=expected_error):
            await use_process()

    await wait_pid_exited(pid)
    assert endpoint.closed
    assert close.finished
    close.assert_closed()
    with pytest.raises(ValueError, match="process object is closed"):
        process.is_alive()


@pytest.mark.asyncio
async def test_owned_child_process_body_error_wins_after_complete_cleanup() -> None:
    process, endpoint, close, pid = _real_owned_process(failures={"async.close"})

    with pytest.raises(ValueError, match="body") as raised:
        async with owned_child_process(
            process,
            endpoints=(endpoint,),  # type: ignore[arg-type]
            close=close,
            stop=(endpoint, "stop"),  # type: ignore[arg-type]
        ):
            raise ValueError("body")

    assert raised.value.__notes__ == ["cleanup error: _InjectedError: async.close"]
    await wait_pid_exited(pid)
    assert endpoint.closed
    assert close.finished
    close.assert_closed()
