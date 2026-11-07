from __future__ import annotations

import asyncio
import os
import pty
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.processes import (
    ChildHandshake,
    child_handshake,
    read_pty_until,
    reap_process,
    wait_for_path_command,
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
        try:
            handshake.wait_sync()
        except AssertionError as error:
            status = reap_process(process)
            raise AssertionError(f"{error}; child exit code: {status}") from error
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
            ready = asyncio.create_task(handshake.wait())
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
        with pytest.raises(AssertionError, match="eof before ready token") as error:
            handshake.wait_sync()
        status = reap_process(process)
        assert f"{error.value}; child exit code: {status}".endswith(
            "child exit code: 7"
        )

    assert _fd_count() == baseline


def test_child_handshake_rejects_a_wrong_token() -> None:
    with child_handshake() as handshake:
        process = _spawn_with_handshake(
            "import os; os.write(int(os.environ['ZETA_TEST_READY_FD']), b'wrong')",
            handshake,
        )
        with pytest.raises(AssertionError, match="wrong token: b'wrong'"):
            handshake.wait_sync()
        assert reap_process(process) == 0


def test_child_handshake_timeout_restores_file_descriptors() -> None:
    baseline = _fd_count()

    with child_handshake() as handshake:
        process = _spawn_with_handshake("import time; time.sleep(120)", handshake)
        with pytest.raises(AssertionError, match="timed out"):
            handshake.wait_sync(timeout=0)
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
