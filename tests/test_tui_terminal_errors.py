from __future__ import annotations

import asyncio
import os
import pty
import selectors
import subprocess
import sys
import threading
import time
from pathlib import Path

from tests.support.owned import await_or_fail
from tests.support.processes import read_pty_until, reap_process
from zeta.tui import loop_errors

NOISE = (
    "ZNOISE-warning",
    "ZNOISE-logging",
    "ZNOISE-unraisable",
    "ZNOISE-thread",
    "ZNOISE-child",
    "ZNOISE-teardown",
)


def _drain_until_eof(master_fd: int, output: bytearray, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(master_fd, selectors.EVENT_READ)
        while (remaining := deadline - time.monotonic()) > 0:
            if not selector.select(remaining):
                continue
            try:
                chunk = os.read(master_fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            output.extend(chunk)
    raise AssertionError(f"PTY did not close; captured: {bytes(output[-2000:])!r}")


def test_stderr_flood_does_not_block_event_loop(tmp_path: Path, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_writer(*_args) -> None:
        started.set()
        release.wait(10)

    monkeypatch.setattr(loop_errors, "write_tui_diagnostic", blocked_writer)
    child = None
    try:
        with loop_errors.stderr_to_log():
            child = subprocess.Popen(
                [sys.executable, "-c", "import os; os.write(2, b'x' * 1048576)"],
                close_fds=False,
            )
            await_or_fail_sync(started)
            os.write(2, b"event-loop-write")
            assert child.wait(timeout=10) == 0
            release.set()
    finally:
        release.set()
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def await_or_fail_sync(event: threading.Event) -> None:
    asyncio.run(await_or_fail(asyncio.to_thread(event.wait, 10), what="stderr writer"))


def test_stray_stderr_never_reaches_the_terminal(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    env = os.environ.copy()
    for name in ("COLORTERM", "NO_COLOR", "TERM"):
        env.pop(name, None)
    env.update(
        ZETA_HOME=str(home),
        TERM="xterm-256color",
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(Path(__file__).parents[1]), env.get("PYTHONPATH")))
        ),
    )
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.support.noisy_tui", "--provider", "codex"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    output = bytearray()
    try:
        read_pty_until(master_fd, b" \xe2\x80\xba ", process=process, output=output)
        start = len(output)
        # The backend makes all its noise before it streams this reply.
        os.write(master_fd, b"hello\r")
        read_pty_until(
            master_fd, b"scripted test backend", process=process, output=output, start=start
        )
        os.write(master_fd, b"\x03")
        os.write(master_fd, b"\x04")
        _drain_until_eof(master_fd, output)
        assert reap_process(process, timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            reap_process(process, timeout=10)
        os.close(master_fd)

    terminal = output.decode("utf-8", "replace")
    leaked = [text for text in (*NOISE, "Traceback") if text in terminal]
    assert leaked == [], f"terminal showed stray stderr: {leaked}"
    assert "resume with:" in terminal
    log = (home / "logs" / "tui-errors.jsonl").read_text()
    assert [text for text in NOISE if text not in log] == []
