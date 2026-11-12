from __future__ import annotations

import os
import pty
import selectors
import subprocess
import sys
import time
from pathlib import Path

from tests.support.processes import read_pty_until, reap_process

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


def test_stderr_flood_does_not_block_event_loop(tmp_path: Path) -> None:
    code = "import os\nfrom zeta.tui.stderr_capture import stderr_to_log\nwith stderr_to_log():\n    os.write(2, b'x' * 1048576)"
    env = os.environ | {"ZETA_HOME": str(tmp_path / "zeta-home")}
    child = subprocess.run([sys.executable, "-c", code], env=env, timeout=10, check=False)
    assert child.returncode == 0
    assert (tmp_path / "zeta-home" / "logs" / "tui-stderr.log").stat().st_size >= 1048576


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
    for text in (*NOISE, "Traceback"):
        assert text not in terminal, f"terminal showed stray stderr: {text}"
    assert "resume with:" in terminal
    log = (home / "logs" / "tui-stderr.log").read_text()
    assert [text for text in NOISE if text not in log] == []
