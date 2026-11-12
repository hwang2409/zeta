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


def _run_pty_python(code: str, env: dict[str, str], timeout: float = 10) -> tuple[bytes, int]:
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    output = bytearray()
    try:
        _drain_until_eof(master_fd, output, timeout)
        return bytes(output), reap_process(process, timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            reap_process(process, timeout=timeout)
        os.close(master_fd)


def _run_pty_python_with_stderr_pipe(
    code: str, env: dict[str, str], timeout: float = 10
) -> tuple[bytes, bytes, int]:
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=subprocess.PIPE,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    output = bytearray()
    try:
        _drain_until_eof(master_fd, output, timeout)
        stderr = process.stderr.read() if process.stderr is not None else b""
        return bytes(output), stderr, reap_process(process, timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            reap_process(process, timeout=timeout)
        os.close(master_fd)


def _capture_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        ZETA_HOME=str(home),
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(Path(__file__).parents[1]), env.get("PYTHONPATH")))
        ),
    )
    return env


def test_capture_uses_temp_fallback_when_home_log_is_unavailable(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "tui-stderr").write_text("not a directory")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    env = _capture_env(home) | {"TMPDIR": str(fallback)}
    code = """
import os
from zeta.tui.stderr_capture import stderr_to_log
with stderr_to_log():
    os.write(2, b'ZFALLBACK\\n')
"""
    terminal, returncode = _run_pty_python(code, env)
    assert returncode == 0
    assert b"ZFALLBACK" not in terminal
    logs = list((fallback / "zeta-tui-stderr").glob("*.log"))
    assert len(logs) == 1
    assert b"ZFALLBACK" in logs[0].read_bytes()


def test_pre_capture_atexit_stderr_stays_captured(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    code = """
import atexit, os
from zeta.tui.stderr_capture import stderr_to_log
atexit.register(lambda: os.write(2, b'ZATEXIT-LEAK\\n'))
with stderr_to_log():
    os.write(2, b'ZATEXIT-SETUP\\n')
"""
    terminal, returncode = _run_pty_python(code, _capture_env(home))

    assert returncode == 0
    assert b"ZATEXIT-LEAK" not in terminal
    logs = list((home / "logs" / "tui-stderr").glob("*.log"))
    assert len(logs) == 1
    assert b"ZATEXIT-LEAK" in logs[0].read_bytes()


def test_four_mib_stderr_flood_is_exact(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    code = """
import os
import zeta.tui.stderr_capture as sc
if hasattr(sc, "_open_log"):
    original = sc._open_log
    def delayed(path):
        os.write(int(os.environ["READY_FD"]), b"ready")
        os.read(int(os.environ["RELEASE_FD"]), 1)
        return original(path)
    sc._open_log = delayed
expected = b"".join(bytes([i]) * 65536 for i in range(64))
with sc.stderr_to_log():
    os.write(2, expected)
"""
    env = _capture_env(home) | {
        "READY_FD": str(ready_write),
        "RELEASE_FD": str(release_read),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=env, pass_fds=(ready_write, release_read)
    )
    os.close(ready_write)
    os.close(release_read)
    selector = selectors.DefaultSelector()
    selector.register(ready_read, selectors.EVENT_READ)
    try:
        events = selector.select(10)
        ready = os.read(ready_read, 5) if events else b""
        if ready:
            assert ready == b"ready"
            os.write(release_write, b"x")
        assert process.wait(timeout=10) == 0
    finally:
        selector.close()
        if process.poll() is None:
            try:
                os.write(release_write, b"x")
            except BrokenPipeError:
                pass
            reap_process(process, timeout=10)
        os.close(ready_read)
        os.close(release_write)
    logs = list((home / "logs" / "tui-stderr").glob("*.log"))
    expected = b"".join(bytes([i]) * 65536 for i in range(64))
    assert len(logs) == 1
    assert logs[0].read_bytes() == expected


def test_twenty_processes_keep_private_complete_logs(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    code = """
import os, sys
from zeta.tui.stderr_capture import stderr_to_log
with stderr_to_log():
    os.write(2, (sys.argv[1] + "\\n").encode())
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, f"ZMARKER-{index:02d}"],
            env=_capture_env(home),
        )
        for index in range(20)
    ]
    assert [process.wait(timeout=20) for process in processes] == [0] * 20
    logs = list((home / "logs" / "tui-stderr").glob("*.log"))
    assert len(logs) == 20
    contents = [path.read_text() for path in logs]
    for index in range(20):
        marker = f"ZMARKER-{index:02d}"
        assert sum(marker in content for content in contents) == 1
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in logs)
    assert (home / "logs" / "tui-stderr").stat().st_mode & 0o777 == 0o700


def test_capture_fails_cleanly_when_home_and_temp_logs_are_unavailable(
    tmp_path: Path,
) -> None:
    home = tmp_path / "zeta-home"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "tui-stderr").write_text("not a directory")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "zeta-tui-stderr").write_text("not a directory")
    sentinel = tmp_path / "lifecycle-called"
    code = f"""
from pathlib import Path
from zeta.cli import main
main._run_tui_lifecycle_captured = lambda *args: Path({str(sentinel)!r}).write_text('called')
raise SystemExit(main._main([]))
"""
    terminal, stderr, returncode = _run_pty_python_with_stderr_pipe(
        code, _capture_env(home) | {"TMPDIR": str(fallback)}
    )
    lines = stderr.decode().splitlines()
    assert returncode == 1
    assert len(lines) == 1
    assert lines[0].startswith(
        "zeta: could not start TUI; stderr log unavailable: "
    )
    assert "Traceback" not in stderr.decode()
    assert b"zeta: could not start TUI" not in terminal
    assert not sentinel.exists()


def test_startup_retention_skips_locked_logs(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    logs = home / "logs" / "tui-stderr"
    logs.mkdir(parents=True)
    old = logs / "1-1.log"
    old.write_bytes(b"x" * (50 * 1024 * 1024))
    old.touch()
    code = """
import os
from zeta.tui.stderr_capture import stderr_to_log
with stderr_to_log():
    os.write(2, b'ZPRIVATE\\n')
"""
    terminal, returncode = _run_pty_python(code, _capture_env(home))
    assert returncode == 0
    assert b"ZPRIVATE" not in terminal
    assert old.exists()
    current = next(path for path in logs.glob("*.log") if path != old)
    assert current.stat().st_mode & 0o777 == 0o600


def test_restart_replacement_gets_terminal_stderr(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    replacement = tmp_path / "replacement.py"
    replacement.write_text("#!/usr/bin/env python3\nimport os; os.write(2, b'AFTER-EXEC\\n')")
    replacement.chmod(0o755)
    code = f"""
from zeta.cli.main import _exec_restart
from zeta.tui.stderr_capture import stderr_to_log
with stderr_to_log():
    _exec_restart({str(replacement)!r}, [], 'session')
"""
    terminal, returncode = _run_pty_python(code, _capture_env(home))

    assert returncode == 0
    assert b"AFTER-EXEC" in terminal


def test_restart_exec_failure_is_user_visible(tmp_path: Path) -> None:
    home = tmp_path / "zeta-home"
    code = """
from zeta.cli.main import _exec_restart
from zeta.tui.stderr_capture import stderr_to_log
with stderr_to_log():
    raise SystemExit(_exec_restart('/missing/zeta', [], 'session'))
"""
    terminal, returncode = _run_pty_python(code, _capture_env(home))

    assert returncode == 1
    assert b"restart failed: " in terminal
    assert b"resume manually: zeta --resume session" in terminal


def test_stderr_flood_does_not_block_event_loop(tmp_path: Path) -> None:
    code = "import os\nfrom zeta.tui.stderr_capture import stderr_to_log\nwith stderr_to_log():\n    os.write(2, b'x' * 1048576)"
    env = os.environ | {"ZETA_HOME": str(tmp_path / "zeta-home")}
    child = subprocess.run([sys.executable, "-c", code], env=env, timeout=10, check=False)
    assert child.returncode == 0
    logs = list((tmp_path / "zeta-home" / "logs" / "tui-stderr").glob("*.log"))
    assert len(logs) == 1
    assert logs[0].stat().st_size >= 1048576


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
    # Check the user-visible terminal before checking that capture persisted the noise.
    for text in (*NOISE, "Traceback"):
        assert text not in terminal, f"terminal showed stray stderr: {text}"
    assert "resume with:" in terminal
    log = next((home / "logs" / "tui-stderr").glob("*.log")).read_text()
    assert [text for text in NOISE if text not in log] == []
