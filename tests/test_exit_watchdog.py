from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# A hang after every test finished: at interpreter exit (after pytest's own
# unconfigure), or in an xdist worker's session finish.
_HANG_AT_EXIT = """
import atexit
import threading


def hang_after_pytest_exit():
    threading.Event().wait()


atexit.register(hang_after_pytest_exit)
"""

_HANG_IN_WORKER_SESSION_FINISH = """
import threading

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        hang_in_worker_session_finish()


def hang_in_worker_session_finish():
    threading.Event().wait()
"""


@pytest.mark.parametrize(
    ("conftest", "extra_args", "hanging_frame"),
    [
        (_HANG_AT_EXIT, (), "hang_after_pytest_exit"),
        (
            _HANG_IN_WORKER_SESSION_FINISH,
            ("-p", "xdist", "-n", "1"),
            "hang_in_worker_session_finish",
        ),
    ],
    ids=["atexit", "xdist-worker"],
)
def test_exit_watchdog_ends_a_hung_exit_with_a_traceback(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest: str,
    extra_args: tuple[str, ...],
    hanging_frame: str,
) -> None:
    # The child loads the real repository conftest, which installs the watchdog.
    monkeypatch.setenv("PYTHONPATH", str(_REPOSITORY_ROOT))
    monkeypatch.setenv("ZETA_TEST_EXIT_WATCHDOG_SECONDS", "1")
    pytester.makeconftest(conftest)
    pytester.makepyfile("def test_passes():\n    pass\n")

    try:
        result = pytester.runpytest_subprocess(
            "-p", "tests.conftest", "-p", "no:cacheprovider", *extra_args, timeout=60
        )
    except pytester.TimeoutExpired:
        pytest.fail("child pytest still hung after 60 s: no exit watchdog fired")

    output = "\n".join(result.outlines + result.errlines)
    assert "Timeout (0:00:01)!" in output, output
    assert hanging_frame in output, output


# A test fails at its timeout guard while a task re-shields every cancellation,
# so pytest-asyncio's loop close (`_cancel_all_tasks`) waits forever. pytest
# cancels its per-test faulthandler timer when the test fails, before teardown.
_HANGING_LOOP_CLOSE = """
import asyncio

_tasks = []


async def resist_cancellation(never):
    while True:
        try:
            await asyncio.shield(never)
        except asyncio.CancelledError:
            pass


async def test_times_out_while_a_task_resists_cancellation():
    _tasks.append(asyncio.create_task(resist_cancellation(asyncio.Future())))
    await asyncio.wait_for(asyncio.Event().wait(), 0.1)
"""


def test_teardown_watchdog_dumps_a_hung_loop_close(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nasyncio_mode = auto\n"
        "faulthandler_timeout = 300\nfaulthandler_exit_on_timeout = true\n"
    )
    (tmp_path / "test_hang.py").write_text(_HANGING_LOOP_CLOSE)
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest"]
    command += ["-p", "no:cacheprovider", "test_hang.py"]
    env = os.environ | {
        "PYTHONPATH": str(_REPOSITORY_ROOT),
        "ZETA_TEST_EXIT_WATCHDOG_SECONDS": "3",
    }
    try:
        done = subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        output = done.stdout + done.stderr
    except subprocess.TimeoutExpired as hung:
        output = _text(hung.stdout) + _text(hung.stderr)

    assert "Timeout (0:00:03)!" in output, f"no stack within 60 s:\n{output}"
    assert "_cancel_all_tasks" in output, output
    # Thread names, to match faulthandler's thread ids, and the stack of the
    # task that resists cancellation.
    assert "MainThread daemon=False" in output, output
    assert "in resist_cancellation" in output, output


def _text(stream: bytes | str | None) -> str:
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream or ""
