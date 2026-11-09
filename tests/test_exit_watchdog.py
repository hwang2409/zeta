from __future__ import annotations

from pathlib import Path

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
