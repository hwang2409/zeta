"""Exit-time watchdog: dump every thread and exit if pytest hangs after the tests."""

from __future__ import annotations

import faulthandler
import os
import sys

import pytest

# Ends a hung exit instead of burning the 30-minute CI job timeout.
EXIT_WATCHDOG_SECONDS = 120


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish() -> None:
    # Never cancelled: a normal exit ends the process before it fires.
    seconds = float(
        os.environ.get("ZETA_TEST_EXIT_WATCHDOG_SECONDS", EXIT_WATCHDOG_SECONDS)
    )
    faulthandler.dump_traceback_later(seconds, exit=True, file=sys.__stderr__)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish() -> None:  # noqa: F811 - DIAG ONLY
    import threading

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        print(
            f"DIAG live-thread worker={worker} name={thread.name} "
            f"daemon={thread.daemon} target={target!r}",
            file=sys.__stderr__,
            flush=True,
        )
