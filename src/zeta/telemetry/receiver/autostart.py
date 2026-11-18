"""Keep the home's built-in receiver running from any zeta process configured for it.

Only the exact built-in endpoint starts a receiver; any other endpoint is an external collector.
The receiver's own lock decides the owner, so concurrent starters are safe: a losing launch exits
and the next health check finds the winner. A launched receiver runs in its own session and
outlives the process that started it, because the home's other zeta processes export to it too;
stop it with a signal. Health checks and launches run on one daemon thread, and ``snapshot()``
only reads an attribute, so status never waits on the receiver.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import ReceiverState, TelemetryError
from .process import OTLP_URL, receiver_ready

POLL_SECONDS = 10.0
STARTUP_SECONDS = 15.0
STARTUP_POLL_SECONDS = 0.2
MAX_LAUNCHES = 5  # consecutive failed launches before the supervisor only watches
MAX_BACKOFF_SECONDS = 30.0
_COMMAND = "import sys; from zeta.cli.main import main; sys.exit(main())"


def supervise(home: Path, endpoint: str) -> ReceiverSupervisor | None:
    """Start supervising ``home``'s receiver when ``endpoint`` is the built-in one."""
    if endpoint != OTLP_URL:
        return None
    if importlib.util.find_spec("uvicorn") is None or importlib.util.find_spec("starlette") is None:
        raise TelemetryError("OpenTelemetry support is not installed; install zeta[otel]")
    supervisor = ReceiverSupervisor(home)
    supervisor.start()
    return supervisor


def _launch(home: Path) -> subprocess.Popen:
    # no pipes: the receiver outlives this process, and a write to a closed pipe would kill it
    return subprocess.Popen(
        [sys.executable, "-c", _COMMAND, "otel", "receiver"],
        env={**os.environ, "ZETA_HOME": str(home)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # a terminal's SIGINT or SIGHUP to this process group does not reach it
    )


class ReceiverSupervisor:
    def __init__(self, home: Path) -> None:
        self._home = home
        self._state = ReceiverState("starting")
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="zeta-otel-receiver", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> ReceiverState:
        return self._state

    def stop(self) -> None:
        self._stopped.set()

    def _run(self) -> None:
        failures = 0
        while not self._stopped.is_set():
            if receiver_ready(self._home):
                failures = 0
                self._state = ReceiverState("ready")
                self._stopped.wait(POLL_SECONDS)
                continue
            if failures >= MAX_LAUNCHES:
                self._state = ReceiverState("error", f"it did not start after {MAX_LAUNCHES} tries; run `zeta otel receiver` to see why")
                self._stopped.wait(POLL_SECONDS)
                continue
            if failures:
                self._state = ReceiverState("stopped")
                if self._stopped.wait(min(2.0 ** (failures - 1), MAX_BACKOFF_SECONDS)):
                    return
            self._state = ReceiverState("starting")
            if not self._launch_once():
                failures += 1

    def _launch_once(self) -> bool:
        """Launch one receiver and wait until some receiver of this home answers."""
        try:
            child = _launch(self._home)
        except OSError:
            return False
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline and not self._stopped.is_set():
            if receiver_ready(self._home):
                return True
            # a launch that lost the lock exits at once; keep watching, the winner may still be starting
            child.poll()
            self._stopped.wait(STARTUP_POLL_SECONDS)
        if self._stopped.is_set():
            return False
        if child.poll() is None:  # stuck before it served: free the lock for the next launch
            child.kill()
            child.wait()
        return False
