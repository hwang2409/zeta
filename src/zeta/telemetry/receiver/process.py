"""``zeta otel receiver``: own the home's receiver lock, bind the fixed port, and serve."""

from __future__ import annotations

import fcntl
import http.client
import json
import math
import os
import re
import secrets
import socket
import sqlite3
import stat
import sys
from pathlib import Path

HOST = "127.0.0.1"
PORT = 4323
AUTHORITY = f"{HOST}:{PORT}"
URL = f"http://{AUTHORITY}/"
OTLP_URL = f"{URL}v1/traces"
HEALTH_TIMEOUT_SECONDS = 1.0
IDLE_EXIT_SECONDS = 15 * 60  # an auto-started receiver with no ingest and no /health for this long exits
_HOME_ID = re.compile(r"[0-9a-f]{32}\Z")


def run_receiver(home: Path, idle_exit: bool = False) -> int:
    """Startup order: home id, lock, bind, migrate, serve. Returns the process exit code.

    With ``idle_exit`` (auto-started receivers), it exits cleanly after ``IDLE_EXIT_SECONDS``
    without an ingest or ``/health`` request; every supervisor checks health every 10 seconds.
    """
    try:
        import uvicorn  # noqa: F401 - fail before taking the lock when the extra is missing

        from . import app  # noqa: F401
    except ModuleNotFoundError:
        print("zeta otel receiver: OpenTelemetry support is not installed; install zeta[otel]", file=sys.stderr)
        return 1
    directory = home / "telemetry"
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        home_id = _home_id(directory)
        lock = os.open(directory / "receiver.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except (OSError, RuntimeError) as exc:
        print(f"zeta otel receiver: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if _running_home_id(HOST, PORT) == home_id:
                print(f"already running at http://{AUTHORITY}")
            else:
                print(f"zeta otel receiver: another receiver holds {directory / 'receiver.lock'} but is not answering", file=sys.stderr)
            return 1
        try:
            listener = socket.create_server((HOST, PORT))
        except OSError as exc:
            print(f"zeta otel receiver: cannot listen on {AUTHORITY}: {exc.strerror or exc}", file=sys.stderr)
            return 1
        with listener:
            return _serve(listener, directory, home_id, IDLE_EXIT_SECONDS if idle_exit else math.inf)
    finally:
        os.close(lock)  # closing the descriptor releases the lock


def _serve(listener: socket.socket, directory: Path, home_id: str, idle_seconds: float) -> int:
    import uvicorn

    from .app import IdleExitServer, IdleTimer, create_app

    idle = IdleTimer(idle_seconds)
    try:
        app = create_app(directory, home_id, idle)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"zeta otel receiver: cannot open the telemetry database: {exc}", file=sys.stderr)
        return 1
    print(f"zeta otel receiver: {OTLP_URL}", flush=True)  # the listener already accepts into its backlog
    server = IdleExitServer(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on"), idle)
    server.run(sockets=[listener])
    return 0


def _running_home_id(host: str, port: int) -> str | None:
    """The home id a live receiver reports, or ``None`` when health is unavailable."""
    connection = http.client.HTTPConnection(host, port, timeout=HEALTH_TIMEOUT_SECONDS)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        if response.status != 200:
            return None
        value = json.loads(response.read(4096)).get("home_id")
    except (OSError, ValueError, AttributeError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    return value if isinstance(value, str) else None


def receiver_ready(home: Path) -> bool:
    """Whether the receiver on the fixed port answers with this home's id; blocks up to the health timeout."""
    try:
        home_id = _read_home_id(home / "telemetry" / "home-id")
    except (OSError, RuntimeError):
        return False
    return _running_home_id(HOST, PORT) == home_id


def _home_id(directory: Path) -> str:
    """A random, non-secret id for this zeta home, created once with mode 0600."""
    path = directory / "home-id"
    try:
        return _read_home_id(path)
    except FileNotFoundError:
        pass
    temporary = directory / f".home-id.{os.getpid()}.{secrets.token_hex(4)}"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(secrets.token_hex(16) + "\n")
    try:
        os.link(temporary, path)  # atomic: a concurrent first start keeps exactly one id
    except FileExistsError:
        pass
    finally:
        temporary.unlink()
    return _read_home_id(path)


def _read_home_id(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise RuntimeError(f"{path} is not a regular file")
        value = file.read(64).strip()
    if not _HOME_ID.fullmatch(value):
        raise RuntimeError(f"{path} does not hold a receiver home id")
    return value
