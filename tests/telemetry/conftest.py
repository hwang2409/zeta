from __future__ import annotations

import fcntl
import os
import socket
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def builtin_port() -> Iterator[None]:
    """Own 127.0.0.1:4323 across xdist workers: tests that start real receivers take turns."""
    descriptor = os.open(Path(tempfile.gettempdir()) / "zeta-tests-receiver-4323.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline, "another test held 127.0.0.1:4323 for 120 s"
                time.sleep(0.1)
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 4323)) == 0:
                pytest.skip("127.0.0.1:4323 is in use on this machine")
        yield
    finally:
        os.close(descriptor)
