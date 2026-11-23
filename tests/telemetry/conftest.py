from __future__ import annotations

import fcntl
import os
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def builtin_port() -> Iterator[None]:
    """Own 127.0.0.1:4323 across xdist workers: tests that start real receivers take turns."""
    descriptor = os.open(Path(tempfile.gettempdir()) / "zeta-tests-receiver-4323.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)  # the holder's exit always releases it, also on a crash
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 4323)) == 0:
                pytest.skip("127.0.0.1:4323 is in use on this machine")
        yield
    finally:
        os.close(descriptor)
