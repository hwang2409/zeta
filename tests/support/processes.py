from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


def wait_for_path_command(
    path: Path,
    *,
    owner_pid: int | None = None,
    timeout: float = 120,
) -> str:
    """Build a marker wait that stops when its test owner exits."""

    owner_pid = os.getpid() if owner_pid is None else owner_pid
    source = f"""
import os
import pathlib
import time

path = pathlib.Path({str(path)!r})
owner_pid = {owner_pid}
deadline = time.monotonic() + {timeout!r}
while not path.exists():
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        raise SystemExit(1)
    if time.monotonic() >= deadline:
        raise SystemExit(1)
    time.sleep(0.1)
""".strip()
    return shlex.join((sys.executable, "-c", source))
