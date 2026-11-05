from __future__ import annotations

import asyncio
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


async def read_pid_file(path: Path, *, timeout: float = 10.0) -> int:
    """Wait until a child has written its pid, not just created the file.

    A child that runs ``Path(p).write_text(pid)`` creates the file before it
    writes, so ``path.exists()`` alone can see an empty file.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            text = path.read_text().split()
            if text:
                return int(text[0])
        except (FileNotFoundError, ValueError):
            pass
        if loop.time() >= deadline:
            raise AssertionError(f"no pid written to {path} within {timeout}s")
        await asyncio.sleep(0.01)
