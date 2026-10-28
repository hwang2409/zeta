from __future__ import annotations

import shlex
import subprocess
import sys

from tests.support.processes import wait_for_path_command


def test_wait_for_path_command_exits_when_owner_is_killed(tmp_path) -> None:
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    waiter = subprocess.Popen(
        shlex.split(
            wait_for_path_command(tmp_path / "never-created", owner_pid=owner.pid)
        )
    )

    try:
        owner.kill()
        owner.wait(timeout=5)
        assert waiter.wait(timeout=5) == 1
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if waiter.poll() is None:
            waiter.kill()
            waiter.wait(timeout=5)
