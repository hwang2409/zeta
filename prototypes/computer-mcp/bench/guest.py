"""Guest setup and grading operations for the desktop benchmark."""

from __future__ import annotations

import shlex
import time

from backend import DockerDesktopBackend
from tasks import Task


class DockerGuestState:
    """Read grader state through docker exec, never through a host mount."""

    def __init__(self, backend: DockerDesktopBackend) -> None:
        self.backend = backend

    def read(self, path: str) -> bytes | None:
        result = self.backend.container_exec(
            "sh",
            "-c",
            f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}",
            check=False,
        )
        return result if result or self._is_empty_file(path) else None

    def _is_empty_file(self, path: str) -> bool:
        output = self.backend.container_exec(
            "sh", "-c", f"test -f {shlex.quote(path)} && printf yes", check=False
        )
        return output == b"yes"

    def is_dir(self, path: str) -> bool:
        output = self.backend.container_exec(
            "sh", "-c", f"test -d {shlex.quote(path)} && printf yes", check=False
        )
        return output == b"yes"


def prepare(backend: DockerDesktopBackend, task: Task) -> None:
    """Copy one task fixture into tmpfs and start its visible applications."""
    fixture_target = f"/tmp/bench-{task.id}"
    backend.copy_to(task.fixture, fixture_target)
    backend.container_exec(
        "sh",
        "-c",
        f"cp -R {shlex.quote(fixture_target)}/. /home/zeta/ && "
        "mkdir -p /home/zeta/Documents /home/zeta/notes /home/zeta/.config",
    )
    for index, command in enumerate(task.launch):
        log = f"/tmp/bench-launch-{index}.log"
        backend.container_exec("sh", "-c", f"({command}) >{log} 2>&1 &", check=True)
        if command.startswith("python3 "):
            time.sleep(0.4)
        else:
            time.sleep(3 if command.startswith("chromium ") else 1.5)
            backend.container_exec(
                "sh",
                "-c",
                "DISPLAY=:99 xdotool getactivewindow windowmaximize",
                check=False,
            )


def apply_reference(backend: DockerDesktopBackend, task: Task) -> None:
    """Apply the deterministic reference solution through docker exec."""
    for command in task.reference_commands:
        backend.container_exec("sh", "-c", command)
