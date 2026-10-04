#!/usr/bin/env python3
"""MCP entry point that seeds and grades one benchmark guest."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT))

from backend import DockerDesktopBackend, Screenshot
from guest import DockerGuestState, prepare
from server import serve
from tasks import TASK_BY_ID, Task


class BenchmarkBackend:
    """Persist grading evidence after every observation, even if MCP is killed."""

    def __init__(
        self, backend: DockerDesktopBackend, task: Task, artifacts: Path
    ) -> None:
        self.backend = backend
        self.task = task
        self.artifacts = artifacts
        self.guest = DockerGuestState(backend)

    def start(self) -> None:
        self.backend.start()

    def reset(self) -> None:
        self.backend.reset()

    def destroy(self) -> None:
        self.backend.destroy()

    def input(self, action: str, arguments: dict[str, object]) -> None:
        self.backend.input(action, arguments)

    def screenshot(self) -> Screenshot:
        shot = self.backend.screenshot()
        self.persist(shot)
        return shot

    def persist(self, shot: Screenshot | None = None) -> None:
        grade = self.task.grade(self.guest)
        temporary = self.artifacts / "grade.json.tmp"
        temporary.write_text(json.dumps(grade, indent=2) + "\n")
        temporary.replace(self.artifacts / "grade.json")
        if shot is not None:
            (self.artifacts / "final.jpg").write_bytes(shot.data)


def main() -> int:
    def terminate(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    task = TASK_BY_ID[os.environ["ZETA_COMPUTER_TASK"]]
    artifacts = Path(os.environ["ZETA_COMPUTER_ARTIFACT_DIR"])
    artifacts.mkdir(parents=True, exist_ok=True)
    desktop = DockerDesktopBackend()
    benchmark = BenchmarkBackend(desktop, task, artifacts)
    try:
        desktop.start()
        prepare(desktop, task)
        benchmark.persist()
        serve(benchmark, destroy_on_exit=False)
    finally:
        if desktop.started_at is not None:
            benchmark.persist()
            try:
                benchmark.persist(desktop.screenshot())
            except RuntimeError as exc:
                (artifacts / "screenshot-error.txt").write_text(f"{exc}\n")
        desktop.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
