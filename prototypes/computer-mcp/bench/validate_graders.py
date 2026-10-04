#!/usr/bin/env python3
"""Prove that each grader rejects untouched state and accepts its reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path[:0] = [str(BENCH), str(ROOT)]

from backend import DockerDesktopBackend
from guest import DockerGuestState, apply_reference, prepare
from tasks import TASK_BY_ID, TASKS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(TASK_BY_ID))
    args = parser.parse_args()
    tasks = [TASK_BY_ID[item] for item in args.task] if args.task else TASKS
    results = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="computer-grader-docker-") as config:
            Path(config, "config.json").write_text("{}\n")
            os.environ["ZETA_COMPUTER_DOCKER_CONFIG"] = config
            os.environ["ZETA_COMPUTER_RUN_ID"] = f"grader-{task.id}"
            backend = DockerDesktopBackend(ttl_seconds=180)
            try:
                backend.start()
                prepare(backend, task)
                guest = DockerGuestState(backend)
                untouched = task.grade(guest)
                apply_reference(backend, task)
                reference = task.grade(guest)
                passed = not untouched["pass"] and reference["pass"]
                result = {
                    "task": task.id,
                    "untouched": untouched["pass"],
                    "reference": reference["pass"],
                    "pass": passed,
                }
                results.append(result)
                print(
                    f"{task.id}: untouched={'PASS' if untouched['pass'] else 'FAIL'}; "
                    f"reference={'PASS' if reference['pass'] else 'FAIL'}"
                )
            finally:
                backend.destroy()
    print(json.dumps(results, indent=2))
    return 0 if all(item["pass"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
