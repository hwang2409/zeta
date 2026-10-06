"""Trusted hidden-grading helpers for the persistent-memory benchmark."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

MEMORY_ROOT = Path(__file__).resolve().parent
_GRADING_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Grade:
    """The executable result of one hidden grading phase."""

    passed: bool
    passed_tests: int
    total_tests: int
    error: str | None
    wall_seconds: float


def _test_node_ids(grader: Path) -> list[str]:
    node_ids: list[str] = []
    for path in sorted(grader.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        node_ids.extend(
            f"{path.name}::{node.name}"
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    return node_ids


def _passed_count(output: str) -> int:
    match = re.search(r"(?:^|\s)(\d+) passed(?:[,.\s]|$)", output)
    return int(match.group(1)) if match else 0


def grade_workspace(workspace: Path, grader: Path) -> Grade:
    """Run grader-owned tests without exposing their path or source to the agent."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="zeta-memory-grader-") as temporary:
        grader_copy = Path(temporary) / "grader"
        shutil.copytree(grader, grader_copy)
        node_ids = [str(grader_copy / item) for item in _test_node_ids(grader_copy)]
        if not node_ids:
            return Grade(False, 0, 0, "grader has no tests", time.monotonic() - started)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(workspace), env.get("PYTHONPATH")) if part
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *node_ids],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=_GRADING_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Grade(
                False,
                0,
                len(node_ids),
                "grader timed out",
                time.monotonic() - started,
            )
    passed_tests = _passed_count(result.stdout)
    error = None if result.returncode == 0 else f"grader exited {result.returncode}"
    return Grade(
        result.returncode == 0 and passed_tests == len(node_ids),
        passed_tests,
        len(node_ids),
        error,
        time.monotonic() - started,
    )
