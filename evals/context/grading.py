"""Trusted grading helpers for the context-management benchmark."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _test_node_ids(grader: Path) -> list[str]:
    node_ids = []
    for path in sorted(grader.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        node_ids.extend(
            f"{path.name}::{node.name}"
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    return node_ids


CONTEXT_ROOT = Path(__file__).resolve().parent
_TIMEOUT_FAILURE = "grader timed out"
_MAX_GRADING_ATTEMPTS = 3
_GRADING_TIMEOUT_SECONDS = 120


def grade_workspace(
    workspace: Path,
    grader: Path,
    expected_passes: int,
    attempts: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """Run grader-owned pytest tests without placing them in the candidate tree."""
    with tempfile.TemporaryDirectory(prefix="zeta-context-grader-") as temporary:
        grader_copy = Path(temporary) / "grader"
        shutil.copytree(grader, grader_copy)
        node_ids = [
            str(grader_copy / node_id) for node_id in _test_node_ids(grader_copy)
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(workspace / "src"),
                str(workspace),
                env.get("PYTHONPATH"),
            )
            if part
        )
        for attempt_number in range(1, _MAX_GRADING_ATTEMPTS + 1):
            started = time.monotonic()
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
                failure = None
                if result.returncode != 0:
                    failure = f"grader exited {result.returncode}"
                elif f"{expected_passes} passed" not in result.stdout:
                    failure = (
                        "grader pass count differed: "
                        f"expected {expected_passes}"
                    )
            except subprocess.TimeoutExpired:
                failure = _TIMEOUT_FAILURE
            timed_out = failure == _TIMEOUT_FAILURE
            if attempts is not None:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "elapsed_seconds": time.monotonic() - started,
                        "timed_out": timed_out,
                    }
                )
            if not timed_out:
                break
    return failure is None, failure


def apply_overlay(workspace: Path, overlay: Path) -> None:
    """Apply a grader-owned reference patch to a fixture copy."""
    shutil.copytree(overlay, workspace, dirs_exist_ok=True)
