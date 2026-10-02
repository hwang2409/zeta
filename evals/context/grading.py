"""Trusted grading helpers for the context-management benchmark."""

from __future__ import annotations

import ast
import shutil
import tempfile
from pathlib import Path


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


from evals.run import _check

CONTEXT_ROOT = Path(__file__).resolve().parent
_TIMEOUT_FAILURE = "command timed out: python"
_MAX_GRADING_ATTEMPTS = 3


def grade_workspace(
    workspace: Path, grader: Path, expected_passes: int
) -> tuple[bool, str | None]:
    """Run grader-owned pytest tests without placing them in the candidate tree."""
    with tempfile.TemporaryDirectory(prefix="zeta-context-grader-") as temporary:
        grader_copy = Path(temporary) / "grader"
        shutil.copytree(grader, grader_copy)
        for _attempt in range(_MAX_GRADING_ATTEMPTS):
            failure = _check(
                workspace,
                {},
                {
                    "command": ["python", "-m", "pytest", "-q"],
                    "expected_passes": expected_passes,
                    "expected_skips": 0,
                    "expected_node_ids": _test_node_ids(grader),
                },
                command_root=workspace,
                grader_root=grader_copy,
            )
            if failure != _TIMEOUT_FAILURE:
                break
    return failure is None, failure


def apply_overlay(workspace: Path, overlay: Path) -> None:
    """Apply a grader-owned reference patch to a fixture copy."""
    shutil.copytree(overlay, workspace, dirs_exist_ok=True)
