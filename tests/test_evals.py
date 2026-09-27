from pathlib import Path

import pytest

from evals.run import _check, _file


def test_eval_paths_stay_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace / "leak").symlink_to(outside)

    with pytest.raises(ValueError):
        _file(workspace, "../outside.txt")
    with pytest.raises(ValueError):
        _file(workspace, "leak")


def test_eval_grades_artifacts_not_model_claims(tmp_path: Path) -> None:
    (tmp_path / "result.txt").write_text("correct\n")
    assert _check(tmp_path, {}, {"path": "result.txt", "equals": "correct\n"}) is None
    assert _check(tmp_path, {}, {"path": "result.txt", "equals": "wrong\n"})
    assert _check(tmp_path, {}, {"path": "missing.txt"}) == "missing file: missing.txt"
    assert _check(tmp_path, {}, {"path": "result.txt", "nonempty_lines": ["correct"]}) is None
