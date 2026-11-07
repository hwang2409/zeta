from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/check_merge_resolution.py"


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(SCRIPT), *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    return tmp_path


def commit(r: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=r, check=True)


def test_dropped_main_function_flagged(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def kept(): pass\ndef lost(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        "def kept(): pass\ndef lost(): pass\ndef newer(): pass\n"
    )
    commit(r, "main")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge")
    result = run(r, "--main-parent", "1")
    assert result.returncode == 1
    assert "newer" in result.stdout


def test_branch_side_intended_deletion_not_flagged(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def old(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("")
    commit(r, "delete")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def old(): pass\ndef new(): pass\n")
    commit(r, "add")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge")
    assert (
        run(r, "--main-parent", "1").returncode == 1
    )  # new is still a loss; old is not reported
    assert "old" not in run(r, "--main-parent", "1").stdout


def test_allowed_drop_passes(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def lost(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def lost(): pass\ndef added(): pass\n")
    commit(r, "main")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge-resolution: src/a.py def added -> intentional")
    assert run(r, "--main-parent", "1").returncode == 0


def test_new_constant_value_flagged(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("WRITER_EPOCH = 1\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("WRITER_EPOCH = 2\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("WRITER_EPOCH = 3\n")
    commit(r, "main")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    (r / "src/a.py").write_text("WRITER_EPOCH = 4\n")
    commit(r, "merge")
    assert run(r, "--main-parent", "1").returncode == 1
    assert "WRITER_EPOCH" in run(r, "--main-parent", "1").stdout


def test_non_merge_is_clean(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("x = 1\n")
    commit(r, "one")
    assert run(r).returncode == 0
    assert "not a two-parent merge" in run(r).stdout


def make_merge(r: Path, *, into: str, from_branch: str) -> tuple[str, str, str]:
    subprocess.run(["git", "checkout", "-q", into], cwd=r, check=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-edit", from_branch], cwd=r, check=True
    )
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    parents = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", merge], cwd=r, text=True
    ).split()
    return merge, parents[1], parents[2]


def test_unresolved_conflict_is_one_fatal_diagnostic(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def branch(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def main(): pass\n")
    commit(r, "main")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text(
        "def main(): pass\n<<<<<<< HEAD\ndef branch(): pass\n>>>>>>> branch\n"
    )
    commit(r, "unresolved")
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    result = run(r, merge, "--main-parent", "1")
    assert result.returncode == 2
    assert "merge does not parse:" in result.stderr
    assert "src/a.py:" in result.stderr
    assert "main:" not in result.stdout


def test_parent_selection_reports_pr_merge(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/base.py").write_text("x = 1\n")
    commit(r, "base")
    subprocess.run(["git", "branch", "branch"], cwd=r, check=True)
    (r / "src/main.py").write_text("x = 2\n")
    commit(r, "main change")
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=r, check=True)
    (r / "src/branch.py").write_text("x = 3\n")
    commit(r, "branch change")
    merge, p1, _p2 = make_merge(r, into="main", from_branch="branch")
    result = run(r, merge)
    assert result.returncode == 0
    assert f"main parent: {p1}" in result.stdout
    assert "origin/main first-parent ancestry" in result.stdout


def test_parent_selection_reports_main_into_branch(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/base.py").write_text("x = 1\n")
    commit(r, "base")
    subprocess.run(["git", "branch", "branch"], cwd=r, check=True)
    (r / "src/main.py").write_text("x = 2\n")
    commit(r, "main change")
    main = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", main], cwd=r, check=True
    )
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=r, check=True)
    (r / "src/branch.py").write_text("x = 3\n")
    commit(r, "branch change")
    merge, p1, _p2 = make_merge(r, into="branch", from_branch="main")
    result = run(r, merge)
    assert result.returncode == 0
    assert f"main parent: {main}" in result.stdout
    assert f"1={p1}" in result.stdout


def test_parent_selection_requires_flag_when_ambiguous(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/base.py").write_text("x = 1\n")
    commit(r, "base")
    subprocess.run(["git", "branch", "branch"], cwd=r, check=True)
    (r / "src/main.py").write_text("x = 2\n")
    commit(r, "main change")
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=r, check=True)
    (r / "src/branch.py").write_text("x = 3\n")
    commit(r, "branch change")
    merge, _p1, _p2 = make_merge(r, into="branch", from_branch="main")
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD~2"], cwd=r, check=True
    )
    result = run(r, merge)
    assert result.returncode == 2
    assert "use --main-parent" in result.stderr
