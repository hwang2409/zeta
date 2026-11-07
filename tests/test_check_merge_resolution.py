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
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge")
    result = run(r)
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
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge")
    assert run(r).returncode == 1  # new is still a loss; old is not reported
    assert "old" not in run(r).stdout


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
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    commit(r, "merge-resolution: src/a.py def added -> intentional")
    assert run(r).returncode == 0


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
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    subprocess.run(["git", "checkout", "branch", "--", "src/a.py"], cwd=r, check=True)
    (r / "src/a.py").write_text("WRITER_EPOCH = 4\n")
    commit(r, "merge")
    assert run(r).returncode == 1
    assert "WRITER_EPOCH" in run(r).stdout


def test_non_merge_is_clean(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("x = 1\n")
    commit(r, "one")
    assert run(r).returncode == 0
    assert "not a two-parent merge" in run(r).stdout


def make_merge(
    r: Path, *, into: str, from_branch: str, restore: str | None = None
) -> tuple[str, str, str]:
    subprocess.run(["git", "checkout", "-q", into], cwd=r, check=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", from_branch], cwd=r, check=True
    )
    if restore is not None:
        subprocess.run(["git", "checkout", into, "--", restore], cwd=r, check=True)
    commit(r, "merge")
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
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
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
    result = run(r, merge)
    assert result.returncode == 2
    assert "merge does not parse:" in result.stderr
    assert "src/a.py:" in result.stderr
    assert "main:" not in result.stdout


def _api_repo(r: Path) -> None:
    (r / "src").mkdir()
    (r / "src/api.py").write_text("def shared(): pass\n")
    commit(r, "base")


def _merge_loss(
    r: Path, *, into: str, from_branch: str, message: str = "merge"
) -> tuple[str, str, str, str]:
    subprocess.run(["git", "checkout", "-q", into], cwd=r, check=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", from_branch], cwd=r, check=True
    )
    p1 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r, text=True).strip()
    p2 = subprocess.check_output(
        ["git", "rev-parse", "MERGE_HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "rm", "-qf", "src/main_only.py"], cwd=r, check=True)
    commit(r, message)
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    return merge, p1, p2, "main_only"


def _direct_case(
    r: Path, *, into: str = "main", message: str = "merge"
) -> tuple[str, str, str, str]:
    _api_repo(r)
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/branch.py").write_text("def branch_only(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/main_only.py").write_text("def main_only(): pass\n")
    commit(r, "main")
    main = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", main], cwd=r, check=True
    )
    return _merge_loss(
        r,
        into=into,
        from_branch="branch" if into == "main" else "main",
        message=message,
    )


def _train_case(r: Path, *, newer: bool, depth: int = 1) -> tuple[str, str, str, str]:
    _api_repo(r)
    initial = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "branch", "origin-side", initial], cwd=r, check=True)
    subprocess.run(["git", "checkout", "-q", "origin-side"], cwd=r, check=True)
    (r / "src/origin.py").write_text("def origin(): pass\n")
    commit(r, "origin")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/main.py").write_text("def main(): pass\n")
    commit(r, "main")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "origin-side"], cwd=r, check=True
    )
    commit(r, "origin merge")
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "branch", "train", base], cwd=r, check=True)
    subprocess.run(["git", "checkout", "-qb", "train-side", base], cwd=r, check=True)
    (r / "src/main_only.py").write_text("def main_only(): pass\n")
    (r / "src/train.py").write_text("def train(): pass\n")
    commit(r, "train change")
    train_change = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    if newer:
        (r / "src/newer.py").write_text("def newer(): pass\n")
        commit(r, "newer main")
        pr_base = "main"
    else:
        subprocess.run(
            ["git", "checkout", "-qb", "pr-side", initial], cwd=r, check=True
        )
        (r / "src/pr.py").write_text("def pr(): pass\n")
        commit(r, "older pr")
        pr_base = "pr-side"
    origin_main = base
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", origin_main],
        cwd=r,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "train"], cwd=r, check=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", train_change], cwd=r, check=True
    )
    commit(r, "train merge 1")
    for index in range(1, depth):
        subprocess.run(
            ["git", "checkout", "-qb", f"train-side-{index}", "HEAD"], cwd=r, check=True
        )
        (r / f"src/train{index}.py").write_text(f"def train{index}(): pass\n")
        commit(r, f"train change {index + 1}")
        subprocess.run(["git", "checkout", "-q", "train"], cwd=r, check=True)
        subprocess.run(
            ["git", "merge", "--no-ff", "--no-commit", f"train-side-{index}"],
            cwd=r,
            check=True,
        )
        commit(r, f"train merge {index + 1}")
    train_merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "checkout", "-q", pr_base], cwd=r, check=True)
    if newer:
        (r / "src/pr.py").write_text("def pr(): pass\n")
        commit(r, "newer pr")
    pr_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(["git", "checkout", "-q", "train"], cwd=r, check=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", pr_head], cwd=r, check=True
    )
    subprocess.run(["git", "rm", "-qf", "src/main_only.py"], cwd=r, check=True)
    commit(r, "train result")
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    return merge, train_merge, pr_head, "main_only"


import pytest


@pytest.mark.parametrize(
    ("shape", "expected", "refusal"),
    [
        ("pr", 0, False),
        ("main-into-branch", 1, False),
        ("train-1", 0, False),
        ("train-older", 0, False),
        ("train-newer", 0, False),
        ("both", None, True),
        ("neither", None, True),
        ("wrong", None, True),
        ("right", 1, False),
        ("update-branch", 1, False),
    ],
)
def test_parent_selection_table(
    tmp_path: Path, shape: str, expected: int | None, refusal: bool
) -> None:
    if shape.startswith("train"):
        r = repo(tmp_path)
        merge, p1, p2, loss = _train_case(
            r, newer=shape == "train-newer", depth=1 if shape == "train-1" else 2
        )
    else:
        r = repo(tmp_path)
        merge, p1, p2, loss = _direct_case(
            r,
            into="branch"
            if shape in {"main-into-branch", "right", "wrong", "update-branch"}
            else "main",
            message="Merge branch 'main' into branch"
            if shape == "update-branch"
            else "merge",
        )
    args: tuple[str, ...] = ()
    if shape == "both":
        subprocess.run(
            ["git", "checkout", "-qb", "later", f"{merge}^1^"], cwd=r, check=True
        )
        (r / "later").write_text("later\n")
        commit(r, "later")
        subprocess.run(
            ["git", "merge", "--no-ff", "--no-edit", merge], cwd=r, check=True
        )
        subprocess.run(
            ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
        )
    elif shape == "neither":
        subprocess.run(
            ["git", "checkout", "-qb", "unrelated", f"{merge}^1^"], cwd=r, check=True
        )
        (r / "unrelated").write_text("unrelated\n")
        commit(r, "unrelated")
        subprocess.run(
            ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
        )
    if shape == "wrong":
        args = ("--main-sha", p1)
        expected = None
        refusal = True
    elif shape == "right":
        args = ("--main-sha", p2)
    result = run(r, merge, *args)
    if refusal:
        assert result.returncode == 2
        if shape == "wrong":
            assert f"disagrees with the identified main parent {p2}" in result.stderr
            assert "main parent:" not in result.stdout
        else:
            assert "cannot identify exactly one main parent" in result.stderr
    else:
        assert result.returncode == 1
        chosen = (p1, p2)[expected]
        assert f"main parent: {chosen}" in result.stdout
        assert loss in result.stdout
