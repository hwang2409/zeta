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


def test_dropped_membership_element_flagged(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text(
        'EVENTS = {"CHILD_MESSAGE_DELIVERED"}\n'
        "def check(event):\n"
        "    return event.type in EVENTS\n"
    )
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"CHILD_MESSAGE_DELIVERED"}\n'
        "def check(event):\n"
        '    return event.type in ("CONFIG_CHANGED", "CHILD_MESSAGE_DELIVERED")\n'
    )
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"CHILD_MESSAGE_DELIVERED", "MAIN"}\n'
        "def check(event):\n"
        "    return event.type in EVENTS\n"
    )
    commit(r, "main")
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text(
        'EVENTS = {"CHILD_MESSAGE_DELIVERED"}\n'
        "def check(event):\n"
        "    return event.type in EVENTS\n"
    )
    commit(r, "bad merge")
    result = run(r)
    assert result.returncode == 1
    assert "CONFIG_CHANGED" in result.stdout
    assert "check|event.type" in result.stdout


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


def test_pr_body_explains_dropped_symbol(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def lost(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def branch(): pass\n")
    commit(r, "branch")
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
    commit(r, "merge")
    body = tmp_path / "pr-body"
    body.write_text("merge-resolution: src/a.py def added -> intentional\n")

    assert run(r).returncode == 1
    explained = run(r, "--pr-body-file", str(body))
    assert explained.returncode == 0
    assert "clean merge" in explained.stdout


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


def _set_origin_main(r: Path) -> None:
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "main"], cwd=r, check=True
    )


def test_collection_lookup_is_local_for_correct_two_file_merge(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text(
        'EVENTS = {"A"}\ndef check(event): return event in EVENTS\n'
    )
    (r / "src/z.py").write_text('EVENTS = {"Z"}\n')
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/z.py").unlink()
    commit(r, "branch deletes z")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"A", "B"}\ndef check(event): return event in EVENTS\n'
    )
    (r / "src/z.py").write_text('EVENTS = {"Z", "C"}\n')
    commit(r, "main adds a and changes z")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/z.py").unlink()
    commit(
        r,
        "correct merge deletes z\n\nmerge-resolution: src/z.py collection element EVENTS -> intentional",
    )
    assert run(r).returncode == 0


def test_membership_imports_resolve_and_report_only_missing_element(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text(
        "from . import render\ndef check(event): return event in render.EVENTS\n"
    )
    (r / "src/render.py").write_text('EVENTS = {"A"}\n')
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/render.py").write_text('EVENTS = {"A", "B"}\n')
    commit(r, "branch adds b")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text('def check(event): return event in ("A", "B")\n')
    commit(r, "main inlines events")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=True
    )
    (r / "src/a.py").write_text(
        "from . import render\ndef check(event): return event in render.EVENTS\n"
    )
    (r / "src/render.py").write_text('EVENTS = {"A"}\n')
    commit(r, "bad merge loses b")
    result = run(r)
    assert result.returncode == 1
    assert "element \"'B'\"" in result.stdout
    assert "element \"'A'\"" not in result.stdout


def test_collection_lookup_cannot_hide_two_file_loss(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text(
        'EVENTS = {"A"}\ndef check(event): return event in EVENTS\n'
    )
    (r / "src/z.py").write_text('EVENTS = {"Z"}\n')
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/z.py").write_text('EVENTS = {"Z", "B"}\n')
    commit(r, "branch adds b")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"A", "B"}\ndef check(event): return event in EVENTS\n'
    )
    commit(r, "main adds a")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "checkout", "-q", "branch", "--", "src/a.py"], cwd=r, check=True
    )
    commit(r, "bad merge")
    result = run(r)
    assert result.returncode == 1
    assert "src/a.py" in result.stdout
    assert "B" in result.stdout


def test_renamed_function_owns_no_merge_element_finding(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text(
        'EVENTS = {"A"}\ndef old(event): return event in EVENTS\n'
    )
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"A"}\ndef new(event): return event in EVENTS\n'
    )
    commit(r, "rename")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        'EVENTS = {"A", "B"}\ndef old(event): return event in EVENTS\n'
    )
    commit(r, "main adds b")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text(
        'EVENTS = {"A", "B"}\ndef new(event): return event in EVENTS\n'
    )
    commit(r, "clean rename merge")
    assert run(r).returncode == 0


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


def _make_conflict_merge(r: Path) -> str:
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "b"], cwd=r, check=True)
    (r / "src/a.py").write_text("def branch(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def main(): pass\n")
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "b"], cwd=r, check=False)
    (r / "src/a.py").write_text(
        "def main(): pass\n<<<<<<< HEAD\ndef branch(): pass\n>>>>>>> b\n"
    )
    commit(r, "unresolved")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()


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


def test_moved_definition_is_info_against_final_head(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def kept(): pass\ndef branch(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def kept(): pass\ndef moved(): pass\n")
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "merge")
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    (r / "src/b.py").write_text("def moved(): pass\n")
    commit(r, "move later")
    final = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    result = run(r, merge, "--final", final)
    assert result.returncode == 0
    assert "moved/restored later: src/b.py:" in result.stdout


def test_definition_absent_from_final_head_still_fails(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def kept(): pass\ndef branch(): pass\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def kept(): pass\ndef lost(): pass\n")
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text("def kept(): pass\n")
    commit(r, "merge")
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    result = run(r, merge, "--final", merge)
    assert result.returncode == 1
    assert "lost" in result.stdout


def test_historical_conflict_fixed_later_is_info(tmp_path: Path):
    r = repo(tmp_path)
    merge = _make_conflict_merge(r)
    (r / "src/a.py").write_text("def main(): pass\ndef branch(): pass\n")
    commit(r, "fix conflict")
    final = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    result = run(r, merge, "--final", final)
    assert result.returncode == 0
    assert "fixed later" in result.stdout


def test_final_conflict_marker_still_fails(tmp_path: Path):
    r = repo(tmp_path)
    merge = _make_conflict_merge(r)
    result = run(r, merge, "--final", merge)
    assert result.returncode == 2
    assert "merge does not parse:" in result.stderr


def test_historical_conflict_marker_still_in_final_fails(tmp_path: Path):
    r = repo(tmp_path)
    merge = _make_conflict_merge(r)
    (r / "src/other.py").write_text("value = 1\n")
    commit(r, "later change without fixing marker")
    final = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    result = run(r, merge, "--final", final)
    assert result.returncode == 2
    assert "merge does not parse:" in result.stderr


def test_parse_diagnostic_can_be_allowed(tmp_path: Path):
    r = repo(tmp_path)
    merge = _make_conflict_merge(r)
    body = r / "body.txt"
    body.write_text("merge-resolution: src/a.py parse -> intentional\n")
    result = run(r, merge, "--pr-body-file", str(body))
    assert result.returncode == 0
    assert "allowed merge parse diagnostic" in result.stdout


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
    assert "pass --main-sha" in result.stderr


def test_main_sha_override_and_removed_parent_flag(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("x = 1\n")
    commit(r, "base")
    subprocess.run(["git", "branch", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("x = 2\ndef main_only(): pass\n")
    commit(r, "main change")
    main = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", main], cwd=r, check=True
    )
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=r, check=True)
    (r / "src/b.py").write_text("x = 3\n")
    commit(r, "branch change")
    merge, p1, p2 = make_merge(r, into="branch", from_branch="main", restore="src/a.py")
    default = run(r, merge)
    assert default.returncode == 1
    assert f"main parent: {p2}" in default.stdout
    assert "main_only" in default.stdout
    wrong_side = run(r, merge, "--main-sha", p1)
    assert wrong_side.returncode == 2
    assert "disagrees with the identified main parent" in wrong_side.stderr
    correct = run(r, merge, "--main-sha", p2)
    assert correct.returncode == 1
    assert f"main parent: {p2}" in correct.stdout
    assert "main_only" in correct.stdout
    abbreviated = run(r, merge, "--main-sha", p2[:8])
    assert abbreviated.returncode == 2
    assert "full 40-hex parent sha" in abbreviated.stderr
    wrong = run(r, merge, "--main-sha", "HEAD~2")
    assert wrong.returncode == 2
    assert "full 40-hex parent sha" in wrong.stderr
    removed = run(r, merge, "--main-parent", "1")
    assert removed.returncode == 2
    assert "unrecognized arguments" in removed.stderr
    assert p1 != p2


def test_both_qualifying_parents_are_rejected(tmp_path: Path):
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("x = 1\n")
    commit(r, "base")
    subprocess.run(["git", "branch", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("x = 2\n")
    commit(r, "main change")
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=r, check=True)
    (r / "src/b.py").write_text("x = 3\n")
    commit(r, "branch change")
    merge, _p1, _p2 = make_merge(r, into="branch", from_branch="main")
    subprocess.run(
        ["git", "checkout", "-qb", "mainline", f"{merge}^1^"], cwd=r, check=True
    )
    (r / "src/c.py").write_text("x = 4\n")
    commit(r, "mainline change")
    subprocess.run(["git", "merge", "--no-ff", "--no-edit", merge], cwd=r, check=True)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=r, check=True
    )
    result = run(r, merge)
    assert result.returncode == 2
    assert "both parents qualify" in result.stderr
