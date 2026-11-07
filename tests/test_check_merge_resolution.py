from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(
    os.environ.get(
        "SCRIPT", Path(__file__).parents[1] / "scripts/check_merge_resolution.py"
    )
)


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
    commit(
        r,
        "bad merge\n\nmerge-resolution: src/a.py collection element EVENTS -> intentional",
    )
    result = run(r)
    assert result.returncode == 1
    assert "src/a.py membership element check|event" in result.stdout


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


def _final_loss(
    r: Path,
    main_text: str,
    branch_text: str,
    merge_text: str,
    final_text: str | None = None,
    final_path: str = "src/a.py",
    merge_extra: tuple[str, str] | None = None,
) -> tuple[str, str]:
    (r / "src").mkdir()
    (r / "src/a.py").write_text("base = 1\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text(branch_text)
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(main_text)
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text(merge_text)
    if merge_extra is not None:
        extra_path, extra_text = merge_extra
        extra = r / extra_path
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(extra_text)
    commit(r, "merge")
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    if final_text is None:
        return merge, merge
    target = r / final_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(final_text)
    commit(r, "final")
    final = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=r, text=True
    ).strip()
    return merge, final


def _final_marker_case(r: Path, case: str) -> tuple[str, str, int, str] | None:
    if case == "fixed marker":
        merge = _make_conflict_merge(r)
        (r / "src/a.py").write_text("def main(): pass\ndef branch(): pass\n")
        commit(r, "fix")
        final = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r, text=True
        ).strip()
        return merge, final, 0, "fixed later"
    if case == "final marker":
        merge = _make_conflict_merge(r)
        return merge, merge, 2, "merge does not parse"
    if case == "parse allow-list":
        merge = _make_conflict_merge(r)
        body = r / "body.txt"
        body.write_text("merge-resolution: src/a.py parse -> intentional\n")
        return merge, body.as_posix(), 0, "allowed merge parse diagnostic"
    return None


def _final_case(r: Path, case: str) -> tuple[str, str, int, str]:
    if case == "unrelated same-name def":
        return (
            *_final_loss(
                r,
                "def helper(): return 1\n",
                "def branch(): pass\n",
                "base = 1\n",
                "def helper(): return 2\n",
                "src/b.py",
            ),
            1,
            "def helper",
        )
    if case == "true move":
        return (
            *_final_loss(
                r,
                "def helper(): return 1\n",
                "def branch(): pass\n",
                "base = 1\n",
                "def helper(): return 1\n",
                "src/b.py",
            ),
            0,
            "moved/restored later: src/b.py",
        )
    if case == "restored in place":
        return (
            *_final_loss(
                r,
                "def helper(): return 1\n",
                "def branch(): pass\n",
                "base = 1\n",
                "def helper(): return 1\n",
            ),
            0,
            "moved/restored later: src/a.py",
        )
    if case == "element unrelated":
        return (
            *_final_loss(
                r,
                'TARGET = {"lost"}\n',
                'TARGET = {"lost"}\n',
                "TARGET = set()\n",
                'OTHER = {"lost"}\n',
                "src/b.py",
            ),
            1,
            "TARGET",
        )
    if case == "element restored":
        return (
            *_final_loss(
                r,
                'TARGET = {"lost"}\n',
                'TARGET = {"lost"}\n',
                "TARGET = set()\n",
                'TARGET = {"lost"}\n',
            ),
            0,
            "moved/restored later: src/a.py",
        )
    if case == "constant different value":
        return (
            *_final_loss(r, "VALUE = 1\n", "VALUE = 2\n", "VALUE = 3\n", "VALUE = 4\n"),
            1,
            "VALUE",
        )
    if case == "constant moved same value":
        return (
            *_final_loss(
                r,
                "VALUE = 1\n",
                "VALUE = 2\n",
                "base = 1\n",
                "VALUE = 1\n",
                "src/b.py",
            ),
            0,
            "moved/restored later: src/b.py",
        )
    if case == "constant duplicate final":
        merge, final = _final_loss(
            r,
            "VALUE = 1\n",
            "VALUE = 2\n",
            "base = 1\n",
            "VALUE = 1\n",
            "src/b.py",
        )
        (r / "src/c.py").write_text("VALUE = 1\n")
        commit(r, "duplicate final constant")
        final = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r, text=True
        ).strip()
        return merge, final, 1, "VALUE"
    if case == "class moved with member":
        return (
            *_final_loss(
                r,
                "class Box:\n    value = 1\n    def method(self): return 1\n",
                "class Branch:\n    pass\n",
                "base = 1\n",
                "class Box:\n    value = 1\n    def method(self): return 1\n",
                "src/b.py",
            ),
            0,
            "moved/restored later: src/b.py",
        )
    if case == "element same-named collection moved":
        merge, final = _final_loss(
            r,
            'TARGET = {"lost"}\n',
            'TARGET = {"lost"}\n',
            "TARGET = set()\n",
            'TARGET = {"lost"}\n',
            "src/b.py",
        )
        (r / "src/a.py").unlink()
        commit(r, "move final collection")
        final = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r, text=True
        ).strip()
        return merge, final, 0, "moved/restored later: src/b.py"
    if case == "duplicate final collection":
        merge, final = _final_loss(
            r,
            'TARGET = {"lost"}\n',
            'TARGET = {"lost"}\n',
            "TARGET = set()\n",
            'TARGET = {"lost"}\n',
            "src/b.py",
        )
        (r / "src/c.py").write_text('TARGET = {"lost"}\n')
        commit(r, "duplicate final collection")
        final = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r, text=True
        ).strip()
        return merge, final, 1, "TARGET"
    if case == "dispatch restored":
        return (
            *_final_loss(
                r,
                'def route(name):\n    return name == "mcp"\n',
                "def branch(): pass\n",
                "base = 1\n",
                'def route(name):\n    return name == "mcp"\n',
            ),
            0,
            "moved/restored later: src/a.py",
        )
    if case == "dispatch other function":
        return (
            *_final_loss(
                r,
                'def route(name):\n    return name == "mcp"\n',
                "def branch(): pass\n",
                "base = 1\n",
                'def other(name):\n    return name == "mcp"\n',
            ),
            1,
            "route",
        )
    if case.startswith("checked merge collision"):
        merge, _ = _final_loss(
            r,
            "def helper(): return 1\n",
            "def branch(): pass\n",
            "base = 1\n",
            merge_extra=("src/b.py", "def helper(): return 1\n"),
        )
        final = {
            "checked merge collision full": merge,
            "checked merge collision short": merge[:7],
            "checked merge collision HEAD": "HEAD",
        }[case]
        return merge, final, 1, "def helper"
    if case == "checked merge":
        return (
            *_final_loss(
                r, "def helper(): return 1\n", "def branch(): pass\n", "base = 1\n"
            ),
            1,
            "def helper",
        )
    marker_case = _final_marker_case(r, case)
    if marker_case is not None:
        return marker_case
    raise AssertionError(f"unknown final case: {case}")


@pytest.mark.parametrize(
    ("case", "expected", "needle"),
    [
        ("unrelated same-name def", 1, "def helper"),
        ("true move", 0, "moved/restored later: src/b.py"),
        ("restored in place", 0, "moved/restored later: src/a.py"),
        ("element unrelated", 1, "TARGET"),
        ("element restored", 0, "moved/restored later: src/a.py"),
        ("constant different value", 1, "VALUE"),
        ("constant moved same value", 0, "moved/restored later: src/b.py"),
        ("constant duplicate final", 1, "VALUE"),
        ("class moved with member", 0, "moved/restored later: src/b.py"),
        ("element same-named collection moved", 0, "moved/restored later: src/b.py"),
        ("duplicate final collection", 1, "TARGET"),
        ("dispatch restored", 0, "moved/restored later: src/a.py"),
        ("dispatch other function", 1, "route"),
        ("checked merge collision full", 1, "def helper"),
        ("checked merge collision short", 1, "def helper"),
        ("checked merge collision HEAD", 1, "def helper"),
        ("checked merge", 1, "def helper"),
        ("fixed marker", 0, "fixed later"),
        ("final marker", 2, "merge does not parse"),
        ("parse allow-list", 0, "allowed merge parse diagnostic"),
    ],
)
def test_final_head_identity_table(
    tmp_path: Path, case: str, expected: int, needle: str
) -> None:
    r = repo(tmp_path)
    merge, final_or_body, expected_code, expected_line = _final_case(r, case)
    assert expected_code == expected
    args = (
        ("--pr-body-file", final_or_body)
        if case == "parse allow-list"
        else ("--final", final_or_body)
    )
    result = run(r, merge, *args)
    output = result.stdout + result.stderr
    assert result.returncode == expected
    assert expected_line in output
    if case.startswith("checked merge collision"):
        assert "moved/restored later" not in output


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
    origin_main = base
    if newer:
        (r / "src/newer.py").write_text("def newer(): pass\n")
        commit(r, "newer main")
        pr_base = "main"
        origin_main = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r, text=True
        ).strip()
    else:
        subprocess.run(
            ["git", "checkout", "-qb", "pr-side", initial], cwd=r, check=True
        )
        (r / "src/pr.py").write_text("def pr(): pass\n")
        commit(r, "older pr")
        pr_base = "pr-side"
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


@pytest.mark.parametrize("case", ["lost line", "renamed owner"])
def test_pr_side_line_table(tmp_path: Path, case: str) -> None:
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def check():\n    return 1\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    added = 2\n    return 1\n")
    commit(r, "branch addition")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text(
        "def renamed():\n    return 1\n" if case == "renamed owner" else "def check():\n    return 3\n"
    )
    commit(r, "main edit")
    _set_origin_main(r)
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False
    )
    (r / "src/a.py").write_text(
        "def renamed():\n    added = 2\n    return 1\n"
        if case == "renamed owner"
        else "def check():\n    return 3\n"
    )
    commit(r, "merge")
    result = run(r)
    assert (result.returncode == 0) is (case == "renamed owner")
    if case == "lost line":
        assert "pr-line" in result.stdout



def test_pr_line_count_subtracts_base_and_main_copies(tmp_path: Path) -> None:
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def check():\n    emit()\n    anchor()\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    emit()\n    anchor()\n    emit()\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    anchor()\n")
    commit(r, "main removes base copy")
    _set_origin_main(r)
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False)
    (r / "src/a.py").write_text("def check():\n    anchor()\n    emit()\n")
    commit(r, "merge keeps pr copy")
    result = run(r)
    assert result.returncode == 0
    assert "pr-line" not in result.stdout


def test_pr_line_final_exact_restore_and_modified_restore(tmp_path: Path) -> None:
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def check():\n    return 1\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    added = 2\n    return 1\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    return 3\n")
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False)
    (r / "src/a.py").write_text("def check():\n    return 3\n")
    commit(r, "lost merge")
    merge = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r, text=True).strip()
    (r / "src/a.py").write_text("def check():\n    added = 2\n    return 3\n")
    commit(r, "exact restoration")
    exact = run(r, merge, "--final", "HEAD")
    assert exact.returncode == 0
    (r / "src/a.py").write_text("def check():\n    added = 999\n    return 3\n")
    commit(r, "modified restoration")
    modified = run(r, merge, "--final", "HEAD")
    assert modified.returncode == 1
    assert "pr-line" in modified.stdout


def test_pr_line_group_marker_requires_current_digest(tmp_path: Path) -> None:
    r = repo(tmp_path)
    (r / "src").mkdir()
    (r / "src/a.py").write_text("def check():\n    return 1\n")
    commit(r, "base")
    subprocess.run(["git", "checkout", "-qb", "branch"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    added = 2\n    return 1\n")
    commit(r, "branch")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    (r / "src/a.py").write_text("def check():\n    return 3\n")
    commit(r, "main")
    _set_origin_main(r)
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=r, check=False)
    (r / "src/a.py").write_text("def check():\n    return 1\n")
    commit(r, "merge")
    result = run(r)
    line = next(line for line in result.stdout.splitlines() if ": pr-line " in line)
    name = line.split("pr-line ", 1)[1].split(":", 1)[0]
    owner, digest = name.rsplit("@", 1)
    marker = next(line for line in result.stdout.splitlines() if line.startswith("suggested override: "))
    printed = marker.removeprefix("suggested override: ")
    assert printed.startswith(f"merge-resolution: src/a.py pr-lines {owner}@")
    group = hashlib.sha256(f"{name}=added = 2".encode()).hexdigest()[:12]
    assert printed.split("@", 1)[1].split(" ", 1)[0] == group
    body = tmp_path / "body"
    body.write_text(f"{printed[:-8]}intentional\n")
    assert run(r, "--pr-body-file", str(body)).returncode == 0
    body.write_text(f"merge-resolution: src/a.py pr-lines {owner}@{digest} -> stale\n")
    assert run(r, "--pr-body-file", str(body)).returncode == 1


@pytest.mark.parametrize(
    ("shape", "expected", "refusal"),
    [
        ("pr", 0, False),
        ("main-into-branch", 1, False),
        ("train-1", 0, False),
        ("train-older", 0, False),
        ("train-newer", 0, False),
        ("train-newer-wrong", None, True),
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
            r,
            newer=shape in {"train-newer", "train-newer-wrong"},
            depth=1 if shape == "train-1" else 2,
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
    if shape == "train-newer-wrong":
        args = ("--main-sha", p2)
    elif shape == "wrong":
        # guard: this direct-shape refusal already held on 707549ed.
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
        elif shape == "train-newer-wrong":
            assert f"disagrees with the identified main parent {p1}" in result.stderr
            assert "main parent:" not in result.stdout
        else:
            assert "cannot identify exactly one main parent" in result.stderr
    else:
        assert result.returncode == 1
        chosen = (p1, p2)[expected]
        assert f"main parent: {chosen}" in result.stdout
        assert loss in result.stdout
