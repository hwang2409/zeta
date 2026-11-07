from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
JOB = ROOT / "scripts/ci_merge_resolution.sh"
CHECKER = ROOT / "scripts/check_merge_resolution.py"
WORKFLOW = ROOT / ".github/workflows/merge-resolution.yml"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, timeout=10)


def commit(repo: Path, message: str) -> None:
    git(repo, "add", ".")
    git(repo, "commit", "-qm", message)


def test_current_pr_body_is_used_instead_of_event_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    (repo / "src").mkdir()
    (repo / "src/a.py").write_text("def lost(): pass\n")
    commit(repo, "base")
    git(repo, "checkout", "-qb", "branch")
    (repo / "src/a.py").write_text("def branch(): pass\n")
    commit(repo, "branch")
    git(repo, "checkout", "-q", "main")
    (repo / "src/a.py").write_text("def lost(): pass\ndef added(): pass\n")
    commit(repo, "main")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "branch"], cwd=repo, check=False, timeout=10)
    git(repo, "checkout", "branch", "--", "src/a.py")
    commit(repo, "merge")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text("#!/bin/sh\nprintf '%s\\n' 'merge-resolution: src/a.py def added -> intentional'\n")
    fake_gh.chmod(0o755)

    result = subprocess.run(
        [str(JOB)],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request",
            "BASE_SHA": subprocess.check_output(["git", "rev-parse", "main^"], cwd=repo, text=True, timeout=10).strip(),
            "HEAD_SHA": "HEAD",
            "PR_NUMBER": "862",
            "FINAL_SHA": "HEAD",
            "CHECK_MERGE_RESOLUTION_SCRIPT": str(CHECKER),
            # This models the stale event payload. The job must ask gh for the body.
            "PR_BODY": "",
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_workflow_runs_body_edits_without_touching_main_ci() -> None:
    workflow = WORKFLOW.read_text()
    assert "types: [opened, synchronize, reopened, edited]" in workflow
    assert "name: merge resolution" in workflow
    assert "github.event.changes.body != null" in workflow
    assert "PR_BODY:" not in workflow
    assert "merge-resolution:" not in Path(ROOT / ".github/workflows/ci.yml").read_text()
