from __future__ import annotations

import os
import re
import subprocess
import textwrap
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


def _merge_resolution_command(tmp_path: Path) -> list[str]:
    workflows = list((ROOT / ".github/workflows").glob("*.y*ml"))
    workflow = next(
        path for path in workflows if "name: merge resolution" in path.read_text()
    )
    text = workflow.read_text()
    script = ROOT / "scripts/ci_merge_resolution.sh"
    if script.exists():
        return [str(script)]

    job = text.split("name: merge resolution", 1)[1]
    run = re.search(r"^        run: \|\n((?:^          .*\n?)+)", job, re.MULTILINE)
    assert run is not None
    extracted = textwrap.dedent(run.group(1))
    command = tmp_path / "merge-resolution.sh"
    command.write_text(extracted)
    command.chmod(0o755)
    return ["bash", str(command)]


def test_current_pr_body_is_used_instead_of_event_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    git(repo, "remote", "add", "origin", str(repo))
    (repo / "src").mkdir()
    (repo / "src/a.py").write_text("def lost(): pass\n")
    (repo / "scripts").mkdir()
    (repo / "scripts/check_merge_resolution.py").write_text(CHECKER.read_text())
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
        _merge_resolution_command(tmp_path),
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request",
            "BASE_SHA": subprocess.check_output(["git", "rev-parse", "main^"], cwd=repo, text=True, timeout=10).strip(),
            "HEAD_SHA": "HEAD",
            "PR_NUMBER": "862",
            "FINAL_SHA": "HEAD",
            # This models the stale event payload. The job must ask gh for the body.
            "PR_BODY": "",
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pull_request_checks_only_first_parent_merges(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    git(repo, "remote", "add", "origin", str(repo))
    git(repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
    (repo / "src").mkdir()
    (repo / "src/a.py").write_text("def base(): pass\n")
    (repo / "scripts").mkdir()
    (repo / "scripts/check_merge_resolution.py").write_text(
        "import sys\n"
        "raise SystemExit(1 if sys.argv[1] == open('expected').read().strip() else 0)\n"
    )
    commit(repo, "base")
    git(repo, "checkout", "-qb", "feature")
    (repo / "src/a.py").write_text("def feature(): pass\n")
    commit(repo, "feature")
    git(repo, "checkout", "-q", "main")
    (repo / "src/a.py").write_text("def main(): pass\n")
    commit(repo, "main")
    main_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()

    # this historical merge is on perf/integration, not on main.
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "feature"],
        cwd=repo,
        check=False,
        timeout=10,
    )
    git(repo, "checkout", "feature", "--", "src/a.py")
    commit(repo, "integration merge")
    inner_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()
    git(repo, "update-ref", "refs/heads/main", main_sha)

    git(repo, "checkout", "-qb", "train", inner_sha)
    (repo / "train").write_text("train\n")
    commit(repo, "train change")
    git(repo, "checkout", "-qb", "target-feature", inner_sha)
    (repo / "target").write_text("target\n")
    commit(repo, "target feature")
    git(repo, "checkout", "train")
    subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "target-feature"],
        cwd=repo,
        check=True,
        timeout=10,
    )
    commit(repo, "train merge")
    train_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()

    git(repo, "update-ref", "refs/heads/perf/integration", inner_sha)
    git(repo, "update-ref", "refs/remotes/origin/main", main_sha)
    git(repo, "update-ref", "refs/remotes/origin/perf/integration", inner_sha)
    (repo / "expected").write_text(train_sha)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text("#!/bin/sh\nexit 0\n")
    fake_gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EVENT_NAME": "pull_request",
        "BASE_REF": "perf/integration",
        "HEAD_SHA": train_sha,
        "PR_NUMBER": "917",
        "FINAL_SHA": train_sha,
        "PR_BODY": "",
        "CHECK_MERGE_RESOLUTION_SCRIPT": str(repo / "scripts/check_merge_resolution.py"),
    }

    old_script = tmp_path / "old-merge-resolution.sh"
    old_script.write_text(
        subprocess.check_output(
            ["git", "show", "origin/perf/integration:scripts/ci_merge_resolution.sh"],
            cwd=ROOT,
            text=True,
            timeout=10,
        )
    )
    old_script.chmod(0o755)
    old_result = subprocess.run(
        [str(old_script)], cwd=repo, env=env, text=True,
        capture_output=True, timeout=20, check=False,
    )
    assert old_result.returncode == 1, old_result.stdout + old_result.stderr
    assert f"checking merge resolution: {inner_sha}" in old_result.stdout
    assert f"checking merge resolution: {train_sha}" in old_result.stdout

    result = subprocess.run(
        _merge_resolution_command(tmp_path),
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"checking merge resolution: {train_sha}" in result.stdout
    assert f"checking merge resolution: {inner_sha}" not in result.stdout


def test_pytest_workflows_fetch_full_history() -> None:
    workflows_dir = ROOT / ".github/workflows"
    for workflow in workflows_dir.glob("*.y*ml"):
        text = workflow.read_text()
        if "pytest" not in text:
            continue
        checkout_before_pytest = re.search(
            r"(?ms)^      - uses: actions/checkout@v4\n(.*?)(?=^      - uses:|\Z).*?(?:^      - uses: ./\.github/workflows/pytest$|^        run: .*pytest)",
            text,
        )
        assert checkout_before_pytest is not None, workflow
        assert "fetch-depth: 0" in checkout_before_pytest.group(1), workflow


def test_title_only_edited_pull_request_still_runs_merge_resolution() -> None:
    workflow = WORKFLOW.read_text()
    job = workflow.split("  merge-resolution:\n", 1)[1]
    title_only_edited_payload = {
        "action": "edited",
        "changes": {},
    }
    assert title_only_edited_payload["action"] == "edited"
    assert "if:" not in job, (
        "a title-only edited payload must still run the merge-resolution job"
    )
    assert "types: [opened, synchronize, reopened, edited]" in workflow
    assert "name: merge resolution" in workflow
    assert "group: merge-resolution-${{ github.event.pull_request.number || github.run_id }}" in workflow
    assert "github.event.action" not in workflow
    assert "github.event.changes" not in workflow
    assert "timeout-minutes: 5" in workflow
    assert "PR_BODY:" not in workflow
    assert "merge-resolution:" not in Path(ROOT / ".github/workflows/ci.yml").read_text()
