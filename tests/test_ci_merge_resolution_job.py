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
    (repo / "src").mkdir()
    (repo / "src/a.py").write_text("def lost(): pass\n")
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
    (repo / "src/a.py").write_text("def lost(): pass\ndef added(): pass\n")
    commit(repo, "main")
    main_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", "feature"], cwd=repo, check=False, timeout=10)
    git(repo, "checkout", "feature", "--", "src/a.py")
    commit(repo, "inner merge")
    inner_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    git(repo, "checkout", "-qb", "train", main_sha)
    (repo / "train").write_text("train\n")
    commit(repo, "train change")
    train_parent_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    subprocess.run(["git", "merge", "--no-ff", "--no-commit", inner_sha], cwd=repo, check=False, timeout=10)
    git(repo, "checkout", inner_sha, "--", "src/a.py")
    commit(repo, "train merge")
    train_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    git(repo, "update-ref", "refs/remotes/origin/main", train_parent_sha)
    (repo / "expected").write_text(train_sha)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text("#!/bin/sh\nexit 0\n")
    fake_gh.chmod(0o755)
    result = subprocess.run(
        _merge_resolution_command(tmp_path),
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request",
            "HEAD_SHA": train_sha,
            "BASE_SHA": train_parent_sha,
            "PR_NUMBER": "917",
            "FINAL_SHA": train_sha,
            "PR_BODY": "",
            "CHECK_MERGE_RESOLUTION_SCRIPT": str(repo / "scripts/check_merge_resolution.py"),
        },
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
    assert "timeout-minutes: 15" in workflow
    assert "PR_BODY:" not in workflow
    assert "merge-resolution:" not in Path(ROOT / ".github/workflows/ci.yml").read_text()


def test_stacked_pr_job_borrows_override_from_base_pr(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    git(repo, "remote", "add", "origin", str(repo))
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
    merge_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()

    pulls = tmp_path / "pulls.json"
    pulls.write_text(
        '[{"number": 972, "body": "merge-resolution: src/a.py def added -> intentional",'
        ' "user": {"login": "hwang2409"},'
        ' "base": {"ref": "main", "repo": {"full_name": "hwang2409/zeta"}},'
        ' "head": {"repo": {"full_name": "hwang2409/zeta"}}}]'
    )
    calls = tmp_path / "gh-calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/bin/sh\n"
        f"echo \"$*\" >> {calls}\n"
        f'if [ "$1" = api ]; then cat {pulls}; else echo "stacked on #972"; fi\n'
    )
    fake_gh.chmod(0o755)

    result = subprocess.run(
        _merge_resolution_command(tmp_path),
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request",
            "HEAD_SHA": merge_sha,
            "PR_NUMBER": "1056",
            "PR_AUTHOR": "hwang2409",
            "GITHUB_REPOSITORY": "hwang2409/zeta",
            "FINAL_SHA": merge_sha,
            "CHECK_MERGE_RESOLUTION_SCRIPT": str(CHECKER),
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert "borrowed overrides: override lines from #972" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"api repos/hwang2409/zeta/commits/{merge_sha}/pulls" in calls.read_text()
