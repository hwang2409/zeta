from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "storage_format_compatibility", Path("scripts/storage_format_compatibility.py")
)
assert _SPEC and _SPEC.loader
_COMPAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPAT)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    (repo / filename).parent.mkdir(parents=True, exist_ok=True)
    (repo / filename).write_text(content)
    subprocess.run(("git", "add", filename), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-m", message), cwd=repo, check=True, capture_output=True)
    return _git(repo, "rev-parse", "HEAD")


def _history(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    root = _commit(repo, "README", "root\n", "root")
    qualifying = _commit(repo, "src/zeta/storage_format.py", "format\n", "qualifying")
    _git(repo, "checkout", "-q", "-b", "side", root)
    _commit(repo, "docs/side", "side\n", "non-qualifying side")
    _git(repo, "checkout", "-q", "master")
    _commit(repo, "docs/main", "main\n", "non-qualifying main")
    _git(repo, "merge", "--no-ff", "side", "-m", "non-qualifying merge", "--no-edit")
    non_qualifying_merge = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "qualifying-merge", "HEAD^1")
    _commit(repo, "src/zeta/storage_format.py", "format2\n", "qualifying side")
    _git(repo, "checkout", "-q", "master")
    _git(repo, "merge", "--no-ff", "qualifying-merge", "-m", "qualifying merge")
    _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "review")
    return repo, qualifying, non_qualifying_merge


def test_qualifying_handles_root_and_merge_parent_diff(tmp_path: Path, monkeypatch) -> None:
    repo, qualifying, non_qualifying_merge = _history(tmp_path)
    monkeypatch.chdir(repo)
    assert _COMPAT.qualifying(qualifying)
    assert not _COMPAT.qualifying(non_qualifying_merge)
    assert _COMPAT.qualifying(_git(repo, "rev-parse", "HEAD"))
    assert not _COMPAT.qualifying(_git(repo, "rev-parse", f"{qualifying}^"))


def test_select_predecessor_checks_adjacent_merges_for_pull_and_push(tmp_path: Path, monkeypatch) -> None:
    repo, qualifying, non_qualifying_merge = _history(tmp_path)
    monkeypatch.chdir(repo)
    head = _git(repo, "rev-parse", "HEAD")
    selected_pull, _ = _COMPAT.select_predecessor(
        event="pull_request", head=head, base=head
    )
    selected_push, _ = _COMPAT.select_predecessor(
        event="push", head=head, base=head
    )
    assert selected_pull == head
    assert selected_push == qualifying
    assert selected_pull != non_qualifying_merge
    assert qualifying != selected_pull
