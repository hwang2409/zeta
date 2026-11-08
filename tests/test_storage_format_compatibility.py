from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.storage_format_compatibility import (
    _assert_mutation_delta,
    _digest_tree,
    _run_roundtrips,
)
from zeta.storage_format import WRITER_EPOCH

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


def test_selected_writer_round_trips_and_reopens_candidate_data() -> None:
    _run_roundtrips(Path(__file__).parents[1], WRITER_EPOCH)


def test_main_runs_selected_predecessor_roundtrip_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = "a" * 40
    phases: list[tuple[bool, int]] = []

    def fake_git(*args: str) -> str:
        if args[0] == "archive":
            Path(args[-1]).touch()
        return predecessor

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "uv":
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    # test-contract: seam isolate the command entry from git and package installation
    monkeypatch.setattr(_COMPAT, "_git", fake_git)
    # test-contract: seam force the selected predecessor at the command boundary
    monkeypatch.setattr(_COMPAT, "select_predecessor", lambda **kwargs: (predecessor, [predecessor]))
    # test-contract: seam exercise the bootstrap predecessor command path
    monkeypatch.setattr(_COMPAT, "has_probe", lambda commit: False)
    # test-contract: seam give the stub predecessor its public writer epoch
    monkeypatch.setattr(_COMPAT, "_writer_epoch", lambda source: WRITER_EPOCH)
    # test-contract: seam replace bounded child processes while testing command wiring
    monkeypatch.setattr(_COMPAT, "_run_phase", fake_run)
    # test-contract: seam provide the selected predecessor lock artifact
    monkeypatch.setattr(
        _COMPAT,
        "_copy_predecessor_lock",
        lambda commit, destination: destination.write_text("stub-lock\n"),
    )
    # test-contract: seam observe whether the command reaches the round-trip gate
    monkeypatch.setattr(
        _COMPAT,
        "_run_roundtrips",
        lambda source, epoch: phases.append((source.is_dir(), epoch)),
    )
    monkeypatch.setattr(sys, "argv", ["storage_format_compatibility.py", "--output-dir", str(tmp_path)])

    assert _COMPAT.main() == 0
    assert phases == [(True, WRITER_EPOCH)]


def test_subprocess_timeout_names_the_failed_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(("uv", "run"), 180)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="candidate writer timed out after 7 seconds"):
        _COMPAT._run_phase(("uv", "run"), phase="candidate writer", timeout=7)


def test_mutation_delta_rejects_non_declared_path(tmp_path: Path) -> None:
    (tmp_path / "owned.json").write_text("before")
    before = _digest_tree(tmp_path)
    (tmp_path / "owned.json").write_text("after")
    (tmp_path / "undeclared.json").write_text("changed")
    after = _digest_tree(tmp_path)
    with pytest.raises(RuntimeError, match="undeclared paths"):
        _assert_mutation_delta(before, after, ("owned.json",))


def test_digest_changes_when_higher_epoch_home_is_mutated(tmp_path: Path) -> None:
    path = tmp_path / "storage-format.json"
    path.write_text('{"writer_epoch":4}\n')
    before = _digest_tree(tmp_path)
    path.write_text('{"writer_epoch":3}\n')
    assert _digest_tree(tmp_path)["storage-format.json"] != before["storage-format.json"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _digest_tree(tmp_path)["storage-format.json"]
