from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import selectors
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.storage_format_compatibility import (
    _assert_mutation_delta,
    _digest_tree,
    _run_roundtrips,
)
from zeta.project_errors import ProjectRegistryBusyError
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.remote_sync import LocalTransport, RemoteSyncError
from zeta.remote_sync.project_publish import project_digest
from zeta.remote_sync.ssh import SshTransport
from zeta.storage_format import (
    P06_FEATURE_EPOCH,
    WRITER_EPOCH,
    StorageFormatError,
    advance,
)

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


# p06-4: the archived release is the exact p06 base source, run in its own
# environment from its own uv.lock; this interpreter is the candidate release.
P06_BASE_SHA = "33dd4857b59332c8ea7e023aa4af7adc4835303d"
_OLD_WRITER = Path(__file__).parent / "compatibility" / "project_lock_old_writer.py"
_OLD_ID = "3" * 32
_NEW_ID = "4" * 32
_POST_CUTOVER_ID = "5" * 32


@pytest.fixture(scope="module")
def archived_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p06-base")
    source = root / "source"
    source.mkdir()
    _COMPAT._git("archive", "--format=tar", P06_BASE_SHA, "-o", str(root / "base.tar"))
    _COMPAT._run_phase(
        ("tar", "xf", str(root / "base.tar"), "-C", str(source)),
        phase="p06 base extraction",
        check=True,
    )
    print(
        f"p06 base {P06_BASE_SHA} uv.lock {_blob(source / 'uv.lock')}; "
        f"candidate {_COMPAT._git('rev-parse', 'HEAD')} uv.lock {_blob(Path('uv.lock'))}"
    )
    return source


def _blob(path: Path) -> str:
    return _COMPAT._git("hash-object", str(path))


def _start_old(source: Path, home: Path, *args: str) -> subprocess.Popen[str]:
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    # Like the 0c harness: a private uv cache keeps installs out of the watched home.
    cache = source.parent / "uv-cache"
    return subprocess.Popen(
        ("uv", "run", "--frozen", "python", str(_OLD_WRITER), "--home", str(home), *args),
        cwd=source,
        env={**environment, "PYTHONPATH": str(source / "src"), "UV_CACHE_DIR": str(cache)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _finish_old(process: subprocess.Popen[str], source: Path) -> dict[str, object]:
    stdout, _ = process.communicate("release\n", timeout=180)
    assert process.returncode == 0, stdout
    result = json.loads(stdout.strip().splitlines()[-1])
    assert Path(result["zeta_file"]).is_relative_to(source), "old writer imported the candidate"
    return result


def _old_send(source: Path, home: Path, ids: dict[str, object], *, gate: bool) -> subprocess.Popen[str]:
    return _start_old(
        source, home, "--operation", "send", "--sender", str(ids["sender"]),
        "--target", str(ids["target"]), "--message-id", _OLD_ID, *(("--gate",) if gate else ()),
    )


def _wait_held(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        ready = selector.select(180)
    line = process.stdout.readline() if ready else ""
    assert line == "held\n", f"old writer did not report the held root lock: {line!r} {process.poll()!r}"


def _kill(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait(timeout=30)
    for stream in (process.stdin, process.stdout):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except BrokenPipeError:
                pass


def _below_epoch_home(source: Path, home: Path) -> dict[str, object]:
    ids = _finish_old(_start_old(source, home, "--operation", "setup"), source)
    assert ids["outcome"] == "done", ids
    old_epoch = _COMPAT._writer_epoch(source)
    assert _manifest_epoch(home) == old_epoch < P06_FEATURE_EPOCH <= WRITER_EPOCH
    return ids


def _manifest_epoch(home: Path) -> int:
    return json.loads((home / "storage-format.json").read_text())["writer_epoch"]


def _outcome(operation: Callable[[], object]) -> str:
    try:
        operation()
    except ProjectRegistryBusyError:
        return "busy"
    except RemoteSyncError as exc:
        return f"refused: {exc}"
    return "done"


def _send(inbox: ProjectInbox, ids: dict[str, object], message_id: str) -> None:
    inbox.send(
        from_project=str(ids["sender"]),
        from_session="b" * 32,
        to_project=str(ids["target"]),
        kind="question",
        title="new-writer",
        body="new-writer-body",
        message_id=message_id,
    )


def _inbox_ids(inbox: ProjectInbox, project_id: str) -> set[str]:
    return {message["id"] for message in inbox.read(project_id)["new"]}


def test_p06_4_archived_old_writer_excludes_new_writer_below_epoch(
    archived_base: Path, tmp_path: Path
) -> None:
    # Guard for p06's below-epoch legacy adapter: base has two legacy root-lock writers.
    home = tmp_path / "home"
    ids = _below_epoch_home(archived_base, home)
    target = str(ids["target"])
    old = _old_send(archived_base, home, ids, gate=True)
    registry = None
    try:
        _wait_held(old)
        with (
            (home / "projects" / ".lock").open("rb") as root_lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(root_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        registry = ProjectRegistry(home / "projects")
        inbox = ProjectInbox(registry, sessions_root=home / "sessions")
        outcomes = [
            _outcome(lambda: _send(inbox, ids, _NEW_ID)),
            _outcome(lambda: registry.update_memory(target, {"brief.md": "new\n"})),
        ]
        assert outcomes == ["busy", "busy"], (
            "new writer mutated while the archived writer held the legacy root lock"
        )

        assert _finish_old(old, archived_base)["outcome"] == "done"
        _send(inbox, ids, _NEW_ID)
        registry.update_memory(target, {"brief.md": "new\n"})
        assert {_OLD_ID, _NEW_ID} <= _inbox_ids(inbox, target)
        assert dict(registry.load_memory(target))["brief.md"] == "new\n"
    finally:
        _kill(old)
        if registry is not None:
            registry._release_storage_lease()


def test_p06_4_public_cutover_excludes_archived_writer(
    archived_base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ids = _below_epoch_home(archived_base, home)
    target = str(ids["target"])
    old_epoch = _manifest_epoch(home)
    old = _old_send(archived_base, home, ids, gate=True)
    try:
        _wait_held(old)
        with pytest.raises(StorageFormatError, match="active writers"):
            advance(home, expected_epoch=old_epoch)
        assert _manifest_epoch(home) == old_epoch
        assert _finish_old(old, archived_base)["outcome"] == "done"
    finally:
        _kill(old)

    advance(home, expected_epoch=old_epoch)
    assert _manifest_epoch(home) == WRITER_EPOCH
    before = _COMPAT._digest_tree(home)
    refused = _finish_old(_old_send(archived_base, home, ids, gate=False), archived_base)
    assert (refused["outcome"], refused.get("error_type")) == ("refused", "StorageFormatError"), refused
    assert "is newer than writer capability" in str(refused["error"])
    assert _COMPAT._digest_tree(home) == before, "refused old writer changed the cut-over home"

    project = home / "projects" / target
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/sh\n[ "$1" = -- ] && shift\nshift\nexec /bin/sh -c "$1"\n')
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    transports = {
        "local": LocalTransport(home),
        "ssh": SshTransport("fake", str(home), name="cloud"),
    }

    def publish(name: str) -> None:
        snapshot = tmp_path / name / target
        shutil.copytree(project, snapshot)
        (snapshot / "sync").mkdir(exist_ok=True)
        (snapshot / "sync" / f"{name}.json").write_text("{}\n")
        transports[name].publish_project(target, snapshot, expected_digest=project_digest(project))

    registry = ProjectRegistry(home / "projects")
    try:
        # At the feature epoch, domain and generated publication no longer take the legacy root lock.
        with (home / "projects" / ".lock").open("rb") as root_lock:
            fcntl.flock(root_lock, fcntl.LOCK_EX)
            outcomes = {
                "memory": _outcome(lambda: registry.update_memory(target, {"brief.md": "post-cutover\n"})),
                "local": _outcome(lambda: publish("local")),
                "ssh": _outcome(lambda: publish("ssh")),
            }
        assert outcomes == dict.fromkeys(outcomes, "done"), (
            "cut-over writer still serialized on the legacy root lock"
        )
        inbox = ProjectInbox(registry, sessions_root=home / "sessions")
        _send(inbox, ids, _POST_CUTOVER_ID)
        assert _POST_CUTOVER_ID in _inbox_ids(inbox, target)
        assert dict(registry.load_memory(target))["brief.md"] == "post-cutover\n"
        for name in transports:
            assert (project / "sync" / f"{name}.json").read_text() == "{}\n"
    finally:
        registry._release_storage_lease()
    assert _manifest_epoch(home) == WRITER_EPOCH
