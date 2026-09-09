from __future__ import annotations

import hashlib
import json
import os
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.checkpoints import workspace as workspace_module
from zeta.core.checkpoints.workspace import (
    DEFAULT_SNAPSHOT_CAP,
    SIZE_NOTICE_THRESHOLD_BYTES,
    SNAPSHOT_MODE_GIT,
    SNAPSHOT_MODE_UNAVAILABLE,
    SNAPSHOT_REF_NAMESPACE,
    SNAPSHOT_SAFETY_REF_NAMESPACE,
    WorkspaceSnapshotStore,
    git_repo_root,
)
from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@zeta.local"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "zeta test"], cwd=root, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True)


def _commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)


def _hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_workspace_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.txt").write_text("initial tracked\n")
    _commit_all(repo, "initial")
    return repo


def _make_tui(store: ConversationStore) -> TUIApp:
    return TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True, color_system="truecolor"),
    )


def test_snapshot_and_restore_round_trip(git_repo: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")

    (git_repo / "tracked.txt").write_text("staged edit\n")
    (git_repo / "untracked.txt").write_text("untracked payload\n")

    snapshot = store.take(git_repo, label="snap-1")
    assert snapshot.mode == SNAPSHOT_MODE_GIT
    assert snapshot.file_count == 2
    assert snapshot.commit_sha
    assert snapshot.tree_sha

    (git_repo / "tracked.txt").write_text("later drift\n")
    (git_repo / "untracked.txt").unlink()
    (git_repo / "added.txt").write_text("added later\n")

    store.restore(snapshot.id, git_repo)

    files = _repo_workspace_files(git_repo)
    assert files == {
        "tracked.txt": "staged edit\n",
        "untracked.txt": "untracked payload\n",
    }
    assert not store.is_dirty(git_repo)


def test_restore_never_deletes_ignored_files(git_repo: Path, tmp_path: Path) -> None:
    (git_repo / ".gitignore").write_text("build/\n")
    _commit_all(git_repo, "add gitignore")
    (git_repo / "build").mkdir()
    (git_repo / "build" / "artifact.bin").write_text("compiled")

    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    snapshot = store.take(git_repo, label="ignored-safe")

    (git_repo / "tracked.txt").write_text("edited")
    (git_repo / "build" / "artifact.bin").write_text("compiled newer")

    store.restore(snapshot.id, git_repo)

    assert (git_repo / "build" / "artifact.bin").read_text() == "compiled newer"
    assert (git_repo / "tracked.txt").read_text() == "initial tracked\n"


def test_snapshot_and_restore_preserve_user_git_state(
    git_repo: Path, tmp_path: Path
) -> None:
    (git_repo / "tracked.txt").write_text("post-init edit")
    index_before = _hash(git_repo / ".git" / "index")
    head_before = (git_repo / ".git" / "HEAD").read_text()
    branches_before = subprocess.check_output(
        ["git", "branch", "-a"], cwd=git_repo, text=True
    )
    reflog_before = subprocess.check_output(
        ["git", "reflog", "HEAD"], cwd=git_repo, text=True
    )

    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    snapshot = store.take(git_repo, label="invariance")
    (git_repo / "tracked.txt").write_text("drift")
    store.restore(snapshot.id, git_repo)

    assert _hash(git_repo / ".git" / "index") == index_before
    assert (git_repo / ".git" / "HEAD").read_text() == head_before
    assert (
        subprocess.check_output(["git", "branch", "-a"], cwd=git_repo, text=True)
        == branches_before
    )
    assert (
        subprocess.check_output(["git", "reflog", "HEAD"], cwd=git_repo, text=True)
        == reflog_before
    )


def test_shadow_refs_live_outside_the_user_git_repo(
    git_repo: Path, tmp_path: Path
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    store.take(git_repo, label="hidden")

    branches = subprocess.check_output(
        ["git", "branch", "-a"], cwd=git_repo, text=True
    )
    tags = subprocess.check_output(["git", "tag"], cwd=git_repo, text=True)
    stash = subprocess.check_output(
        ["git", "stash", "list"], cwd=git_repo, text=True
    )
    every_user_ref = subprocess.check_output(
        ["git", "for-each-ref"], cwd=git_repo, text=True
    )

    assert "refs/zeta" not in branches
    assert tags.strip() == ""
    assert stash.strip() == ""
    assert "refs/zeta" not in every_user_ref

    shadow_dir = session_dir / "workspace_shadow.git"
    assert shadow_dir.is_dir()
    shadow_refs = subprocess.check_output(
        ["git", "for-each-ref", "refs/zeta/"],
        cwd=shadow_dir,
        text=True,
        env={**os.environ, "GIT_DIR": str(shadow_dir)},
    )
    assert "refs/zeta/checkpoints/" in shadow_refs


def test_undo_redo_stack_navigation(git_repo: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")

    (git_repo / "tracked.txt").write_text("v1")
    snap_1 = store.take(git_repo, label="v1")
    (git_repo / "tracked.txt").write_text("v2")
    snap_2 = store.take(git_repo, label="v2")
    (git_repo / "tracked.txt").write_text("v3")
    snap_3 = store.take(git_repo, label="v3")

    assert store.current_id == snap_3.id
    target = store.undo_target()
    assert target and target.id == snap_2.id
    store.restore(snap_2.id, git_repo)
    assert (git_repo / "tracked.txt").read_text() == "v2"
    target = store.undo_target()
    assert target and target.id == snap_1.id
    store.restore(snap_1.id, git_repo)
    assert (git_repo / "tracked.txt").read_text() == "v1"

    forward = store.redo_target()
    assert forward and forward.id == snap_2.id
    store.restore(snap_2.id, git_repo)
    assert (git_repo / "tracked.txt").read_text() == "v2"
    forward = store.redo_target()
    assert forward and forward.id == snap_3.id
    store.restore(snap_3.id, git_repo)
    assert (git_repo / "tracked.txt").read_text() == "v3"
    assert store.redo_target() is None


def test_is_dirty_detects_workspace_drift(git_repo: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    store.take(git_repo, label="base")
    assert not store.is_dirty(git_repo)
    (git_repo / "tracked.txt").write_text("drift\n")
    assert store.is_dirty(git_repo)


def test_non_git_workspace_records_unavailable_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "plain"
    workspace.mkdir()
    (workspace / "file.txt").write_text("hi")
    assert git_repo_root(workspace) is None

    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    snapshot = store.take(workspace, label="conv-only")
    assert snapshot.mode == SNAPSHOT_MODE_UNAVAILABLE
    (workspace / "file.txt").write_text("edited")
    restored = store.restore(snapshot.id, workspace)
    assert restored.mode == SNAPSHOT_MODE_UNAVAILABLE
    assert (workspace / "file.txt").read_text() == "edited"


def test_snapshot_state_survives_reopen(git_repo: Path, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("persisted")
    snapshot = store.take(git_repo, label="persist")

    reopened = WorkspaceSnapshotStore(session_dir, "session-1")
    assert reopened.current_id == snapshot.id
    assert reopened.by_id(snapshot.id) is not None


def _make_store(session_dir: Path, cwd: Path) -> ConversationStore:
    return ConversationStore(
        session_dir=session_dir,
        session_id="tui",
        cwd=cwd,
        bash_cwd=cwd,
    )


def test_slash_checkpoint_captures_workspace(git_repo: Path, tmp_path: Path) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)

    result = app.slash_checkpoint("save")
    assert "workspace snapshot" in result
    assert "1 files" in result


def test_slash_fork_restores_tree(git_repo: Path, tmp_path: Path) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)
    app.slash_checkpoint("baseline")

    (git_repo / "tracked.txt").write_text("post-checkpoint edit")
    result = app.slash_fork("baseline --force")
    assert "workspace restored" in result
    assert (git_repo / "tracked.txt").read_text() == "initial tracked\n"


def test_slash_fork_refuses_dirty_workspace_without_force(
    git_repo: Path, tmp_path: Path
) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)
    app.slash_checkpoint("baseline")

    (git_repo / "tracked.txt").write_text("drift")
    result = app.slash_fork("baseline")
    assert "uncommitted edits" in result
    assert "--force" in result
    assert (git_repo / "tracked.txt").read_text() == "drift"


def test_slash_undo_and_redo_flow(git_repo: Path, tmp_path: Path) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("u1")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("a1")]))
    app = _make_tui(store)
    app.slash_checkpoint("first")
    (git_repo / "tracked.txt").write_text("edit-1\n")
    store.append_message(Message(MessageRole.USER, [TextContent("u2")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("a2")]))
    app.slash_checkpoint("second")

    undo_result = app.slash_undo("")
    assert "undone" in undo_result
    assert (git_repo / "tracked.txt").read_text() == "initial tracked\n"

    redo_result = app.slash_redo("")
    assert "redone" in redo_result
    assert (git_repo / "tracked.txt").read_text() == "edit-1\n"


def test_slash_undo_refuses_when_no_earlier_snapshot(
    git_repo: Path, tmp_path: Path
) -> None:
    store = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(store)
    result = app.slash_undo("")
    assert "no earlier" in result


def test_slash_checkpoint_soft_fails_on_git_error(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)

    real_run = subprocess.run

    def failing_run(cmd, *args, **kwargs):
        if (
            len(cmd) >= 2
            and cmd[0] == "git"
            and cmd[1] == "add"
            and kwargs.get("check")
        ):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=cmd, stderr="fatal: forced failure\n"
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_run)
    result = app.slash_checkpoint("break")
    assert "workspace snapshot skipped" in result
    assert "forced failure" in result
    assert "Traceback" not in result


def test_slash_undo_soft_fails_on_git_error(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.types import Message, MessageRole, TextContent

    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)
    app.slash_checkpoint("first")
    (git_repo / "tracked.txt").write_text("edit-1\n")
    store.append_message(Message(MessageRole.USER, [TextContent("u2")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("a2")]))
    app.slash_checkpoint("second")

    real_run = subprocess.run

    def failing_run(cmd, *args, **kwargs):
        if (
            len(cmd) >= 2
            and cmd[0] == "git"
            and cmd[1] == "read-tree"
            and kwargs.get("check")
        ):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=cmd, stderr="fatal: read-tree boom\n"
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_run)
    result = app.slash_undo("--force")
    assert "workspace restore failed" in result
    assert "undone" not in result
    assert "read-tree boom" in result
    assert "Traceback" not in result


def test_slash_checkpoint_in_non_git_directory(tmp_path: Path) -> None:
    from zeta.types import Message, MessageRole, TextContent

    workspace = tmp_path / "plain"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("hi")
    store = _make_store(tmp_path / "session", workspace)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)
    result = app.slash_checkpoint("noop")
    assert "not a git repository" in result


def test_snapshot_size_notice_flags_large_working_tree(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui import checkpoints as checkpoints_module
    from zeta.types import Message, MessageRole, TextContent

    monkeypatch.setattr(checkpoints_module, "SIZE_NOTICE_THRESHOLD_BYTES", 8)
    store = _make_store(tmp_path / "session", git_repo)
    store.append_message(Message(MessageRole.USER, [TextContent("hi")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("hey")]))
    app = _make_tui(store)
    (git_repo / "big.bin").write_bytes(b"0" * 128)
    result = app.slash_checkpoint("big")
    assert "WARNING: workspace snapshot is large" in result


def test_size_threshold_constant_is_bounded() -> None:
    assert SIZE_NOTICE_THRESHOLD_BYTES == 100 * 1024 * 1024


def _shadow_refs(session_dir: Path, namespace: str) -> list[str]:
    shadow = session_dir / "workspace_shadow.git"
    if not shadow.exists():
        return []
    result = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", namespace],
        cwd=shadow,
        text=True,
        env={**os.environ, "GIT_DIR": str(shadow)},
    )
    return [line for line in result.splitlines() if line]


def _cat_shadow_file(session_dir: Path, tree_sha: str, entry: str) -> str:
    shadow = session_dir / "workspace_shadow.git"
    return subprocess.check_output(
        ["git", "cat-file", "-p", f"{tree_sha}:{entry}"],
        cwd=shadow,
        text=True,
        env={**os.environ, "GIT_DIR": str(shadow)},
    )


def test_restore_interrupt_leaves_pre_restore_state_recoverable(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("baseline\n")
    (git_repo / "keep.txt").write_text("keep-baseline\n")
    baseline = store.take(git_repo, label="baseline")

    (git_repo / "tracked.txt").write_text("dirty-content\n")
    (git_repo / "keep.txt").write_text("dirty-keep\n")
    (git_repo / "new_stale.txt").write_text("only-in-dirty\n")

    real_run_git = workspace_module._run_git

    def interrupt_at(target: str):
        def wrapper(args, **kwargs):
            if args and args[0] == target:
                raise KeyboardInterrupt(f"interrupt at git {target}")
            return real_run_git(args, **kwargs)

        return wrapper

    for target in ("checkout-index", "read-tree"):
        # Reset the working tree to the dirty state each iteration
        (git_repo / "tracked.txt").write_text("dirty-content\n")
        (git_repo / "keep.txt").write_text("dirty-keep\n")
        (git_repo / "new_stale.txt").write_text("only-in-dirty\n")

        monkeypatch.setattr(workspace_module, "_run_git", interrupt_at(target))
        with pytest.raises(KeyboardInterrupt):
            store.restore(baseline.id, git_repo)
        monkeypatch.setattr(workspace_module, "_run_git", real_run_git)

        # Safety ref must exist and reference blobs matching the pre-restore state
        safety_refs = _shadow_refs(session_dir, SNAPSHOT_SAFETY_REF_NAMESPACE)
        assert safety_refs, f"no safety ref after interrupt at {target}"
        safety_commit = subprocess.check_output(
            ["git", "rev-parse", safety_refs[-1]],
            cwd=session_dir / "workspace_shadow.git",
            text=True,
            env={
                **os.environ,
                "GIT_DIR": str(session_dir / "workspace_shadow.git"),
            },
        ).strip()
        safety_tree = subprocess.check_output(
            ["git", "rev-parse", f"{safety_commit}^{{tree}}"],
            cwd=session_dir / "workspace_shadow.git",
            text=True,
            env={
                **os.environ,
                "GIT_DIR": str(session_dir / "workspace_shadow.git"),
            },
        ).strip()

        # Every dirty file's pre-restore blob is intact and readable
        assert _cat_shadow_file(session_dir, safety_tree, "tracked.txt") == (
            "dirty-content\n"
        )
        assert _cat_shadow_file(session_dir, safety_tree, "keep.txt") == (
            "dirty-keep\n"
        )
        assert _cat_shadow_file(session_dir, safety_tree, "new_stale.txt") == (
            "only-in-dirty\n"
        )

        # Clean up safety ref for the next iteration
        for ref in safety_refs:
            subprocess.run(
                ["git", "update-ref", "-d", ref],
                cwd=session_dir / "workspace_shadow.git",
                env={
                    **os.environ,
                    "GIT_DIR": str(session_dir / "workspace_shadow.git"),
                },
                check=True,
            )


def test_restore_succeeds_and_cleans_up_safety_ref(
    git_repo: Path, tmp_path: Path
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("v1\n")
    snap = store.take(git_repo, label="v1")
    (git_repo / "tracked.txt").write_text("v2\n")
    store.restore(snap.id, git_repo)

    # Successful restore leaves no orphan safety refs behind
    assert _shadow_refs(session_dir, SNAPSHOT_SAFETY_REF_NAMESPACE) == []


def test_git_log_all_and_push_mirror_ignore_shadow_refs(
    git_repo: Path, tmp_path: Path
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("snap-content\n")
    store.take(git_repo, label="private")

    log_all = subprocess.check_output(
        ["git", "log", "--all", "--pretty=%s"], cwd=git_repo, text=True
    )
    assert "zeta shadow checkpoint" not in log_all
    assert "pre-restore safety" not in log_all

    fsck_out = subprocess.run(
        ["git", "fsck", "--full", "--strict"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "zeta" not in (fsck_out.stdout + fsck_out.stderr)

    # Simulate `git push --mirror` — it advertises every ref under refs/*
    mirror_refs = subprocess.check_output(
        ["git", "for-each-ref", "refs/"], cwd=git_repo, text=True
    )
    assert "refs/zeta" not in mirror_refs


def test_take_after_restore_earlier_truncates_abandoned_tail(
    git_repo: Path, tmp_path: Path
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("v1")
    snap_1 = store.take(git_repo, label="v1")
    (git_repo / "tracked.txt").write_text("v2")
    snap_2 = store.take(git_repo, label="v2")
    (git_repo / "tracked.txt").write_text("v3")
    snap_3 = store.take(git_repo, label="v3")

    store.restore(snap_1.id, git_repo)
    (git_repo / "tracked.txt").write_text("branch-checkpoint")
    new_snap = store.take(git_repo, label="branch")

    ids = [snap.id for snap in store.snapshots]
    assert ids == [snap_1.id, new_snap.id]
    assert snap_2.id not in ids
    assert snap_3.id not in ids

    # Orphaned snapshot refs are actually removed from the shadow repo
    ref_names = _shadow_refs(session_dir, SNAPSHOT_REF_NAMESPACE)
    assert f"{SNAPSHOT_REF_NAMESPACE}/{snap_2.id}" not in ref_names
    assert f"{SNAPSHOT_REF_NAMESPACE}/{snap_3.id}" not in ref_names
    assert f"{SNAPSHOT_REF_NAMESPACE}/{snap_1.id}" in ref_names
    assert f"{SNAPSHOT_REF_NAMESPACE}/{new_snap.id}" in ref_names

    # /undo now stops at snap_1's predecessor chain, never at v3
    store.restore(new_snap.id, git_repo)
    undo_target = store.undo_target()
    assert undo_target is not None and undo_target.id == snap_1.id
    assert store.by_id(snap_3.id) is None


def test_snapshot_ring_cap_drops_oldest_and_frees_refs(
    git_repo: Path, tmp_path: Path
) -> None:
    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1", cap=3)
    ids: list[str] = []
    for index in range(5):
        (git_repo / "tracked.txt").write_text(f"v{index}\n")
        ids.append(store.take(git_repo, label=f"v{index}").id)

    surviving = [snap.id for snap in store.snapshots]
    assert surviving == ids[-3:]

    ref_names = _shadow_refs(session_dir, SNAPSHOT_REF_NAMESPACE)
    for dropped in ids[:-3]:
        assert f"{SNAPSHOT_REF_NAMESPACE}/{dropped}" not in ref_names
    for kept in ids[-3:]:
        assert f"{SNAPSHOT_REF_NAMESPACE}/{kept}" in ref_names


def test_default_snapshot_cap_is_bounded() -> None:
    assert DEFAULT_SNAPSHOT_CAP == 50


def test_dirty_guard_still_confirms_when_current_id_is_stale(
    git_repo: Path, tmp_path: Path
) -> None:
    from zeta.tui.checkpoints import _dirty_guard

    session_dir = tmp_path / "session"
    store = WorkspaceSnapshotStore(session_dir, "session-1")
    (git_repo / "tracked.txt").write_text("clean\n")
    store.take(git_repo, label="baseline")

    # Corrupt the state file so current_id points at a snapshot that no
    # longer exists — a scenario that previously bypassed the guard.
    state = json.loads(store.state_path.read_text())
    state["current_id"] = "does-not-exist"
    store.state_path.write_text(json.dumps(state))
    reopened = WorkspaceSnapshotStore(session_dir, "session-1")

    # Even with the stale current_id, the guard sees the restorable
    # snapshot and detects the drift.
    (git_repo / "tracked.txt").write_text("drift\n")
    warning = _dirty_guard(reopened, str(git_repo), forced=False)
    assert warning is not None and "--force" in warning


@pytest.mark.parametrize(
    "row",
    [
        {"id": "x", "size_bytes": {}},
        {
            "id": "x",
            "created_at": "2026-09-09T00:00:00+00:00",
            "mode": "unavailable",
            "size_bytes": {},
        },
        {"id": "x", "created_at": {}, "mode": "unavailable"},
        {
            "id": "x",
            "created_at": "2026-09-09T00:00:00+00:00",
            "mode": "unavailable",
            "repo_root": {},
        },
    ],
)
def test_corrupt_snapshot_rows_degrade_in_store_and_tui(
    tmp_path: Path, row: dict
) -> None:
    conversation = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    path = conversation.session_dir / "workspace_snapshots.json"
    path.write_text(json.dumps({"snapshots": [row], "current_id": "x"}))
    snapshots = WorkspaceSnapshotStore(
        conversation.session_dir, conversation.session_id
    )
    assert snapshots.snapshots == ()
    assert snapshots.current_id is None
    app = _make_tui(conversation)
    result = app.slash_checkpoint("save")
    assert "checkpoint 'save'" in result
    assert "snapshot skipped" in result and "corrupt" in result
    assert app._snapshots().snapshots == ()


def test_checkpoint_reports_snapshot_setup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = _make_tui(conversation)

    def fail():
        raise workspace_module.WorkspaceSnapshotError(
            "cannot initialize snapshot store"
        )

    monkeypatch.setattr(app, "_snapshots", fail)
    assert (
        "workspace snapshot skipped: cannot initialize snapshot store"
        in app.slash_checkpoint("save")
    )


@pytest.mark.parametrize("field", ["tree_sha", "commit_sha", "repo_root"])
@pytest.mark.parametrize("invalid", ["missing", None, ""])
def test_incomplete_current_snapshot_cannot_overwrite_dirty_files(
    git_repo: Path, tmp_path: Path, field: str, invalid: str | None
) -> None:
    from zeta.tui.checkpoints import _dirty_guard

    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("older")
    (git_repo / "tracked.txt").write_text("current snapshot\n")
    app.slash_checkpoint("current")
    path = conversation.session_dir / "workspace_snapshots.json"
    state = json.loads(path.read_text())
    if invalid == "missing":
        del state["snapshots"][-1][field]
    else:
        state["snapshots"][-1][field] = invalid
    path.write_text(json.dumps(state))
    (git_repo / "tracked.txt").write_text("unsaved work\n")

    reopened = _make_tui(conversation)
    result = reopened.slash_undo("")
    assert (git_repo / "tracked.txt").read_text() == "unsaved work\n"
    assert "undone" not in result
    snapshots = reopened._snapshots()
    assert [snap.label for snap in snapshots.snapshots] == ["older"]
    assert snapshots.current_id is None
    assert snapshots.is_dirty(git_repo)
    assert _dirty_guard(snapshots, str(git_repo), forced=False) is not None
    assert _dirty_guard(snapshots, str(git_repo), forced=True) is None


@pytest.mark.parametrize("row", [None, [], "invalid", {"id": "invalid"}])
def test_invalid_snapshot_row_preserves_valid_rows(
    git_repo: Path, tmp_path: Path, row: object
) -> None:
    store = WorkspaceSnapshotStore(tmp_path / "session", "session-1")
    store.take(git_repo, label="valid")
    state = json.loads(store.state_path.read_text())
    state["snapshots"].append(row)
    store.state_path.write_text(json.dumps(state))
    reopened = WorkspaceSnapshotStore(store.session_dir, "session-1")
    assert [snap.label for snap in reopened.snapshots] == ["valid"]
    assert reopened.current_id == reopened.snapshots[0].id
    assert reopened.is_corrupt
    assert reopened.is_dirty(git_repo)


def test_empty_snapshot_store_does_not_report_clean(git_repo: Path, tmp_path: Path) -> None:
    from zeta.tui.checkpoints import _dirty_guard

    store = WorkspaceSnapshotStore(tmp_path / "session", "session-1")
    assert store.is_dirty(git_repo)
    assert _dirty_guard(store, str(git_repo), forced=False) is not None


def test_failed_dirty_comparison_cannot_overwrite_files(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("older")
    (git_repo / "tracked.txt").write_text("current snapshot\n")
    app.slash_checkpoint("current")
    (git_repo / "tracked.txt").write_text("unsaved work\n")

    def fail(*args):
        raise workspace_module.WorkspaceSnapshotError("cannot compare workspace")

    monkeypatch.setattr(workspace_module, "_current_tree_sha", fail)
    result = app.slash_undo("")
    assert (git_repo / "tracked.txt").read_text() == "unsaved work\n"
    assert "--force" in result


def test_corrupt_current_snapshot_allows_only_forced_recovery(
    git_repo: Path, tmp_path: Path
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("oldest")
    (git_repo / "tracked.txt").write_text("older snapshot\n")
    app.slash_checkpoint("older")
    (git_repo / "tracked.txt").write_text("current snapshot\n")
    app.slash_checkpoint("current")
    path = app._snapshots().state_path
    state = json.loads(path.read_text())
    del state["snapshots"][-1]["tree_sha"]
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    (git_repo / "tracked.txt").write_text("unsaved work\n")
    reopened = _make_tui(conversation)

    result = reopened.slash_undo("")
    assert "corrupt" in result
    assert "--force" in result and "/checkpoint --reset" in result
    assert (git_repo / "tracked.txt").read_text() == "unsaved work\n"
    assert "workspace restored" in reopened.slash_undo("--force")
    assert (git_repo / "tracked.txt").read_text() == "older snapshot\n"
    # Matching a valid tree must not clear the corrupt-state safety guard.
    assert reopened._snapshots().is_dirty(git_repo)
    assert "corrupt" in reopened.slash_undo("")
    assert "workspace restored" in reopened.slash_undo("--force")
    assert (git_repo / "tracked.txt").read_text() == "initial tracked\n"
    assert path.read_bytes() == original


def test_checkpoint_preserves_corrupt_manifest_and_all_shadow_refs(
    git_repo: Path, tmp_path: Path
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    for label in ("oldest", "older", "current"):
        (git_repo / "tracked.txt").write_text(label)
        app.slash_checkpoint(label)
    path = app._snapshots().state_path
    state = json.loads(path.read_text())
    state["snapshots"][-1]["size_bytes"] = {}
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    shadow = conversation.session_dir / workspace_module.SHADOW_REPO_DIRNAME
    refs_command = ["git", "--git-dir", str(shadow), "show-ref"]
    refs_before = subprocess.check_output(refs_command)
    reopened = _make_tui(conversation)
    reopened._workspace_snapshot_cap = 1

    result = reopened.slash_checkpoint("after-corruption")
    assert "snapshot skipped" in result and "corrupt" in result
    assert path.read_bytes() == original
    assert subprocess.check_output(refs_command) == refs_before
    loaded = WorkspaceSnapshotStore(conversation.session_dir, conversation.session_id)
    assert [snap.label for snap in loaded.snapshots] == ["oldest", "older"]


def test_corrupt_snapshot_marker_clears_only_through_explicit_reset(
    git_repo: Path, tmp_path: Path
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("older")
    app.slash_checkpoint("current")
    path = app._snapshots().state_path
    state = json.loads(path.read_text())
    del state["snapshots"][-1]["tree_sha"]
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    reopened = _make_tui(conversation)
    snapshots = reopened._snapshots()
    assert snapshots.is_corrupt
    older = snapshots.snapshots[0]
    with pytest.raises(workspace_module.WorkspaceSnapshotError, match="corrupt"):
        snapshots.restore(older.id, git_repo)
    snapshots.restore(older.id, git_repo, force=True)
    snapshots.set_current(older.id)
    reopened.slash_checkpoint("blocked-write")
    assert snapshots.is_corrupt
    assert path.read_bytes() == original
    assert _make_tui(conversation)._snapshots().is_corrupt

    result = reopened.slash_checkpoint("--reset")
    assert "snapshot state reset" in result
    assert not snapshots.is_corrupt
    backups = list(conversation.session_dir.glob("workspace_snapshots.corrupt.*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    loaded = _make_tui(conversation)._snapshots()
    assert not loaded.is_corrupt
    assert loaded.snapshots == (older,)
    assert not loaded.is_dirty(git_repo)
    assert "snapshot skipped" not in reopened.slash_checkpoint("after-reset")


@pytest.mark.parametrize("damage", ["ref", "pruned", "commit", "tree"])
@pytest.mark.parametrize("malformed_row", [False, True])
def test_reset_discards_unrestorable_snapshots(
    git_repo: Path, tmp_path: Path, damage: str, malformed_row: bool
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("valid")
    (git_repo / "tracked.txt").write_text("damaged snapshot\n")
    app.slash_checkpoint("damaged")
    snapshots = app._snapshots()
    valid, damaged = snapshots.snapshots
    shadow = conversation.session_dir / "workspace_shadow.git"
    state = json.loads(snapshots.state_path.read_text())
    if damage in {"ref", "pruned"}:
        subprocess.run(
            ["git", f"--git-dir={shadow}", "update-ref", "-d",
             f"{SNAPSHOT_REF_NAMESPACE}/{damaged.id}"], check=True,
        )
        if damage == "pruned":
            subprocess.run(
                ["git", f"--git-dir={shadow}", "prune", "--expire=now"], check=True,
            )
    elif damage == "commit":
        state["snapshots"][-1]["commit_sha"] = valid.commit_sha
    else:
        state["snapshots"][-1]["tree_sha"] = valid.tree_sha
    if malformed_row:
        state["snapshots"].append({"id": "invalid"})
    snapshots.state_path.write_text(json.dumps(state))
    original = snapshots.state_path.read_bytes()
    (git_repo / "tracked.txt").write_text("unsaved work\n")
    reopened = _make_tui(conversation)

    assert "snapshot state reset" in reopened.slash_checkpoint("--reset")
    loaded = _make_tui(conversation)
    assert loaded._snapshots().snapshots == (valid,)
    assert loaded._snapshots().current_id is None
    assert not loaded._snapshots().is_corrupt
    backups = list(conversation.session_dir.glob("workspace_snapshots.corrupt.*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert "undone" in loaded.slash_undo("--force")
    assert (git_repo / "tracked.txt").read_text() == "initial tracked\n"
    assert loaded._snapshots().current_id == valid.id


@pytest.mark.parametrize("prune", [False, True])
def test_failed_forced_undo_reports_failure_without_changing_workspace(
    git_repo: Path, tmp_path: Path, prune: bool
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("older")
    (git_repo / "tracked.txt").write_text("current snapshot\n")
    app.slash_checkpoint("current")
    snapshots = app._snapshots()
    older, current = snapshots.snapshots
    shadow = conversation.session_dir / "workspace_shadow.git"
    subprocess.run(
        ["git", f"--git-dir={shadow}", "update-ref", "-d",
         f"{SNAPSHOT_REF_NAMESPACE}/{older.id}"], check=True,
    )
    if prune:
        subprocess.run(
            ["git", f"--git-dir={shadow}", "prune", "--expire=now"], check=True,
        )
    (git_repo / "tracked.txt").write_text("unsaved work\n")
    original = snapshots.state_path.read_bytes()

    result = app.slash_undo("--force")

    assert "restore failed" in result
    assert "undone" not in result and "workspace restored" not in result
    assert "rev-parse" in result
    assert (git_repo / "tracked.txt").read_text() == "unsaved work\n"
    assert snapshots.current_id == current.id
    assert snapshots.state_path.read_bytes() == original


@pytest.mark.parametrize("args", ["", "--force"])
def test_all_invalid_history_points_undo_to_reset(tmp_path: Path, args: str) -> None:
    conversation = _make_store(tmp_path / "session", tmp_path)
    path = conversation.session_dir / "workspace_snapshots.json"
    path.write_text(json.dumps({"snapshots": [{"id": "invalid"}], "current_id": "invalid"}))
    original = path.read_bytes()

    result = _make_tui(conversation).slash_undo(args)

    assert "no valid" in result and "snapshot" in result
    assert "/checkpoint --reset" in result
    assert "--force" not in result
    assert path.read_bytes() == original


@pytest.mark.parametrize("root_kind", ["missing", "non-git", "file"])
def test_reset_discards_invalid_restore_root(
    git_repo: Path, tmp_path: Path, root_kind: str
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    app.slash_checkpoint("valid")
    app.slash_checkpoint("bad-root")
    snapshots = app._snapshots()
    valid = snapshots.snapshots[0]
    root = tmp_path / "invalid-root"
    if root_kind == "non-git":
        root.mkdir()
    elif root_kind == "file":
        root.write_text("not a directory")
    state = json.loads(snapshots.state_path.read_text())
    state["snapshots"][-1]["repo_root"] = str(root)
    snapshots.state_path.write_text(json.dumps(state))
    original = snapshots.state_path.read_bytes()

    reopened = _make_tui(conversation)
    assert "snapshot state reset" in reopened.slash_checkpoint("--reset")
    loaded = _make_tui(conversation)._snapshots()
    assert loaded.snapshots == (valid,)
    assert loaded.current_id is None
    archives = list(conversation.session_dir.glob("workspace_snapshots.corrupt.*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == original


@pytest.mark.parametrize("malformed_row", [False, True])
def test_reset_preserves_unavailable_snapshots(tmp_path: Path, malformed_row: bool) -> None:
    conversation = _make_store(tmp_path / "session", tmp_path)
    app = _make_tui(conversation)
    app.slash_checkpoint("older")
    app.slash_checkpoint("current")
    snapshots = app._snapshots()
    expected = snapshots.snapshots
    if malformed_row:
        state = json.loads(snapshots.state_path.read_text())
        state["snapshots"].append({"id": "invalid"})
        snapshots.state_path.write_text(json.dumps(state))
    original = snapshots.state_path.read_bytes()
    original_stat = snapshots.state_path.stat()

    reopened = _make_tui(conversation)
    assert "snapshot state reset" in reopened.slash_checkpoint("--reset")
    loaded = _make_tui(conversation)._snapshots()
    assert loaded.snapshots == expected
    assert loaded.current_id == expected[-1].id
    assert not loaded.is_corrupt
    archives = list(conversation.session_dir.glob("workspace_snapshots.corrupt.*.json"))
    if malformed_row:
        assert len(archives) == 1 and archives[0].read_bytes() == original
    else:
        assert archives == []
        assert snapshots.state_path.read_bytes() == original
        assert snapshots.state_path.stat().st_ino == original_stat.st_ino
        assert snapshots.state_path.stat().st_mtime_ns == original_stat.st_mtime_ns


@pytest.mark.parametrize("target_kind", ["unavailable", "non-git-root"])
def test_failed_forced_undo_preserves_cursor_and_files(
    git_repo: Path, tmp_path: Path, target_kind: str
) -> None:
    conversation = _make_store(tmp_path / "session", git_repo)
    app = _make_tui(conversation)
    snapshots = app._snapshots()
    snapshots.take(tmp_path if target_kind == "unavailable" else git_repo, label="older")
    snapshots.take(git_repo, label="current")
    state = json.loads(snapshots.state_path.read_text())
    non_git_root = tmp_path / "non-git"
    non_git_root.mkdir()
    (non_git_root / "tracked.txt").write_text("keep non-git files")
    if target_kind == "non-git-root":
        state["snapshots"][0]["repo_root"] = str(non_git_root)
    snapshots.state_path.write_text(json.dumps(state))
    original = snapshots.state_path.read_bytes()
    (git_repo / "tracked.txt").write_text("unsaved work")
    before = _repo_workspace_files(git_repo)
    non_git_before = _repo_workspace_files(non_git_root)
    reopened = _make_tui(conversation)

    result = reopened.slash_undo("--force")

    assert "undo failed" in result
    assert "undone" not in result
    assert reopened._snapshots().current_id == state["current_id"]
    assert snapshots.state_path.read_bytes() == original
    assert _repo_workspace_files(git_repo) == before
    assert _repo_workspace_files(non_git_root) == non_git_before


def test_corruption_warning_rejects_missing_restore_root(git_repo: Path, tmp_path: Path) -> None:
    store = WorkspaceSnapshotStore(tmp_path / "session", "session")
    store.take(git_repo, label="missing-root")
    state = json.loads(store.state_path.read_text())
    state["snapshots"][0]["repo_root"] = str(tmp_path / "missing")
    state["snapshots"].append({"id": "invalid"})
    store.state_path.write_text(json.dumps(state))
    reopened = WorkspaceSnapshotStore(store.session_dir, "session")

    assert "--force" not in reopened.corruption_message
    assert "no valid workspace snapshot" in reopened.corruption_message


@pytest.mark.parametrize("snapshot_count", [2, 50])
def test_corruption_warning_stops_after_first_restorable_snapshot(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snapshot_count: int
) -> None:
    store = WorkspaceSnapshotStore(tmp_path / "session", "session")
    for index in range(snapshot_count):
        store.take(git_repo, label=str(index))
    state = json.loads(store.state_path.read_text())
    state["snapshots"].append({"id": "invalid"})
    store.state_path.write_text(json.dumps(state))
    reopened = WorkspaceSnapshotStore(store.session_dir, "session")
    calls = []
    run_git = workspace_module._run_git

    def counted_git(args, **kwargs):
        calls.append(args)
        return run_git(args, **kwargs)

    monkeypatch.setattr(workspace_module, "_run_git", counted_git)
    assert "--force" in reopened.corruption_message
    # One root lookup and one ref/commit/tree/object resolution, independent
    # of history length. Full history validation belongs only to reset.
    assert len(calls) <= 4
