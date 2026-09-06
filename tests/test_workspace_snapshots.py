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
