from __future__ import annotations

import hashlib
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.checkpoints.workspace import (
    SIZE_NOTICE_THRESHOLD_BYTES,
    SNAPSHOT_MODE_GIT,
    SNAPSHOT_MODE_UNAVAILABLE,
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


def test_shadow_refs_live_outside_branches_and_tags(
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
    zeta_refs = subprocess.check_output(
        ["git", "for-each-ref", "refs/zeta/"], cwd=git_repo, text=True
    )

    assert "refs/zeta" not in branches
    assert tags.strip() == ""
    assert stash.strip() == ""
    assert "refs/zeta/checkpoints/" in zeta_refs


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
