"""Shadow-git workspace snapshots for /checkpoint, /fork, /undo, /redo.

Snapshots capture the tracked + non-ignored-untracked working-tree contents
as loose git objects pinned by a ref under ``refs/zeta/checkpoints/``. That
namespace is invisible to ``git branch``, ``git tag``, and HEAD's reflog, so
the user's index, HEAD, branches, and reflog-visible state are never touched.

Limits: snapshots respect ``.gitignore``, so ignored files are never captured
and never deleted on restore. Repositories without a ``.git`` directory
degrade to conversation-only checkpoints. Very large working sets trigger a
size notice on the caller.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_STATE_FILE = "workspace_snapshots.json"
SNAPSHOT_STATE_TMP_PREFIX = ".workspace_snapshots."
SNAPSHOT_REF_NAMESPACE = "refs/zeta/checkpoints"
SNAPSHOT_MODE_GIT = "git-shadow"
SNAPSHOT_MODE_UNAVAILABLE = "unavailable"
SIZE_NOTICE_THRESHOLD_BYTES = 100 * 1024 * 1024


class WorkspaceSnapshotError(RuntimeError):
    """Raised when git shadow snapshot machinery fails."""


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """One captured working-tree snapshot."""

    id: str
    created_at: str
    mode: str
    commit_sha: str | None
    tree_sha: str | None
    size_bytes: int
    file_count: int
    repo_root: str | None
    label: str | None = None
    checkpoint_entry_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "mode": self.mode,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "repo_root": self.repo_root,
            "label": self.label,
            "checkpoint_entry_id": self.checkpoint_entry_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkspaceSnapshot:
        return cls(
            id=str(value["id"]),
            created_at=str(value["created_at"]),
            mode=str(value["mode"]),
            commit_sha=value.get("commit_sha"),
            tree_sha=value.get("tree_sha"),
            size_bytes=int(value.get("size_bytes", 0)),
            file_count=int(value.get("file_count", 0)),
            repo_root=value.get("repo_root"),
            label=value.get("label"),
            checkpoint_entry_id=value.get("checkpoint_entry_id"),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reserve_shadow_index_path(session_dir: Path) -> Path:
    """Reserve a unique path for a shadow git index, without creating the file.

    git refuses an empty index file, so we create-and-delete atomically to
    hold a name in the session dir without leaving a zero-byte index behind.
    """

    session_dir.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=str(session_dir), prefix=".shadow-index.", suffix=".idx"
    )
    os.close(handle)
    path = Path(name)
    path.unlink(missing_ok=True)
    return path


def _run_git(
    args: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["GIT_AUTHOR_NAME"] = merged_env.get("GIT_AUTHOR_NAME", "zeta")
    merged_env["GIT_AUTHOR_EMAIL"] = merged_env.get("GIT_AUTHOR_EMAIL", "zeta@localhost")
    merged_env["GIT_COMMITTER_NAME"] = merged_env.get("GIT_COMMITTER_NAME", "zeta")
    merged_env["GIT_COMMITTER_EMAIL"] = merged_env.get(
        "GIT_COMMITTER_EMAIL", "zeta@localhost"
    )
    if env is not None:
        merged_env.update(env)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=merged_env,
        input=input_text,
        capture_output=True,
        text=True,
        check=check,
    )


def git_repo_root(cwd: str | Path) -> str | None:
    """Return the working tree root git manages, or None outside a repo."""

    try:
        result = _run_git(
            ["rev-parse", "--show-toplevel"], cwd=cwd, check=False
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return root or None


def _tracked_and_untracked(repo_root: str) -> list[str]:
    result = _run_git(
        ["ls-files", "-co", "--exclude-standard", "-z"],
        cwd=repo_root,
    )
    if not result.stdout:
        return []
    return [entry for entry in result.stdout.split("\x00") if entry]


def _tree_files(repo_root: str, tree_sha: str) -> list[str]:
    result = _run_git(
        ["ls-tree", "-r", "--name-only", "-z", tree_sha],
        cwd=repo_root,
    )
    if not result.stdout:
        return []
    return [entry for entry in result.stdout.split("\x00") if entry]


def _write_shadow_commit(
    repo_root: str,
    session_dir: Path,
    label: str,
) -> tuple[str, str, int, int]:
    """Build a shadow commit that captures the current worktree.

    Returns ``(commit_sha, tree_sha, size_bytes, file_count)``. The commit is
    pinned by the caller via a ref; without a ref, git's periodic gc could
    reap it.
    """

    session_dir.mkdir(parents=True, exist_ok=True)
    entries = _tracked_and_untracked(repo_root)
    total_size = 0
    for entry in entries:
        path = Path(repo_root) / entry
        try:
            total_size += path.stat().st_size
        except (FileNotFoundError, OSError):
            continue

    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {"GIT_INDEX_FILE": str(index_path)}
        _run_git(
            ["add", "--all"],
            cwd=repo_root,
            env=env,
        )
        tree_result = _run_git(
            ["write-tree"],
            cwd=repo_root,
            env=env,
        )
        tree_sha = tree_result.stdout.strip()
        commit_result = _run_git(
            ["commit-tree", tree_sha, "-m", f"zeta shadow checkpoint: {label}"],
            cwd=repo_root,
        )
        commit_sha = commit_result.stdout.strip()
    finally:
        index_path.unlink(missing_ok=True)
    if not tree_sha or not commit_sha:
        raise WorkspaceSnapshotError("git produced empty tree or commit sha")
    return commit_sha, tree_sha, total_size, len(entries)


def _pin_ref(repo_root: str, snapshot_id: str, commit_sha: str) -> None:
    _run_git(
        ["update-ref", f"{SNAPSHOT_REF_NAMESPACE}/{snapshot_id}", commit_sha],
        cwd=repo_root,
    )


def _restore_from_tree(
    repo_root: str,
    session_dir: Path,
    tree_sha: str,
) -> None:
    current_entries = set(_tracked_and_untracked(repo_root))
    target_entries = set(_tree_files(repo_root, tree_sha))
    stale = current_entries - target_entries
    for entry in stale:
        target = Path(repo_root) / entry
        try:
            target.unlink()
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            shutil.rmtree(target, ignore_errors=True)
    _prune_empty_parents(Path(repo_root), stale)

    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {"GIT_INDEX_FILE": str(index_path)}
        _run_git(["read-tree", tree_sha], cwd=repo_root, env=env)
        _run_git(
            ["checkout-index", "--all", "--force", f"--prefix={repo_root}/"],
            cwd=repo_root,
            env=env,
        )
    finally:
        index_path.unlink(missing_ok=True)


def _prune_empty_parents(repo_root: Path, entries: Iterable[str]) -> None:
    seen: set[Path] = set()
    for entry in entries:
        parent = (repo_root / entry).parent
        while parent != repo_root and parent not in seen:
            seen.add(parent)
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _current_tree_sha(repo_root: str, session_dir: Path) -> str:
    session_dir.mkdir(parents=True, exist_ok=True)
    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {"GIT_INDEX_FILE": str(index_path)}
        _run_git(["add", "--all"], cwd=repo_root, env=env)
        tree_result = _run_git(["write-tree"], cwd=repo_root, env=env)
    finally:
        index_path.unlink(missing_ok=True)
    return tree_result.stdout.strip()


class WorkspaceSnapshotStore:
    """Persistent shadow-git snapshots with undo/redo linear history."""

    def __init__(self, session_dir: str | Path, session_id: str) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.state_path = self.session_dir / SNAPSHOT_STATE_FILE
        self._snapshots: list[WorkspaceSnapshot] = []
        self._current_id: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        snapshots = raw.get("snapshots", [])
        if isinstance(snapshots, list):
            self._snapshots = [
                WorkspaceSnapshot.from_dict(entry)
                for entry in snapshots
                if isinstance(entry, dict) and "id" in entry
            ]
        current_id = raw.get("current_id")
        if isinstance(current_id, str) and any(
            snap.id == current_id for snap in self._snapshots
        ):
            self._current_id = current_id

    def _persist(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshots": [snapshot.to_dict() for snapshot in self._snapshots],
            "current_id": self._current_id,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.session_dir,
                prefix=SNAPSHOT_STATE_TMP_PREFIX,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, separators=(",", ":"), sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.state_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    @property
    def snapshots(self) -> tuple[WorkspaceSnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def current_id(self) -> str | None:
        return self._current_id

    def current(self) -> WorkspaceSnapshot | None:
        if self._current_id is None:
            return None
        return self.by_id(self._current_id)

    def by_id(self, snapshot_id: str) -> WorkspaceSnapshot | None:
        for snapshot in self._snapshots:
            if snapshot.id == snapshot_id:
                return snapshot
        return None

    def by_checkpoint(self, checkpoint_entry_id: str) -> WorkspaceSnapshot | None:
        for snapshot in self._snapshots:
            if snapshot.checkpoint_entry_id == checkpoint_entry_id:
                return snapshot
        return None

    def take(
        self,
        cwd: str | Path,
        *,
        label: str,
        checkpoint_entry_id: str | None = None,
    ) -> WorkspaceSnapshot:
        """Capture the current working tree as a shadow-git snapshot."""

        repo_root = git_repo_root(cwd)
        snapshot_id = uuid.uuid4().hex
        if repo_root is None:
            snapshot = WorkspaceSnapshot(
                id=snapshot_id,
                created_at=_now(),
                mode=SNAPSHOT_MODE_UNAVAILABLE,
                commit_sha=None,
                tree_sha=None,
                size_bytes=0,
                file_count=0,
                repo_root=None,
                label=label,
                checkpoint_entry_id=checkpoint_entry_id,
            )
        else:
            commit_sha, tree_sha, size_bytes, file_count = _write_shadow_commit(
                repo_root, self.session_dir, label
            )
            _pin_ref(repo_root, snapshot_id, commit_sha)
            snapshot = WorkspaceSnapshot(
                id=snapshot_id,
                created_at=_now(),
                mode=SNAPSHOT_MODE_GIT,
                commit_sha=commit_sha,
                tree_sha=tree_sha,
                size_bytes=size_bytes,
                file_count=file_count,
                repo_root=repo_root,
                label=label,
                checkpoint_entry_id=checkpoint_entry_id,
            )
        self._snapshots.append(snapshot)
        self._current_id = snapshot.id
        self._persist()
        return snapshot

    def restore(self, snapshot_id: str, cwd: str | Path) -> WorkspaceSnapshot:
        snapshot = self.by_id(snapshot_id)
        if snapshot is None:
            raise WorkspaceSnapshotError(f"snapshot not found: {snapshot_id}")
        if snapshot.mode == SNAPSHOT_MODE_UNAVAILABLE:
            self._current_id = snapshot.id
            self._persist()
            return snapshot
        repo_root = snapshot.repo_root or git_repo_root(cwd)
        if repo_root is None or snapshot.tree_sha is None:
            raise WorkspaceSnapshotError(
                f"snapshot {snapshot_id} has no restorable tree"
            )
        _restore_from_tree(repo_root, self.session_dir, snapshot.tree_sha)
        self._current_id = snapshot.id
        self._persist()
        return snapshot

    def set_current(self, snapshot_id: str | None) -> None:
        if snapshot_id is not None and self.by_id(snapshot_id) is None:
            raise WorkspaceSnapshotError(f"snapshot not found: {snapshot_id}")
        self._current_id = snapshot_id
        self._persist()

    def is_dirty(self, cwd: str | Path) -> bool:
        """Return True when the working tree differs from the current snapshot."""

        current = self.current()
        if current is None or current.mode != SNAPSHOT_MODE_GIT:
            return False
        repo_root = current.repo_root or git_repo_root(cwd)
        if repo_root is None:
            return False
        try:
            current_tree = _current_tree_sha(repo_root, self.session_dir)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return False
        return current_tree != current.tree_sha

    def undo_target(self) -> WorkspaceSnapshot | None:
        if self._current_id is None:
            if self._snapshots:
                return self._snapshots[-1]
            return None
        try:
            index = next(
                i
                for i, snap in enumerate(self._snapshots)
                if snap.id == self._current_id
            )
        except StopIteration:
            return None
        if index == 0:
            return None
        return self._snapshots[index - 1]

    def redo_target(self) -> WorkspaceSnapshot | None:
        if self._current_id is None:
            return None
        try:
            index = next(
                i
                for i, snap in enumerate(self._snapshots)
                if snap.id == self._current_id
            )
        except StopIteration:
            return None
        if index + 1 >= len(self._snapshots):
            return None
        return self._snapshots[index + 1]
