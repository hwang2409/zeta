"""Shadow-git workspace snapshots for /checkpoint, /fork, /undo, /redo.

Snapshots capture the tracked + non-ignored-untracked working-tree contents
as loose git objects pinned by refs inside a private bare shadow repo under
the session directory. The user's ``.git`` — index, HEAD, branches, reflog,
object DB — is never touched, so ``git log --all``, ``git for-each-ref``,
``git push --mirror``, and ``git fsck`` all stay quiet.

Limits: snapshots respect ``.gitignore``, so ignored files are never captured
and never deleted on restore. Repositories without a ``.git`` directory
degrade to conversation-only checkpoints. Very large working sets trigger a
size notice on the caller. The store keeps at most ``DEFAULT_SNAPSHOT_CAP``
snapshots per session (settings-overridable) and drops the oldest past the
cap. Restores are atomic: a safety commit is written first, the destructive
prune of stale files happens last, and any interruption leaves both the
partial state and the pre-restore blobs recoverable from shadow refs.
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

from ..process_env import subprocess_env
from . import ConversationIntegrityError, load_session_json

SNAPSHOT_STATE_FILE = "workspace_snapshots.json"
SNAPSHOT_STATE_TMP_PREFIX = ".workspace_snapshots."
SNAPSHOT_REF_NAMESPACE = "refs/zeta/checkpoints"
SNAPSHOT_SAFETY_REF_NAMESPACE = "refs/zeta/safety"
SHADOW_REPO_DIRNAME = "workspace_shadow.git"
SNAPSHOT_MODE_GIT = "git-shadow"
SNAPSHOT_MODE_UNAVAILABLE = "unavailable"
SIZE_NOTICE_THRESHOLD_BYTES = 100 * 1024 * 1024
DEFAULT_SNAPSHOT_CAP = 50


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
        for field in ("id", "created_at", "mode"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ConversationIntegrityError(
                    f"snapshot {field} must be a nonempty string"
                )
        if value["mode"] not in {SNAPSHOT_MODE_GIT, SNAPSHOT_MODE_UNAVAILABLE}:
            raise ConversationIntegrityError("unsupported snapshot mode")
        for field in ("size_bytes", "file_count"):
            number = value.get(field, 0)
            if type(number) is not int or number < 0:
                raise ConversationIntegrityError(
                    f"snapshot {field} must be a nonnegative integer"
                )
        for field in ("commit_sha", "tree_sha", "repo_root", "label", "checkpoint_entry_id"):
            if value.get(field) is not None and not isinstance(value[field], str):
                raise ConversationIntegrityError(
                    f"snapshot {field} must be a string or null"
                )
        if value["mode"] == SNAPSHOT_MODE_GIT:
            for field in ("commit_sha", "tree_sha", "repo_root"):
                if not value.get(field):
                    raise ConversationIntegrityError(
                        f"git snapshot {field} must be a nonempty string"
                    )
        return cls(
            id=value["id"],
            created_at=value["created_at"],
            mode=value["mode"],
            commit_sha=value.get("commit_sha"),
            tree_sha=value.get("tree_sha"),
            size_bytes=value.get("size_bytes", 0),
            file_count=value.get("file_count", 0),
            repo_root=value.get("repo_root"),
            label=value.get("label"),
            checkpoint_entry_id=value.get("checkpoint_entry_id"),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _shadow_dir(session_dir: Path) -> Path:
    return session_dir / SHADOW_REPO_DIRNAME


def _ensure_shadow_repo(session_dir: Path) -> Path:
    """Initialise the per-session bare repo that pins shadow objects and refs."""

    shadow = _shadow_dir(session_dir)
    if not (shadow / "HEAD").exists():
        session_dir.mkdir(parents=True, exist_ok=True)
        shadow.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(shadow)],
                check=True,
                capture_output=True,
                env=subprocess_env(),
            )
        except subprocess.CalledProcessError as exc:
            raise WorkspaceSnapshotError(
                f"git init failed for shadow repo: {_git_error_detail(exc)}"
            ) from exc
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceSnapshotError(
                f"git init failed for shadow repo: {exc}"
            ) from exc
    return shadow


def _shadow_env(session_dir: Path, repo_root: str | Path) -> dict[str, str]:
    """Env that routes objects, refs, and work-tree lookups to the shadow repo."""

    shadow = _ensure_shadow_repo(session_dir)
    return {
        "GIT_DIR": str(shadow),
        "GIT_WORK_TREE": str(repo_root),
    }


def _shadow_ref_env(session_dir: Path) -> dict[str, str]:
    """Env for ref-only operations against the shadow bare repo."""

    shadow = _ensure_shadow_repo(session_dir)
    return {"GIT_DIR": str(shadow)}


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


def _git_error_detail(exc: subprocess.CalledProcessError) -> str:
    stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
    reason = stderr or f"exit {exc.returncode}"
    return reason.splitlines()[0] if reason else f"exit {exc.returncode}"


def _run_git(
    args: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = subprocess_env()
    merged_env["GIT_AUTHOR_NAME"] = merged_env.get("GIT_AUTHOR_NAME", "zeta")
    merged_env["GIT_AUTHOR_EMAIL"] = merged_env.get("GIT_AUTHOR_EMAIL", "zeta@localhost")
    merged_env["GIT_COMMITTER_NAME"] = merged_env.get("GIT_COMMITTER_NAME", "zeta")
    merged_env["GIT_COMMITTER_EMAIL"] = merged_env.get(
        "GIT_COMMITTER_EMAIL", "zeta@localhost"
    )
    if env is not None:
        merged_env.update(env)
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=merged_env,
            input=input_text,
            capture_output=True,
            text=True,
            check=check,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkspaceSnapshotError(
            f"git {args[0] if args else ''} failed: {_git_error_detail(exc)}"
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        if not check:
            raise
        raise WorkspaceSnapshotError(f"git invocation failed: {exc}") from exc


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


def _tree_files(session_dir: Path, repo_root: str, tree_sha: str) -> list[str]:
    result = _run_git(
        ["ls-tree", "-r", "--name-only", "-z", tree_sha],
        cwd=repo_root,
        env=_shadow_ref_env(session_dir),
    )
    if not result.stdout:
        return []
    return [entry for entry in result.stdout.split("\x00") if entry]


def _write_shadow_commit(
    session_dir: Path,
    repo_root: str,
    label: str,
) -> tuple[str, str, int, int]:
    """Build a shadow commit that captures the current worktree.

    Returns ``(commit_sha, tree_sha, size_bytes, file_count)``. The commit
    lives in the shadow bare repo; the caller pins it with a ref there so
    the shadow repo's own gc will not reap it.
    """

    _ensure_shadow_repo(session_dir)
    entries = _tracked_and_untracked(repo_root)
    total_size = 0
    for entry in entries:
        path = Path(repo_root) / entry
        try:
            total_size += path.stat().st_size
        except (FileNotFoundError, OSError):
            continue

    commit_sha, tree_sha = _write_current_shadow_commit(
        session_dir, repo_root, message=f"zeta shadow checkpoint: {label}"
    )
    return commit_sha, tree_sha, total_size, len(entries)


def _write_current_shadow_commit(
    session_dir: Path,
    repo_root: str,
    *,
    message: str,
) -> tuple[str, str]:
    """Snapshot the working tree into a shadow commit; return (commit, tree)."""

    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {**_shadow_env(session_dir, repo_root), "GIT_INDEX_FILE": str(index_path)}
        _run_git(["add", "--all"], cwd=repo_root, env=env)
        tree_result = _run_git(["write-tree"], cwd=repo_root, env=env)
        tree_sha = tree_result.stdout.strip()
        commit_result = _run_git(
            ["commit-tree", tree_sha, "-m", message],
            cwd=repo_root,
            env=env,
        )
        commit_sha = commit_result.stdout.strip()
    finally:
        index_path.unlink(missing_ok=True)
    if not tree_sha or not commit_sha:
        raise WorkspaceSnapshotError("git produced empty tree or commit sha")
    return commit_sha, tree_sha


def _update_shadow_ref(session_dir: Path, ref: str, commit_sha: str) -> None:
    _run_git(
        ["update-ref", ref, commit_sha],
        cwd=session_dir,
        env=_shadow_ref_env(session_dir),
    )


def _delete_shadow_ref(session_dir: Path, ref: str) -> None:
    _run_git(
        ["update-ref", "-d", ref],
        cwd=session_dir,
        env=_shadow_ref_env(session_dir),
        check=False,
    )


def _pin_ref(session_dir: Path, snapshot_id: str, commit_sha: str) -> None:
    _update_shadow_ref(
        session_dir, f"{SNAPSHOT_REF_NAMESPACE}/{snapshot_id}", commit_sha
    )


def _restore_from_tree(
    session_dir: Path,
    repo_root: str,
    tree_sha: str,
) -> None:
    """Restore the working tree to ``tree_sha`` without losing pre-restore blobs.

    Ordering guarantees atomicity: before any mutation we pin the current
    tree under a safety ref; then we materialise target files (writes and
    overwrites only); the destructive prune of stale files is the last step.
    Any interruption leaves the pre-restore blobs pinned in the shadow repo,
    so nothing is permanently lost.
    """

    safety_ref = f"{SNAPSHOT_SAFETY_REF_NAMESPACE}/{uuid.uuid4().hex}"
    safety_commit, _safety_tree = _write_current_shadow_commit(
        session_dir, repo_root, message="zeta pre-restore safety"
    )
    _update_shadow_ref(session_dir, safety_ref, safety_commit)
    committed = False
    try:
        _materialise_tree(session_dir, repo_root, tree_sha)
        _prune_stale_entries(session_dir, repo_root, tree_sha)
        committed = True
    finally:
        if committed:
            _delete_shadow_ref(session_dir, safety_ref)


def _materialise_tree(session_dir: Path, repo_root: str, tree_sha: str) -> None:
    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {**_shadow_env(session_dir, repo_root), "GIT_INDEX_FILE": str(index_path)}
        _run_git(["read-tree", tree_sha], cwd=repo_root, env=env)
        _run_git(
            ["checkout-index", "--all", "--force", f"--prefix={repo_root}/"],
            cwd=repo_root,
            env=env,
        )
    finally:
        index_path.unlink(missing_ok=True)


def _prune_stale_entries(session_dir: Path, repo_root: str, tree_sha: str) -> None:
    current_entries = set(_tracked_and_untracked(repo_root))
    target_entries = set(_tree_files(session_dir, repo_root, tree_sha))
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


def _current_tree_sha(session_dir: Path, repo_root: str) -> str:
    index_path = _reserve_shadow_index_path(session_dir)
    try:
        env = {**_shadow_env(session_dir, repo_root), "GIT_INDEX_FILE": str(index_path)}
        _run_git(["add", "--all"], cwd=repo_root, env=env)
        tree_result = _run_git(["write-tree"], cwd=repo_root, env=env)
    finally:
        index_path.unlink(missing_ok=True)
    return tree_result.stdout.strip()


class WorkspaceSnapshotStore:
    """Persistent shadow-git snapshots with undo/redo linear history."""

    def __init__(
        self,
        session_dir: str | Path,
        session_id: str,
        *,
        cap: int | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.state_path = self.session_dir / SNAPSHOT_STATE_FILE
        resolved_cap = DEFAULT_SNAPSHOT_CAP if cap is None else int(cap)
        if resolved_cap < 1:
            raise ValueError("workspace snapshot cap must be >= 1")
        self._cap = resolved_cap
        self._snapshots: list[WorkspaceSnapshot] = []
        self._current_id: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = load_session_json(self.state_path)
        except ConversationIntegrityError:
            return
        if not isinstance(raw, dict):
            return
        snapshots = raw.get("snapshots", [])
        if not isinstance(snapshots, list) or any(
            not isinstance(entry, dict) for entry in snapshots
        ):
            return
        try:
            self._snapshots = [WorkspaceSnapshot.from_dict(entry) for entry in snapshots]
        except ConversationIntegrityError:
            return
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

    @property
    def cap(self) -> int:
        return self._cap

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

        self._truncate_tail_from_current()
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
                self.session_dir, repo_root, label
            )
            _pin_ref(self.session_dir, snapshot_id, commit_sha)
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
        self._enforce_cap()
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
        _restore_from_tree(self.session_dir, repo_root, snapshot.tree_sha)
        self._current_id = snapshot.id
        self._persist()
        return snapshot

    def set_current(self, snapshot_id: str | None) -> None:
        if snapshot_id is not None and self.by_id(snapshot_id) is None:
            raise WorkspaceSnapshotError(f"snapshot not found: {snapshot_id}")
        self._current_id = snapshot_id
        self._persist()

    def is_dirty(self, cwd: str | Path) -> bool:
        """Return True when the working tree differs or cannot be verified clean.

        Falls back to the latest git snapshot when ``current_id`` is missing
        or stale. An absent reference or failed comparison must not bypass
        the dirty guard.
        """

        reference = self.current()
        if reference is None or reference.mode != SNAPSHOT_MODE_GIT:
            reference = next(
                (
                    snap
                    for snap in reversed(self._snapshots)
                    if snap.mode == SNAPSHOT_MODE_GIT
                ),
                None,
            )
        if reference is None or reference.tree_sha is None:
            return True
        repo_root = reference.repo_root or git_repo_root(cwd)
        if repo_root is None:
            return True
        try:
            current_tree = _current_tree_sha(self.session_dir, repo_root)
        except WorkspaceSnapshotError:
            return True
        return current_tree != reference.tree_sha

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

    def _truncate_tail_from_current(self) -> None:
        """Drop any snapshots after ``_current_id`` so /undo can't jump to them."""

        if self._current_id is None or not self._snapshots:
            return
        try:
            index = next(
                i
                for i, snap in enumerate(self._snapshots)
                if snap.id == self._current_id
            )
        except StopIteration:
            return
        tail = self._snapshots[index + 1 :]
        if not tail:
            return
        self._snapshots = self._snapshots[: index + 1]
        for dropped in tail:
            self._delete_snapshot_ref(dropped)

    def _enforce_cap(self) -> None:
        if len(self._snapshots) <= self._cap:
            return
        overflow = len(self._snapshots) - self._cap
        dropped = self._snapshots[:overflow]
        self._snapshots = self._snapshots[overflow:]
        for old in dropped:
            self._delete_snapshot_ref(old)
        if self._current_id is not None and not any(
            snap.id == self._current_id for snap in self._snapshots
        ):
            # The current snapshot was the one we dropped; anchor to the
            # oldest survivor so undo/redo remain coherent.
            self._current_id = self._snapshots[0].id if self._snapshots else None

    def _delete_snapshot_ref(self, snapshot: WorkspaceSnapshot) -> None:
        if snapshot.mode != SNAPSHOT_MODE_GIT:
            return
        _delete_shadow_ref(
            self.session_dir, f"{SNAPSHOT_REF_NAMESPACE}/{snapshot.id}"
        )
