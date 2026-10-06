"""Explicit, credential-free session and project-memory transfer.

The public functions form the transfer seam.  A transport owns only publication
and retrieval below another ZETA_HOME; snapshot, validation, conflict, and cwd
rules remain here so local tests and SSH use the same behavior.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..core.session import SessionManager
from ..core.session_files import SessionError, SessionInUseError, session_directory

_SCHEMA = "zeta.session-transfer.v1"
_MEMORY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
_EXCLUDED_NAMES = frozenset({".lock", ".spill.lock"})
_CREDENTIAL_NAMES = frozenset(
    {"oauth.json", "credentials.json", "tokens.json", "auth.json"}
)


class RemoteSyncError(ValueError):
    """A transfer was unsafe or conflicted with newer state."""


@dataclass(frozen=True, slots=True)
class SessionTransferResult:
    session_id: str
    digest: str
    last_seq: int
    resume_notice: str


@dataclass(frozen=True, slots=True)
class MemoryTransferResult:
    project_id: str
    updated: tuple[str, ...]
    conflicts: tuple[str, ...]


class Transport(Protocol):
    """Publication seam implemented by local tests and SSH."""

    name: str

    def publish_session(self, snapshot: Path, *, force: bool) -> Path: ...

    def fetch_session(self, session_id: str, destination: Path) -> Path: ...

    def push_memory(self, source_home: Path, project_id: str) -> MemoryTransferResult: ...

    def pull_memory(self, destination_home: Path, project_id: str) -> MemoryTransferResult: ...


@dataclass(slots=True)
class LocalTransport:
    """A second ZETA_HOME adapter used by tests and local workflows."""

    _home: Path
    name: str = "local"

    def __init__(self, home: str | Path, name: str = "local") -> None:
        self._home = Path(home).expanduser().resolve()
        self.name = name

    @property
    def home(self) -> Path:
        return self._home

    def publish_session(self, snapshot: Path, *, force: bool) -> Path:
        manifest = _read_manifest(snapshot)
        session_id = manifest["session_id"]
        sessions = self.home / "sessions"
        sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = sessions / session_id
        if destination.exists():
            source_state = _tree_state(snapshot)
            destination_state = _tree_state(destination)
            if not force and destination_state != source_state:
                if destination_state[0] >= source_state[0]:
                    raise RemoteSyncError(
                        "newer remote session exists; use --force to replace it"
                    )
                raise RemoteSyncError(
                    "remote session differs from the snapshot; use --force to replace it"
                )
        staging = sessions / f".{session_id}.incoming-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(snapshot, staging)
        _map_missing_cwd(staging, self.home / "remote-workspaces" / session_id)
        _atomic_replace_directory(staging, destination)
        return destination

    def fetch_session(self, session_id: str, destination: Path) -> Path:
        source = self.home / "sessions" / _safe_component(session_id, "session id")
        if not source.is_dir():
            raise RemoteSyncError(f"remote session {session_id} was not found")
        with _snapshot_locks(source):
            _copy_tree(source, destination)
        return destination

    def push_memory(self, source_home: Path, project_id: str) -> MemoryTransferResult:
        return _sync_memory(
            source_home,
            self.home,
            project_id=project_id,
            peer=self.name,
            source_label="local",
        )

    def pull_memory(
        self, destination_home: Path, project_id: str
    ) -> MemoryTransferResult:
        return _sync_memory(
            self.home,
            destination_home,
            project_id=project_id,
            peer=self.name,
            source_label=self.name,
        )


def resolve_transport(
    home: str | Path, target: str, *, remote_home: str | None = None
) -> Transport:
    """Resolve a configured alias or an explicitly supplied SSH host."""

    if not isinstance(target, str) or not target.strip():
        raise RemoteSyncError("remote host must be nonempty")
    settings = Path(home).expanduser() / "settings.toml"
    remotes: object = {}
    if settings.exists():
        try:
            remotes = tomllib.loads(settings.read_text(encoding="utf-8")).get(
                "remotes", {}
            )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise RemoteSyncError(f"cannot read remote settings: {exc}") from exc
    if not isinstance(remotes, dict):
        raise RemoteSyncError("[remotes] must be a table of ssh:// URLs")
    configured = remotes.get(target)
    from .ssh import SshTransport

    if configured is not None:
        if not isinstance(configured, str):
            raise RemoteSyncError(f"remote {target!r} must be an ssh:// URL")
        transport = SshTransport.from_url(target, configured)
        if remote_home is not None:
            transport.remote_home = remote_home
        return transport
    if "://" in target:
        raise RemoteSyncError(
            "unknown remote URL; configure an alias or pass an SSH host explicitly"
        )
    return SshTransport(target, remote_home or "~/.zeta", name=target)


def push_session(
    home: str | Path,
    transport: Transport,
    *,
    session_id: str | None = None,
    force: bool = False,
) -> SessionTransferResult:
    """Push one consistent session snapshot and its linked project memory."""

    local_home = Path(home).expanduser().resolve()
    manager = SessionManager(local_home)
    if session_id is None:
        metadata = manager.find_most_recent()
    else:
        metadata = manager.read_metadata(manager.resolve_id(session_id))
    session_id = metadata.session_id
    source = manager.sessions_dir / session_id
    with tempfile.TemporaryDirectory(prefix="zeta-session-push-") as temporary:
        snapshot = Path(temporary) / session_id
        with _snapshot_locks(source):
            _copy_tree(source, snapshot)
            manifest = _make_manifest(snapshot, metadata.cwd)
            _write_json(snapshot / "transfer.json", manifest)
        if metadata.project_id is not None:
            transport.push_memory(local_home, metadata.project_id)
        published = transport.publish_session(snapshot, force=force)
        final = _read_manifest(published)
    return _result(final)


def pull_session(
    home: str | Path,
    transport: Transport,
    *,
    session_id: str,
    cwd: str | Path | None = None,
    force: bool = False,
) -> SessionTransferResult:
    """Pull one session atomically and map its cwd to an existing local path."""

    local_home = Path(home).expanduser().resolve()
    safe_id = _safe_component(session_id, "session id")
    sessions = local_home / "sessions"
    sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="zeta-session-pull-") as temporary:
        snapshot = Path(temporary) / safe_id
        transport.fetch_session(safe_id, snapshot)
        _validate_snapshot(snapshot, safe_id)
        metadata = _read_json(snapshot / "meta.json")
        project_id = metadata.get("project_id")
        if project_id is not None:
            if not isinstance(project_id, str):
                raise RemoteSyncError("session project id is invalid")
            transport.pull_memory(local_home, _safe_component(project_id, "project id"))
        destination = sessions / safe_id
        lease = (
            session_directory(sessions, safe_id, exclusive=True)
            if destination.exists()
            else nullcontext()
        )
        try:
            with lease:
                if destination.exists():
                    source_state = _tree_state(snapshot)
                    destination_state = _tree_state(destination)
                    if not force and destination_state != source_state:
                        if destination_state[0] >= source_state[0]:
                            raise RemoteSyncError(
                                "newer local session exists; use --force to replace it"
                            )
                        raise RemoteSyncError(
                            "local session differs from remote; use --force to replace it"
                        )
                mapped = (
                    Path(cwd).expanduser().resolve()
                    if cwd is not None
                    else local_home / "remote-workspaces" / safe_id
                )
                mapped.mkdir(parents=True, exist_ok=True, mode=0o700)
                previous = _read_manifest(snapshot)
                _rewrite_cwd(snapshot, mapped)
                _append_resume_hint(snapshot, mapped, previous)
                manifest = _make_manifest(snapshot, str(mapped), previous=previous)
                _write_json(snapshot / "transfer.json", manifest)
                _atomic_replace_directory(snapshot, destination)
        except SessionInUseError as exc:
            raise RemoteSyncError(
                "local session is active; stop it before replacement"
            ) from exc
        except SessionError as exc:
            raise RemoteSyncError(str(exc)) from exc
    return _result(manifest)


def push_project_memory(
    home: str | Path,
    transport: Transport,
    *,
    project_id: str,
) -> MemoryTransferResult:
    """Push standard memory files with per-file three-way CAS semantics."""

    return transport.push_memory(
        Path(home).expanduser().resolve(),
        _safe_component(project_id, "project id"),
    )


def pull_project_memory(
    home: str | Path,
    transport: Transport,
    *,
    project_id: str,
) -> MemoryTransferResult:
    """Pull standard memory files with per-file three-way CAS semantics."""

    return transport.pull_memory(
        Path(home).expanduser().resolve(),
        _safe_component(project_id, "project id"),
    )


def _sync_memory(
    source_home: Path,
    destination_home: Path,
    *,
    project_id: str,
    peer: str,
    source_label: str,
) -> MemoryTransferResult:
    source = source_home / "projects" / project_id
    destination = destination_home / "projects" / project_id
    if not source.is_dir():
        raise RemoteSyncError(f"project {project_id} was not found")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _copy_project_tree(source, destination)
    source_memory = source / "memory"
    destination_memory = destination / "memory"
    destination_memory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir = source / "sync"
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = state_dir / f"{_safe_component(peer, 'remote name')}.json"
    baseline = _read_json(state_path) if state_path.exists() else {}
    baseline_files = baseline.get("files", {}) if isinstance(baseline, dict) else {}
    updated: list[str] = []
    conflicts: list[str] = []
    next_files: dict[str, str | None] = {}
    for name in _MEMORY_FILES:
        source_path = source_memory / name
        destination_path = destination_memory / name
        source_digest = _file_digest(source_path)
        destination_digest = _file_digest(destination_path)
        old = baseline_files.get(name) if isinstance(baseline_files, dict) else None
        if source_digest == destination_digest:
            next_files[name] = source_digest
            continue
        source_changed = old is None or source_digest != old
        destination_changed = old is None or destination_digest != old
        if source_changed and destination_changed:
            _keep_conflict(source_path, destination_memory, name, peer=source_label)
            conflicts.append(name)
            next_files[name] = destination_digest
            continue
        if destination_changed:
            _keep_conflict(source_path, destination_memory, name, peer=source_label)
            conflicts.append(name)
            next_files[name] = destination_digest
            continue
        if source_path.exists():
            _atomic_copy_file(source_path, destination_path)
        elif destination_path.exists():
            destination_path.unlink()
        updated.append(name)
        next_files[name] = source_digest
    state = {"schema": 1, "project_id": project_id, "files": next_files}
    _write_json(state_path, state)
    destination_state = destination / "sync"
    destination_state.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_json(destination_state / f"{_safe_component(peer, 'remote name')}.json", state)
    return MemoryTransferResult(project_id, tuple(updated), tuple(conflicts))


def _keep_conflict(source: Path, destination: Path, name: str, *, peer: str) -> None:
    if not source.exists():
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    _atomic_copy_file(source, destination / f"{name}.conflict-{peer}-{stamp}")


@contextmanager
def _snapshot_locks(root: Path) -> Iterator[None]:
    """Hold every append lock while copying, with the root acquired first."""

    if not root.is_dir() or root.is_symlink():
        raise RemoteSyncError("session directory is missing or unsafe")
    with ExitStack() as stack:
        root_lock = root / ".lock"
        candidates = [root_lock] if root_lock.is_file() else []
        candidates.extend(
            path
            for path in sorted(root.rglob(".lock"))
            if path != root_lock and path.is_file() and not path.is_symlink()
        )
        for lock_path in candidates:
            handle = stack.enter_context(lock_path.open("rb"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _copy_project_tree(source: Path, destination: Path) -> None:
    """Copy project identity plus standard memory and memory history only."""

    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = source / "project.json"
    if not record.is_file() or record.is_symlink():
        raise RemoteSyncError("project record is missing or unsafe")
    _atomic_copy_file(record, destination / "project.json")
    source_memory = source / "memory"
    destination_memory = destination / "memory"
    destination_memory.mkdir(mode=0o700)
    for name in _MEMORY_FILES:
        path = source_memory / name
        if path.is_file() and not path.is_symlink():
            _atomic_copy_file(path, destination_memory / name)
    for relative_history in (Path("memory/history"), Path("history")):
        history = source / relative_history
        if history.exists():
            _copy_tree(history, destination / relative_history)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, mode=0o700)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {"..", ""} for part in relative.parts):
            raise RemoteSyncError("unsafe path in session snapshot")
        if path.is_symlink():
            raise RemoteSyncError(f"session snapshot contains a symlink: {relative}")
        if path.name in _EXCLUDED_NAMES:
            continue
        if path.name.lower() in _CREDENTIAL_NAMES and "spill" not in relative.parts:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(path, target)
            target.chmod(0o600)
        else:
            raise RemoteSyncError(f"unsupported file in session snapshot: {relative}")


def _make_manifest(
    snapshot: Path, cwd: str, *, previous: dict[str, object] | None = None
) -> dict[str, object]:
    session_id = snapshot.name
    git = _git_metadata(Path(cwd))
    if previous and not git.get("remote_url"):
        prior_git = previous.get("git")
        if isinstance(prior_git, dict):
            git = prior_git
    last_seq, digest = _tree_state(snapshot)
    source_cwd = cwd
    if previous is not None:
        prior_source = previous.get("source_cwd", previous.get("stored_cwd"))
        if isinstance(prior_source, str) and prior_source:
            source_cwd = prior_source
    remote_url = git.get("remote_url")
    notice = (
        f"The stored cwd was unavailable. Clone {remote_url} into this cwd before work."
        if remote_url
        else "The stored cwd was unavailable. Restore or clone the project into this cwd before work."
    )
    return {
        "schema": _SCHEMA,
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "last_seq": last_seq,
        "digest": digest,
        "source_cwd": source_cwd,
        "resume_cwd": cwd,
        "git": git,
        "includes_spill_files": True,
        "resume_notice": notice,
    }


def _git_metadata(cwd: Path) -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            value = subprocess.check_output(
                ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        return value or None

    return {
        "remote_url": run("remote", "get-url", "origin"),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
    }


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in _EXCLUDED_NAMES:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_state(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    last_seq = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in _EXCLUDED_NAMES:
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "transfer.json":
            continue
        data = path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + data)
        if path.name == "conversation.jsonl":
            last_seq += max(0, len(data.splitlines()) - 1)
    return last_seq, digest.hexdigest()


def _rewrite_cwd(snapshot: Path, cwd: Path) -> None:
    cwd_text = str(cwd.resolve())
    meta_path = snapshot / "meta.json"
    meta = _read_json(meta_path)
    meta["cwd"] = cwd_text
    _write_json(meta_path, meta)
    for log in snapshot.rglob("conversation.jsonl"):
        lines = log.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RemoteSyncError(f"empty conversation log: {log}")
        header = json.loads(lines[0])
        header["data"]["cwd"] = cwd_text
        log.write_text(
            "\n".join([json.dumps(header, separators=(",", ":")), *lines[1:]]) + "\n",
            encoding="utf-8",
        )
    for state_path in snapshot.rglob("session_state.json"):
        state = _read_json(state_path)
        state["bash_cwd"] = cwd_text
        _write_json(state_path, state)


def _append_resume_hint(
    snapshot: Path, cwd: Path, manifest: dict[str, object]
) -> None:
    meta_path = snapshot / "meta.json"
    metadata = _read_json(meta_path)
    prompt = metadata.get("system_prompt", "")
    if not isinstance(prompt, str):
        raise RemoteSyncError("session system prompt is invalid")
    git = manifest.get("git", {})
    remote_url = git.get("remote_url") if isinstance(git, dict) else None
    repository = remote_url if isinstance(remote_url, str) else "the original repository"
    start = "<zeta-remote-resume>"
    if start in prompt:
        prompt = prompt.split(start, 1)[0].rstrip()
    hint = (
        f"{start}\n"
        f"This session was transferred to another machine. Its mapped cwd is {cwd}. "
        f"If the project files are absent, clone {repository} into that directory "
        "before editing code.\n"
        "</zeta-remote-resume>"
    )
    metadata["system_prompt"] = f"{prompt}\n\n{hint}" if prompt else hint
    _write_json(meta_path, metadata)


def _map_missing_cwd(snapshot: Path, placeholder: Path) -> None:
    manifest = _read_manifest(snapshot)
    cwd = manifest.get("resume_cwd", manifest.get("source_cwd"))
    if not isinstance(cwd, str) or not Path(cwd).is_dir():
        placeholder.mkdir(parents=True, exist_ok=True, mode=0o700)
        _rewrite_cwd(snapshot, placeholder)
        _append_resume_hint(snapshot, placeholder, manifest)
        updated = _make_manifest(snapshot, str(placeholder), previous=manifest)
        _write_json(snapshot / "transfer.json", updated)


def _validate_snapshot(snapshot: Path, session_id: str) -> None:
    manifest = _read_manifest(snapshot)
    if manifest.get("session_id") != session_id:
        raise RemoteSyncError("session manifest id does not match the requested session")
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise RemoteSyncError("remote session contains a symlink")


def _read_manifest(snapshot: Path) -> dict[str, object]:
    value = _read_json(snapshot / "transfer.json")
    if value.get("schema") != _SCHEMA:
        raise RemoteSyncError("unsupported or missing session transfer manifest")
    return value


def _result(manifest: dict[str, object]) -> SessionTransferResult:
    return SessionTransferResult(
        session_id=str(manifest["session_id"]),
        digest=str(manifest["digest"]),
        last_seq=int(manifest["last_seq"]),
        resume_notice=str(manifest["resume_notice"]),
    )


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.parent / f".{destination.name}.replaced-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, destination)


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteSyncError(f"cannot read transfer data: {path}") from exc
    if not isinstance(value, dict):
        raise RemoteSyncError(f"transfer data is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _safe_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).parts != (value,)
        or "\x00" in value
    ):
        raise RemoteSyncError(f"{label} must be one safe path component")
    return value


__all__ = [
    "LocalTransport",
    "MemoryTransferResult",
    "RemoteSyncError",
    "SessionTransferResult",
    "pull_project_memory",
    "pull_session",
    "push_project_memory",
    "push_session",
    "resolve_transport",
]
