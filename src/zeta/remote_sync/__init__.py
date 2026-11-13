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
import stat
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
from ..core.session_execution import RUNTIME_LEASE_NAME
from ..core.session_files import (
    SessionError,
    SessionInUseError,
    reference_publication,
    session_directory,
)
from ..core.session_metadata import SessionMetadata
from ..core.store import ConversationStore
from ..core.transcript_recovery import RepairTarget, validate_recovery_records
from ..locking import shape_lock
from ..session_retention.pin import RETENTION_PIN_FILENAME
from ..storage_format import WriterLease, open_writer
from .errors import RemoteSyncError
from .memory import (
    MemoryTransferResult,
    _machine_id,
    fetch_local_project,
    publish_local_project,
    resolve_project_memory,
    sync_project_memory,
)

_SCHEMA = "zeta.session-transfer.v1"
_MAX_CONVERSATION_HEADER_BYTES = 1024 * 1024
_EXCLUDED_NAMES = frozenset({".lock", ".spill.lock", RUNTIME_LEASE_NAME})
_CREDENTIAL_NAMES = frozenset(
    {"oauth.json", "credentials.json", "tokens.json", "auth.json"}
)


@dataclass(frozen=True, slots=True)
class SessionTransferResult:
    session_id: str
    digest: str
    last_seq: int
    resume_notice: str


class _Digest(Protocol):
    def update(self, data: bytes) -> object: ...


class Transport(Protocol):
    """Publication seam implemented by local tests and SSH."""

    name: str

    def publish_session(self, snapshot: Path, *, force: bool) -> Path: ...

    def fetch_session(self, session_id: str, destination: Path) -> Path: ...

    def fetch_project(self, project_id: str, destination: Path) -> str: ...

    def publish_project(
        self, project_id: str, snapshot: Path, *, expected_digest: str
    ) -> None: ...


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

    @property
    def machine_id(self) -> str:
        return _machine_id(self.home)

    def publish_session(self, snapshot: Path, *, force: bool) -> Path:
        with open_writer(self.home) as lease:
            manifest = _read_manifest(snapshot)
            installed_ids, referenced_ids = _snapshot_reference_ids(
                snapshot, str(manifest["session_id"])
            )
            with reference_publication(
                self.home,
                lease,
                installed_ids=installed_ids,
                referenced_ids=referenced_ids,
            ):
                return self._publish_session(snapshot, force=force)

    def _publish_session(self, snapshot: Path, *, force: bool) -> Path:
        manifest = _read_manifest(snapshot)
        session_id = manifest["session_id"]
        sessions = self.home / "sessions"
        sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = sessions / session_id
        lease = (
            session_directory(sessions, session_id, exclusive=True)
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
        except SessionInUseError as exc:
            raise RemoteSyncError(
                "remote session is active; stop it before replacement"
            ) from exc
        except SessionError as exc:
            raise RemoteSyncError(str(exc)) from exc
        return destination

    def fetch_session(self, session_id: str, destination: Path) -> Path:
        source = self.home / "sessions" / _safe_component(session_id, "session id")
        if not source.is_dir():
            raise RemoteSyncError(f"remote session {session_id} was not found")
        with _snapshot_locks(source):
            _copy_tree(source, destination)
        return destination

    def fetch_project(self, project_id: str, destination: Path) -> str:
        with open_writer(self.home):
            return fetch_local_project(
                self.home, project_id, destination, peer=self.name
            )

    def publish_project(
        self, project_id: str, snapshot: Path, *, expected_digest: str
    ) -> None:
        with open_writer(self.home):
            publish_local_project(
                self.home,
                project_id,
                snapshot,
                expected_digest=expected_digest,
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
    with tempfile.TemporaryDirectory(
        dir=source.parent, prefix=f".{session_id}.push-staging-"
    ) as staging_temporary:
        staging = Path(staging_temporary) / session_id
        with _snapshot_locks(source):
            _copy_tree(source, staging)
        with tempfile.TemporaryDirectory(prefix="zeta-session-push-") as temporary:
            snapshot = Path(temporary) / session_id
            _copy_tree(staging, snapshot)
            manifest = _make_manifest(snapshot, metadata.cwd)
            _write_json(snapshot / "transfer.json", manifest)
            with open_writer(local_home) as storage_lease:
                _validate_snapshot(snapshot, session_id, storage_lease)
            if metadata.project_id is not None:
                sync_project_memory(
                    local_home,
                    transport,
                    project_id=metadata.project_id,
                    direction="push",
                )
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
    with open_writer(local_home) as storage_lease:
        return _pull_session(
            local_home,
            transport,
            session_id,
            cwd=cwd,
            force=force,
            storage_lease=storage_lease,
        )


def _pull_session(
    local_home: Path,
    transport: Transport,
    session_id: str,
    *,
    cwd: str | Path | None,
    force: bool,
    storage_lease: WriterLease,
) -> SessionTransferResult:
    safe_id = _safe_component(session_id, "session id")
    sessions = local_home / "sessions"
    sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="zeta-session-pull-") as temporary:
        snapshot = Path(temporary) / safe_id
        transport.fetch_session(safe_id, snapshot)
        _validate_snapshot(snapshot, safe_id, storage_lease)
        metadata = _read_json(snapshot / "meta.json")
        project_id = metadata.get("project_id")
        if project_id is not None:
            if not isinstance(project_id, str):
                raise RemoteSyncError("session project id is invalid")
            sync_project_memory(
                local_home,
                transport,
                project_id=_safe_component(project_id, "project id"),
                direction="pull",
            )
        destination = sessions / safe_id
        installed_ids, referenced_ids = _snapshot_reference_ids(snapshot, safe_id)
        lease = (
            session_directory(sessions, safe_id, exclusive=True)
            if destination.exists()
            else nullcontext()
        )
        try:
            with reference_publication(
                local_home,
                storage_lease,
                installed_ids=installed_ids,
                referenced_ids=referenced_ids,
            ), lease:
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
                _append_resume_hint(snapshot)
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

    return sync_project_memory(
        Path(home).expanduser().resolve(),
        transport,
        project_id=_safe_component(project_id, "project id"),
        direction="push",
    )


def pull_project_memory(
    home: str | Path,
    transport: Transport,
    *,
    project_id: str,
) -> MemoryTransferResult:
    """Pull standard memory files with per-file three-way CAS semantics."""

    return sync_project_memory(
        Path(home).expanduser().resolve(),
        transport,
        project_id=_safe_component(project_id, "project id"),
        direction="pull",
    )


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
            stack.enter_context(shape_lock())
            handle = stack.enter_context(lock_path.open("rb"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


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
                ["git", *args],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
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
        _stream_file(path, digest)
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
        digest.update(relative.encode("utf-8") + b"\0")
        lines = _stream_file(
            path, digest, count_lines=path.name == "conversation.jsonl"
        )
        if path.name == "conversation.jsonl":
            last_seq += max(0, lines - 1)
    return last_seq, digest.hexdigest()


def _stream_file(path: Path, digest: _Digest, *, count_lines: bool = False) -> int:
    lines = 0
    last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            if count_lines:
                lines += chunk.count(b"\n")
                last = chunk[-1:]
    if count_lines and last and last != b"\n":
        lines += 1
    return lines


def _rewrite_cwd(snapshot: Path, cwd: Path) -> None:
    cwd_text = str(cwd.resolve())
    meta_path = snapshot / "meta.json"
    meta = _read_json(meta_path)
    meta["cwd"] = cwd_text
    _write_metadata(meta_path, meta)
    for log in snapshot.rglob("conversation.jsonl"):
        temporary = log.with_name(f".{log.name}.{os.getpid()}.tmp")
        try:
            with log.open("rb") as source:
                first = source.readline(_MAX_CONVERSATION_HEADER_BYTES + 1)
                if not first:
                    raise RemoteSyncError(f"empty conversation log: {log}")
                if len(first) > _MAX_CONVERSATION_HEADER_BYTES or not first.endswith(
                    b"\n"
                ):
                    raise RemoteSyncError(f"conversation header is too large: {log}")
                header = json.loads(first)
                header["data"]["cwd"] = cwd_text
                with temporary.open("xb") as output:
                    output.write(
                        json.dumps(header, separators=(",", ":")).encode("utf-8")
                        + b"\n"
                    )
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.chmod(0o600)
            os.replace(temporary, log)
        finally:
            temporary.unlink(missing_ok=True)
    for state_path in snapshot.rglob("session_state.json"):
        state = _read_json(state_path)
        state["bash_cwd"] = cwd_text
        _write_json(state_path, state)


def _append_resume_hint(snapshot: Path) -> None:
    meta_path = snapshot / "meta.json"
    metadata = _read_json(meta_path)
    prompt = metadata.get("system_prompt", "")
    if not isinstance(prompt, str):
        raise RemoteSyncError("session system prompt is invalid")
    start = "<zeta-remote-resume>"
    if start in prompt:
        prompt = prompt.split(start, 1)[0].rstrip()
    hint = (
        f"{start}\n"
        "This session was transferred to another machine. Before editing code, "
        "verify that the mapped working directory contains the intended repository.\n"
        "</zeta-remote-resume>"
    )
    metadata["system_prompt"] = f"{prompt}\n\n{hint}" if prompt else hint
    _write_metadata(meta_path, metadata)


def _map_missing_cwd(snapshot: Path, placeholder: Path) -> None:
    manifest = _read_manifest(snapshot)
    cwd = manifest.get("resume_cwd", manifest.get("source_cwd"))
    if not isinstance(cwd, str) or not Path(cwd).is_dir():
        placeholder.mkdir(parents=True, exist_ok=True, mode=0o700)
        _rewrite_cwd(snapshot, placeholder)
        _append_resume_hint(snapshot)
        updated = _make_manifest(snapshot, str(placeholder), previous=manifest)
        _write_json(snapshot / "transfer.json", updated)


def _validate_snapshot(
    snapshot: Path, session_id: str, storage_lease: WriterLease
) -> None:
    manifest = _read_manifest(snapshot)
    if manifest.get("session_id") != session_id:
        raise RemoteSyncError(
            "session manifest id does not match the requested session"
        )
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise RemoteSyncError("remote session contains a symlink")
        if path.name in _EXCLUDED_NAMES:
            raise RemoteSyncError("remote session contains a runtime-only file")
    target_label = session_id
    try:
        for directory, target in _snapshot_transcripts(snapshot, session_id):
            target_label = target.child_run_id or session_id
            store = ConversationStore(
                directory.parent,
                session_id=directory.name,
                _read_only=True,
                _must_exist=True,
                _storage_lease=storage_lease,
            )
            try:
                validate_recovery_records(store.directory_fd, target)
            finally:
                store.close()
    except (OSError, ValueError) as exc:
        raise RemoteSyncError(f"session transcript is invalid: {target_label}") from exc


def _snapshot_reference_ids(
    snapshot: Path, session_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    metadata = _read_json(snapshot / "meta.json")
    parent = metadata.get("parent_session_id")
    if parent is None:
        return (session_id,), ()
    if type(parent) is not str:
        raise RemoteSyncError("session parent id is invalid")
    return (session_id,), (_safe_component(parent, "parent session id"),)


def _snapshot_transcripts(
    snapshot: Path, session_id: str
) -> tuple[tuple[Path, RepairTarget], ...]:
    transcripts = [(snapshot, RepairTarget(session_id))]
    for path in sorted(snapshot.rglob("conversation.jsonl")):
        directory = path.parent
        if directory == snapshot:
            continue
        relative = directory.relative_to(snapshot).parts
        if len(relative) % 2 or any(
            relative[index] != "agents" for index in range(0, len(relative), 2)
        ):
            raise RemoteSyncError("remote session contains an unsafe child transcript")
        segments = relative[1::2]
        if any(not segment.isdigit() or int(segment) < 1 for segment in segments):
            raise RemoteSyncError("remote session contains an unsafe child transcript")
        child_run_id = ":".join((session_id, *segments))
        transcripts.append((directory, RepairTarget(session_id, child_run_id)))
    return tuple(transcripts)


def _read_manifest(snapshot: Path) -> dict[str, object]:
    value = _read_json(snapshot / "transfer.json")
    if value.get("schema") != _SCHEMA:
        raise RemoteSyncError("unsupported or missing session transfer manifest")
    _manifest_text(value, "session_id", 255)
    _manifest_text(value, "created_at", 128)
    _manifest_text(value, "digest", 128)
    _manifest_text(value, "source_cwd", 4096)
    _manifest_text(value, "resume_cwd", 4096)
    _manifest_text(value, "resume_notice", 8192)
    if type(value.get("last_seq")) is not int or value["last_seq"] < 0:
        raise RemoteSyncError("session manifest last_seq is invalid")
    if type(value.get("includes_spill_files")) is not bool:
        raise RemoteSyncError("session manifest includes_spill_files is invalid")
    git = value.get("git")
    if not isinstance(git, dict):
        raise RemoteSyncError("session manifest git is invalid")
    for name in ("remote_url", "branch", "head"):
        item = git.get(name)
        if item is not None:
            _validated_text(item, f"git.{name}", 4096 if name == "remote_url" else 512)
    return value


def _manifest_text(value: dict[str, object], name: str, limit: int) -> str:
    return _validated_text(value.get(name), name, limit)


def _validated_text(value: object, name: str, limit: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > limit
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RemoteSyncError(f"session manifest {name} is invalid")
    return value


def _result(manifest: dict[str, object]) -> SessionTransferResult:
    return SessionTransferResult(
        session_id=str(manifest["session_id"]),
        digest=str(manifest["digest"]),
        last_seq=int(manifest["last_seq"]),
        resume_notice=str(manifest["resume_notice"]),
    )


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RemoteSyncError(f"cannot fsync file: {path}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise RemoteSyncError(f"cannot fsync directory: {path}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise RemoteSyncError(f"cannot fsync directory: {path}") from exc
    finally:
        os.close(fd)


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    _preserve_retention_pin(destination, staging)
    backup = destination.parent / f".{destination.name}.replaced-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if destination.exists():
            destination.rename(backup)
            _fsync_directory(destination.parent)
        staging.rename(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
            _fsync_directory(destination.parent)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _preserve_retention_pin(destination: Path, staging: Path) -> None:
    # The pin sidecar is retention-owned local state. A snapshot may carry its
    # own pin (the transfer does not exclude it) and the destination may have
    # one too; copy the destination's pin forward so the result stays pinned
    # whenever the destination or the snapshot was pinned, never silently
    # clearing a local pin. A destination pin that exists but is a symlink or
    # not a regular single-link file cannot be copied safely; skipping it would
    # silently unpin, so refuse the replace and leave the destination untouched.
    pin = destination / RETENTION_PIN_FILENAME
    if not os.path.lexists(pin):
        return
    info = pin.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RemoteSyncError(
            f"destination retention pin is not a regular single-link file: {pin}"
        )
    target = staging / RETENTION_PIN_FILENAME
    shutil.copyfile(pin, target)
    target.chmod(0o600)
    _fsync_file(target)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, destination)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteSyncError(f"cannot read transfer data: {path}") from exc
    if not isinstance(value, dict):
        raise RemoteSyncError(f"transfer data is not an object: {path}")
    return value


def _write_metadata(path: Path, value: dict[str, object]) -> None:
    # The same reader validation SessionManager._publish_metadata applies, so a
    # transferred session never lands with a meta.json the reader rejects.
    try:
        SessionMetadata.from_dict(value, path=path)
    except SessionError as exc:
        raise RemoteSyncError(f"session metadata is unreadable: {exc}") from exc
    _write_json(path, value)


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
    "resolve_project_memory",
    "resolve_transport",
]
