"""SSH transport for remote session and memory publication."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse

from .. import memory_migration_plan, project_schema
from ..storage_format import (
    ACCEPTED_KEY_SETS,
    FEATURE_EPOCHS,
    GATED_HOME_PATHS,
    StorageFeature,
    writer_epoch,
)
from . import (
    RemoteSyncError,
    _append_resume_hint,
    _copy_tree,
    _directory_digest,
    _make_manifest,
    _read_manifest,
    _rewrite_cwd,
    _safe_component,
    _tree_state,
    _write_json,
    project_publish,
)
from .project_publish import PreparedTransfer, prepare_project_transfer

_HOST = re.compile(r"[A-Za-z0-9_.@-]+\Z")
DEFAULT_MAX_ARCHIVE_BYTES = 1 << 30
DEFAULT_MAX_ARCHIVE_MEMBERS = 200_000

_SESSION_FETCH_SCRIPT = r'''
import fcntl, pathlib, sys, tarfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
ident = sys.argv[2]
if pathlib.Path(ident).parts != (ident,): sys.exit(45)
def _reject(reason): print(f"zeta-remote: {reason}", file=sys.stderr); sys.exit(46)
parent = home / "sessions"
if not parent.is_dir(): sys.exit(44)
root = parent / ident
if not root.is_dir() or root.is_symlink(): sys.exit(44)
locks = []
try:
    for path in [root / ".lock", *sorted(root.rglob(".lock"))]:
        if path.is_file() and not path.is_symlink():
            handle = path.open("rb"); fcntl.flock(handle.fileno(), fcntl.LOCK_EX); locks.append(handle)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in [root, *sorted(root.rglob("*"))]:
            relative = path.relative_to(root)
            if path.is_symlink(): _reject(f"{path.relative_to(home)}: symlink in session")
            if path.name in {".lock", ".spill.lock", "runtime.lease"}: continue
            if path.is_dir() or path.is_file():
                archive.add(path, arcname=str(pathlib.Path("payload") / relative), recursive=False)
            else: _reject(f"{path.relative_to(home)}: not a file or directory")
finally:
    for handle in reversed(locks): handle.close()
'''


_PROJECT_AUTHORITY = rf'''
import contextlib, fcntl, json, os, pathlib, secrets, stat, sys
_PROJECT_LOCKS_FEATURE_EPOCH = {FEATURE_EPOCHS[StorageFeature.PROJECT_LOCKS]}
_WRITER_EPOCH = {writer_epoch()}
_GATED_HOME_PATHS = {tuple(pattern for pattern, _owner in GATED_HOME_PATHS)!r}
_LEASE = None

def _reject(reason): print(f"zeta-remote: {{reason}}", file=sys.stderr); sys.exit(46)

def _has_gated_state(home):
    for pattern in _GATED_HOME_PATHS:
        listed = pattern.endswith("/*")
        path = home / (pattern[:-2] if listed else pattern)
        if not os.path.lexists(path): continue
        if not listed or path.is_symlink() or not path.is_dir(): return True
        try:
            if next(path.iterdir(), None) is not None: return True
        except OSError: return True
    return False

def _publish_manifest(home):
    temporary = home / f".storage-format.json.{{secrets.token_hex(16)}}.tmp"
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps({{"writer_epoch": _WRITER_EPOCH}}, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, home / "storage-format.json")
        directory_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass

def _writer_epoch(home):
    # Mirror storage_format.open_writer: bootstrap only an empty home, then
    # retain a shared stored-format lease for the life of this program.
    global _LEASE
    if _LEASE is not None: return _LEASE[1]
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = home / "storage-format.json"
    try:
        lock_fd = os.open(home / ".storage-format.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError: sys.exit(53)
    try:
        if manifest_path.exists():
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
        else:
            try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: sys.exit(53)
            if not manifest_path.exists():
                if _has_gated_state(home): sys.exit(53)
                _publish_manifest(home)
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        os.close(lock_fd); sys.exit(53)
    if type(manifest) is not dict or set(manifest) != {{"writer_epoch"}}: sys.exit(53)
    epoch = manifest.get("writer_epoch")
    if type(epoch) is not int or epoch < 1 or epoch > _WRITER_EPOCH: sys.exit(53)
    _LEASE = (lock_fd, epoch)
    return epoch

def _lock_file(directory_fd, name, operation):
    try:
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        _reject(f"lock {{name}}: cannot open ({{exc.strerror}})")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        _reject(f"lock {{name}}: not a regular single-link file")
    os.fchmod(fd, 0o600)
    try:
        fcntl.flock(fd, operation)
    except BlockingIOError:
        os.close(fd)
        sys.exit(52)
    return fd

def _validate_project(project_fd, ident):
    try:
        record_fd = os.open("project.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=project_fd)
        with os.fdopen(record_fd) as stream:
            record = json.load(stream)
    except (OSError, ValueError):
        sys.exit(44)
    if record.get("project_id") != ident:
        _reject(f"projects/{{ident}}/project.json: project_id does not match")

def _project_fd(projects_fd, ident):
    try:
        fd = os.open(ident, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=projects_fd)
    except OSError:
        sys.exit(44)
    _validate_project(fd, ident)
    return fd

@contextlib.contextmanager
def _project_authority(home, ident, *, identity, exclusive, nonblocking=False):
    epoch = _writer_epoch(home)
    try:
        projects_fd = os.open(home / "projects", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        sys.exit(44)
    project_fd = None
    lock_fd = None
    try:
        project_locks_enabled = epoch >= _PROJECT_LOCKS_FEATURE_EPOCH
        legacy = not project_locks_enabled
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            operation |= fcntl.LOCK_NB
        if legacy or identity:
            lock_fd = _lock_file(projects_fd, ".lock", operation)
            if not identity:
                project_fd = _project_fd(projects_fd, ident)
        else:
            project_fd = _project_fd(projects_fd, ident)
            lock_fd = _lock_file(project_fd, ".memory.lock", operation)
            _validate_project(project_fd, ident)
        yield projects_fd if identity else project_fd
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if project_fd is not None:
            os.close(project_fd)
        os.close(projects_fd)

class ProjectRegistry:
    def __init__(self, projects):
        self.projects = pathlib.Path(projects)
    def _identity_write(self):
        return _project_authority(
            self.projects.parent, "", identity=True, exclusive=True
        )
    def _project_lock(self, project_id, domain, *, exclusive):
        if domain != "memory":
            raise ValueError("generated publisher supports memory only")
        return _project_authority(
            self.projects.parent,
            project_id,
            identity=False,
            exclusive=exclusive,
        )
'''


_PROJECT_FETCH_SCRIPT = _PROJECT_AUTHORITY + r'''
import tarfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
ident = sys.argv[2]
if pathlib.Path(ident).parts != (ident,): sys.exit(45)
root = home / "projects" / ident
with _project_authority(home, ident, identity=False, exclusive=False, nonblocking=True):
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in [root, *sorted(root.rglob("*"))]:
            relative = path.relative_to(root)
            if path.is_symlink(): _reject(f"{path.relative_to(home)}: symlink in project")
            if path.name in {".lock", ".memory.lock", ".sessions.lock", ".spill.lock", "runtime.lease"}: continue
            if path.is_dir() or path.is_file():
                archive.add(path, arcname=str(pathlib.Path("payload") / relative), recursive=False)
            else: _reject(f"{path.relative_to(home)}: not a file or directory")
'''

_IDENTITY_SCRIPT = r'''
import fcntl, os, pathlib, secrets, sys, tempfile
path = pathlib.Path(os.path.expanduser(sys.argv[1])) / ".machine-id"
path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = path.with_name(path.name + ".lock")
lock_path.touch(mode=0o600, exist_ok=True)
lock_path.chmod(0o600)
with lock_path.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        value = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        value = ""
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        value = secrets.token_hex(16)
        fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
        temporary = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as output:
                output.write(value + "\n")
            temporary.chmod(0o600)
            current = path.read_text(encoding="ascii").strip() if path.exists() else ""
            if len(current) == 32 and all(character in "0123456789abcdef" for character in current):
                value = current
            else:
                os.replace(temporary, path)
                value = path.read_text(encoding="ascii").strip()
        finally:
            temporary.unlink(missing_ok=True)
    elif path.stat().st_mode & 0o777 != 0o600:
        path.chmod(0o600)
print(value, end="")
'''

_SESSION_INSTALL_SCRIPT = r'''
import fcntl, hashlib, json, os, pathlib, shutil, sys, tarfile, tempfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
ident, expected = sys.argv[2], sys.argv[3]
max_members, max_bytes = int(sys.argv[4]), int(sys.argv[5])
if pathlib.Path(ident).parts != (ident,): sys.exit(45)
def _reject(reason): print(f"zeta-remote: {reason}", file=sys.stderr); sys.exit(46)
parent = home / "sessions"; parent.mkdir(parents=True, exist_ok=True, mode=0o700)
destination = parent / ident
lease = None
if destination.exists():
    lease = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try: fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(48)
def digest(root):
    value = hashlib.sha256()
    if not root.exists(): return "missing"
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): _reject(f"{path.relative_to(home)}: symlink in session")
        if path.is_file() and path.name not in {".lock", ".memory.lock", ".sessions.lock", ".spill.lock", "runtime.lease"}:
            value.update(path.relative_to(root).as_posix().encode() + b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024): value.update(chunk)
    return value.hexdigest()
if expected != digest(destination): sys.exit(47)
staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{ident}.incoming-", dir=parent))
try:
    members = declared = extracted = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            members += 1
            if members > max_members: sys.exit(49)
            if member.size < 0: _reject(f"archive member {member.name}: negative size")
            declared += member.size
            if declared > max_bytes: sys.exit(50)
            if shutil.disk_usage(parent).free < declared - extracted: sys.exit(51)
            parts = pathlib.PurePosixPath(member.name).parts
            if not parts or parts[0] != "payload" or any(p in {"", ".", ".."} for p in parts) or member.issym() or member.islnk(): _reject(f"archive member {member.name}: unsafe path or link")
            target = staging.joinpath(*parts[1:])
            if target.name == "runtime.lease": _reject(f"archive member {member.name}: runtime lease")
            if member.isdir(): target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None: _reject(f"archive member {member.name}: unreadable")
                with target.open("wb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk: _reject(f"archive member {member.name}: truncated")
                        extracted += len(chunk); remaining -= len(chunk)
                        if extracted > max_bytes: sys.exit(50)
                        output.write(chunk)
                    if source.read(1): sys.exit(50)
                target.chmod(0o600)
            else: _reject(f"archive member {member.name}: not a file or directory")
    manifest = json.loads((staging / "transfer.json").read_text())
    resume_cwd = pathlib.Path(manifest.get("resume_cwd", ""))
    expected_cwd = home / "remote-workspaces" / ident
    if resume_cwd != expected_cwd: _reject("transfer.json: resume_cwd does not match the remote workspace")
    expected_cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = parent / f".{ident}.replaced-{os.getpid()}"
    if backup.exists(): shutil.rmtree(backup)
    if destination.exists(): destination.rename(backup)
    try: staging.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists(): backup.rename(destination)
        raise
    if backup.exists(): shutil.rmtree(backup)
finally:
    if staging.exists(): shutil.rmtree(staging)
'''


_PROJECT_INSTALL_WRAPPER = r"""
import sys, tarfile, tempfile
home = Path(sys.argv[1]).expanduser().resolve()
ident, expected, transfer = sys.argv[2], sys.argv[3], sys.argv[4]
max_members, max_bytes = int(sys.argv[5]), int(sys.argv[6])
if Path(ident).parts != (ident,): sys.exit(45)
home.mkdir(parents=True, exist_ok=True, mode=0o700)
_writer_epoch(home)
with tempfile.TemporaryDirectory(prefix=f".{ident}.upload-", dir=home) as temporary:
    staging = Path(temporary) / ident
    staging.mkdir(mode=0o700)
    members = declared = extracted = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            members += 1
            if members > max_members: sys.exit(49)
            if member.size < 0: _reject(f"archive member {member.name}: negative size")
            declared += member.size
            if declared > max_bytes: sys.exit(50)
            if shutil.disk_usage(home).free < declared - extracted: sys.exit(51)
            parts = tuple(member.name.split("/"))
            if not member.isfile() or len(parts) < 2 or parts[0] != "payload": _reject(f"archive member {member.name}: not a payload file")
            try: _canonical_snapshot_path(parts[1:])
            except ProjectPublicationError as exc:
                _reject(f"archive member {member.name}: {exc}")
            target = staging.joinpath(*parts[1:])
            if target.exists(): _reject(f"archive member {member.name}: duplicate")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = archive.extractfile(member)
            if source is None: _reject(f"archive member {member.name}: unreadable")
            with target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk: _reject(f"archive member {member.name}: truncated")
                    extracted += len(chunk); remaining -= len(chunk)
                    if extracted > max_bytes: sys.exit(50)
                    output.write(chunk)
                if source.read(1): sys.exit(50)
            target.chmod(0o600)
    for directory in staging.rglob("*"):
        if directory.is_dir(): directory.chmod(0o700)
    if transfer_digest(staging) != transfer: _reject("archive does not match its transfer digest")
    try:
        publish_local_project(home, ident, staging, expected_digest=expected)
    except ProjectPublicationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(47)
"""


def _project_install_script() -> str:
    """Ship the dependency-free schema and publisher to an uninstalled peer."""

    schema_filename = project_schema.__file__
    migration_filename = memory_migration_plan.__file__
    publisher_filename = project_publish.__file__
    if None in {schema_filename, migration_filename, publisher_filename}:
        raise RemoteSyncError("project publication module source is unavailable")
    schema_source = Path(schema_filename).read_text(encoding="utf-8")
    accepted = {
        "entry_memory_snapshot_reference": sorted(
            ACCEPTED_KEY_SETS["entry_memory_snapshot_reference"]
        ),
        "legacy_memory_snapshot": sorted(ACCEPTED_KEY_SETS["legacy_memory_snapshot"]),
        "memory_state_root": sorted(ACCEPTED_KEY_SETS["memory_state_root"]),
        "project_registry_record": sorted(ACCEPTED_KEY_SETS["project_registry_record"]),
    }
    dependency_free_registry = (
        f"\nACCEPTED_KEY_SETS = {accepted!r}\n"
        "def reject_unknown_keys(payload, shape):\n"
        "    accepted = ACCEPTED_KEY_SETS.get(shape)\n"
        "    if accepted is None or set(payload) - set(accepted):\n"
        "        raise ValueError(f'unknown keys for {shape}')\n"
    )
    schema_source += dependency_free_registry
    migration_source = Path(migration_filename).read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n\n", "", 1
    )
    publisher_source = Path(publisher_filename).read_text(encoding="utf-8")
    dependencies = (
        "from .. import project_schema\n",
        "from ..project_registry import ProjectRegistry\n",
        (
            "from ..memory_migration_plan import (\n"
            "    LEGACY_MEMORY_FILES,\n"
            "    build_migration_plan,\n"
            "    legacy_memory_digest,\n"
            "    reachable_version_pruning_plan,\n"
            ")\n"
        ),
        "from ..storage_format import reject_unknown_keys\n",
    )
    if any(dependency not in publisher_source for dependency in dependencies):
        raise RemoteSyncError("project publication dependency import is unavailable")
    publisher_source = publisher_source.replace(
        "from __future__ import annotations\n\n", "", 1
    )
    for dependency in dependencies:
        publisher_source = publisher_source.replace(dependency, "", 1)
    return (
        schema_source
        + "\n"
        + _PROJECT_AUTHORITY
        + "\nproject_schema = sys.modules[__name__]\n"
        + migration_source
        + publisher_source
        + "\n"
        + _PROJECT_INSTALL_WRAPPER
    )


@dataclass(slots=True)
class SshTransport:
    """Transfer snapshots through one configured SSH host and remote ZETA_HOME."""

    host: str
    remote_home: str = "~/.zeta"
    name: str | None = None
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES
    max_archive_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS
    _resolved_home: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _HOST.fullmatch(self.host) or self.host.startswith("-"):
            raise RemoteSyncError("SSH host is invalid")
        if "\x00" in self.remote_home or "\n" in self.remote_home:
            raise RemoteSyncError("remote ZETA_HOME is invalid")
        if self.name is None:
            self.name = self.host
        _safe_component(self.name, "remote name")
        if type(self.max_archive_bytes) is not int or self.max_archive_bytes < 1:
            raise RemoteSyncError("archive byte limit must be positive")
        if type(self.max_archive_members) is not int or self.max_archive_members < 1:
            raise RemoteSyncError("archive member limit must be positive")

    @classmethod
    def from_url(cls, name: str, value: str) -> SshTransport:
        parsed = urlparse(value)
        if parsed.scheme != "ssh" or not parsed.hostname or parsed.query or parsed.fragment:
            raise RemoteSyncError(f"remote {name!r} must be an ssh:// URL")
        host = parsed.netloc
        path = unquote(parsed.path) if parsed.path else "~/.zeta"
        return cls(host=host, remote_home=path, name=name)

    @property
    def machine_id(self) -> str:
        result = self._run(_IDENTITY_SCRIPT, [self.remote_home])
        value = result.stdout.decode("ascii").strip()
        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise RemoteSyncError("remote machine identity is invalid")
        return value

    def publish_session(self, snapshot: Path, *, force: bool) -> Path:
        session_id = _safe_component(snapshot.name, "session id")
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-publish-") as temporary:
            outgoing = Path(temporary) / session_id
            _copy_tree(snapshot, outgoing)
            remote_cwd = Path(self._home()) / "remote-workspaces" / session_id
            previous = _read_manifest(outgoing)
            _rewrite_cwd(outgoing, remote_cwd)
            _append_resume_hint(outgoing)
            _write_json(
                outgoing / "transfer.json",
                _make_manifest(outgoing, str(remote_cwd), previous=previous),
            )
            expected = self._existing_session_state(session_id, outgoing, force)
            self._install_session(session_id, outgoing, expected)
            shutil.copyfile(outgoing / "transfer.json", snapshot / "transfer.json")
        return snapshot

    def fetch_session(self, session_id: str, destination: Path) -> Path:
        self._fetch_session(_safe_component(session_id, "session id"), destination)
        return destination

    def fetch_project(self, project_id: str, destination: Path) -> str:
        project_id = _safe_component(project_id, "project id")
        try:
            self._fetch_project(project_id, destination)
        except RemoteSyncError as exc:
            if "was not found" in str(exc):
                return "missing"
            raise
        return project_publish.project_digest(destination)

    def publish_project(
        self, project_id: str, snapshot: Path, *, expected_digest: str
    ) -> None:
        project_id = _safe_component(project_id, "project id")
        prepared = prepare_project_transfer(snapshot)
        self._install_project(project_id, prepared, expected_digest)

    def _existing_session_state(
        self, ident: str, source: Path, force: bool
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-existing-") as temporary:
            current = Path(temporary) / ident
            try:
                self._fetch_session(ident, current)
            except RemoteSyncError as exc:
                if "was not found" in str(exc):
                    return "missing"
                raise
            expected = _directory_digest(current)
            if not force:
                source_state = _tree_state(source)
                current_state = _tree_state(current)
                if current_state != source_state:
                    if current_state[0] >= source_state[0]:
                        raise RemoteSyncError(
                            "newer remote session exists; use --force to replace it"
                        )
                    raise RemoteSyncError(
                        "remote session differs from the snapshot; use --force to replace it"
                    )
            return expected

    def _home(self) -> str:
        if self._resolved_home is None:
            script = "import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())"
            result = self._run(script, [self.remote_home])
            self._resolved_home = result.stdout.decode().strip()
            if not self._resolved_home.startswith("/"):
                raise RemoteSyncError("remote ZETA_HOME did not resolve absolutely")
        return self._resolved_home

    def _fetch_session(self, ident: str, destination: Path) -> None:
        self._fetch_archive(
            _SESSION_FETCH_SCRIPT, "session", ident, destination
        )

    def _fetch_project(self, ident: str, destination: Path) -> None:
        self._fetch_archive(
            _PROJECT_FETCH_SCRIPT, "project", ident, destination
        )

    def _fetch_archive(
        self, script: str, kind: str, ident: str, destination: Path
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-fetch-") as temporary:
            archive = Path(temporary) / "snapshot.tar.gz"
            result = self._run_to_file(
                script,
                [self._home(), ident],
                archive,
            )
            if result.returncode == 44:
                raise RemoteSyncError(f"remote {kind} {ident} was not found")
            if result.returncode == 52:
                raise RemoteSyncError(f"project registry busy on {self.name}; retry")
            if result.returncode:
                raise RemoteSyncError(
                    f"SSH snapshot failed on {self.host} (exit {result.returncode}): "
                    f"{result.stderr.decode(errors='replace').strip()}"
                )
            _unpack(
                archive,
                destination,
                max_members=self.max_archive_members,
                max_bytes=self.max_archive_bytes,
            )

    def _install_project(
        self, ident: str, prepared: PreparedTransfer, expected: str
    ) -> None:
        with tempfile.TemporaryFile() as incoming:
            incoming.write(prepared.archive_bytes)
            incoming.seek(0)
            result = self._run(
                _project_install_script(),
                [
                    self._home(),
                    ident,
                    expected,
                    prepared.transfer_digest,
                    str(self.max_archive_members),
                    str(self.max_archive_bytes),
                ],
                stdin=incoming,
                check=False,
            )
        self._raise_install_error(result, project=True)

    def _install_session(self, ident: str, source: Path, expected: str) -> None:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-install-") as temporary:
            archive = Path(temporary) / "snapshot.tar.gz"
            _pack(source, archive)
            with archive.open("rb") as incoming:
                result = self._run(
                    _SESSION_INSTALL_SCRIPT,
                    [
                        self._home(),
                        ident,
                        expected,
                        str(self.max_archive_members),
                        str(self.max_archive_bytes),
                    ],
                    stdin=incoming,
                    check=False,
                )
        self._raise_install_error(result, project=False)

    def _raise_install_error(
        self, result: subprocess.CompletedProcess[bytes], *, project: bool
    ) -> None:
        if result.returncode == 47:
            raise RemoteSyncError(
                "remote changed during transfer; retry after inspection"
            )
        if result.returncode == 48:
            raise RemoteSyncError(
                "remote session is active; stop it before replacement"
            )
        if result.returncode == 49:
            raise RemoteSyncError("remote archive exceeds the member limit")
        if result.returncode == 50:
            raise RemoteSyncError("remote archive exceeds the uncompressed byte limit")
        if result.returncode == 51:
            raise RemoteSyncError("remote has insufficient free space for the archive")
        if result.returncode:
            kind = "project" if project else "session"
            raise RemoteSyncError(
                f"SSH {kind} publication failed on {self.host} "
                f"(exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )

    def _run(
        self,
        script: str,
        arguments: list[str],
        *,
        stdin: BinaryIO | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        command = self._command(script, arguments)
        try:
            result = subprocess.run(
                command,
                stdin=stdin if stdin is not None else subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise RemoteSyncError(f"could not run ssh: {exc}") from exc
        if check and result.returncode:
            raise RemoteSyncError(
                f"SSH command failed on {self.host} (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result

    def _run_to_file(
        self,
        script: str,
        arguments: list[str],
        destination: Path,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            process = subprocess.Popen(
                self._command(script, arguments),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            total = 0
            with destination.open("xb") as output:
                destination.chmod(0o600)
                while chunk := process.stdout.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_archive_bytes:
                        process.kill()
                        process.wait()
                        raise RemoteSyncError(
                            "remote archive exceeds the compressed byte limit"
                        )
                    output.write(chunk)
            stderr = process.stderr.read()
            return subprocess.CompletedProcess(
                process.args,
                process.wait(),
                b"",
                stderr,
            )
        except OSError as exc:
            raise RemoteSyncError(f"could not run ssh: {exc}") from exc

    def _command(self, script: str, arguments: list[str]) -> list[str]:
        remote_command = " ".join(
            shlex.quote(value)
            for value in ("/usr/bin/python3", "-B", "-c", script, *arguments)
        )
        return ["ssh", "--", self.host, remote_command]


def _pack(source: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w:gz") as archive:
        for path in [source, *sorted(source.rglob("*"))]:
            if path.is_symlink():
                raise RemoteSyncError("cannot upload a symlink")
            relative = path.relative_to(source)
            archive.add(
                path,
                arcname=str(Path("payload") / relative),
                recursive=False,
            )


def _unpack(
    archive_path: Path,
    destination: Path,
    *,
    max_members: int,
    max_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incoming-",
            dir=destination.parent,
        )
    )
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > max_members:
                raise RemoteSyncError("remote archive exceeds the member limit")
            total = 0
            for member in members:
                if member.size < 0:
                    raise RemoteSyncError("remote archive member has an invalid size")
                total += member.size
                if total > max_bytes:
                    raise RemoteSyncError(
                        "remote archive exceeds the uncompressed byte limit"
                    )
                _validate_member(member)
            if shutil.disk_usage(destination.parent).free < total:
                raise RemoteSyncError(
                    "local destination has insufficient free space for the archive"
                )
            extracted = 0
            for member in members:
                target = staging.joinpath(*Path(member.name).parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise RemoteSyncError("remote archive member is unreadable")
                with target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RemoteSyncError(
                                "remote archive member ended before its declared size"
                            )
                        extracted += len(chunk)
                        remaining -= len(chunk)
                        if extracted > max_bytes:
                            raise RemoteSyncError(
                                "remote archive exceeds the uncompressed byte limit"
                            )
                        output.write(chunk)
                    if source.read(1):
                        raise RemoteSyncError(
                            "remote archive member exceeds its declared size"
                        )
                target.chmod(0o600)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_member(member: tarfile.TarInfo) -> None:
    parts = Path(member.name).parts
    if (
        not parts
        or parts[0] != "payload"
        or any(part in {"", ".", ".."} for part in parts)
        or member.issym()
        or member.islnk()
    ):
        raise RemoteSyncError("remote archive contains an unsafe path")
    if not member.isdir() and not member.isfile():
        raise RemoteSyncError("remote archive contains an unsupported file")


__all__ = ["SshTransport"]
