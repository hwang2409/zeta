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

from .. import project_schema
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

_FETCH_SCRIPT = r'''
import fcntl, io, os, pathlib, sys, tarfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
kind, ident = sys.argv[2], sys.argv[3]
if kind not in {"sessions", "projects"} or pathlib.Path(ident).parts != (ident,): sys.exit(45)
parent = home / kind
if not parent.is_dir(): sys.exit(44)
root = parent / ident
locks = []
try:
    if kind == "projects":
        fd = os.open(parent / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600); handle = os.fdopen(fd, "r+b")
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: sys.exit(52)
        locks.append(handle)
    if not root.is_dir() or root.is_symlink(): sys.exit(44)
    for path in [root / ".lock", *sorted(root.rglob(".lock"))]:
        if path.is_file() and not path.is_symlink():
            handle = path.open("rb"); fcntl.flock(handle.fileno(), fcntl.LOCK_EX); locks.append(handle)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in [root, *sorted(root.rglob("*"))]:
            relative = path.relative_to(root)
            if path.is_symlink(): sys.exit(46)
            if path.name in {".lock", ".spill.lock", "runtime.lease"}: continue
            if path.is_dir() or path.is_file():
                archive.add(path, arcname=str(pathlib.Path("payload") / relative), recursive=False)
            else: sys.exit(46)
finally:
    for handle in reversed(locks): handle.close()
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

_INSTALL_SCRIPT = r'''
import fcntl, hashlib, json, os, pathlib, shutil, sys, tarfile, tempfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
kind, ident, expected = sys.argv[2], sys.argv[3], sys.argv[4]
max_members, max_bytes = int(sys.argv[5]), int(sys.argv[6])
if kind not in {"sessions", "projects"} or pathlib.Path(ident).parts != (ident,): sys.exit(45)
parent = home / kind; parent.mkdir(parents=True, exist_ok=True, mode=0o700)
destination = parent / ident
registry_lease = None
if kind == "projects":
    fd = os.open(parent / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600); registry_lease = os.fdopen(fd, "r+b")
    fcntl.flock(registry_lease.fileno(), fcntl.LOCK_EX)
lease = None
if destination.exists() and kind == "sessions":
    lease = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try: fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(48)
def digest(root):
    value = hashlib.sha256()
    if not root.exists(): return "missing"
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): sys.exit(46)
        if path.is_file() and path.name not in {".lock", ".spill.lock", "runtime.lease"}:
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
            if member.size < 0: sys.exit(46)
            declared += member.size
            if declared > max_bytes: sys.exit(50)
            if shutil.disk_usage(parent).free < declared - extracted: sys.exit(51)
            parts = pathlib.PurePosixPath(member.name).parts
            if not parts or parts[0] != "payload" or any(p in {"", ".", ".."} for p in parts) or member.issym() or member.islnk(): sys.exit(46)
            target = staging.joinpath(*parts[1:])
            if kind == "sessions" and target.name == "runtime.lease": sys.exit(46)
            if member.isdir(): target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None: sys.exit(46)
                with target.open("wb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk: sys.exit(46)
                        extracted += len(chunk); remaining -= len(chunk)
                        if extracted > max_bytes: sys.exit(50)
                        output.write(chunk)
                    if source.read(1): sys.exit(50)
                target.chmod(0o600)
            else: sys.exit(46)
    if kind == "sessions":
        manifest = json.loads((staging / "transfer.json").read_text())
        resume_cwd = pathlib.Path(manifest.get("resume_cwd", ""))
        expected_cwd = home / "remote-workspaces" / ident
        if resume_cwd != expected_cwd: sys.exit(46)
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
with tempfile.TemporaryDirectory(prefix=f".{ident}.upload-", dir=home) as temporary:
    staging = Path(temporary) / ident
    staging.mkdir(mode=0o700)
    members = declared = extracted = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            members += 1
            if members > max_members: sys.exit(49)
            if member.size < 0: sys.exit(46)
            declared += member.size
            if declared > max_bytes: sys.exit(50)
            if shutil.disk_usage(home).free < declared - extracted: sys.exit(51)
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) < 2 or parts[0] != "payload" or any(p in {"", ".", ".."} for p in parts): sys.exit(46)
            target = staging.joinpath(*parts[1:])
            if target.exists(): sys.exit(46)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = archive.extractfile(member)
            if source is None: sys.exit(46)
            with target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk: sys.exit(46)
                    extracted += len(chunk); remaining -= len(chunk)
                    if extracted > max_bytes: sys.exit(50)
                    output.write(chunk)
                if source.read(1): sys.exit(50)
            target.chmod(0o600)
    for directory in staging.rglob("*"):
        if directory.is_dir(): directory.chmod(0o700)
    if transfer_digest(staging) != transfer: sys.exit(46)
    try:
        publish_local_project(home, ident, staging, expected_digest=expected)
    except ProjectPublicationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(47)
"""


def _project_install_script() -> str:
    """Ship the dependency-free schema and publisher to an uninstalled peer."""

    schema_filename = project_schema.__file__
    publisher_filename = project_publish.__file__
    if schema_filename is None or publisher_filename is None:
        raise RemoteSyncError("project publication module source is unavailable")
    schema_source = Path(schema_filename).read_text(encoding="utf-8")
    publisher_source = Path(publisher_filename).read_text(encoding="utf-8")
    dependency_import = "from .. import project_schema\n"
    if dependency_import not in publisher_source:
        raise RemoteSyncError("project publication schema import is unavailable")
    publisher_source = publisher_source.replace(
        "from __future__ import annotations\n\n", "", 1
    ).replace(dependency_import, "", 1)
    return (
        schema_source
        + "\nimport sys\nproject_schema = sys.modules[__name__]\n"
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
            _append_resume_hint(outgoing, remote_cwd, previous)
            _write_json(
                outgoing / "transfer.json",
                _make_manifest(outgoing, str(remote_cwd), previous=previous),
            )
            expected = self._existing_state("sessions", session_id, outgoing, force)
            self._install("sessions", session_id, outgoing, expected)
            shutil.copyfile(outgoing / "transfer.json", snapshot / "transfer.json")
        return snapshot

    def fetch_session(self, session_id: str, destination: Path) -> Path:
        self._fetch("sessions", _safe_component(session_id, "session id"), destination)
        return destination

    def fetch_project(self, project_id: str, destination: Path) -> str:
        project_id = _safe_component(project_id, "project id")
        try:
            self._fetch("projects", project_id, destination)
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

    def _existing_state(
        self, kind: str, ident: str, source: Path, force: bool
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-existing-") as temporary:
            current = Path(temporary) / ident
            try:
                self._fetch(kind, ident, current)
            except RemoteSyncError as exc:
                if "was not found" in str(exc):
                    return "missing"
                raise
            expected = _directory_digest(current)
            if kind == "sessions" and not force:
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

    def _fetch(self, kind: str, ident: str, destination: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-fetch-") as temporary:
            archive = Path(temporary) / "snapshot.tar.gz"
            result = self._run_to_file(
                _FETCH_SCRIPT,
                [self._home(), kind, ident],
                archive,
            )
            if result.returncode == 44:
                raise RemoteSyncError(f"remote {kind[:-1]} {ident} was not found")
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

    def _install(self, kind: str, ident: str, source: Path, expected: str) -> None:
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-install-") as temporary:
            archive = Path(temporary) / "snapshot.tar.gz"
            _pack(source, archive)
            with archive.open("rb") as incoming:
                result = self._run(
                    _INSTALL_SCRIPT,
                    [
                        self._home(),
                        kind,
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
                stdin=stdin,
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
