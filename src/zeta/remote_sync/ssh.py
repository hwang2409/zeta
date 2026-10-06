"""SSH transport for remote session and memory publication."""

from __future__ import annotations

import io
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import (
    MemoryTransferResult,
    RemoteSyncError,
    _copy_project_tree,
    _copy_tree,
    _directory_digest,
    _map_missing_cwd,
    _safe_component,
    _sync_memory,
    _tree_state,
)

_HOST = re.compile(r"[A-Za-z0-9_.@-]+\Z")

_FETCH_SCRIPT = r'''
import fcntl, io, os, pathlib, sys, tarfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
kind, ident = sys.argv[2], sys.argv[3]
if kind not in {"sessions", "projects"} or pathlib.Path(ident).parts != (ident,): sys.exit(45)
root = home / kind / ident
if not root.is_dir() or root.is_symlink(): sys.exit(44)
locks = []
try:
    for path in [root / ".lock", *sorted(root.rglob(".lock"))]:
        if path.is_file() and not path.is_symlink():
            handle = path.open("rb"); fcntl.flock(handle.fileno(), fcntl.LOCK_EX); locks.append(handle)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in [root, *sorted(root.rglob("*"))]:
            relative = path.relative_to(root)
            if path.is_symlink(): sys.exit(46)
            if path.name in {".lock", ".spill.lock"}: continue
            if path.is_dir() or path.is_file():
                archive.add(path, arcname=str(pathlib.Path("payload") / relative), recursive=False)
            else: sys.exit(46)
finally:
    for handle in reversed(locks): handle.close()
'''

_INSTALL_SCRIPT = r'''
import fcntl, hashlib, json, os, pathlib, shutil, sys, tarfile, tempfile
home = pathlib.Path(sys.argv[1]).expanduser().resolve()
kind, ident, expected = sys.argv[2], sys.argv[3], sys.argv[4]
if kind not in {"sessions", "projects"} or pathlib.Path(ident).parts != (ident,): sys.exit(45)
parent = home / kind; parent.mkdir(parents=True, exist_ok=True, mode=0o700)
destination = parent / ident
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
        if path.is_file() and path.name not in {".lock", ".spill.lock"}:
            value.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return value.hexdigest()
if expected != digest(destination): sys.exit(47)
staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{ident}.incoming-", dir=parent))
try:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            parts = pathlib.PurePosixPath(member.name).parts
            if not parts or parts[0] != "payload" or any(p in {"", ".", ".."} for p in parts) or member.issym() or member.islnk(): sys.exit(46)
            target = staging.joinpath(*parts[1:])
            if member.isdir(): target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None: sys.exit(46)
                with target.open("wb") as output: shutil.copyfileobj(source, output)
                target.chmod(0o600)
            else: sys.exit(46)
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


@dataclass(slots=True)
class SshTransport:
    """Transfer snapshots through one configured SSH host and remote ZETA_HOME."""

    host: str
    remote_home: str = "~/.zeta"
    name: str | None = None
    _resolved_home: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _HOST.fullmatch(self.host) or self.host.startswith("-"):
            raise RemoteSyncError("SSH host is invalid")
        if "\x00" in self.remote_home or "\n" in self.remote_home:
            raise RemoteSyncError("remote ZETA_HOME is invalid")
        if self.name is None:
            self.name = self.host
        _safe_component(self.name, "remote name")

    @classmethod
    def from_url(cls, name: str, value: str) -> SshTransport:
        parsed = urlparse(value)
        if parsed.scheme != "ssh" or not parsed.hostname or parsed.query or parsed.fragment:
            raise RemoteSyncError(f"remote {name!r} must be an ssh:// URL")
        host = parsed.netloc
        path = unquote(parsed.path) if parsed.path else "~/.zeta"
        return cls(host=host, remote_home=path, name=name)

    def publish_session(self, snapshot: Path, *, force: bool) -> Path:
        session_id = _safe_component(snapshot.name, "session id")
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-publish-") as temporary:
            outgoing = Path(temporary) / session_id
            _copy_tree(snapshot, outgoing)
            _map_missing_cwd(
                outgoing, Path(self._home()) / "remote-workspaces" / session_id
            )
            expected = self._existing_state("sessions", session_id, outgoing, force)
            self._install("sessions", session_id, outgoing, expected)
            shutil.copyfile(outgoing / "transfer.json", snapshot / "transfer.json")
        return snapshot

    def fetch_session(self, session_id: str, destination: Path) -> Path:
        self._fetch("sessions", _safe_component(session_id, "session id"), destination)
        return destination

    def push_memory(self, source_home: Path, project_id: str) -> MemoryTransferResult:
        return self._memory(source_home, project_id, pull=False)

    def pull_memory(
        self, destination_home: Path, project_id: str
    ) -> MemoryTransferResult:
        return self._memory(destination_home, project_id, pull=True)

    def _memory(
        self, local_home: Path, project_id: str, *, pull: bool
    ) -> MemoryTransferResult:
        project_id = _safe_component(project_id, "project id")
        with tempfile.TemporaryDirectory(prefix="zeta-ssh-memory-") as temporary:
            mirror_home = Path(temporary) / "remote"
            remote_project = mirror_home / "projects" / project_id
            try:
                self._fetch("projects", project_id, remote_project)
                expected = _directory_digest(remote_project)
            except RemoteSyncError as exc:
                if "was not found" not in str(exc):
                    raise
                expected = "missing"
                source = local_home / "projects" / project_id
                if pull or not source.is_dir():
                    raise
                _copy_project_tree(source, remote_project)
            result = (
                _sync_memory(
                    mirror_home,
                    local_home,
                    project_id=project_id,
                    peer=str(self.name),
                    source_label=str(self.name),
                )
                if pull
                else _sync_memory(
                    local_home,
                    mirror_home,
                    project_id=project_id,
                    peer=str(self.name),
                    source_label="local",
                )
            )
            if not pull:
                self._install("projects", project_id, remote_project, expected)
            return result

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
        result = self._run(_FETCH_SCRIPT, [self._home(), kind, ident], check=False)
        if result.returncode == 44:
            raise RemoteSyncError(f"remote {kind[:-1]} {ident} was not found")
        if result.returncode:
            raise RemoteSyncError(
                f"SSH snapshot failed on {self.host} (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        _unpack(result.stdout, destination)

    def _install(self, kind: str, ident: str, source: Path, expected: str) -> None:
        archive = _pack(source)
        result = self._run(
            _INSTALL_SCRIPT,
            [self._home(), kind, ident, expected],
            input=archive,
            check=False,
        )
        if result.returncode == 47:
            raise RemoteSyncError("remote changed during transfer; retry after inspection")
        if result.returncode == 48:
            raise RemoteSyncError("remote session is active; stop it before replacement")
        if result.returncode:
            raise RemoteSyncError(
                f"SSH publication failed on {self.host} (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )

    def _run(
        self,
        script: str,
        arguments: list[str],
        *,
        input: bytes | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        command = " ".join(
            shlex.quote(value)
            for value in ("/usr/bin/python3", "-B", "-c", script, *arguments)
        )
        try:
            result = subprocess.run(
                ["ssh", "--", self.host, command],
                input=input,
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


def _pack(source: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path in [source, *sorted(source.rglob("*"))]:
            if path.is_symlink():
                raise RemoteSyncError("cannot upload a symlink")
            relative = path.relative_to(source)
            archive.add(
                path,
                arcname=str(Path("payload") / relative),
                recursive=False,
            )
    return output.getvalue()


def _unpack(data: bytes, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != "payload"
                or any(part in {"", ".", ".."} for part in parts)
                or member.issym()
                or member.islnk()
            ):
                raise RemoteSyncError("remote archive contains an unsafe path")
            target = destination.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise RemoteSyncError("remote archive member is unreadable")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o600)
            else:
                raise RemoteSyncError("remote archive contains an unsupported file")


__all__ = ["SshTransport"]
