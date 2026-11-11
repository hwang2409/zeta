"""Secure descriptors for hosted read-only observer sockets."""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_SOCKET_BYTES = 90
MAX_DESCRIPTOR_BYTES = 8192
MAX_DISPLAY_BYTES = 256


class DescriptorError(ValueError):
    pass


def observer_socket_dir(directory: Path | None = None) -> Path:
    directory = directory or Path(f"/tmp/zeta-{os.getuid()}")
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
    except OSError as exc:
        raise DescriptorError("observer socket directory is unavailable") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise DescriptorError("observer socket directory has unsafe ownership, mode, or type")
    return directory


def create_observer_socket(path: Path) -> socket.socket:
    observer_socket_dir(path.parent)
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        path_stat = None
    if path_stat is not None:
        if not stat.S_ISSOCK(path_stat.st_mode):
            raise DescriptorError(f"refusing to replace non-socket path: {path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise DescriptorError(f"could not inspect socket: {path}") from exc
        else:
            raise DescriptorError(f"a server is already listening on {path}")
        finally:
            probe.close()
        _unlink_if_owned(path, (path_stat.st_dev, path_stat.st_ino))
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    identity: tuple[int, int] | None = None
    try:
        unix_socket.bind(str(path))
        os.chmod(path, 0o600)
        unix_socket.listen(socket.SOMAXCONN)
        socket_stat = path.lstat()
        identity = (socket_stat.st_dev, socket_stat.st_ino)
        return unix_socket
    except BaseException:
        unix_socket.close()
        if identity is not None:
            _unlink_if_owned(path, identity)
        raise


def unlink_observer_socket(path: Path, identity: tuple[int, int]) -> None:
    _unlink_if_owned(path, identity)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if (path_stat.st_dev, path_stat.st_ino) == identity:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def socket_path(home: Path, session_id: str) -> Path:
    digest = hashlib.sha256(f"{home.resolve()}\0{session_id}".encode()).hexdigest()[:32]
    path = Path(f"/tmp/zeta-{os.getuid()}") / f"{digest}.sock"
    if len(os.fsencode(path)) > MAX_SOCKET_BYTES:
        raise DescriptorError("observer socket path is too long")
    return path


@dataclass(frozen=True, slots=True)
class Descriptor:
    socket: str
    session_id: str
    pid: int
    name: str = ""
    cwd: str = ""
    provider: str = ""
    model: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "socket": self.socket,
            "session_id": self.session_id,
            "pid": self.pid,
            "name": self.name[:MAX_DISPLAY_BYTES],
            "cwd": self.cwd[:MAX_DISPLAY_BYTES],
            "provider": self.provider[:MAX_DISPLAY_BYTES],
            "model": self.model[:MAX_DISPLAY_BYTES],
        }

    def encode(self) -> bytes:
        value = json.dumps(self.payload(), separators=(",", ":"), sort_keys=True).encode()
        if len(value) > MAX_DESCRIPTOR_BYTES:
            raise DescriptorError("observer descriptor is too large")
        return value

    @classmethod
    def decode(cls, raw: bytes) -> Descriptor:
        if len(raw) > MAX_DESCRIPTOR_BYTES:
            raise DescriptorError("observer descriptor is too large")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DescriptorError("invalid observer descriptor") from exc
        if not isinstance(value, dict):
            raise DescriptorError("invalid observer descriptor")
        allowed = ("socket", "session_id", "pid", "name", "cwd", "provider", "model")
        if any(key not in allowed for key in value):
            raise DescriptorError("unknown observer descriptor field")
        if not all(isinstance(value.get(key), str) for key in ("socket", "session_id")):
            raise DescriptorError("invalid observer descriptor")
        if any(key in value and (not isinstance(value[key], str) or len(value[key].encode()) > MAX_DISPLAY_BYTES) for key in ("name", "cwd", "provider", "model")):
            raise DescriptorError("invalid observer descriptor")
        if not isinstance(value.get("pid"), int) or isinstance(value["pid"], bool):
            raise DescriptorError("invalid observer descriptor")
        return cls(
            socket=value["socket"], session_id=value["session_id"], pid=value["pid"],
            name=value.get("name", ""), cwd=value.get("cwd", ""),
            provider=value.get("provider", ""), model=value.get("model", ""),
        )


def descriptor_dir(home: Path) -> Path:
    return home / "run" / "remote"


def descriptor_path(home: Path, session_id: str) -> Path:
    return descriptor_dir(home) / f"{session_id}.json"


def publish(home: Path, descriptor: Descriptor) -> Path:
    directory = descriptor_dir(home)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = descriptor_path(home, descriptor.session_id)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(descriptor.encode())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    return path


def remove(home: Path, session_id: str) -> None:
    descriptor_path(home, session_id).unlink(missing_ok=True)


def validate(path: Path, *, expected_session_id: str | None = None) -> Descriptor:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode):
            raise DescriptorError("observer descriptor has unsafe ownership or mode")
        raw = os.read(fd, MAX_DESCRIPTOR_BYTES + 1)
    except OSError as exc:
        raise DescriptorError("observer descriptor is unavailable") from exc
    finally:
        with contextlib.suppress(UnboundLocalError, OSError):
            os.close(fd)
    descriptor = Descriptor.decode(raw)
    if expected_session_id is not None and descriptor.session_id != expected_session_id:
        raise DescriptorError("observer session does not match descriptor")
    socket = Path(descriptor.socket)
    if not socket.is_absolute() or len(os.fsencode(socket)) > MAX_SOCKET_BYTES:
        raise DescriptorError("observer socket path is invalid")
    try:
        socket_info = socket.lstat()
    except OSError as exc:
        raise DescriptorError("observer socket is unavailable") from exc
    if socket_info.st_uid != os.getuid() or stat.S_IMODE(socket_info.st_mode) != 0o600 or not stat.S_ISSOCK(socket_info.st_mode):
        raise DescriptorError("observer socket has unsafe ownership, mode, or type")
    if descriptor.pid <= 0:
        raise DescriptorError("observer pid is invalid")
    try:
        os.kill(descriptor.pid, 0)
    except OSError as exc:
        raise DescriptorError("observer host is not running") from exc
    return descriptor
