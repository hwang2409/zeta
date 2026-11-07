"""Bounded, generation-stamped OAuth credential authority for MCP servers."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

from ..core.session import env_home
from .client import MCPError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_CREDENTIAL_RECORD_BYTES = 64 * 1024
_LOCK_TIMEOUT_SECONDS = 1.0
REFRESH_LEEWAY_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class MCPOAuthToken:
    """One persisted OAuth token bundle for a configured MCP server."""

    access_token: str
    refresh_token: str | None
    expires_at: float | None
    token_type: str
    scope: str | None
    authorization_server: str
    token_endpoint: str
    authorization_endpoint: str
    client_id: str
    client_secret: str | None
    redirect_uri: str
    resource: str
    refresh_error: str | None = None


@dataclass(frozen=True, slots=True)
class CredentialSnapshot:
    generation: int
    token: MCPOAuthToken | None


class SaveDisposition(Enum):
    COMMITTED = "committed"
    STALE = "stale"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class SaveIfCurrentResult:
    disposition: SaveDisposition
    snapshot: CredentialSnapshot


class CredentialStoreErrorReason(str, Enum):
    BUSY = "busy"
    UNSAFE_FILE = "unsafe_file"
    OVERSIZED = "oversized"
    CANDIDATE_OVERSIZED = "candidate_oversized"
    MALFORMED = "malformed"
    IO = "io"
    DURABILITY = "durability"


CredentialOperation = Literal["load", "save_login", "save_if_current", "delete"]


@dataclass(frozen=True, slots=True)
class MCPCredentialStoreError(MCPError):
    reason: CredentialStoreErrorReason
    server: str
    credential_file: str
    operation: CredentialOperation


class PublicCredentialStoreError(TypedDict):
    code: str
    reason: str
    server: str
    credential_file: str
    operation: str
    recovery: str
    text: str


def public_credential_store_error(
    error: MCPCredentialStoreError,
) -> PublicCredentialStoreError:
    name = error.server
    path = error.credential_file
    details = {
        CredentialStoreErrorReason.BUSY: (
            "retry",
            f"MCP credential store is busy for '{name}'; try again.",
        ),
        CredentialStoreErrorReason.UNSAFE_FILE: (
            "move_aside_then_auth",
            (
                f"MCP credential file is unsafe for '{name}': {path}. "
                f"Move this file aside, then run '/mcp auth {name}'."
            ),
        ),
        CredentialStoreErrorReason.OVERSIZED: (
            "move_aside_then_auth",
            (
                f"MCP credential file is too large for '{name}': {path}. "
                f"Move this file aside, then run '/mcp auth {name}'."
            ),
        ),
        CredentialStoreErrorReason.CANDIDATE_OVERSIZED: (
            "retry",
            (
                f"MCP credentials are too large for '{name}'; no credential was saved. "
                "Retry authentication."
            ),
        ),
        CredentialStoreErrorReason.MALFORMED: (
            "move_aside_then_auth",
            (
                f"MCP credential file is malformed for '{name}': {path}. "
                f"Move this file aside, then run '/mcp auth {name}'."
            ),
        ),
        CredentialStoreErrorReason.IO: (
            "check_file_then_retry",
            (
                f"Could not access MCP credential file for '{name}': {path}. "
                "Check the file and retry."
            ),
        ),
        CredentialStoreErrorReason.DURABILITY: (
            "check_file_then_retry",
            (
                f"Could not durably update MCP credentials for '{name}': {path}. "
                "No success was reported; check the file and retry."
            ),
        ),
    }
    recovery, text = details[error.reason]
    return {
        "code": "mcp_credential_store",
        "reason": error.reason.value,
        "server": name,
        "credential_file": path,
        "operation": error.operation,
        "recovery": recovery,
        "text": text,
    }


def _safe_slug(name: str) -> str:
    return _SAFE_NAME.sub("_", name) or "server"


def token_store_dir(home: str | Path | None = None) -> Path:
    root = Path(home).expanduser() if home is not None else env_home()
    return root / "mcp-tokens"


def token_store_path(name: str, home: str | Path | None = None) -> Path:
    return token_store_dir(home) / f"{_safe_slug(name)}.json"


class MCPCredentialStore:
    """Own bounded credential reads, durable commits, and cross-process ordering."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        lock_timeout: float = _LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._home = home
        self._lock_timeout = lock_timeout

    async def load(self, name: str) -> CredentialSnapshot:
        return await asyncio.to_thread(self._load_transaction, name, "load")

    async def save_login(self, name: str, token: MCPOAuthToken) -> CredentialSnapshot:
        return await asyncio.to_thread(self._save_login_transaction, name, token)

    async def save_if_current(
        self,
        name: str,
        expected_generation: int,
        replacement: MCPOAuthToken,
    ) -> SaveIfCurrentResult:
        return await asyncio.to_thread(
            self._save_if_current_transaction,
            name,
            expected_generation,
            replacement,
        )

    async def delete(self, name: str) -> CredentialSnapshot:
        return await asyncio.to_thread(self._delete_transaction, name)

    def _error(
        self,
        name: str,
        operation: CredentialOperation,
        reason: CredentialStoreErrorReason,
    ) -> MCPCredentialStoreError:
        return MCPCredentialStoreError(
            reason,
            name,
            str(token_store_path(name, self._home).expanduser().absolute()),
            operation,
        )

    def _lock(
        self, name: str, operation: CredentialOperation, *, exclusive: bool
    ) -> int:
        directory = token_store_dir(self._home)
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock_path = token_store_path(name, self._home).with_suffix(".lock")
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags, 0o600)
            try:
                info = os.fstat(fd)
            except OSError:
                os.close(fd)
                raise
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(fd)
                raise self._error(
                    name, operation, CredentialStoreErrorReason.UNSAFE_FILE
                )
        except MCPCredentialStoreError:
            raise
        except OSError as exc:
            reason = (
                CredentialStoreErrorReason.UNSAFE_FILE
                if exc.errno in {errno.ELOOP, errno.ENXIO, errno.ENODEV}
                else CredentialStoreErrorReason.IO
            )
            raise self._error(name, operation, reason) from exc
        deadline = time.monotonic() + self._lock_timeout
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                return fd
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    os.close(fd)
                    raise self._error(
                        name, operation, CredentialStoreErrorReason.BUSY
                    ) from exc
                time.sleep(min(0.05, remaining))
            except OSError as exc:
                os.close(fd)
                raise self._error(
                    name, operation, CredentialStoreErrorReason.IO
                ) from exc

    @staticmethod
    def _unlock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _load_transaction(
        self, name: str, operation: CredentialOperation
    ) -> CredentialSnapshot:
        lock_fd = self._lock(name, operation, exclusive=False)
        try:
            return self._read_locked(name, operation)
        finally:
            self._unlock(lock_fd)

    def _read_locked(
        self, name: str, operation: CredentialOperation
    ) -> CredentialSnapshot:
        path = token_store_path(name, self._home)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            return CredentialSnapshot(0, None)
        except OSError as exc:
            reason = (
                CredentialStoreErrorReason.UNSAFE_FILE
                if exc.errno in {errno.ELOOP, errno.ENXIO, errno.ENODEV}
                else CredentialStoreErrorReason.IO
            )
            raise self._error(name, operation, reason) from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise self._error(
                    name, operation, CredentialStoreErrorReason.UNSAFE_FILE
                )
            data = os.read(fd, _MAX_CREDENTIAL_RECORD_BYTES + 1)
        except MCPCredentialStoreError:
            raise
        except OSError as exc:
            raise self._error(name, operation, CredentialStoreErrorReason.IO) from exc
        finally:
            os.close(fd)
        if len(data) > _MAX_CREDENTIAL_RECORD_BYTES:
            raise self._error(name, operation, CredentialStoreErrorReason.OVERSIZED)
        try:
            value = json.loads(data.decode("utf-8"))
            return _snapshot_from_json(value)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise self._error(
                name, operation, CredentialStoreErrorReason.MALFORMED
            ) from exc

    def _save_login_transaction(
        self, name: str, token: MCPOAuthToken
    ) -> CredentialSnapshot:
        operation: CredentialOperation = "save_login"
        lock_fd = self._lock(name, operation, exclusive=True)
        try:
            current = self._read_locked(name, operation)
            snapshot = CredentialSnapshot(current.generation + 1, token)
            self._write_locked(name, operation, snapshot)
            return snapshot
        finally:
            self._unlock(lock_fd)

    def _save_if_current_transaction(
        self,
        name: str,
        expected_generation: int,
        replacement: MCPOAuthToken,
    ) -> SaveIfCurrentResult:
        operation: CredentialOperation = "save_if_current"
        lock_fd = self._lock(name, operation, exclusive=True)
        try:
            current = self._read_locked(name, operation)
            if current.token is None:
                return SaveIfCurrentResult(SaveDisposition.DELETED, current)
            if current.generation != expected_generation:
                return SaveIfCurrentResult(SaveDisposition.STALE, current)
            snapshot = CredentialSnapshot(current.generation + 1, replacement)
            self._write_locked(name, operation, snapshot)
            return SaveIfCurrentResult(SaveDisposition.COMMITTED, snapshot)
        finally:
            self._unlock(lock_fd)

    def _delete_transaction(self, name: str) -> CredentialSnapshot:
        operation: CredentialOperation = "delete"
        lock_fd = self._lock(name, operation, exclusive=True)
        try:
            current = self._read_locked(name, operation)
            snapshot = CredentialSnapshot(current.generation + 1, None)
            self._write_locked(name, operation, snapshot)
            return snapshot
        finally:
            self._unlock(lock_fd)

    def _write_locked(
        self,
        name: str,
        operation: CredentialOperation,
        snapshot: CredentialSnapshot,
    ) -> None:
        directory = token_store_dir(self._home)
        target = token_store_path(name, self._home)
        payload = _snapshot_json(snapshot)
        if len(payload) > _MAX_CREDENTIAL_RECORD_BYTES:
            raise self._error(
                name, operation, CredentialStoreErrorReason.CANDIDATE_OVERSIZED
            )
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".mcp-token.", suffix=".json.tmp", dir=str(directory)
            )
            tmp_path = Path(tmp_name)
            os.fchmod(fd, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                try:
                    os.fsync(fd)
                except OSError as exc:
                    raise self._error(
                        name, operation, CredentialStoreErrorReason.DURABILITY
                    ) from exc
            finally:
                os.close(fd)
            os.replace(tmp_path, target)
            tmp_path = None
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                try:
                    os.fsync(directory_fd)
                except OSError as exc:
                    raise self._error(
                        name, operation, CredentialStoreErrorReason.DURABILITY
                    ) from exc
            finally:
                os.close(directory_fd)
        except MCPCredentialStoreError:
            raise
        except OSError as exc:
            raise self._error(name, operation, CredentialStoreErrorReason.IO) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _snapshot_json(snapshot: CredentialSnapshot) -> bytes:
    token: dict[str, object] | None = None
    if snapshot.token is not None:
        token = asdict(snapshot.token)
        token.pop("client_secret", None)
    return (
        json.dumps({"generation": snapshot.generation, "token": token}, sort_keys=True)
        + "\n"
    ).encode()


def _snapshot_from_json(value: object) -> CredentialSnapshot:
    if type(value) is not dict:
        raise ValueError("credential record must be an object")
    if "generation" not in value:
        return CredentialSnapshot(0, _token_from_json(value))
    generation = value.get("generation")
    if (
        type(generation) is not int
        or generation < 0
        or set(value) != {"generation", "token"}
    ):
        raise ValueError("invalid credential generation")
    raw_token = value.get("token")
    return CredentialSnapshot(
        generation,
        None if raw_token is None else _token_from_json(raw_token),
    )


def _token_from_json(data: object) -> MCPOAuthToken:
    if type(data) is not dict:
        raise ValueError("credential token must be an object")
    return MCPOAuthToken(
        access_token=_required_str(data, "access_token"),
        refresh_token=_optional_str(data.get("refresh_token")),
        expires_at=_optional_float(data.get("expires_at")),
        token_type=_required_str(data, "token_type", default="Bearer"),
        scope=_optional_str(data.get("scope")),
        authorization_server=_required_str(data, "authorization_server"),
        token_endpoint=_required_str(data, "token_endpoint"),
        authorization_endpoint=_required_str(data, "authorization_endpoint"),
        client_id=_required_str(data, "client_id"),
        client_secret=_optional_str(data.get("client_secret")),
        redirect_uri=_required_str(data, "redirect_uri"),
        resource=_required_str(data, "resource"),
        refresh_error=_optional_str(data.get("refresh_error")),
    )


def _required_str(
    data: dict[object, object], key: str, *, default: str | None = None
) -> str:
    value = data.get(key, default)
    if type(value) is not str or not value:
        raise ValueError(f"missing {key}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("expected string or null")
    return value or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise ValueError("expected number or null")
    return float(value)  # type: ignore[arg-type]


def token_is_expired(
    token: MCPOAuthToken,
    *,
    now: float | None = None,
    leeway: float = REFRESH_LEEWAY_SECONDS,
) -> bool:
    if token.expires_at is None:
        return False
    current = now if now is not None else time.time()
    return current + leeway >= token.expires_at


def token_state(token: MCPOAuthToken | None) -> str:
    if token is None:
        return "unauthorized"
    if token.refresh_error is not None:
        return "refresh-failed"
    if token_is_expired(token):
        return "expired"
    return "authorized"


def redact_token(token: MCPOAuthToken) -> MCPOAuthToken:
    return replace(
        token,
        access_token="<redacted>",
        refresh_token=None if token.refresh_token is None else "<redacted>",
        client_secret=None if token.client_secret is None else "<redacted>",
    )


__all__ = [
    "REFRESH_LEEWAY_SECONDS",
    "CredentialSnapshot",
    "CredentialStoreErrorReason",
    "MCPCredentialStore",
    "MCPCredentialStoreError",
    "MCPOAuthToken",
    "PublicCredentialStoreError",
    "SaveDisposition",
    "SaveIfCurrentResult",
    "public_credential_store_error",
    "redact_token",
    "token_is_expired",
    "token_state",
    "token_store_dir",
    "token_store_path",
]
