"""Persistent OAuth token storage for MCP HTTP servers.

Tokens live under ``~/.zeta/mcp-tokens/<safe_name>.json`` with mode ``0600``
and are always written via a tmp+rename so a torn write cannot corrupt them.
Values never enter provider context, transcripts, or logs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..core.session import env_home

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

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


def _safe_slug(name: str) -> str:
    return _SAFE_NAME.sub("_", name) or "server"


def token_store_dir(home: str | Path | None = None) -> Path:
    """Return the directory that holds every persisted MCP token."""

    root = Path(home).expanduser() if home is not None else env_home()
    return root / "mcp-tokens"


def token_store_path(name: str, home: str | Path | None = None) -> Path:
    """Return the token file path for one server."""

    return token_store_dir(home) / f"{_safe_slug(name)}.json"


def load_token(
    name: str, *, home: str | Path | None = None
) -> MCPOAuthToken | None:
    """Load one server's persisted token bundle, or None if absent or corrupt."""

    path = token_store_path(name, home)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read MCP token file %s", path)
        return None
    if type(data) is not dict:
        return None
    try:
        return MCPOAuthToken(
            access_token=str(data["access_token"]),
            refresh_token=_optional_str(data.get("refresh_token")),
            expires_at=_optional_float(data.get("expires_at")),
            token_type=str(data.get("token_type", "Bearer")),
            scope=_optional_str(data.get("scope")),
            authorization_server=str(data["authorization_server"]),
            token_endpoint=str(data["token_endpoint"]),
            authorization_endpoint=str(data["authorization_endpoint"]),
            client_id=str(data["client_id"]),
            client_secret=_optional_str(data.get("client_secret")),
            redirect_uri=str(data["redirect_uri"]),
            resource=str(data["resource"]),
            refresh_error=_optional_str(data.get("refresh_error")),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("MCP token file %s is malformed; ignoring", path)
        return None


def _optional_str(value: object) -> str | None:
    if type(value) is str and value:
        return value
    return None


def _optional_float(value: object) -> float | None:
    if type(value) in {int, float}:
        return float(value)  # type: ignore[arg-type]
    return None


def save_token(
    name: str,
    token: MCPOAuthToken,
    *,
    home: str | Path | None = None,
) -> Path:
    """Atomically persist a token bundle under mode 0600."""

    directory = token_store_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    target = token_store_path(name, home)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mcp-token.", suffix=".json.tmp", dir=str(directory)
    )
    tmp_path = Path(tmp_name)
    payload = json.dumps(asdict(token), indent=2, sort_keys=True) + "\n"
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def delete_token(name: str, *, home: str | Path | None = None) -> None:
    """Remove one server's token file, if it exists."""

    token_store_path(name, home).unlink(missing_ok=True)


def token_is_expired(
    token: MCPOAuthToken,
    *,
    now: float | None = None,
    leeway: float = REFRESH_LEEWAY_SECONDS,
) -> bool:
    """Return True if the access token has expired (with a refresh leeway)."""

    if token.expires_at is None:
        return False
    current = now if now is not None else time.time()
    return current + leeway >= token.expires_at


def token_state(token: MCPOAuthToken | None) -> str:
    """Return the auth state string for /mcp status output."""

    if token is None:
        return "unauthorized"
    if token.refresh_error is not None:
        return "refresh-failed"
    if token_is_expired(token):
        return "expired"
    return "authorized"


def redact_token(token: MCPOAuthToken) -> MCPOAuthToken:
    """Return a copy safe to include in log or debug output."""

    return replace(
        token,
        access_token="<redacted>",
        refresh_token=None if token.refresh_token is None else "<redacted>",
        client_secret=None if token.client_secret is None else "<redacted>",
    )


__all__ = [
    "REFRESH_LEEWAY_SECONDS",
    "MCPOAuthToken",
    "delete_token",
    "load_token",
    "redact_token",
    "save_token",
    "token_is_expired",
    "token_state",
    "token_store_dir",
    "token_store_path",
]
