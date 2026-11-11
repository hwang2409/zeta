"""Remote host and observer row discovery."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

MAX_ROWS = 128
MAX_HOST_BYTES = 255
MAX_PREFIX_BYTES = 128

@dataclass(frozen=True, slots=True)
class RemoteTarget:
    name: str
    host: str
    remote_home: str = "~/.zeta"

@dataclass(frozen=True, slots=True)
class RemoteSession:
    host: str
    session_id: str
    name: str
    state: str | None
    machine_id: str = ""


def _validate_component(value: str, label: str, limit: int) -> str:
    encoded = value.encode("ascii", "strict")
    if not 1 <= len(encoded) <= limit or not value[0].isalnum() or any(
        not (char.isalnum() or char in "._-@") for char in value
    ):
        raise ValueError(f"invalid remote {label}")
    return value


def validate_host(value: str) -> str:
    return _validate_component(value, "host", MAX_HOST_BYTES)


def validate_prefix(value: str) -> str:
    return _validate_component(value, "session prefix", MAX_PREFIX_BYTES)


def split_target(value: str) -> tuple[str, str | None]:
    host, sep, prefix = value.partition(":")
    validate_host(host)
    if sep:
        validate_prefix(prefix)
    return host, prefix if sep else None


def choose_session(rows: Iterable[RemoteSession], prefix: str) -> RemoteSession:
    matches = [row for row in rows if row.session_id.startswith(prefix) or row.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("remote session prefix is not unique")
    return matches[0]


def configured_hosts(home: Path) -> tuple[str, ...]:
    path = home / "settings.toml"
    # Keep this parser deliberately narrow: remotes are read from the simple
    # table used by the settings loader, without executing arbitrary config.
    try:
        import tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    values = data.get("remotes", {})
    if not isinstance(values, dict):
        return ()
    targets: list[RemoteTarget] = []
    for name, url in values.items():
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        parsed = urlparse(url)
        if parsed.scheme != "ssh" or not parsed.hostname or parsed.query or parsed.fragment:
            continue
        try:
            host = validate_host(parsed.hostname)
        except (UnicodeEncodeError, ValueError):
            continue
        targets.append(RemoteTarget(name, host, unquote(parsed.path) if parsed.path else "~/.zeta"))
    return tuple(targets[:MAX_ROWS])


def dedupe_rows(rows: Iterable[RemoteSession]) -> tuple[RemoteSession, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for row in rows:
        key = (row.machine_id or row.host, row.session_id)
        if key not in seen:
            seen.add(key); result.append(row)
        if len(result) >= MAX_ROWS:
            break
    return tuple(result)
