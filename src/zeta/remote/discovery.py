"""Remote host and observer row discovery."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAX_ROWS = 128

@dataclass(frozen=True, slots=True)
class RemoteSession:
    host: str
    session_id: str
    name: str
    state: str | None
    machine_id: str = ""


def validate_host(value: str) -> str:
    if not value or value != value.strip() or any(c.isspace() or c in "/'\";\n\x00" for c in value):
        raise ValueError("invalid remote host")
    return value


def split_target(value: str) -> tuple[str, str | None]:
    host, sep, prefix = value.partition(":")
    validate_host(host)
    if sep and (not prefix or any(c in prefix for c in "/\x00") or any(c.isspace() for c in prefix)):
        raise ValueError("invalid remote session prefix")
    return host, prefix if sep else None


def choose_session(rows: Iterable[RemoteSession], prefix: str) -> RemoteSession:
    matches = [row for row in rows if row.session_id.startswith(prefix) or row.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("remote session prefix is not unique")
    return matches[0]


def ssh_config_aliases(path: Path | None = None) -> tuple[str, ...]:
    path = path or Path.home() / ".ssh" / "config"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    result: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "host":
            result.extend(alias for alias in parts[1:] if "*" not in alias and "?" not in alias and alias not in result)
    return tuple(result)


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
    hosts = values.get("hosts", values.keys())
    if isinstance(hosts, dict):
        hosts = hosts.keys()
    if not isinstance(hosts, (list, tuple)):
        return ()
    return tuple(validate_host(str(host)) for host in hosts)[:MAX_ROWS]


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
