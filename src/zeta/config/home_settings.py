"""The one writer of the home ``settings.toml``: safe reads, compare-and-set updates, restart discovery.

A client sees an allowlisted projection plus ``revision``, the SHA-256 of the exact file bytes
(an absent file hashes as empty bytes). An update names that revision; it takes the sibling
``settings.toml.lock``, rereads the bytes under the lock, and refuses a stale revision, so an edit
finished before the reread is never lost. The lock serializes zeta writers only; an editor that
writes after the reread is not stopped. The edit touches only the built-in endpoint line (under
``[otel]``, or a top-level ``otel.endpoint``), checked against a ``tomlkit`` parse, and a
candidate is published only after ``load_settings``'s own validation accepts it.

Every function blocks (file locks, fsync, a directory scan); call it off the event loop.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import TOMLKitError

from ..core.session_execution import execution_owner_state
from ..core.session_files import (
    SessionError,
    child_directory,
    open_session_file,
    require_off_loop_lock,
    write_session_file,
)
from ..telemetry.receiver.process import OTLP_URL
from .settings import SETTINGS_FILENAME, validate_home_settings

LOCK_NAME = f"{SETTINGS_FILENAME}.lock"
LOCK_SECONDS = 0.25
LOCK_POLL_SECONDS = 0.01
RESTART_SCAN_SECONDS = 1.0
_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEADER = re.compile(rb"[ \t]*\[")  # a table or array-of-tables header line
_OTEL_HEADER = re.compile(rb"""[ \t]*\[[ \t]*(?:otel|"otel"|'otel')[ \t]*\][ \t]*(?:#[^\r\n]*)?\r?\n?\Z""")
_KEY = rb"""[ \t]*(?:endpoint|"endpoint"|'endpoint')[ \t]*="""
_ENDPOINT = re.compile(_KEY)
_DOTTED = re.compile(rb"""[ \t]*(?:otel|"otel"|'otel')[ \t]*\.""" + _KEY)


class HomeSettingsError(Exception):
    """A refused read or update; ``code`` is the wire ``data.code``, ``data`` its extra fields."""

    def __init__(self, code: str, **data: object) -> None:
        super().__init__(code)
        self.code = code
        self.data = data


def get(home: Path) -> dict[str, Any]:
    """The safe projection of ``home``'s settings and their revision."""
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        raw = _read(home_fd)
    finally:
        os.close(home_fd)
    try:
        settings = _project(raw, home)
    except HomeSettingsError:
        settings = {"telemetry": {"source": "unknown", "editable": False}}
        return {"revision": _revision(raw), "settings": settings, "settings_malformed": True}
    return {"revision": _revision(raw), "settings": settings, "settings_malformed": False}


def update(home: Path, expected_revision: object, changes: object) -> dict[str, Any]:
    """Apply ``changes`` if the file still has ``expected_revision``; report what must restart."""
    enabled = _parse_changes(expected_revision, changes)
    require_off_loop_lock("home settings")
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        lock_fd = _open_file(home_fd, LOCK_NAME, os.O_RDWR | os.O_CREAT)
        try:
            _lock(lock_fd)
            raw = _read(home_fd)
            if _revision(raw) != expected_revision:
                raise HomeSettingsError("stale_revision", current_revision=_revision(raw))
            current = _project(raw, home)["telemetry"]
            if current["source"] == "custom":
                raise HomeSettingsError("custom_home_setting")
            candidate = raw if current["enabled"] == enabled else _with_telemetry(raw, enabled)
            _project(candidate, home)  # the same validation load_settings runs, before publication
            if candidate != raw:
                write_session_file(home_fd, SETTINGS_FILENAME, candidate)
            raw = _read(home_fd)
        finally:
            os.close(lock_fd)  # closing the descriptor releases the lock
        result = {"revision": _revision(raw), "settings": _project(raw, home)}
        result["restart"] = _restart(home_fd)
        return result
    finally:
        os.close(home_fd)


def _parse_changes(expected_revision: object, changes: object) -> bool:
    if type(expected_revision) is not str or _REVISION.fullmatch(expected_revision) is None:
        raise HomeSettingsError("invalid_params")
    if type(changes) is not list or len(changes) != 1:
        raise HomeSettingsError("invalid_params")
    change = changes[0]
    if (
        type(change) is not dict
        or change.keys() != {"key", "value"}
        or change["key"] != "telemetry.enabled"
        or type(change["value"]) is not bool
    ):
        raise HomeSettingsError("invalid_params")
    return change["value"]


def _revision(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _open_file(home_fd: int, name: str, flags: int) -> int:
    """Open a regular, unshared file; never follow a link or block on a FIFO."""
    try:
        return open_session_file(home_fd, name, flags)
    except FileNotFoundError:
        raise
    except SessionError:
        raise HomeSettingsError("settings_unsafe") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise HomeSettingsError("settings_unsafe") from None
        raise


def _read(home_fd: int) -> bytes:
    try:
        fd = _open_file(home_fd, SETTINGS_FILENAME, os.O_RDONLY)
    except FileNotFoundError:
        return b""
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _lock(fd: int) -> None:
    deadline = time.monotonic() + LOCK_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise HomeSettingsError("settings_busy", retryable=True) from None
            time.sleep(LOCK_POLL_SECONDS)


def _project(raw: bytes, home: Path) -> dict[str, Any]:
    try:
        endpoint = validate_home_settings(raw, home).otel_endpoint
    except (ValueError, TypeError):
        raise HomeSettingsError("settings_malformed") from None
    if endpoint is None:
        telemetry = {"enabled": False, "source": "off", "editable": True}
    elif endpoint == OTLP_URL:
        telemetry = {"enabled": True, "source": "builtin", "editable": True}
    else:
        telemetry = {"enabled": True, "source": "custom", "editable": False}
    return {"telemetry": telemetry}


def _with_telemetry(raw: bytes, enabled: bool) -> bytes:
    """Add or remove only the built-in endpoint line; every other byte stays as it was.

    With an ``[otel]`` table, on inserts ``endpoint`` under its header and off removes that line;
    the header is never added or removed. Without one, on inserts a top-level
    ``otel.endpoint`` line before the first table (or at the end) and off removes it, so
    on-then-off restores the exact bytes. Line endings and the trailing-newline state are kept.
    Each edit must parse to the same document as the requested change, else the layout (an
    inline table, a line inside a multiline string) is refused as a hand-configured setting.
    """
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    unterminated = bool(raw) and not raw.endswith(b"\n")
    lines = (raw + newline if unterminated else raw).splitlines(keepends=True)
    expected = _document(raw, enabled)
    for edited in _on_edits(lines, newline) if enabled else _off_edits(lines):
        candidate = b"".join(edited)
        if unterminated and candidate.endswith(newline):
            candidate = candidate[: -len(newline)]
        try:
            if _document(candidate) == expected:
                return candidate
        except HomeSettingsError:
            pass  # the edit does not parse here; try the next one
    raise HomeSettingsError("custom_home_setting")


def _on_edits(lines: list[bytes], newline: bytes) -> list[list[bytes]]:
    """Insert the endpoint under an ``[otel]`` header, or as a dotted line before the first table."""
    value = b' = "' + OTLP_URL.encode() + b'"' + newline
    headers = [i for i, line in enumerate(lines) if _OTEL_HEADER.match(line)]
    if headers:
        return [lines[: i + 1] + [b"endpoint" + value] + lines[i + 1 :] for i in headers]
    first = _first_table(lines, 0)
    return [lines[:first] + [b"otel.endpoint" + value] + lines[first:]]


def _off_edits(lines: list[bytes]) -> list[list[bytes]]:
    """Remove one endpoint line: a top-level dotted one, or one inside an ``[otel]`` table."""
    found = [i for i in range(_first_table(lines, 0)) if _DOTTED.match(lines[i])]
    for header in (i for i, line in enumerate(lines) if _OTEL_HEADER.match(line)):
        found += [i for i in range(header + 1, _first_table(lines, header + 1)) if _ENDPOINT.match(lines[i])]
    return [lines[:i] + lines[i + 1 :] for i in found]


def _first_table(lines: list[bytes], start: int) -> int:
    return next((i for i in range(start, len(lines)) if _HEADER.match(lines[i])), len(lines))


def _document(raw: bytes, enabled: bool | None = None) -> dict[str, Any]:
    """``raw`` as plain data, with the endpoint set (``enabled``) or removed, and no empty ``otel``."""
    try:
        document = tomlkit.parse(raw.decode("utf-8")).unwrap()
    except (TOMLKitError, UnicodeDecodeError):
        raise HomeSettingsError("settings_malformed") from None
    if enabled is not None and isinstance(document.get("otel", {}), dict):
        otel = document.setdefault("otel", {})
        if enabled:
            otel["endpoint"] = OTLP_URL
        else:
            otel.pop("endpoint", None)
    if document.get("otel") == {}:
        del document["otel"]
    return document


def _restart(home_fd: int) -> dict[str, Any]:
    """Every top-level session a live process owns, by the kernel lease lock; never by a pid hint.

    No row cap and no recency filter; a scan that hits its deadline or an unreadable directory
    reports ``complete: false`` with the owners it confirmed. ``serve`` is always true: the
    accepting process configured telemetry once, at startup.
    """
    deadline = time.monotonic() + RESTART_SCAN_SECONDS
    sessions: list[str] = []
    complete = True
    try:
        with child_directory(home_fd, "sessions") as root_fd:
            for name in os.listdir(root_fd):
                if time.monotonic() >= deadline:
                    complete = False
                    break
                try:
                    with child_directory(root_fd, name) as session_fd:
                        if execution_owner_state(session_fd) is not None:
                            sessions.append(name)
                except OSError as exc:
                    if exc.errno not in (errno.ENOTDIR, errno.ELOOP, errno.ENOENT):
                        complete = False
                except SessionError:
                    complete = False
    except FileNotFoundError:
        pass
    except OSError:
        complete = False
    return {"serve": True, "sessions": sorted(sessions), "complete": complete}
