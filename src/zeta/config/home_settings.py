"""The one writer of the home ``settings.toml``: safe reads and compare-and-set updates.

A client sees an allowlisted projection plus ``revision``, the SHA-256 of the exact file bytes
(an absent file hashes as empty bytes). An update names that revision; it takes the sibling
``settings.toml.lock``, rereads the bytes under the lock, and refuses a stale revision, so an edit
finished before the reread is never lost. The lock serializes zeta writers only; an editor that writes after the reread is not stopped.
Edits preserve bytes outside the changed setting and are validated by the startup parser.

Every function blocks (file locks and fsync); call it off the event loop.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import TOMLKitError

from ..agent.budget import HARD_MAX_AGENT_DEPTH
from ..core.session_files import (
    SessionError,
    open_session_file,
    require_off_loop_lock,
    write_session_file,
)
from ..telemetry.receiver.process import OTLP_URL
from .settings import SETTINGS_FILENAME, Settings, validate_home_settings

LOCK_NAME = f"{SETTINGS_FILENAME}.lock"
LOCK_SECONDS = 0.25
LOCK_POLL_SECONDS = 0.01
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
    with _locked_home(home) as home_fd:
        raw = _read(home_fd)
        revision = _revision(raw)
    try:
        settings = _project(raw, home)
    except HomeSettingsError:
        settings = {"telemetry": {"source": "unknown", "editable": False}}
        return {"revision": revision, "settings": settings, "settings_malformed": True}
    return {"revision": revision, "settings": settings, "settings_malformed": False}


def update(home: Path, expected_revision: object, changes: object) -> dict[str, Any]:
    """Apply one change if the file still has the expected revision."""
    key, value = _parse_changes(expected_revision, changes)
    with _locked_home(home) as home_fd:
        raw = _read(home_fd)
        revision = _revision(raw)
        if revision != expected_revision:
            raise HomeSettingsError("stale_revision")
        settings = _project(raw, home)
        candidate = _edit(raw, settings, key, value)
        changed = candidate != raw
        if changed:
            settings = _project(candidate, home)  # validate before publication
            write_session_file(home_fd, SETTINGS_FILENAME, candidate)
            revision = _revision(candidate)
    return {
        "revision": revision,
        "settings": settings,
        "changed": changed,
        "restart_serve": changed and key == "telemetry.enabled",
    }


@contextmanager
def _locked_home(home: Path) -> Iterator[int]:
    require_off_loop_lock("home settings")
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        lock_fd = _open_file(home_fd, LOCK_NAME, os.O_RDWR | os.O_CREAT)
        try:
            _lock(lock_fd)
            yield home_fd
        finally:
            os.close(lock_fd)
    finally:
        os.close(home_fd)


def _valid_model(value: object) -> bool:
    return (type(value) is str and 0 < len(value) <= 256 and value == value.strip()
            and all(char.isprintable() for char in value))


_VALUE_VALIDATORS: dict[str, Callable[[object], bool]] = {
    "telemetry.enabled": lambda value: type(value) is bool,
    "agents.explicit_finish": lambda value: type(value) is bool,
    "agents.max_depth": lambda value: type(value) is int and 0 <= value <= HARD_MAX_AGENT_DEPTH,
    "model": _valid_model,
}


def _parse_changes(expected_revision: object, changes: object) -> tuple[str, Any]:
    if type(expected_revision) is not str or _REVISION.fullmatch(expected_revision) is None:
        raise HomeSettingsError("invalid_params")
    if type(changes) is not list or len(changes) != 1:
        raise HomeSettingsError("invalid_params")
    change = changes[0]
    if type(change) is not dict or change.keys() != {"key", "value"}:
        raise HomeSettingsError("invalid_params")
    key, value = change["key"], change["value"]
    validate = _VALUE_VALIDATORS.get(key) if type(key) is str else None
    if validate is None or not validate(value):
        raise HomeSettingsError("invalid_params")
    return key, value


def _edit(raw: bytes, settings: dict[str, Any], key: str, value: Any) -> bytes:
    if key == "telemetry.enabled":
        telemetry = settings["telemetry"]
        if not telemetry["editable"]:
            raise HomeSettingsError("invalid_params")
        return raw if telemetry["enabled"] == value else _with_telemetry(raw, value)
    table, _, name = key.rpartition(".")
    current = settings[table].get(name) if table else settings[name]
    return raw if current == value else _with_scalar(raw, table, name, value)


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


def _read_home(home: Path) -> bytes:
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return _read(home_fd)
    finally:
        os.close(home_fd)


def read_settings(home: Path) -> Settings:
    """Read validated home settings without taking a write lock."""
    return validate_home_settings(_read_home(home), home)


def _project(raw: bytes, home: Path) -> dict[str, Any]:
    try:
        settings = validate_home_settings(raw, home)
        return {
            "telemetry": telemetry_projection(settings),
            "model": settings.model,
            "agents": {
                "max_depth": settings.agents_max_depth,
                "explicit_finish": settings.agents_explicit_finish is True,
            },
        }
    except (ValueError, TypeError):
        raise HomeSettingsError("settings_malformed") from None


def telemetry_projection(settings: Settings) -> dict[str, Any]:
    endpoint = settings.otel_endpoint
    if endpoint is None:
        telemetry = {"enabled": False, "source": "off", "editable": True}
    elif endpoint == OTLP_URL:
        telemetry = {"enabled": True, "source": "builtin", "editable": True}
    else:
        telemetry = {"enabled": True, "source": "custom", "editable": False}
    return telemetry


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
    raise HomeSettingsError("invalid_params")


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


def _with_scalar(raw: bytes, table: str, name: str, value: str | int | bool) -> bytes:
    """Replace only a scalar's bytes, or insert a line without formatting existing lines."""
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    unterminated = bool(raw) and not raw.endswith(b"\n")
    lines = (raw + newline if unterminated else raw).splitlines(keepends=True)
    expected = _document(raw)
    parent = expected.setdefault(table, {}) if table else expected
    if not isinstance(parent, dict):
        raise HomeSettingsError("invalid_params")
    present = name in parent
    parent[name] = value
    encoded = json.dumps(value, ensure_ascii=False).encode()
    # Only accept a byte edit that changes the intended TOML value, not a matching decoy.
    for candidate in _scalar_edits(lines, newline, table, name, encoded, present=present):
        if unterminated:
            candidate = candidate.removesuffix(newline)
        try:
            if _document(candidate) == expected:
                return candidate
        except HomeSettingsError:
            pass
    raise HomeSettingsError("invalid_params")


def _scalar_edits(
    lines: list[bytes], newline: bytes, table: str, name: str, encoded: bytes, *, present: bool,
) -> Iterator[bytes]:
    def atom(name: bytes) -> bytes:
        return rb"(?:" + name + rb"|'" + name + rb"'|\"" + name + rb"\")"

    key = name.encode()
    dotted = (table + "." + name).encode() if table else key
    scalar = rb'''("(?:[^"\\\r\n]|\\.)*"|'[^'\r\n]*'|[-+a-zA-Z0-9_]+)'''
    plain = atom(key)
    patterns = [plain]
    if table:
        namespace = atom(table.encode())
        header = re.compile(rb"[ \t]*\[[ \t]*" + namespace + rb"[ \t]*\][ \t]*(?:#[^\r\n]*)?\r?\n?\Z")
        headers = [i for i, line in enumerate(lines) if header.match(line)]
        patterns.append(namespace + rb"[ \t]*\.[ \t]*" + plain)
    else:
        headers = []

    if not present:
        for i, spelling in [(0, dotted), *((i + 1, key) for i in headers)]:
            yield b"".join(lines[:i] + [spelling + b" = " + encoded + newline] + lines[i:])
        return
    for pattern in patterns:
        line_pattern = re.compile(rb"[ \t]*" + pattern + rb"[ \t]*=[ \t]*" + scalar + rb"[ \t]*(?:#.*)?\r?\n?\Z")
        for i, line in enumerate(lines):
            if match := line_pattern.match(line):
                start, end = match.span(1)
                yield b"".join(lines[:i] + [line[:start] + encoded + line[end:]] + lines[i + 1:])


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
