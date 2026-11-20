"""The one writer of skill and agent profile files: compare-and-set create, update, and delete.

A revision is the SHA-256 of the file's exact bytes, the same value ``skill_read`` and
``agent_profile_read`` return. Every write takes one lock file per catalog root (``.library.lock``),
rereads the target under it, and refuses a stale revision. A document is validated by the catalog
loader before it is written, and published atomically (temp file, fsync, rename), so a cancelled
write leaves the old file or the new one. The lock serializes zeta writers only.

Every function blocks (file locks and fsync); call it off the event loop.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from ..agent.presets import AGENT_PRESETS
from ..core.session_files import (
    SessionError,
    descend_directory,
    open_session_file,
    require_off_loop_lock,
    unlink_session_file,
    write_session_file,
    write_session_file_if_absent,
)
from ..project_schema import MAX_MEMORY_FILE_SIZE
from . import agent_catalog, catalog
from .discovery import MarkdownDocument, NamedItem, parse_document

LOCK_NAME = ".library.lock"
LOCK_SECONDS = 0.25
LOCK_POLL_SECONDS = 0.01


class LibraryError(Exception):
    """A refused write; ``code`` is the wire reason, ``detail`` a public loader message."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def write(root: Path, *, profiles: bool, name: str, revision: str | None, document: str) -> str:
    """Create (no ``revision``) or update one definition named ``name`` under ``root``; return its revision."""
    try:
        raw = document.encode("utf-8")
    except UnicodeError:
        raise LibraryError("invalid_document") from None
    if len(raw) > MAX_MEMORY_FILE_SIZE:
        raise LibraryError("document_too_large")
    kind = "agent" if profiles else "skill"
    filename = Path(f"{name}.md")
    try:
        parsed = parse_document(raw, filename, filename, filename.parent, kind)
        if profiles:
            agent_catalog.build_agent(parsed, "", None)
        else:
            catalog.build_skill(parsed, "")
    except (ValueError, UnicodeError, RecursionError, yaml.YAMLError) as exc:
        raise LibraryError("invalid_document", str(exc)) from None
    if parsed.metadata["name"] != name:
        raise LibraryError("name_mismatch")
    if revision is None:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        _check(None, revision, name, profiles=profiles)
    with _target(root, profiles=profiles, name=name) as (directory_fd, target, current):
        _check(current, revision, name, profiles=profiles)
        if current is None:
            if not write_session_file_if_absent(directory_fd, target, raw):
                raise LibraryError("stale_revision")
        else:
            write_session_file(directory_fd, target, raw)
    return hashlib.sha256(raw).hexdigest()


def delete(root: Path, *, profiles: bool, name: str, revision: str) -> None:
    """Delete one definition if its file still has ``revision``. A skill directory keeps its other files."""
    if not root.is_dir():
        _check(None, revision, name, profiles=profiles)
    with _target(root, profiles=profiles, name=name) as (directory_fd, target, current):
        _check(current, revision, name, profiles=profiles)
        unlink_session_file(directory_fd, target)


def _check(current: bytes | None, revision: str | None, name: str, *, profiles: bool) -> None:
    if revision is None:
        if current is not None:
            raise LibraryError("stale_revision")
        return
    if current is None:
        packaged = AGENT_PRESETS if profiles else {skill.name for skill in catalog.discover_packaged_skills().skills}
        raise LibraryError("packaged_read_only" if name in packaged else "resource_not_found")
    if hashlib.sha256(current).hexdigest() != revision:
        raise LibraryError("stale_revision")


@contextmanager
def _target(root: Path, *, profiles: bool, name: str) -> Iterator[tuple[int, str, bytes | None]]:
    """Lock ``root``, find the file that defines ``name``, and pin its directory.

    Yields ``(directory_fd, filename, current bytes or None)``. An existing definition is found with
    the catalog loader, which keeps it inside the root; a new one is ``<name>.md`` in the root.
    """
    require_off_loop_lock("library")
    base = root.resolve(strict=True)
    with ExitStack() as stack:
        base_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, base_fd)
        # The lock lives in the root, so writers from every zeta home that shares it serialize.
        lock_fd = open_session_file(base_fd, LOCK_NAME, os.O_RDWR | os.O_CREAT)
        stack.callback(os.close, lock_fd)
        _lock(lock_fd)
        path = _find(base, profiles=profiles, name=name) or base / f"{name}.md"
        if not path.parent.is_relative_to(base):
            raise LibraryError("unsafe_path")
        # Walk from the root without following links, so a parent swapped for one is refused.
        try:
            directory_fd = stack.enter_context(descend_directory(base_fd, path.parent.relative_to(base).parts))
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise LibraryError("unsafe_path") from None
            raise
        yield directory_fd, path.name, _read(directory_fd, path.name)


def _lock(fd: int) -> None:
    deadline = time.monotonic() + LOCK_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise LibraryError("busy") from None
            time.sleep(LOCK_POLL_SECONDS)


def _find(base: Path, *, profiles: bool, name: str) -> Path | None:
    found: list[Path] = []

    def capture(item: NamedItem, document: MarkdownDocument) -> None:
        if item.name == name:
            found.append(document.path)

    if profiles:
        agent_catalog.discover_agents(base.parent, source="", agents_dir=base, capture=capture)
    else:
        catalog.discover_skills(base.parent, source="", skills_dir=base, capture=capture)
    return found[0] if found else None


def _read(directory_fd: int, name: str) -> bytes | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except SessionError:
        raise LibraryError("unsafe_path") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise LibraryError("unsafe_path") from None
        raise
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


__all__ = ["LibraryError", "delete", "write"]
