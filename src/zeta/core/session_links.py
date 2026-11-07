"""Durable project-linkage intents and their reconciliation.

A session records a small ``project_link_pending.json`` intent before touching
the project registry, so a crash at either boundary is recoverable.  Root
sessions reconstruct their own link from immutable metadata; child/grandchild
lineage is recorded in a durable *index* the ROOT session owns -- a flat
``pending_child_links`` directory holding one small intent file per pending
child link -- so reconciliation lists a single directory and never walks the
``agents`` subtree.  Everything here is idempotent: the registry dedupes by
session id and a published intent is removed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..project_registry import ProjectRegistry, ProjectRegistryError
from .session_files import (
    SessionError,
    child_directory,
    open_session_file,
    session_directory,
    write_session_json,
)

logger = logging.getLogger(__name__)

_PROJECT_ROLES = {"session", "orchestrator", "worker"}

# The ROOT session owns a flat directory of pending child-lineage intents.  A
# child (or grandchild) publishes one small file here -- named by a reversible,
# path-safe encoding of its own global session id -- before appending to the
# registry, and removes it once the append is durable.  Reconciliation lists
# only this directory, so it never traverses the ``agents`` subtree.
PENDING_CHILD_LINKS_DIRNAME = "pending_child_links"

# Bound one reconciliation pass: at most this many index entries are read and
# published per open.  Because every published entry is removed from the index,
# each pass makes forward progress and the remainder is reconciled on the next
# open.  The scan uses os.scandir and stops at this bound instead of
# materializing (and sorting) an unbounded directory listing.
_CHILD_LINK_MAX_ENTRIES_PER_PASS = 256

# Per-entry size cap.  An intent file is a tiny JSON document; anything larger
# is refused rather than allocated while holding the registry lock.
_MAX_PENDING_LINK_SIZE = 8192

# A temp file from write_session_file's atomic publish is transient, but one
# left by a crashed writer never completes.  Reclaim only clearly abandoned
# temp files (older than ~1 hour by mtime) so an in-flight publish is untouched.
_STALE_TEMP_AGE_SECONDS = 3600.0
_INVALID_CHILD_LINKS_DIRNAME = "pending_child_links.invalid"
_TEMP_NAME_RE = re.compile(r"^\..+\.[0-9a-f]{32}\.tmp$")


def valid_pending_link(value: object) -> bool:
    """A pending link is authoritative only when structurally complete.

    A valid-JSON but semantically wrong document (``{}``, a non-dict, wrong
    field types) must never be treated as the source of truth; the caller falls
    back to immutable metadata instead.
    """

    if not isinstance(value, dict):
        return False
    project_id = value.get("project_id")
    role = value.get("role")
    transcript_path = value.get("transcript_path")
    parent = value.get("parent_session_id")
    return (
        isinstance(project_id, str)
        and bool(project_id)
        and isinstance(role, str)
        and role in _PROJECT_ROLES
        and isinstance(transcript_path, str)
        and bool(transcript_path)
        and (parent is None or (isinstance(parent, str) and bool(parent)))
    )


def valid_child_pending_link(value: object) -> bool:
    """A child-lineage pending link additionally carries its own session id.

    Child stores are not top-level sessions, so their identity cannot be
    reconstructed from a session directory name; the durable intent must record
    the ``session_id`` itself for the root's reconciliation to publish it.
    """

    return (
        valid_pending_link(value)
        and isinstance(value, dict)
        and isinstance(value.get("session_id"), str)
        and bool(value["session_id"])
    )


def _encode_link_name(session_id: str) -> str:
    """Encode a global session id as one path-safe filename component.

    Child instance ids contain ``:`` and could in principle contain other
    characters; a urlsafe base64 encoding yields a collision-free name with no
    ``/`` or ``\\x00`` so the index stays a flat, well-formed directory.
    """

    return base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii")


def persist_pending_root_link(
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    project_id: str,
    role: str,
    parent_session_id: str | None,
) -> None:
    """Durably publish a root-session linkage intent before registry mutation."""

    link = {
        "project_id": project_id,
        "role": role,
        "parent_session_id": parent_session_id,
        "transcript_path": str(Path(sessions_dir) / session_id),
    }
    with session_directory(sessions_dir, session_id) as (_, directory_fd):
        write_session_json(directory_fd, "project_link_pending.json", link)
        os.fsync(directory_fd)


def persist_pending_child_link(root_session_dir: os.PathLike[str], link: dict[str, object]) -> None:
    """Durably record a child-lineage intent in the ROOT's pending index.

    Written and directory-fsynced BEFORE the registry append (in place of any
    per-child intent file), so a crash or a failed append is recovered by the
    root's reconciliation on the next open.  Because the index lives in the root
    session directory, nested grandchildren record here too and reconciliation
    never has to walk the ``agents`` subtree.
    """
    root = Path(root_session_dir)
    name = _encode_link_name(str(link["session_id"]))
    with (
        session_directory(root.parent, root.name) as (_, root_fd),
        child_directory(root_fd, PENDING_CHILD_LINKS_DIRNAME, create=True) as pending_fd,
    ):
        # Ensuring the index directory added its entry to the root session dir;
        # fsync root_fd so that directory entry is durable before the registry
        # append.  Otherwise a crash after this returns but before the append
        # could lose the whole index -- write_session_json below fsyncs the
        # intent file and the index dir, but never the parent that names it.
        os.fsync(root_fd)
        write_session_json(pending_fd, name, link)
        # Durably land the new directory entry before the registry append;
        # write_session_json fsyncs the file and its rename, not the dir.
        os.fsync(pending_fd)


_ReconcileOutcome = Literal["drained", "more", "failed"]


class ChildLineagePublisher:
    """Durably admit and single-flight live child-lineage publication."""

    def __init__(
        self,
        project_registry: ProjectRegistry,
        root_session_dir: os.PathLike[str] | str,
    ) -> None:
        self._project_registry = project_registry
        self._root_session_dir = Path(root_session_dir)
        self._mutex = threading.Lock()
        self._generation = 0
        self._owner_active = False

    def publish(self, link: Mapping[str, object]) -> None:
        value = dict(link)
        persist_pending_child_link(self._root_session_dir, value)
        with self._mutex:
            self._generation += 1
            if self._owner_active:
                return
            self._owner_active = True
            observed_generation = self._generation

        def ready_collected() -> None:
            nonlocal observed_generation
            with self._mutex:
                observed_generation = self._generation

        try:
            while True:
                outcome = _reconcile_child_links_pass(
                    self._project_registry,
                    self._root_session_dir.parent,
                    self._root_session_dir.name,
                    ready_collected=ready_collected,
                )
                if outcome == "failed":
                    raise ProjectRegistryError("child project lineage remains pending")
                with self._mutex:
                    current_generation = self._generation
                    if (
                        outcome == "drained"
                        and current_generation == observed_generation
                    ):
                        self._owner_active = False
                        return
                    observed_generation = current_generation
        except BaseException:
            with self._mutex:
                self._owner_active = False
            raise


def reconcile_child_links(
    project_registry: ProjectRegistry, sessions_dir: os.PathLike[str] | str, session_id: str
) -> None:
    """Publish durable child-lineage intents from the ROOT's pending index.

    The root keeps a flat ``pending_child_links`` directory holding one small
    intent file per pending child link.  Reconciliation lists that single
    directory (bounded to ``_CHILD_LINK_MAX_ENTRIES_PER_PASS`` entries per pass,
    via os.scandir without sorting an unbounded list), publishes the missing
    links in a single registry read, and removes the published entries.  Because
    published entries leave the index, every pass makes forward progress and a
    crash, a failed append, or a bounded pass is recovered on the next open
    without ever double-recording a link.
    """
    outcome = _reconcile_child_links_pass(project_registry, sessions_dir, session_id)
    if outcome == "failed":
        logger.warning("child project linkage reconciliation failed for %s", session_id)


def _reconcile_child_links_pass(
    project_registry: ProjectRegistry,
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    *,
    ready_collected: Callable[[], None] | None = None,
) -> _ReconcileOutcome:
    try:
        with session_directory(sessions_dir, session_id) as (_, session_fd):
            try:
                pending_fd = os.open(
                    PENDING_CHILD_LINKS_DIRNAME,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=session_fd,
                )
            except (FileNotFoundError, NotADirectoryError):
                return "drained"
            except OSError:
                return "failed"
            try:
                if ready_collected is not None:
                    ready_collected()
                intents, more, failed = _collect_pending_intents(pending_fd, session_fd)
                if failed:
                    return "failed"
                if intents and not _publish_pending_intents(
                    project_registry, pending_fd, intents
                ):
                    return "failed"
                return "more" if more else "drained"
            finally:
                os.close(pending_fd)
    except (SessionError, OSError):
        return "failed"


def _collect_pending_intents(
    pending_fd: int, session_fd: int
) -> tuple[list[tuple[str, dict[str, object]]], bool, bool]:
    """Scan one bounded pass, quarantining only permanently invalid entries.

    A failed metadata lookup or read is transient: stop immediately and leave the
    entry in place so the next open can retry it.  Only structural and content
    invalidity is quarantined.
    """
    names, temps, stopped, more = _scan_pending_entries(pending_fd)
    if stopped:
        return [], more, True
    for name in temps:
        if not _remove_stale_temp(pending_fd, name):
            return [], more, True
    try:
        quarantine_fd = _open_quarantine(session_fd)
    except OSError as exc:
        logger.warning("cannot open pending-link quarantine: %s", exc)
        return [], more, True
    try:
        intents: list[tuple[str, dict[str, object]]] = []
        for name in names:
            status, link = _classify_pending_entry(pending_fd, name)
            if status == "publish" and link is not None:
                intents.append((name, link))
            elif status == "invalid":
                if not _quarantine_entry(pending_fd, quarantine_fd, name):
                    return [], more, True
                logger.warning("quarantined unpublishable child-link entry %r", name)
            elif status == "stop":
                return [], more, True
        return intents, more, False
    finally:
        os.close(quarantine_fd)


def _scan_pending_entries(
    pending_fd: int,
) -> tuple[list[str], list[str], bool, bool]:
    """Read up to the per-pass budget, recognizing only writer temp names."""
    names: list[str] = []
    temps: list[str] = []
    seen = 0
    stopped = False
    more = False
    # os.scandir iterates lazily; the pass stops at the bound instead of
    # materializing an unbounded listing.
    with os.scandir(pending_fd) as entries:
        for entry in entries:
            if seen >= _CHILD_LINK_MAX_ENTRIES_PER_PASS:
                more = True
                logger.warning(
                    "child project linkage reconciliation reached the %d-entry "
                    "pass limit; the remaining pending links will be reconciled "
                    "in a later pass",
                    _CHILD_LINK_MAX_ENTRIES_PER_PASS,
                )
                break
            seen += 1
            # A temp file from write_session_file's atomic publish still counts
            # toward the budget, but is only reclaimed once clearly abandoned.
            if _TEMP_NAME_RE.fullmatch(entry.name):
                temps.append(entry.name)
            else:
                names.append(entry.name)
    return names, temps, stopped, more


def _remove_stale_temp(pending_fd: int, name: str) -> bool:
    """Reclaim a temp file left by a crashed atomic publish, once stale by mtime."""
    try:
        info = os.stat(name, dir_fd=pending_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if time.time() - info.st_mtime < _STALE_TEMP_AGE_SECONDS:
        return True
    return _remove_entry(pending_fd, name)


def _read_pending_bytes(fd: int) -> bytes | None:
    """Read an intent to EOF without trusting a single ``os.read`` call.

    Regular-file reads are allowed to be short (for example when a file is
    being observed through a wrapper or under unusual filesystem conditions).
    A short first read must not turn a complete JSON intent into a malformed
    one.  Keep the buffer bounded while also detecting a file that grows past
    the per-entry cap after the initial stat.
    """
    data = bytearray()
    while len(data) <= _MAX_PENDING_LINK_SIZE:
        chunk = os.read(fd, min(4096, _MAX_PENDING_LINK_SIZE + 1 - len(data)))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > _MAX_PENDING_LINK_SIZE:
            return None
    return None


def _classify_pending_entry(
    pending_fd: int, name: str
) -> tuple[str, dict[str, object] | None]:
    """Classify one entry without deleting anything on transient I/O failure."""
    try:
        info = os.stat(name, dir_fd=pending_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "gone", None
    except OSError:
        return "stop", None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
        return "invalid", None
    if info.st_size > _MAX_PENDING_LINK_SIZE:
        return "invalid", None
    try:
        fd = open_session_file(pending_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return "gone", None
    except SessionError:
        return "invalid", None
    except OSError:
        return "stop", None
    try:
        raw = _read_pending_bytes(fd)
    except OSError:
        return "stop", None
    finally:
        os.close(fd)
    if raw is None:
        return "invalid", None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "invalid", None
    if not valid_child_pending_link(value):
        return "invalid", None
    return "publish", value


def _record_child_lineage_batch(
    project_registry: ProjectRegistry,
    project_id: str,
    records: list[Mapping[str, object]],
) -> None:
    """Acquire today's registry lock through one seam for p06 conversion."""
    project_registry.record_sessions(project_id, records)


def _publish_pending_intents(
    project_registry: ProjectRegistry,
    pending_fd: int,
    intents: list[tuple[str, dict[str, object]]],
) -> bool:
    # Group by project so the registry's JSONL file is read exactly once per
    # project, then published in a single deduped batch.
    by_project: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for name, link in intents:
        by_project.setdefault(str(link["project_id"]), []).append((name, link))
    for project_id, group in by_project.items():
        records = [
            {
                "session_id": link["session_id"],
                "transcript_path": link["transcript_path"],
                "role": link["role"],
                "parent_session_id": link.get("parent_session_id"),
            }
            for _name, link in group
        ]
        try:
            _record_child_lineage_batch(project_registry, project_id, records)
        except (ProjectRegistryError, OSError, ValueError, KeyError, TypeError) as exc:
            # Stop the pass; a failed append leaves the intents in place and the
            # next open retries them (the registry dedupes on session id).
            logger.warning("child project linkage remains pending: %s", exc)
            return False
        # The batch landed (or every id was already present). A failed cleanup
        # defers the exact intent to durable recovery rather than declaring drained.
        for name, _link in group:
            if not _remove_entry(pending_fd, name):
                return False
    return True


def _open_quarantine(session_fd: int) -> int:
    try:
        os.mkdir(_INVALID_CHILD_LINKS_DIRNAME, 0o700, dir_fd=session_fd)
    except FileExistsError:
        pass
    return os.open(
        _INVALID_CHILD_LINKS_DIRNAME,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=session_fd,
    )


def _quarantine_destination(name: str) -> str:
    """Return a collision-resistant quarantine name bounded below NAME_MAX."""
    prefix: list[str] = []
    prefix_size = 0
    for character in name:
        character_size = len(os.fsencode(character))
        if prefix_size + character_size > 200:
            break
        prefix.append(character)
        prefix_size += character_size
    return f"{''.join(prefix)}.{uuid4().hex}"


def _quarantine_entry(pending_fd: int, quarantine_fd: int, name: str) -> bool:
    try:
        destination = _quarantine_destination(name)
        os.rename(name, destination, src_dir_fd=pending_fd, dst_dir_fd=quarantine_fd)
        os.fsync(pending_fd)
        os.fsync(quarantine_fd)
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning("could not quarantine pending-link entry %r: %s", name, exc)
        return False


def _remove_entry(pending_fd: int, name: str) -> bool:
    try:
        os.unlink(name, dir_fd=pending_fd)
        os.fsync(pending_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


__all__ = [
    "PENDING_CHILD_LINKS_DIRNAME",
    "_PROJECT_ROLES",
    "persist_pending_child_link",
    "reconcile_child_links",
    "valid_child_pending_link",
    "valid_pending_link",
]
