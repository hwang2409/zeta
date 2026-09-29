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
from pathlib import Path

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
        write_session_json(pending_fd, name, link)
        # Durably land the new directory entry before the registry append;
        # write_session_json fsyncs the file and its rename, not the dir.
        os.fsync(pending_fd)


def remove_pending_child_link(root_session_dir: os.PathLike[str], session_id: str) -> None:
    """Drop a published intent from the ROOT's index, directory-fsynced.

    Called only after the registry append is durable.  A missing entry (already
    reconciled, or never written) is harmless and swallowed.
    """
    root = Path(root_session_dir)
    name = _encode_link_name(session_id)
    try:
        with (
            session_directory(root.parent, root.name) as (_, root_fd),
            child_directory(root_fd, PENDING_CHILD_LINKS_DIRNAME) as pending_fd,
        ):
            os.unlink(name, dir_fd=pending_fd)
            os.fsync(pending_fd)
    except (FileNotFoundError, NotADirectoryError, SessionError, OSError):
        pass


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
    try:
        with session_directory(sessions_dir, session_id) as (_, session_fd):
            try:
                pending_fd = os.open(
                    PENDING_CHILD_LINKS_DIRNAME,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=session_fd,
                )
            except (FileNotFoundError, NotADirectoryError, OSError):
                return
            try:
                names = _list_pending_entries(pending_fd)
                intents: list[tuple[str, dict[str, object]]] = []
                for name in names:
                    link = _read_pending_entry(pending_fd, name)
                    if link is not None:
                        intents.append((name, link))
                if intents:
                    _publish_pending_intents(project_registry, pending_fd, intents)
            finally:
                os.close(pending_fd)
    except (SessionError, OSError) as exc:
        logger.warning(
            "child project linkage reconciliation failed for %s: %s",
            session_id,
            exc,
        )


def _list_pending_entries(pending_fd: int) -> list[str]:
    """List up to a bounded number of index entries without sorting the whole dir."""
    names: list[str] = []
    # os.scandir iterates lazily; the pass stops at the bound instead of
    # materializing an unbounded listing.
    with os.scandir(pending_fd) as entries:
        for entry in entries:
            name = entry.name
            # Skip in-flight temp files from write_session_file's atomic publish.
            if name.startswith("."):
                continue
            if len(names) >= _CHILD_LINK_MAX_ENTRIES_PER_PASS:
                logger.warning(
                    "child project linkage reconciliation reached the %d-entry "
                    "pass limit; the remaining pending links will be reconciled "
                    "on a later open",
                    _CHILD_LINK_MAX_ENTRIES_PER_PASS,
                )
                break
            names.append(name)
    return names


def _read_pending_entry(pending_fd: int, name: str) -> dict[str, object] | None:
    try:
        fd = open_session_file(pending_fd, name, os.O_RDONLY)
    except (FileNotFoundError, SessionError, OSError):
        return None
    try:
        if os.fstat(fd).st_size > _MAX_PENDING_LINK_SIZE:
            return None
        raw = os.read(fd, _MAX_PENDING_LINK_SIZE + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(raw) > _MAX_PENDING_LINK_SIZE:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not valid_child_pending_link(value):
        return None
    return value


def _publish_pending_intents(
    project_registry: ProjectRegistry,
    pending_fd: int,
    intents: list[tuple[str, dict[str, object]]],
) -> None:
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
            project_registry.record_sessions(project_id, records)
        except (ProjectRegistryError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("child project linkage remains pending: %s", exc)
            continue
        # The batch landed (or every id was already present); the index entries
        # are now redundant.  A failed unlink is harmless -- the registry dedupes
        # on the next pass -- so it is swallowed.
        for name, _link in group:
            _remove_entry(pending_fd, name)


def _remove_entry(pending_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=pending_fd)
        os.fsync(pending_fd)
    except (FileNotFoundError, OSError):
        pass


__all__ = [
    "PENDING_CHILD_LINKS_DIRNAME",
    "_PROJECT_ROLES",
    "persist_pending_child_link",
    "reconcile_child_links",
    "remove_pending_child_link",
    "valid_child_pending_link",
    "valid_pending_link",
]
