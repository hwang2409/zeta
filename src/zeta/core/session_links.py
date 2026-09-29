"""Durable project-linkage intents and their reconciliation.

A session records a small ``project_link_pending.json`` intent before touching
the project registry, so a crash at either boundary is recoverable.  Root
sessions reconstruct their own link from immutable metadata; child/grandchild
lineage lives inside the root's ``agents`` subtree and is republished from the
durable intents each carries.  Everything here is idempotent: the registry
dedupes by session id and a published intent is removed.
"""

from __future__ import annotations

import json
import logging
import os

from ..project_registry import ProjectRegistry, ProjectRegistryError
from .session_files import (
    SessionError,
    read_session_file,
    session_directory,
)

logger = logging.getLogger(__name__)

_PROJECT_ROLES = {"session", "orchestrator", "worker"}

# Bound the child-lineage walk: agent nesting is shallow, so a small depth
# limit covers every real tree while refusing to follow a pathological one.
_CHILD_LINK_MAX_DEPTH = 8

# Bound the total work of one reconciliation pass.  Every directory entry the
# traversal observes (via os.scandir) and every child directory it opens spends
# one unit of this shared budget, so a single writable-root open can never turn
# into an unbounded scan of an attacker-influenced ``agents`` subtree.  When the
# budget is exhausted (or the depth limit truncates a deeper subtree) the pass
# stops deterministically and logs that it was incomplete; the durable intents
# it did not reach stay on disk and are picked up on a later open.
_CHILD_LINK_VISIT_BUDGET = 4096

PENDING_LINK_FILENAME = "project_link_pending.json"


class _VisitBudget:
    """A shared allowance for one bounded reconciliation pass.

    Each observed directory entry and each opened child directory spends one
    unit.  ``truncated`` latches once the pass stops early -- whether from an
    exhausted budget or the depth limit -- so the caller can log an incomplete
    pass exactly once.
    """

    __slots__ = ("_remaining", "truncated")

    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self.truncated = False

    def spend(self) -> bool:
        if self._remaining <= 0:
            self.truncated = True
            return False
        self._remaining -= 1
        return True



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


def reconcile_child_links(
    project_registry: ProjectRegistry, sessions_dir: os.PathLike[str] | str, session_id: str
) -> None:
    """Publish durable child-lineage intents found in the agent subtree.

    Child stores persist a pending intent carrying their own identity; the root
    walks its bounded ``agents`` subtree, collects every durable intent within a
    shared visit budget, and publishes them in a single batch per project so the
    registry is read only once.  Everything is idempotent: the registry dedupes
    by session id and every published intent is removed, so a crash, a failed
    append, or a budget-truncated pass is recovered on the next open without ever
    double-recording a link.
    """
    budget = _VisitBudget(_CHILD_LINK_VISIT_BUDGET)
    try:
        with session_directory(sessions_dir, session_id) as (_, session_fd):
            intents: list[tuple[tuple[str, ...], dict[str, object]]] = []
            _walk_agent_links(session_fd, (), depth=0, budget=budget, intents=intents)
            if intents:
                _publish_child_intents(project_registry, session_fd, intents)
    except (SessionError, OSError) as exc:
        logger.warning(
            "child project linkage reconciliation failed for %s: %s",
            session_id,
            exc,
        )
        return
    if budget.truncated:
        logger.warning(
            "child project linkage reconciliation incomplete for %s: "
            "visit budget or depth limit reached; remaining intents will be "
            "reconciled on a later open",
            session_id,
        )


def _walk_agent_links(
    parent_fd: int,
    parts: tuple[str, ...],
    *,
    depth: int,
    budget: _VisitBudget,
    intents: list[tuple[tuple[str, ...], dict[str, object]]],
) -> None:
    if depth > _CHILD_LINK_MAX_DEPTH:
        # A deeper subtree beyond the depth bound is refused; note it as a
        # truncated pass only when such a subtree actually exists.
        if _agents_dir_has_entries(parent_fd):
            budget.truncated = True
        return
    try:
        agents_fd = os.open(
            "agents",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return
    try:
        # os.scandir iterates lazily; entries are consumed one at a time and the
        # walk stops at the budget instead of materializing (and sorting) an
        # unbounded directory listing.
        with os.scandir(agents_fd) as entries:
            for entry in entries:
                if not budget.spend():
                    return
                name = entry.name
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=agents_fd,
                    )
                except (FileNotFoundError, NotADirectoryError, OSError):
                    continue
                if not budget.spend():
                    os.close(child_fd)
                    return
                try:
                    child_parts = parts + ("agents", name)
                    link = _read_pending_child_link(child_fd)
                    if link is not None:
                        intents.append((child_parts, link))
                    _walk_agent_links(
                        child_fd,
                        child_parts,
                        depth=depth + 1,
                        budget=budget,
                        intents=intents,
                    )
                finally:
                    os.close(child_fd)
    finally:
        os.close(agents_fd)


def _agents_dir_has_entries(parent_fd: int) -> bool:
    try:
        agents_fd = os.open(
            "agents",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return False
    try:
        with os.scandir(agents_fd) as entries:
            return any(True for _ in entries)
    finally:
        os.close(agents_fd)


def _read_pending_child_link(directory_fd: int) -> dict[str, object] | None:
    try:
        raw = read_session_file(directory_fd, PENDING_LINK_FILENAME)
    except (FileNotFoundError, SessionError, OSError):
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not valid_child_pending_link(value):
        return None
    return value


def _publish_child_intents(
    project_registry: ProjectRegistry,
    session_fd: int,
    intents: list[tuple[tuple[str, ...], dict[str, object]]],
) -> None:
    # Group by project so the registry's JSONL file is read exactly once per
    # project, then published in a single deduped batch.
    by_project: dict[str, list[tuple[tuple[str, ...], dict[str, object]]]] = {}
    for parts, link in intents:
        by_project.setdefault(str(link["project_id"]), []).append((parts, link))
    for project_id, group in by_project.items():
        records = [
            {
                "session_id": link["session_id"],
                "transcript_path": link["transcript_path"],
                "role": link["role"],
                "parent_session_id": link.get("parent_session_id"),
            }
            for _parts, link in group
        ]
        try:
            project_registry.record_sessions(project_id, records)
        except (ProjectRegistryError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("child project linkage remains pending: %s", exc)
            continue
        # The batch landed (or every id was already present); the intents are
        # now redundant.  A failed unlink is harmless -- the registry dedupes on
        # the next pass -- so it is swallowed.
        for parts, _link in group:
            _remove_pending_intent(session_fd, parts)


def _remove_pending_intent(session_fd: int, parts: tuple[str, ...]) -> None:
    directory_fd = _open_relative(session_fd, parts)
    if directory_fd is None:
        return
    try:
        os.unlink(PENDING_LINK_FILENAME, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except (FileNotFoundError, OSError):
        pass
    finally:
        os.close(directory_fd)


def _open_relative(base_fd: int, parts: tuple[str, ...]) -> int | None:
    """Reopen a descendant directory by path parts, refusing symlinks."""
    opened: list[int] = []
    current = base_fd
    try:
        for name in parts:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            opened.append(fd)
            current = fd
    except OSError:
        for fd in opened:
            os.close(fd)
        return None
    if not opened:
        return None
    for fd in opened[:-1]:
        os.close(fd)
    return opened[-1]


__all__ = [
    "PENDING_LINK_FILENAME",
    "_PROJECT_ROLES",
    "reconcile_child_links",
    "valid_child_pending_link",
    "valid_pending_link",
]
