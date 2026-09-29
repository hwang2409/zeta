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

PENDING_LINK_FILENAME = "project_link_pending.json"


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
    walks its bounded ``agents`` subtree and publishes each intent idempotently,
    so a crash or a failed registry append is recovered on the next open without
    ever double-recording a link.
    """
    try:
        with session_directory(sessions_dir, session_id) as (_, session_fd):
            _walk_agent_links(project_registry, session_fd, depth=0)
    except (SessionError, OSError) as exc:
        logger.warning(
            "child project linkage reconciliation failed for %s: %s",
            session_id,
            exc,
        )


def _walk_agent_links(
    project_registry: ProjectRegistry, parent_fd: int, *, depth: int
) -> None:
    if depth > _CHILD_LINK_MAX_DEPTH:
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
        for name in sorted(os.listdir(agents_fd)):
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=agents_fd,
                )
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue
            try:
                _publish_pending_child_link(project_registry, child_fd)
                _walk_agent_links(project_registry, child_fd, depth=depth + 1)
            finally:
                os.close(child_fd)
    finally:
        os.close(agents_fd)


def _publish_pending_child_link(
    project_registry: ProjectRegistry, directory_fd: int
) -> None:
    try:
        raw = read_session_file(directory_fd, PENDING_LINK_FILENAME)
    except (FileNotFoundError, SessionError, OSError):
        return
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not valid_child_pending_link(value):
        return
    try:
        project_registry.record_session(
            value["project_id"],
            session_id=value["session_id"],
            transcript_path=value["transcript_path"],
            role=value["role"],
            parent_session_id=value.get("parent_session_id"),
        )
    except (ProjectRegistryError, OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("child project linkage remains pending: %s", exc)
        return
    try:
        os.unlink(PENDING_LINK_FILENAME, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except (FileNotFoundError, OSError):
        pass


__all__ = [
    "PENDING_LINK_FILENAME",
    "_PROJECT_ROLES",
    "reconcile_child_links",
    "valid_child_pending_link",
    "valid_pending_link",
]
