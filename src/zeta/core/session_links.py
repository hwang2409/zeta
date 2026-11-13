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

import asyncio
import base64
import fcntl
import json
import logging
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..project_registry import ProjectRegistry, ProjectRegistryError
from ..storage_format import WriterLease, mapping_mutations
from .session_files import (
    SessionError,
    SessionReferenceAuthority,
    child_directory,
    open_session_file,
    reference_publication,
    session_directory,
    write_session_file,
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
    previous_project_id = value.get("previous_project_id")
    return (
        isinstance(project_id, str)
        and bool(project_id)
        and isinstance(role, str)
        and role in _PROJECT_ROLES
        and isinstance(transcript_path, str)
        and bool(transcript_path)
        and (parent is None or (isinstance(parent, str) and bool(parent)))
        and (
            previous_project_id is None
            or (isinstance(previous_project_id, str) and bool(previous_project_id))
        )
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


@contextmanager
def project_links_lock(directory_fd: int) -> Iterator[None]:
    """Serialize complete project-link intent mutation inside session shape."""

    lock_fd = open_session_file(
        directory_fd, ".project-links.lock", os.O_RDWR | os.O_CREAT
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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
    lease: WriterLease,
    *,
    previous_project_id: str | None = None,
) -> None:
    with reference_publication(
        Path(sessions_dir).parent,
        lease,
        installed_ids=(session_id,),
        referenced_ids=(parent_session_id,) if parent_session_id else (),
    ) as authority:
        _persist_pending_root_link(
            sessions_dir, session_id, project_id, role, parent_session_id, lease,
            previous_project_id=previous_project_id, authority=authority,
        )


def _persist_pending_root_link(
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    project_id: str,
    role: str,
    parent_session_id: str | None,
    lease: WriterLease,
    *,
    previous_project_id: str | None = None,
    authority: SessionReferenceAuthority,
) -> None:
    """Durably publish a root-session linkage intent before registry mutation."""

    link = {
        "project_id": project_id,
        "role": role,
        "parent_session_id": parent_session_id,
        "previous_project_id": previous_project_id,
        "transcript_path": str(Path(sessions_dir) / session_id),
    }
    referenced_ids = (parent_session_id,) if parent_session_id is not None else ()
    with (
        session_directory(sessions_dir, session_id) as (_, directory_fd),
        project_links_lock(directory_fd),
    ):
        assert authority is not None
        authority.validate((session_id, *referenced_ids))
        lease.publish_json(
            directory_fd,
            "project_link_pending.json",
            mapping_mutations(link),
            write_session_file,
        )
        os.fsync(directory_fd)


def persist_pending_child_link(
    root_session_dir: os.PathLike[str] | str,
    link: dict[str, object],
    lease: WriterLease,
) -> None:
    root = Path(root_session_dir)
    with reference_publication(
        root.parent.parent, lease,
        installed_ids=(str(link["session_id"]),),
        referenced_ids=tuple(value for value in (link.get("parent_session_id"), root.name) if isinstance(value, str)),
    ) as authority:
        _persist_pending_child_link(root_session_dir, link, lease, authority=authority)


def _persist_pending_child_link(
    root_session_dir: os.PathLike[str],
    link: dict[str, object],
    lease: WriterLease,
    *,
    authority: SessionReferenceAuthority,
) -> None:
    """Durably record a child-lineage intent in the ROOT's pending index.

    Written and directory-fsynced BEFORE the registry append (in place of any
    per-child intent file), so a crash or a failed append is recovered by the
    root's reconciliation on the next open.  Because the index lives in the root
    session directory, nested grandchildren record here too and reconciliation
    never has to walk the ``agents`` subtree.
    """
    root = Path(root_session_dir)
    name = _encode_link_name(str(link["session_id"]))
    installed_ids = (str(link["session_id"]),)
    referenced_ids = tuple(
        session_id
        for session_id in (link.get("parent_session_id"), root.name)
        if isinstance(session_id, str)
    )
    with (
        session_directory(root.parent, root.name) as (_, root_fd),
        project_links_lock(root_fd),
        child_directory(root_fd, PENDING_CHILD_LINKS_DIRNAME, create=True) as pending_fd,
    ):
        assert authority is not None
        authority.validate((*installed_ids, *referenced_ids))
        # Ensuring the index directory added its entry to the root session dir;
        # fsync root_fd so that directory entry is durable before the registry
        # append. Otherwise a crash after this returns but before the append
        # could lose the whole index.
        os.fsync(root_fd)
        lease.publish_json(
            pending_fd, name, mapping_mutations(link), write_session_file
        )
        os.fsync(pending_fd)


_ReconcileOutcome = Literal["drained", "more", "failed"]
_CHILD_RECONCILIATION_RETRY_DELAYS = (0.1, 1.0, 5.0, 30.0)
_child_reconciliation_tasks: dict[tuple[str, str], object] = {}
_child_reconciliation_tasks_lock = threading.Lock()


class ChildLineagePublisher:
    """Durably admit and single-flight live child-lineage publication."""

    def __init__(
        self,
        project_registry: ProjectRegistry,
        root_session_dir: os.PathLike[str] | str,
        storage_lease: WriterLease,
    ) -> None:
        self._project_registry = project_registry
        self._root_session_dir = Path(root_session_dir)
        self._storage_lease = storage_lease
        self._mutex = threading.Lock()
        self._generation = 0
        self._owner_active = False

    def publish(self, link: Mapping[str, object]) -> None:
        value = dict(link)
        installed_ids = (str(value["session_id"]),)
        referenced_ids = tuple(
            session_id
            for session_id in (value.get("parent_session_id"), self._root_session_dir.name)
            if isinstance(session_id, str)
        )
        with reference_publication(
            self._root_session_dir.parent.parent,
            self._storage_lease,
            installed_ids=installed_ids,
            referenced_ids=referenced_ids,
        ) as authority:
            self._publish_admitted(value, authority)

    def _publish_admitted(
        self, value: dict[str, object], authority: SessionReferenceAuthority
    ) -> None:
        _persist_pending_child_link(
            self._root_session_dir,
            value,
            self._storage_lease,
            authority=authority,
        )
        with self._mutex:
            self._generation += 1
            if self._owner_active:
                return
            self._owner_active = True
            observed_generation = self._generation

        def ready_collection_started() -> None:
            nonlocal observed_generation
            with self._mutex:
                observed_generation = self._generation

        try:
            while True:
                outcome = _reconcile_child_links_pass(
                    self._project_registry,
                    self._root_session_dir.parent,
                    self._root_session_dir.name,
                    ready_collection_started=ready_collection_started,
                    authority=authority,
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
    project_registry: ProjectRegistry,
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    *,
    _reference_authority: SessionReferenceAuthority | None = None,
) -> _ReconcileOutcome:
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
    if _reference_authority is None:
        with reference_publication(
            Path(sessions_dir).parent,
            project_registry._storage_lease,
            referenced_ids=(session_id,),
        ) as authority:
            outcome = _reconcile_child_links_pass(
                project_registry, sessions_dir, session_id,
                ready_collection_started=None, authority=authority,
            )
    else:
        outcome = _reconcile_child_links_pass(
            project_registry, sessions_dir, session_id,
            ready_collection_started=None, authority=_reference_authority,
        )
    if outcome == "failed":
        logger.warning("child project linkage reconciliation failed for %s", session_id)
    if outcome != "drained":
        _schedule_child_links_reconciliation(project_registry, sessions_dir, session_id)
    return outcome


def _schedule_child_links_reconciliation(
    project_registry: ProjectRegistry,
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
) -> None:
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    key = (str(Path(sessions_dir).absolute()), session_id)
    with _child_reconciliation_tasks_lock:
        if key in _child_reconciliation_tasks:
            return

    async def drain() -> None:
        delay_index = 0
        while True:
            await asyncio.sleep(_CHILD_RECONCILIATION_RETRY_DELAYS[delay_index])
            outcome = await asyncio.to_thread(
                reconcile_child_links,
                project_registry,
                sessions_dir,
                session_id,
            )
            if outcome == "drained":
                return
            delay_index = min(
                delay_index + 1, len(_CHILD_RECONCILIATION_RETRY_DELAYS) - 1
            )

    if loop is not None and loop.is_running():
        task = loop.create_task(drain())
        with _child_reconciliation_tasks_lock:
            _child_reconciliation_tasks[key] = task

        def task_done(_: asyncio.Task[None]) -> None:
            with _child_reconciliation_tasks_lock:
                _child_reconciliation_tasks.pop(key, None)

        task.add_done_callback(task_done)
        return

    def drain_off_loop() -> None:
        delay_index = 0
        try:
            while True:
                time.sleep(_CHILD_RECONCILIATION_RETRY_DELAYS[delay_index])
                outcome = reconcile_child_links(project_registry, sessions_dir, session_id)
                if outcome == "drained":
                    return
                delay_index = min(
                    delay_index + 1, len(_CHILD_RECONCILIATION_RETRY_DELAYS) - 1
                )
        except Exception as exc:  # noqa: BLE001 - keep the worker from leaking its slot
            logger.warning(
                "child project linkage retry stopped for %s: %s: %s",
                session_id,
                type(exc).__name__,
                exc,
            )
        finally:
            with _child_reconciliation_tasks_lock:
                _child_reconciliation_tasks.pop(key, None)

    with _child_reconciliation_tasks_lock:
        _child_reconciliation_tasks[key] = threading.Thread(
            target=drain_off_loop, daemon=True
        )
        thread = _child_reconciliation_tasks[key]
    assert isinstance(thread, threading.Thread)
    thread.start()


def _reconcile_child_links_pass(
    project_registry: ProjectRegistry,
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    *,
    ready_collection_started: Callable[[], None] | None = None,
    authority: SessionReferenceAuthority,
) -> _ReconcileOutcome:
    return _reconcile_child_links_pass_admitted(
        project_registry,
        sessions_dir,
        session_id,
        authority=authority,
        ready_collection_started=ready_collection_started,
    )

def _reconcile_child_links_pass_admitted(
    project_registry: ProjectRegistry,
    sessions_dir: os.PathLike[str] | str,
    session_id: str,
    *,
    authority: SessionReferenceAuthority,
    ready_collection_started: Callable[[], None] | None = None,
) -> _ReconcileOutcome:
    try:
        with (
            session_directory(sessions_dir, session_id) as (_, session_fd),
            project_links_lock(session_fd),
        ):
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
                if ready_collection_started is not None:
                    ready_collection_started()
                intents, more, failed = _collect_pending_intents(pending_fd, session_fd)
                if failed:
                    return "failed"
            finally:
                os.close(pending_fd)
        if not intents:
            return "more" if more else "drained"

        authority.validate(
            tuple(
                reference
                for _name, link in intents
                for reference in (
                    str(link["session_id"]),
                    link.get("parent_session_id"),
                    session_id,
                )
                if isinstance(reference, str)
            )
        )
        project_ids = {
            str(project_id)
            for _name, link in intents
            for project_id in (link["project_id"], link.get("previous_project_id"))
            if project_id is not None
        }
        domains = tuple((project_id, "sessions") for project_id in sorted(project_ids))
        with project_registry._project_locks(
            domains, exclusive=True
        ) as project_fds, session_directory(
            sessions_dir, session_id
        ) as (_, session_fd), project_links_lock(session_fd):
            try:
                pending_fd = os.open(
                    PENDING_CHILD_LINKS_DIRNAME,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=session_fd,
                )
            except (FileNotFoundError, NotADirectoryError):
                return "drained"
            try:
                current: list[tuple[str, dict[str, object]]] = []
                for name, link in intents:
                    status, value = _classify_pending_entry(pending_fd, name)
                    if status == "publish" and value == link:
                        current.append((name, link))
                if not current:
                    return "more"

                by_project: dict[str, dict[str, object]] = {
                    project_id: {"remove": set(), "records": []}
                    for project_id in project_ids
                }
                for _name, link in current:
                    project_id = str(link["project_id"])
                    previous = link.get("previous_project_id")
                    if previous is not None and previous != project_id:
                        removals = by_project[str(previous)]["remove"]
                        assert isinstance(removals, set)
                        removals.add(str(link["session_id"]))
                    records = by_project[project_id]["records"]
                    assert isinstance(records, list)
                    records.append(link)

                fd_by_project = {
                    project_id: project_fd
                    for (project_id, _domain), project_fd in zip(
                        domains, project_fds, strict=True
                    )
                }
                ordered_projects = sorted(
                    project_ids,
                    key=lambda value: (
                        bool(by_project[value]["records"]),
                        value,
                    ),
                )
                for project_id in ordered_projects:
                    update = by_project[project_id]
                    records = update["records"]
                    removals = update["remove"]
                    assert isinstance(records, list) and isinstance(removals, set)
                    prepared = project_registry._prepare_session_records(
                        project_id, records
                    )
                    project_registry._record_child_lineage_batch(
                        fd_by_project[project_id], removals, prepared
                    )
                for name, _link in current:
                    if not _remove_entry(pending_fd, name):
                        return "failed"
            finally:
                os.close(pending_fd)
        return "more" if more or len(current) != len(intents) else "drained"
    except (
        KeyError,
        ProjectRegistryError,
        SessionError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "child project linkage reconciliation failed for %s: %s: %s",
            session_id,
            type(exc).__name__,
            exc,
        )
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


def associate_project(manager, metadata, project_id: str):
    """Associate metadata and its registry row under one reference authority."""
    previous_project_id = metadata.project_id
    registry = manager.project_registry
    if registry is None:
        raise ProjectRegistryError("project registry capability is unavailable")
    registry.show_project(project_id)
    referenced_ids = tuple(
        value
        for value in (metadata.session_id, metadata.parent_session_id)
        if value is not None
    )
    authority_holder: list[SessionReferenceAuthority] = []

    @contextmanager
    def admission():
        with reference_publication(
            manager.home,
            manager._storage_lease,
            referenced_ids=referenced_ids,
        ) as authority:
            authority_holder.append(authority)
            yield authority

    def publish(current):
        publish_project_association(
            manager,
            current,
            project_id=project_id,
            previous_project_id=previous_project_id,
            authority=authority_holder[-1],
        )

    bind = lambda item: setattr(item, "project_id", project_id) or item
    current = manager._mutate(
        metadata.session_id,
        bind,
        execution_lease=metadata._execution_lease,
        authority=admission(),
        before_release=publish,
    )
    manager._copy_metadata(metadata, current)
    return current


def publish_project_association(
    manager,
    current,
    *,
    project_id: str,
    previous_project_id: str | None,
    authority: SessionReferenceAuthority,
) -> None:
    """Finish the metadata-to-registry handoff under reference authority."""
    try:
        _persist_pending_root_link(
            manager.sessions_dir,
            current.session_id,
            project_id,
            current.project_role or "session",
            current.parent_session_id,
            manager._storage_lease,
            previous_project_id=previous_project_id,
            authority=authority,
        )
    except (SessionError, OSError) as exc:
        logger.warning("could not persist project linkage intent: %s", exc)
    manager._reconcile_project_link(current.session_id, _reference_authority=authority)
    if previous_project_id is not None and previous_project_id != project_id:
        from ..transcript_search.index import delete_indexed_session

        delete_indexed_session(
            manager.project_registry.root, previous_project_id, current.session_id
        )


__all__ = [
    "PENDING_CHILD_LINKS_DIRNAME",
    "_PROJECT_ROLES",
    "persist_pending_child_link",
    "reconcile_child_links",
    "valid_child_pending_link",
    "valid_pending_link",
]
