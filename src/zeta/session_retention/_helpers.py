"""Internal retention scan helpers."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..attention_forks import read_attention_fork
from ..attention_records import AttentionRecord
from ..core.agent_state import _parse_agent_state
from ..core.child_lifecycle import ChildRunPhase, parse_legacy_child_lifecycle
from ..core.session import SessionMetadata
from ..core.session_files import (
    SessionError,
    SessionInUseError,
    SessionNotFoundError,
    child_directory,
    descend_directory,
    open_session_file,
    read_reference_generation,
)
from ..core.session_links import valid_child_pending_link, valid_pending_link
from ..core.todo import parse_todo_items
from ..core.transcript_recovery import RepairTarget, classify_recovery_records
from ..memory.reconciliation_state import classify_reconciliation_records
from ..project_inbox import InboxError, validate_inbox_record
from ..project_inbox_sent import SentMessageTracker, _Broken, _Loaded
from ..storage_format import StorageFormatError, validate_retention_reservation
from . import (
    _AGENTS_DIRNAME,
    _ATTENTION_FORK_NAME,
    _LIFECYCLE_NAME,
    _MAX_TREE_DEPTH,
    _METADATA_NAME,
    _PENDING_CHILD_LINKS_DIRNAME,
    _PENDING_LINK_NAME,
    _SESSION_STATE_NAME,
    PreviewSnapshot,
    _Budget,
    _ClosureBudget,
    _InFlight,
    _read_json,
    _Sweep,
    _Walk,
)
from . import (
    RetentionPlannerError as PlannerError,
)

_descend = descend_directory
_TIMESTAMP_FIELDS = ("created_at", "updated_at", "started_at", "finished_at")


class _EntryScan:
    """The only directory enumerator: resumable, no-follow, and charged per entry."""

    def __init__(self, directory_fd: int, charge: Callable[[int], None]):
        try:
            self._fd = os.open(
                ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            self._entries = os.scandir(self._fd)
        except OSError as exc:
            raise PlannerError("unsafe retention directory") from exc
        self.charge = charge
        self._pending = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._pending is None:
            try:
                entry = next(self._entries)
                info = entry.stat(follow_symlinks=False)
            except StopIteration:
                raise
            except OSError as exc:
                raise PlannerError("unsafe retention entry") from exc
            self._pending = (entry, info.st_size if stat.S_ISREG(info.st_mode) else 0)
        entry, size = self._pending
        self.charge(size)
        self._pending = None
        return entry

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._entries, "close", None)
        if close is not None:
            close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _next_root(sweep: _Sweep) -> str | None:
    assert sweep.root_scan is not None
    for entry in sweep.root_scan:
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise PlannerError("unsafe session root entry")
        return entry.name
    sweep.discovery_done = True
    return None


def _flock_nb(fd) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SessionInUseError("session is open or in use") from exc


def _lease_file(stack, directory_fd, name) -> None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PlannerError(f"unsafe store lock: {name}") from exc
    stack.callback(os.close, fd)
    _flock_nb(fd)


def _walk_directory(flight: _InFlight, walk: _Walk, budget: _Budget):
    comps = walk.frontier.pop()
    if len(comps) > _MAX_TREE_DEPTH:
        raise PlannerError("tree is too deep")
    if comps[-2:-1] == (_AGENTS_DIRNAME,) and comps not in flight.store_dirs:
        raise PlannerError("child store appeared during walk")
    with _descend(flight.root_fd, comps) as dfd:
        info = os.fstat(dfd)
        entries = _EntryScan(dfd, budget.charge_entry)
    walk.fingerprints.append(["/".join(comps), info.st_dev, info.st_ino, info.st_mtime_ns])
    walk.max_activity = max(walk.max_activity, info.st_mtime_ns)
    return comps, entries


def _visit(walk: _Walk, dfd, comps, entry) -> None:
    name = entry.name
    info = entry.stat(follow_symlinks=False)
    walk.entries += 1
    if stat.S_ISDIR(info.st_mode):
        walk.frontier.append((*comps, name))
        return
    if not stat.S_ISREG(info.st_mode):
        raise PlannerError("unsafe entry in tree")
    if info.st_nlink != 1:
        raise PlannerError("hard-linked file in tree")
    walk.bytes += info.st_size
    walk.max_activity = max(walk.max_activity, info.st_mtime_ns)


def _fingerprints_intact(flight: _InFlight) -> bool:
    for relpath, dev, ino, mtime in flight.walk.fingerprints:
        comps = tuple(p for p in relpath.split("/") if p)
        try:
            with _descend(flight.root_fd, comps) as dfd:
                cur = os.fstat(dfd)
        except (OSError, SessionError):
            return False
        if (cur.st_dev, cur.st_ino, cur.st_mtime_ns) != (dev, ino, mtime):
            return False
    return True


def _failure_reason(exc: BaseException) -> str:
    """Map one root's filesystem or safety failure to a conservative reason."""
    if isinstance(exc, SessionInUseError):
        return "session is open or in use"
    if isinstance(exc, (SessionNotFoundError, FileNotFoundError)):
        return "session or entry disappeared"
    if isinstance(exc, OSError):
        return _bounded_reason(f"unreadable entry: {exc.strerror or exc}")
    return _bounded_reason(str(exc))


def _sent_file_reason(directory_fd, scan) -> str | None:
    value = _read_json(directory_fd, "project-inbox-sent.json", scan)
    if value is None:
        return None
    tracker = object.__new__(SentMessageTracker)
    tracker.path = Path("project-inbox-sent.json")
    state = tracker._classify_value(value, charge_record=lambda: scan.charge(0))
    if isinstance(state, _Broken):
        return "broken sent tracker"
    assert isinstance(state, _Loaded)
    for record in state.records:
        if "done" not in record["reported"]:
            return "unacknowledged sent message"
    return None


def _authority_transcript_recovery_reason(
    flight: _InFlight, stores
) -> str | None:
    for comps, names in stores:
        with _descend(flight.root_fd, comps) as dfd:
            child = ":".join(part for part in comps if part != _AGENTS_DIRNAME)
            target = RepairTarget(
                flight.root_id, f"{flight.root_id}:{child}" if child else None
            )
            try:
                if classify_recovery_records(dfd, target, names):
                    return "unresolved transcript recovery"
            except (OSError, SessionError, ValueError) as exc:
                raise PlannerError("broken transcript recovery") from exc
    return None


def _read_authority_bytes(directory_fd: int, name: str) -> bytes | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except (OSError, SessionError) as exc:
        raise PlannerError(f"unsafe retention read: {name}") from exc
    try:
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            return stream.read()
    except OSError as exc:
        raise PlannerError(f"unsafe retention read: {name}") from exc
    finally:
        if fd != -1:
            os.close(fd)


def _authority_memory_reconciliation_reason(
    flight: _InFlight, stores
) -> str | None:
    for comps, store_names in stores:
        names = set(store_names)
        with _descend(flight.root_fd, comps) as dfd:
            cursor = (
                _read_authority_bytes(dfd, "memory-reconcile.json")
                if "memory-reconcile.json" in names else None
            )
            receipts = (
                _read_authority_bytes(dfd, "memory-reconcile-receipts.jsonl")
                if "memory-reconcile-receipts.jsonl" in names else None
            )
            if classify_reconciliation_records(cursor, receipts):
                return "unresolved memory reconciliation"
    return None


def _authority_store_reason(flight: _InFlight, scan: _ClosureBudget) -> str | None:
    """Store-local blockers; also folds metadata timestamps into activity."""
    for comps in flight.store_dirs:
        with _descend(flight.root_fd, comps) as dfd:
            pending = _read_json(dfd, _PENDING_LINK_NAME, scan)
            if pending is not None:
                if not valid_pending_link(pending):
                    raise PlannerError("malformed pending project link")
                return "pending project link"
            if _pending_child_link(dfd, scan):
                return "pending child link"
            if _has_child_marker(dfd, scan):
                return "live or recoverable child"
            lifecycle = _read_json(dfd, _LIFECYCLE_NAME, scan)
            if comps and not _lifecycle_terminal(lifecycle):
                return "live or recoverable child"
            metadata = _read_json(dfd, _METADATA_NAME, scan)
            if not comps:
                _canonical_metadata(metadata, flight.root_id)
            for value in (metadata, lifecycle):
                flight.walk.max_activity = max(flight.walk.max_activity, _metadata_activity(value))
    return None


def _has_child_marker(directory_fd, scan) -> bool:
    value = _read_json(directory_fd, _SESSION_STATE_NAME, scan)
    if value is None:
        return False
    if type(value) is not dict:
        raise PlannerError("malformed session state")
    try:
        bash_cwd = value.get("bash_cwd")
        if type(bash_cwd) is not str or not bash_cwd:
            raise ValueError
        parse_todo_items(value.get("todo_items", []))
        if type(value.get("todo_dismissed", False)) is not bool:
            raise ValueError
        parsed = _parse_agent_state(value, Path(_SESSION_STATE_NAME))
    except (TypeError, ValueError, SessionError) as exc:
        raise PlannerError("malformed session state") from exc
    return bool(parsed["agent_children"] or parsed["agent_parent"])


def _lifecycle_terminal(value) -> bool:
    if type(value) is not dict:
        raise PlannerError("malformed child lifecycle")
    try:
        record = parse_legacy_child_lifecycle(value)
    except (TypeError, ValueError) as exc:
        raise PlannerError("malformed child lifecycle") from exc
    return record.phase in {
        ChildRunPhase.COMPLETED, ChildRunPhase.CANCELED, ChildRunPhase.FAILED
    }


def _canonical_metadata(value, session_id: str) -> SessionMetadata:
    if type(value) is not dict:
        raise PlannerError("malformed metadata")
    try:
        metadata = SessionMetadata.from_dict(
            value, path=Path("sessions") / session_id / _METADATA_NAME
        )
    except (TypeError, ValueError, SessionError) as exc:
        raise PlannerError("malformed metadata") from exc
    if metadata.session_id != session_id:
        raise PlannerError("metadata id does not match directory")
    return metadata


def _metadata_activity(value) -> int:
    """Latest metadata timestamp in ns; a malformed one makes the root ineligible."""
    if value is None:
        return 0
    if type(value) is not dict:
        raise PlannerError("malformed metadata")
    latest = 0
    for name in _TIMESTAMP_FIELDS:
        text = value.get(name)
        if text is None:
            continue
        try:
            moment = datetime.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise PlannerError("malformed metadata timestamp") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        latest = max(latest, int(moment.timestamp() * 1_000_000_000))
    return latest


def _authority_parent_id(sessions_fd, session_id, scan) -> str | None:
    """Return the canonical parent id; a missing chain member is unsafe."""
    try:
        with child_directory(sessions_fd, session_id) as session_fd:
            meta = _read_json(session_fd, _METADATA_NAME, scan)
    except FileNotFoundError as exc:
        raise PlannerError("missing metadata ancestor") from exc
    if meta is None:
        raise PlannerError("malformed metadata")
    return _canonical_metadata(meta, session_id).parent_session_id


def _authority_inbox_reason(home_fd, ids, scan: _ClosureBudget) -> str | None:
    for status_fd, bodies_fd in _authority_inbox_dirs(home_fd, scan):
        with scan.iter_entries(status_fd) as entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(".json"):
                    raise PlannerError("unexpected inbox file")
                if _classify_inbox_record(
                    _read_json(status_fd, name, scan), name, ids, bodies_fd
                ):
                    return "pending inbox reference"
    return None


def _authority_inbox_dirs(home_fd, scan):
    """Yield each project's inbox ``new`` and ``claimed`` fds, never following links."""
    with ExitStack() as stack:
        projects_fd = _open_dir(stack, home_fd, "projects")
        if projects_fd is None:
            return
        with scan.iter_entries(projects_fd) as projects:
            for entry in projects:
                project = entry.name
                if entry.is_symlink():
                    raise PlannerError("unsafe project directory")
                if not entry.is_dir(follow_symlinks=False):
                    continue  # e.g. projects/.lock
                with ExitStack() as inner:
                    project_fd = _open_dir(inner, projects_fd, project)
                    inbox_fd = _open_dir(inner, project_fd, "inbox") if project_fd is not None else None
                    bodies_fd = _open_dir(inner, inbox_fd, "bodies") if inbox_fd is not None else None
                    for status in ("new", "claimed") if inbox_fd is not None else ():
                        status_fd = _open_dir(inner, inbox_fd, status)
                        if status_fd is not None:
                            yield status_fd, bodies_fd


def _open_dir(stack, directory_fd, name) -> int | None:
    """Open one directory without following links; absent is None."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PlannerError(f"unsafe directory: {name}") from exc
    stack.callback(os.close, fd)
    return fd


def _authority_dependents(sessions_fd, root_id, scan: _ClosureBudget) -> str | None:
    """Scan every other root through the canonical metadata authorities."""
    with scan.iter_entries(sessions_fd) as roots:
        for entry in roots:
            other = entry.name
            if other == root_id or other.startswith("."):
                continue
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise PlannerError("unsafe possible dependent")
            if _dependent_reason(sessions_fd, other, root_id, scan):
                return "published top-level dependent"
    return None


def _dependent_reason(sessions_fd, other, root_id, scan) -> bool:
    try:
        with child_directory(sessions_fd, other) as other_fd:
            meta = _canonical_metadata(_read_json(other_fd, _METADATA_NAME, scan), other)
            fork_value = _read_json(other_fd, _ATTENTION_FORK_NAME, scan)
            try:
                fork = (
                    read_attention_fork(Path(other), directory_fd=other_fd)
                    if fork_value is not None else None
                )
            except (TypeError, ValueError, SessionError) as exc:
                raise PlannerError("malformed attention fork") from exc
            attention = _attention_reference(other_fd, root_id, scan)
    except FileNotFoundError:
        return False
    return bool(
        attention
        or meta.parent_session_id == root_id
        or (fork is not None and fork.forked_from_session == root_id)
    )


def _attention_reference(other_fd, root_id, scan) -> bool:
    with ExitStack() as stack:
        attention_fd = _open_dir(stack, other_fd, "attention")
        if attention_fd is None:
            return False
        with scan.iter_entries(attention_fd) as records:
            for entry in records:
                if not entry.name.endswith(".json"):
                    raise PlannerError("malformed attention record")
                value = _read_json(attention_fd, entry.name, scan)
                try:
                    record = AttentionRecord.from_dict(value)
                except (TypeError, ValueError, KeyError) as exc:
                    raise PlannerError("malformed attention record") from exc
                if entry.name != f"{record.id}.json":
                    raise PlannerError("attention id does not match filename")
                if record.fork_session_id == root_id:
                    return True
    return False


def _authority_store_dirs(root_fd, scan, prefix=()):
    """Yield component paths for every child store directory under a root."""
    if len(prefix) > _MAX_TREE_DEPTH:
        raise PlannerError("tree is too deep")
    try:
        with (
            _descend(root_fd, (*prefix, _AGENTS_DIRNAME)) as agents_fd,
            scan.iter_entries(agents_fd) as entries,
        ):
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                comps = (*prefix, _AGENTS_DIRNAME, entry.name)
                yield comps
                yield from _authority_store_dirs(root_fd, scan, comps)
    except (FileNotFoundError, NotADirectoryError):
        return


def _generation_value(home, scan) -> int:
    value = read_reference_generation(home)
    scan.charge(0)
    return value


def _reservation_present(home_fd, root_id, scan) -> bool:
    with ExitStack() as stack:
        trash_fd = _open_dir(stack, home_fd, "session-trash")
        if trash_fd is None:
            return False
        reservations_fd = _open_dir(stack, trash_fd, "reservations")
        if reservations_fd is not None:
            value = _read_json(reservations_fd, f"{root_id}.json", scan)
            if value is not None:
                if type(value) is not dict:
                    raise PlannerError("malformed retention reservation")
                try:
                    validate_retention_reservation(value)
                except StorageFormatError as exc:
                    raise PlannerError("malformed retention reservation") from exc
                if value["session_id"] != root_id:
                    raise PlannerError("reservation id does not match filename")
                return True
        try:
            info = os.stat(root_id, dir_fd=trash_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            raise PlannerError("unsafe retention trash evidence")
        scan.charge_entry(0)
        return True


def _pending_child_link(directory_fd, scan) -> bool:
    with ExitStack() as stack:
        fd = _open_dir(stack, directory_fd, _PENDING_CHILD_LINKS_DIRNAME)
        if fd is None:
            return False
        with scan.iter_entries(fd) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    raise PlannerError("malformed pending child link")
                value = _read_json(fd, entry.name, scan)
                if not valid_child_pending_link(value):
                    raise PlannerError("malformed pending child link")
                return True
    return False


def _classify_inbox_record(record, name, ids, bodies_fd) -> bool:
    try:
        record = validate_inbox_record(record, name, bodies_fd=bodies_fd)
    except (InboxError, TypeError, ValueError) as exc:
        raise PlannerError("malformed inbox record") from exc
    sender = record["from"]
    return (
        sender["session"] in ids
        or record.get("to_session") in ids
        or record.get("claimer_session") in ids
    )


def _preview_snapshot(preview) -> PreviewSnapshot | None:
    if not preview:
        return None
    return PreviewSnapshot(
        preview_id=preview["preview_id"],
        days=preview["days"],
        cutoff=preview["cutoff"],
        root_ids=tuple(preview["root_ids"]),
        count=preview["count"],
        bytes=preview["bytes"],
        created_at=preview["created_at"],
    )


def _bounded_reason(message: str) -> str:
    return message[:120]
