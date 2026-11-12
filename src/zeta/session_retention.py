"""Durable retention-owned session sidecars and the dormant retention planner.

The planner is a library-only, bounded, resumable preview walk. It is reachable
only from tests; no scheduler, startup hook, slash command, setting, or event
calls it. ``test_session_retention`` asserts no production module imports the
planner entry points.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from .attention_forks import read_attention_fork
from .attention_records import AttentionRecord
from .core.agent_state import _parse_agent_state
from .core.child_lifecycle import ChildRunPhase, parse_legacy_child_lifecycle
from .core.session import SessionMetadata
from .core.session_files import (
    SessionError,
    SessionInUseError,
    SessionNotFoundError,
    atomic_publish_file,
    child_directory,
    descend_directory,
    open_session_file,
    read_reference_generation,
    unlink_session_file,
)
from .core.session_links import valid_child_pending_link, valid_pending_link
from .core.todo import parse_todo_items
from .project_inbox import InboxError, validate_inbox_record
from .project_inbox_sent import SentMessageTracker, _Broken, _Loaded
from .storage_format import (
    StorageFeature,
    StorageFormatError,
    WriterLease,
    feature_enabled,
    validate_retention_reservation,
    validate_retention_state,
)

RETENTION_PIN_FILENAME = ".retention-pin.json"
_MAX_PIN_BYTES = 256
_PIN_DATA = b'{"pinned":true,"schema_version":1}\n'


def is_retention_pinned(directory_fd: int, *, scan=None) -> bool:
    """Return pin state, treating every unsafe present sidecar as pinned."""
    try:
        fd = open_session_file(directory_fd, RETENTION_PIN_FILENAME, os.O_RDONLY)
    except FileNotFoundError:
        try:
            charge = scan.charge_entry if scan is not None else (lambda _size: None)
            with _EntryScan(directory_fd, charge) as entries:
                return any(
                    entry.name.startswith(f".{RETENTION_PIN_FILENAME}.")
                    and entry.name.endswith(".tmp")
                    for entry in entries
                )
        except OSError:
            return True
    except (OSError, SessionError):
        return True
    try:
        info = os.fstat(fd)
        if (
            info.st_size > _MAX_PIN_BYTES
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            return True
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            try:
                json.loads(stream.read(_MAX_PIN_BYTES + 1))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                return True
        return True
    except (OSError, ValueError):
        return True
    finally:
        if fd != -1:
            os.close(fd)


def write_retention_pin(directory_fd: int) -> None:
    """Atomically publish the v1 pin sidecar."""
    atomic_publish_file(directory_fd, RETENTION_PIN_FILENAME, _PIN_DATA)


def remove_retention_pin(directory_fd: int) -> None:
    """Remove the pin sidecar; an absent sidecar is already unpinned."""
    unlink_session_file(directory_fd, RETENTION_PIN_FILENAME)


_STATE_FILENAME = "retention-state.json"
_METADATA_NAME = "meta.json"
_SESSION_STATE_NAME = "session_state.json"
_PENDING_LINK_NAME = "project_link_pending.json"
_PENDING_CHILD_LINKS_DIRNAME = "pending_child_links"
_ATTENTION_FORK_NAME = "attention_fork.json"
_STORE_LOCK_NAME = ".lock"
_RUNTIME_LEASE_NAME = "runtime.lease"
_AGENTS_DIRNAME = "agents"
_LIFECYCLE_NAME = "agent_lifecycle.json"
_TERMINAL_AGENT_STATES = frozenset({"completed", "canceled", "failed"})
_MAX_JSON_BYTES = 1 << 20
_MAX_TREE_DEPTH = 64
_NS_PER_DAY = 86_400 * 1_000_000_000


@dataclass(frozen=True)
class RetentionBounds:
    """Hard per-pass work bounds (design §3.9).

    Defaults are the production bounds. Tests pass smaller values through the
    same code path; nothing else configures them.
    """

    metadata_candidates: int = 32
    roots_advanced: int = 8
    entries_per_root: int = 4096
    entries_total: int = 32_768
    reference_records_per_root: int = 4096
    reference_bytes_per_root: int = 8 << 20
    reference_records_total: int = 32_768
    reference_bytes_total: int = 64 << 20
    worker_ns: int = 30 * 1_000_000_000


@dataclass(frozen=True)
class RootOutcome:
    """Why one root is or is not a preview candidate this sweep."""

    root_id: str
    closed: bool
    reason: str | None
    bytes: int = 0


@dataclass(frozen=True)
class PreviewSnapshot:
    """A completed sweep's preview (design §3.1)."""

    preview_id: str
    days: int
    cutoff: str
    root_ids: tuple[str, ...]
    count: int
    bytes: int
    created_at: str


@dataclass(frozen=True)
class PassResult:
    """The outcome of one bounded planner pass."""

    enabled: bool
    complete: bool
    reasons: tuple[RootOutcome, ...] = ()
    preview: PreviewSnapshot | None = None
    in_progress_root: str | None = None


class RetentionPlannerError(Exception):
    """Raised when the planner cannot proceed safely."""


def _read_json(directory_fd: int, name: str, scan=None) -> object | None:
    """Read one bounded regular JSON file, or None when it is absent.

    Any unsafe entry (symlink, nonregular, multi-link, oversized, malformed)
    raises so the caller can treat the root conservatively. ``scan`` charges
    the record and its size to the reference budget before the read.
    """
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except (OSError, SessionError) as exc:
        raise RetentionPlannerError(f"unsafe retention read: {name}") from exc
    try:
        info = os.fstat(fd)
        if info.st_size > _MAX_JSON_BYTES:
            raise RetentionPlannerError(f"retention read too large: {name}")
        if scan is not None:
            scan.charge(info.st_size)
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            return json.loads(stream.read())
    except (OSError, ValueError) as exc:
        raise RetentionPlannerError(f"malformed retention read: {name}") from exc
    finally:
        if fd != -1:
            os.close(fd)


def _load_state(home_fd: int) -> dict | None:
    """Return validated planner state, or None when absent or unusable."""
    try:
        value = _read_json(home_fd, _STATE_FILENAME)
    except RetentionPlannerError:
        return None
    if value is None:
        return None
    if type(value) is not dict:
        return None
    try:
        validate_retention_state(value)
    except StorageFormatError:
        return None
    return value


def _store_state(home_fd: int, state: dict) -> None:
    validate_retention_state(state)
    atomic_publish_file(
        home_fd,
        _STATE_FILENAME,
        (json.dumps(state, separators=(",", ":"), sort_keys=True) + "\n").encode(),
    )

def _now_ns() -> int:
    return time.time_ns()


class _Deferred(Exception):
    """The shared pass budget ran out; the in-flight root resumes next pass."""


@dataclass
class _Budget:
    """One pass's shared work budget, checked after each entry and record."""

    bounds: RetentionBounds
    clock: Callable[[], int]
    deadline: int
    entries: int = 0
    records: int = 0
    record_bytes: int = 0

    def charge_entry(self, size: int = 0) -> None:
        self.entries += 1
        if self.entries > self.bounds.entries_total or self.clock() >= self.deadline:
            raise _Deferred

    def walk_exhausted(self, root_entries: int) -> bool:
        return (
            root_entries >= self.bounds.entries_per_root
            or self.entries >= self.bounds.entries_total
            or self.clock() >= self.deadline
        )


@dataclass
class _ClosureBudget:
    """One root's reference-scan counters; every record also charges the pass."""

    budget: _Budget
    records: int = 0
    record_bytes: int = 0

    def iter_entries(self, directory_fd):
        """Yield directory entries through the shared bounded scanner."""
        return _EntryScan(directory_fd, self.charge_entry)

    def charge_entry(self, size: int) -> None:
        self.budget.charge_entry(size)
        self.charge(size)

    def charge(self, size: int) -> None:
        bounds, budget = self.budget.bounds, self.budget
        if budget.clock() >= budget.deadline:
            raise _Deferred
        self.records += 1
        self.record_bytes += size
        if (
            self.records > bounds.reference_records_per_root
            or self.record_bytes > bounds.reference_bytes_per_root
        ):
            raise RetentionPlannerError("references not fully scanned")
        budget.records += 1
        budget.record_bytes += size
        if (
            budget.records > bounds.reference_records_total
            or budget.record_bytes > bounds.reference_bytes_total
            or budget.clock() >= budget.deadline
        ):
            raise _Deferred


@dataclass
class _Walk:
    """Authoritative walk progress; valid only while its lease set is held."""

    frontier: list[tuple[str, ...]] = field(default_factory=lambda: [()])
    listing: tuple[tuple[str, ...], _EntryScan] | None = None
    fingerprints: list[list] = field(default_factory=list)
    max_activity: int = 0
    bytes: int = 0
    entries: int = 0

    @property
    def done(self) -> bool:
        return self.listing is None and not self.frontier


@dataclass
class _InFlight:
    """The single root whose whole lease set this planner holds across passes."""

    root_id: str
    stack: ExitStack
    root_fd: int
    store_dirs: list[tuple[str, ...]]
    device: int
    inode: int
    walk: _Walk = field(default_factory=_Walk)
    rewalked: bool = False
    deferred: bool = False


@dataclass
class _Sweep:
    """One in-memory sweep over every root that existed at its start."""

    days: int
    anchor: int
    preview_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    root_scan: _EntryScan | None = None
    current_root: str | None = None
    discovery_done: bool = False
    discovery_blocked: bool = False
    selected: list[str] = field(default_factory=list)
    bytes: int = 0

    @property
    def cutoff_ns(self) -> int:
        return self.anchor - self.days * _NS_PER_DAY


_UNLOADED = object()


class RetentionPlanner:
    """A bounded, resumable, dormant preview planner for one home.

    The sweep cursor, the in-progress accumulator, and the in-flight root's walk
    live in memory. The in-flight root's whole lease set stays held across
    passes, so its walk progress stays authoritative; ``close()`` releases it. A
    new instance models a process restart and starts a new sweep. The persisted
    ``preview`` holds only the latest complete preview.
    """

    def __init__(
        self,
        home: str | Path,
        lease: WriterLease,
        *,
        bounds: RetentionBounds | None = None,
        now_ns=_now_ns,
    ) -> None:
        self.home = Path(home)
        self.lease = lease
        self.bounds = bounds or RetentionBounds()
        self._now_ns = now_ns
        self._lease_epoch = uuid.uuid4().hex
        self._sweep: _Sweep | None = None
        self._flight: _InFlight | None = None
        self._latest: object = _UNLOADED

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Release the in-flight root's lease set; the next pass rewalks it."""
        flight, self._flight = self._flight, None
        if flight is not None:
            if flight.walk.listing is not None:
                flight.walk.listing[1].close()
            flight.stack.close()

    @property
    def enabled(self) -> bool:
        return feature_enabled(self.lease.observed_epoch, StorageFeature.SESSION_RETENTION)

    def run_pass(self, *, days: int) -> PassResult:
        """Advance one bounded sweep step and return what it proved."""
        if type(days) is not int or days <= 0 or days > 3650:
            raise RetentionPlannerError("retention days must be in 1..3650")
        if not self.enabled:
            return PassResult(enabled=False, complete=False)
        home_fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return self._run_pass(home_fd, days)
        finally:
            os.close(home_fd)

    def _run_pass(self, home_fd: int, days: int) -> PassResult:
        if self._latest is _UNLOADED:
            state = _load_state(home_fd)
            self._latest = state["preview"] if state is not None else None
        budget = _Budget(self.bounds, self._now_ns, self._now_ns() + self.bounds.worker_ns)
        sweep = self._sweep
        if sweep is None or sweep.days != days:
            self.close()
            if sweep is not None and sweep.root_scan is not None:
                sweep.root_scan.close()
            sweep = self._sweep = _Sweep(days, self._now_ns())
        reasons = self._advance(home_fd, sweep, budget)
        complete = (
            sweep.discovery_done
            and not sweep.discovery_blocked
            and sweep.current_root is None
        )
        if complete:
            self._latest = {
                "preview_id": sweep.preview_id, "days": days,
                "cutoff": str(sweep.cutoff_ns), "root_ids": list(sweep.selected),
                "count": len(sweep.selected), "bytes": sweep.bytes,
                "created_at": str(sweep.anchor),
            }
            if sweep.root_scan is not None:
                sweep.root_scan.close()
            self._sweep = None
        _store_state(home_fd, self._state_payload())
        return PassResult(
            enabled=True, complete=complete, reasons=tuple(reasons),
            preview=_preview_snapshot(self._latest),
            in_progress_root=None if complete else sweep.current_root,
        )

    def _advance(self, home_fd, sweep: _Sweep, budget: _Budget) -> list[RootOutcome]:
        reasons: list[RootOutcome] = []
        with ExitStack() as stack:
            try:
                sessions_fd = stack.enter_context(child_directory(home_fd, "sessions"))
            except FileNotFoundError:
                sweep.discovery_done = True
                return reasons
            if sweep.root_scan is None:
                sweep.root_scan = _EntryScan(sessions_fd, budget.charge_entry)
            else:
                sweep.root_scan.charge = budget.charge_entry
            while len(reasons) < min(self.bounds.roots_advanced, self.bounds.metadata_candidates):
                try:
                    if sweep.current_root is None:
                        try:
                            sweep.current_root = _next_root(sweep)
                        except RetentionPlannerError:
                            sweep.discovery_blocked = True
                            sweep.root_scan.close()
                            break
                        if sweep.current_root is None:
                            break
                    outcome = self._decide(home_fd, sessions_fd, sweep, budget)
                except _Deferred:
                    break
                if outcome is None:
                    break
                reasons.append(outcome)
                sweep.current_root = None
                if outcome.closed:
                    sweep.selected.append(outcome.root_id)
                    sweep.bytes += outcome.bytes
        return reasons

    def _decide(self, home_fd, sessions_fd, sweep: _Sweep, budget: _Budget) -> RootOutcome | None:
        """Decide the cursor root, or return None when it resumes next pass."""
        root_id = sweep.current_root
        assert root_id is not None
        scan = _ClosureBudget(budget)
        try:
            flight = self._hold(sessions_fd, root_id, scan)
            if not self._walk(flight, budget):
                return None
            reason = self._closure_reason(home_fd, sessions_fd, flight, sweep, scan)
        except _Deferred:
            if self._flight is None:
                reason = "references not fully scanned"
            elif not self._flight.deferred:
                self._flight.deferred = True
                return None
            else:
                reason = "references not fully scanned"
        except (OSError, SessionError, RetentionPlannerError) as exc:
            reason = _failure_reason(exc)
        bytes_ = self._flight.walk.bytes if reason is None else 0
        self.close()
        return RootOutcome(root_id, reason is None, reason, bytes_)

    def _hold(self, sessions_fd, root_id, scan) -> _InFlight:
        """Return the held lease set for ``root_id``, taking it when needed."""
        flight = self._flight
        if flight is not None and flight.root_id == root_id:
            info = os.stat(root_id, dir_fd=sessions_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (flight.device, flight.inode):
                raise RetentionPlannerError("session changed during walk")
            return flight
        self.close()
        stack = ExitStack()
        try:
            try:
                root_fd = stack.enter_context(child_directory(sessions_fd, root_id))
            except FileNotFoundError as exc:
                raise SessionNotFoundError(f"session {root_id} was not found") from exc
            store_dirs = [(), *_authority_store_dirs(root_fd, scan)]
            for comps in store_dirs:
                dfd = stack.enter_context(_descend(root_fd, comps))
                _flock_nb(dfd)
                if os.fstat(dfd).st_nlink == 0:
                    raise SessionNotFoundError(f"session {root_id} was not found")
                _lease_file(stack, dfd, _STORE_LOCK_NAME)
                _lease_file(stack, dfd, _RUNTIME_LEASE_NAME)
            info = os.fstat(root_fd)
        except BaseException:
            stack.close()
            raise
        self._flight = _InFlight(root_id, stack, root_fd, store_dirs, info.st_dev, info.st_ino)
        return self._flight

    def _walk(self, flight: _InFlight, budget: _Budget) -> bool:
        """Walk one bounded step; True once a walk finished and still matches."""
        walk = flight.walk
        root_entries = 0
        while not walk.done:
            if walk.listing is None:
                walk.listing = _walk_directory(flight, walk, budget)
            comps, entries = walk.listing
            with _descend(flight.root_fd, comps) as dfd:
                while root_entries < self.bounds.entries_per_root:
                    entries.charge = budget.charge_entry
                    try:
                        entry = next(entries)
                    except StopIteration:
                        entries.close()
                        walk.listing = None
                        break
                    _visit(walk, dfd, comps, entry)
                    root_entries += 1
                else:
                    return False
        if _fingerprints_intact(flight):
            return True
        if flight.rewalked:
            raise RetentionPlannerError("session changed during walk")
        flight.walk, flight.rewalked = _Walk(), True
        return False

    def _closure_reason(self, home_fd, sessions_fd, flight, sweep, scan) -> str | None:
        return _ClosureAuthority(self, home_fd, sessions_fd, flight, sweep, scan).classify()

    def _authority_sent_reason(self, flight, scan) -> str | None:
        for comps in flight.store_dirs:
            with _descend(flight.root_fd, comps) as directory_fd:
                reason = _sent_file_reason(directory_fd, scan)
                if reason is not None:
                    return reason
        return None

    def _pinned(self, sessions_fd, root_fd, root_id, scan) -> bool:
        """Own or inherited pin; an exhausted hop bound pins conservatively."""
        if is_retention_pinned(root_fd, scan=scan):
            return True
        seen = {root_id}
        current = root_id
        for _ in range(self.bounds.metadata_candidates):
            parent = _authority_parent_id(sessions_fd, current, scan)
            if parent is None:
                return False
            if parent in seen:
                return True
            seen.add(parent)
            try:
                with child_directory(sessions_fd, parent) as parent_fd:
                    if is_retention_pinned(parent_fd, scan=scan):
                        return True
            except FileNotFoundError:
                return False
            current = parent
        return True

    def _state_payload(self) -> dict:
        """Project memory state onto the registered shape (observability only)."""
        sweep, flight = self._sweep, self._flight
        walk = flight.walk if flight is not None else _Walk()
        anchor = sweep.anchor if sweep is not None else 0
        return {
            "root_id": flight.root_id if flight is not None else "pending",
            "device": flight.device if flight is not None else 0,
            "inode": flight.inode if flight is not None else 0,
            "cutoff": str(sweep.cutoff_ns) if sweep is not None else "0",
            "walk_started_at": str(anchor),
            "lease_epoch": self._lease_epoch,
            "frontier": [list(c) for c in walk.frontier]
            + ([list(walk.listing[0])] if walk.listing is not None else []),
            "completed_directories": [list(f) for f in walk.fingerprints],
            "max_activity": walk.max_activity,
            "bytes": walk.bytes,
            "descendant_ids": ["/".join(c) for c in flight.store_dirs if c] if flight else [],
            "entry_count": walk.entries,
            "reference_cursor": {},
            "start_generation": 0,
            "references": {},
            "phase": "discover" if flight is None else ("walk" if not walk.done else "references"),
            "preview": self._latest,
        }


class _ClosureAuthority:
    """One owner for candidate closure decisions and bounded references."""

    def __init__(self, planner, home_fd, sessions_fd, flight, sweep, scan):
        self.planner = planner
        self.home_fd = home_fd
        self.sessions_fd = sessions_fd
        self.flight = flight
        self.sweep = sweep
        self.scan = scan

    def classify(self) -> str | None:
        root_id = self.flight.root_id
        generation = read_reference_generation(self.planner.home)
        ids = {root_id}
        for comps in self.flight.store_dirs:
            if comps:
                child_parts = [part for part in comps if part != _AGENTS_DIRNAME]
                ids.add(root_id + ":" + ":".join(child_parts))
        checks = (
            lambda: "session is being deleted" if _reservation_present(self.home_fd, root_id, self.scan) else None,
            lambda: _authority_store_reason(self.flight, self.scan),
            lambda: self.planner._authority_sent_reason(self.flight, self.scan),
            lambda: "pinned" if self.planner._pinned(self.sessions_fd, self.flight.root_fd, root_id, self.scan) else None,
            lambda: _authority_inbox_reason(self.home_fd, ids, self.scan),
            lambda: _authority_dependents(self.sessions_fd, root_id, self.scan),
        )
        if (reason := next((item for check in checks if (item := check()) is not None), None)) is not None:
            return reason
        if min(self.flight.walk.max_activity, self.sweep.anchor) >= self.sweep.cutoff_ns:
            return "not old enough"
        if read_reference_generation(self.planner.home) != generation:
            return "references changed during scan"
        return None


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
            raise RetentionPlannerError("unsafe retention directory") from exc
        self.charge = charge
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            entry = next(self._entries)
            info = entry.stat(follow_symlinks=False)
        except StopIteration:
            raise
        except OSError as exc:
            raise RetentionPlannerError("unsafe retention entry") from exc
        self.charge(info.st_size if stat.S_ISREG(info.st_mode) else 0)
        return entry

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._entries.close()
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
            raise RetentionPlannerError("unsafe session root entry")
        return entry.name
    sweep.discovery_done = True
    return None


def _is_directory(directory_fd, name) -> bool:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode)


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
        raise RetentionPlannerError(f"unsafe store lock: {name}") from exc
    stack.callback(os.close, fd)
    _flock_nb(fd)


def _walk_directory(flight: _InFlight, walk: _Walk, budget: _Budget):
    comps = walk.frontier.pop()
    if len(comps) > _MAX_TREE_DEPTH:
        raise RetentionPlannerError("tree is too deep")
    if comps[-2:-1] == (_AGENTS_DIRNAME,) and comps not in flight.store_dirs:
        raise RetentionPlannerError("child store appeared during walk")
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
        raise RetentionPlannerError("unsafe entry in tree")
    if info.st_nlink != 1:
        raise RetentionPlannerError("hard-linked file in tree")
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
    state = tracker._classify_value(value)
    if isinstance(state, _Broken):
        return "broken sent tracker"
    assert isinstance(state, _Loaded)
    for record in state.records:
        scan.charge(0)
        if "done" not in record["reported"]:
            return "unacknowledged sent message"
    return None


def _authority_store_reason(flight: _InFlight, scan: _ClosureBudget) -> str | None:
    """Store-local blockers; also folds metadata timestamps into activity."""
    for comps in flight.store_dirs:
        with _descend(flight.root_fd, comps) as dfd:
            pending = _read_json(dfd, _PENDING_LINK_NAME, scan)
            if pending is not None:
                if not valid_pending_link(pending):
                    raise RetentionPlannerError("malformed pending project link")
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
        raise RetentionPlannerError("malformed session state")
    try:
        bash_cwd = value.get("bash_cwd")
        if type(bash_cwd) is not str or not bash_cwd:
            raise ValueError
        parse_todo_items(value.get("todo_items", []))
        if type(value.get("todo_dismissed", False)) is not bool:
            raise ValueError
        parsed = _parse_agent_state(value, Path(_SESSION_STATE_NAME))
    except (TypeError, ValueError, SessionError) as exc:
        raise RetentionPlannerError("malformed session state") from exc
    return bool(parsed["agent_children"] or parsed["agent_parent"])


def _lifecycle_terminal(value) -> bool:
    if type(value) is not dict:
        raise RetentionPlannerError("malformed child lifecycle")
    try:
        record = parse_legacy_child_lifecycle(value)
    except (TypeError, ValueError) as exc:
        raise RetentionPlannerError("malformed child lifecycle") from exc
    return record.phase in {
        ChildRunPhase.COMPLETED, ChildRunPhase.CANCELED, ChildRunPhase.FAILED
    }


def _canonical_metadata(value, session_id: str) -> SessionMetadata:
    if type(value) is not dict:
        raise RetentionPlannerError("malformed metadata")
    try:
        return SessionMetadata.from_dict(value, path=Path("sessions") / session_id / _METADATA_NAME)
    except (TypeError, ValueError, SessionError) as exc:
        raise RetentionPlannerError("malformed metadata") from exc


def _metadata_activity(value) -> int:
    """Latest metadata timestamp in ns; a malformed one makes the root ineligible."""
    if value is None:
        return 0
    if type(value) is not dict:
        raise RetentionPlannerError("malformed metadata")
    latest = 0
    for name in _TIMESTAMP_FIELDS:
        text = value.get(name)
        if text is None:
            continue
        try:
            moment = datetime.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise RetentionPlannerError("malformed metadata timestamp") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        latest = max(latest, int(moment.timestamp() * 1_000_000_000))
    return latest


def _authority_parent_id(sessions_fd, session_id, scan) -> str | None:
    """The canonical metadata parent id; missing ancestors end the chain."""
    try:
        with child_directory(sessions_fd, session_id) as session_fd:
            meta = _read_json(session_fd, _METADATA_NAME, scan)
    except FileNotFoundError:
        return None
    if meta is None:
        raise RetentionPlannerError("malformed metadata")
    return _canonical_metadata(meta, session_id).parent_session_id


def _authority_inbox_reason(home_fd, ids, scan: _ClosureBudget) -> str | None:
    for status_fd in _authority_inbox_dirs(home_fd, scan):
        for entry in scan.iter_entries(status_fd):
            name = entry.name
            if name.endswith(".json") and _classify_inbox_record(
                _read_json(status_fd, name, scan), name, ids
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
                    raise RetentionPlannerError("unsafe project directory")
                if not entry.is_dir(follow_symlinks=False):
                    continue  # e.g. projects/.lock
                with ExitStack() as inner:
                    project_fd = _open_dir(inner, projects_fd, project)
                    inbox_fd = _open_dir(inner, project_fd, "inbox") if project_fd is not None else None
                    for status in ("new", "claimed") if inbox_fd is not None else ():
                        status_fd = _open_dir(inner, inbox_fd, status)
                        if status_fd is not None:
                            yield status_fd


def _open_dir(stack, directory_fd, name) -> int | None:
    """Open one directory without following links; absent is None."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RetentionPlannerError(f"unsafe directory: {name}") from exc
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
                raise RetentionPlannerError("unsafe possible dependent")
            try:
                with child_directory(sessions_fd, other) as other_fd:
                    meta = _canonical_metadata(
                        _read_json(other_fd, _METADATA_NAME, scan), other
                    )
                    fork_value = _read_json(other_fd, _ATTENTION_FORK_NAME, scan)
                    fork = None
                    if fork_value is not None:
                        try:
                            fork = read_attention_fork(Path(other), directory_fd=other_fd)
                        except (TypeError, ValueError, SessionError) as exc:
                            raise RetentionPlannerError("malformed attention fork") from exc
                    if _attention_reference(other_fd, root_id, scan):
                        return "published top-level dependent"
            except FileNotFoundError:
                continue
            if meta.parent_session_id == root_id:
                return "published top-level dependent"
            if fork is not None and fork.forked_from_session == root_id:
                return "published top-level dependent"
    return None


def _attention_reference(other_fd, root_id, scan) -> bool:
    with ExitStack() as stack:
        attention_fd = _open_dir(stack, other_fd, "attention")
        if attention_fd is None:
            return False
        with scan.iter_entries(attention_fd) as records:
            for entry in records:
                if not entry.name.endswith(".json"):
                    raise RetentionPlannerError("malformed attention record")
                value = _read_json(attention_fd, entry.name, scan)
                try:
                    record = AttentionRecord.from_dict(value)
                except (TypeError, ValueError, KeyError) as exc:
                    raise RetentionPlannerError("malformed attention record") from exc
                if entry.name != f"{record.id}.json":
                    raise RetentionPlannerError("attention id does not match filename")
                if record.fork_session_id == root_id:
                    return True
    return False


def _authority_store_dirs(root_fd, scan, prefix=()):
    """Yield component paths for every child store directory under a root."""
    if len(prefix) > _MAX_TREE_DEPTH:
        raise RetentionPlannerError("tree is too deep")
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
                    raise RetentionPlannerError("malformed retention reservation")
                try:
                    validate_retention_reservation(value)
                except StorageFormatError as exc:
                    raise RetentionPlannerError("malformed retention reservation") from exc
                if value["session_id"] != root_id:
                    raise RetentionPlannerError("reservation id does not match filename")
                return True
        try:
            info = os.stat(root_id, dir_fd=trash_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            raise RetentionPlannerError("unsafe retention trash evidence")
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
                    raise RetentionPlannerError("malformed pending child link")
                value = _read_json(fd, entry.name, scan)
                if not valid_child_pending_link(value):
                    raise RetentionPlannerError("malformed pending child link")
                return True
    return False


def _classify_inbox_record(record, name, ids) -> bool:
    try:
        record = validate_inbox_record(record, name)
    except (InboxError, TypeError, ValueError) as exc:
        raise RetentionPlannerError("malformed inbox record") from exc
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
