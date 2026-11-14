"""Durable retention-owned session sidecars and the dormant retention planner.

The planner is a library-only, bounded, resumable preview walk. It is reachable
only from tests; no scheduler, startup hook, slash command, setting, or event
calls it. ``test_session_retention`` asserts no production module imports the
planner entry points.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Self

from ..core.session_files import (
    SessionError,
    SessionNotFoundError,
    SessionReferenceGate,
    SessionReferenceUnavailable,
    atomic_publish_file,
    child_directory,
    open_session_file,
    read_reference_generation,
    unlink_session_file,
)
from ..locking import RETENTION_HOME_ID, domain_authority
from ..storage_format import (
    StorageFeature,
    StorageFormatError,
    WriterLease,
    feature_enabled,
    validate_retention_arm,
    validate_retention_history,
    validate_retention_state,
)
from .handshake import planner_request_pending
from .pin import is_retention_pinned
from .transaction import (
    Candidate,
    CandidateRejected,
    TransactionOutcome,
    _QuarantineTransaction,
)

_STATE_FILENAME = "retention-state.json"
_HISTORY_FILENAME = "retention-history.jsonl"
_MAX_HISTORY = 128
_ARM_FILENAME = "retention-arm.json"
_COORDINATOR_DOMAIN = "coordinator"
_ARM_WARNING = "restart every zeta process on every host that pushes or pulls sessions to this home before arming"
_METADATA_NAME = "meta.json"
_SESSION_STATE_NAME = "session_state.json"
_PENDING_LINK_NAME = "project_link_pending.json"
_PENDING_CHILD_LINKS_DIRNAME = "pending_child_links"
_ATTENTION_FORK_NAME = "attention_fork.json"
_STORE_LOCK_NAME = ".lock"
_RUNTIME_LEASE_NAME = "runtime.lease"
_AGENTS_DIRNAME = "agents"
_LIFECYCLE_NAME = "agent_lifecycle.json"
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
    mode: str | None = None
    reasons: tuple[RootOutcome, ...] = ()
    preview: PreviewSnapshot | None = None
    in_progress_root: str | None = None


class RetentionPlannerError(Exception):
    """Raised when the planner cannot proceed safely."""


def retention_result(
    *,
    timestamp: str,
    mode: str,
    days: int,
    root_ids: tuple[str, ...] = (),
    count: int | None = None,
    preview_id: str | None = None,
    cutoff: str | None = None,
    armed_by_zeta_version: str | None = None,
    descendant_count: int = 0,
    bytes: int = 0,
    skip_counts: dict[str, int] | None = None,
    progress_counts: dict[str, int] | None = None,
    errors: tuple[str, ...] = (),
    disable_pending: bool = False,
) -> dict[str, object]:
    """Build the single strict result projection used by all surfaces."""
    result = {
        "timestamp": timestamp, "mode": mode, "days": days,
        "armed_by_zeta_version": armed_by_zeta_version, "preview_id": preview_id,
        "cutoff": cutoff, "root_ids": list(root_ids),
        "count": len(root_ids) if count is None else count,
        "descendant_count": descendant_count, "bytes": bytes,
        "skip_counts": dict(skip_counts or {}),
        "progress_counts": dict(progress_counts or {}),
        "errors": list(errors), "disable_pending": disable_pending,
    }
    validate_retention_history(result)
    return result


def project_retention_result(result: PassResult, *, days: int, timestamp: str) -> dict[str, object]:
    """Project one planner result into the shared client-visible mapping."""
    preview = result.preview
    if not result.enabled:
        return retention_result(timestamp=timestamp, mode="disabled", days=0)
    if result.mode is not None:
        root_ids = tuple(reason.root_id for reason in result.reasons if reason.root_id != "recovery")
        return retention_result(
            timestamp=timestamp, mode=result.mode, days=days,
            root_ids=root_ids,
            errors=tuple(reason.reason for reason in result.reasons if reason.reason),
        )
    if preview is None:
        errors = tuple(reason.reason for reason in result.reasons if reason.reason)
        return retention_result(
            timestamp=timestamp, mode="preview", days=days,
            errors=errors, progress_counts={"roots": len(result.reasons)},
        )
    return retention_result(
        timestamp=timestamp, mode="preview", days=days,
        root_ids=preview.root_ids, count=preview.count, bytes=preview.bytes,
        preview_id=preview.preview_id, cutoff=preview.cutoff,
    )


def append_retention_history(
    home: str | Path, lease: WriterLease, result: Mapping[str, object], *, authority_held: bool = False
) -> None:
    """Durably replace the bounded, validated retention history."""
    if not feature_enabled(lease.observed_epoch, StorageFeature.SESSION_RETENTION_ACTIVATION):
        return
    validate_retention_history(result)
    if not authority_held:
        with domain_authority(Path(home), RETENTION_HOME_ID, _COORDINATOR_DOMAIN, exclusive=True):
            return append_retention_history(home, lease, result, authority_held=True)
    path = Path(home) / _HISTORY_FILENAME
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if type(value) is not dict:
                    raise RetentionPlannerError("malformed retention history")
                validate_retention_history(value)
                rows.append(value)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, StorageFormatError, RetentionPlannerError):
        # A corrupt history row must not strand the activation owner.
        rows = []
    rows = (rows + [dict(result)])[-_MAX_HISTORY:]
    def publish() -> None:
        home_fd = os.open(Path(home), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            atomic_publish_file(
                home_fd, _HISTORY_FILENAME,
                ("".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)).encode(),
            )
        finally:
            os.close(home_fd)
    publish()


def retention_arm_warning() -> str:
    return _ARM_WARNING


@dataclass(frozen=True)
class RetentionAuthorityResult:
    message: str
    projection: dict[str, object]
    applied: bool

    def __str__(self) -> str:
        return self.message


@contextmanager
def _try_authority(root: Path, *, exclusive: bool, deadline: float):
    while True:
        try:
            with domain_authority(
                root, RETENTION_HOME_ID, _COORDINATOR_DOMAIN,
                exclusive=exclusive, nonblocking=True,
            ):
                yield
                return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError
            time.sleep(min(0.01, deadline - time.monotonic()))


def retention_authority(
    home: str | Path, lease: WriterLease, *, days: int,
    zeta_version: str | None = None,
    now: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
) -> RetentionAuthorityResult:
    try:
        import asyncio
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise SessionError("retention authority requested on an asyncio event-loop thread")
    if type(days) is not int or not 0 <= days <= 3650:
        raise RetentionPlannerError("retention days must be in 0..3650")
    empty = retention_result(timestamp=now(), mode="none", days=days)
    if not feature_enabled(lease.observed_epoch, StorageFeature.SESSION_RETENTION_ACTIVATION):
        return RetentionAuthorityResult("retention unavailable: upgrade the home before arming", empty, False)
    root = Path(home)
    try:
        if days == 0:
            with (
                _try_authority(root, exclusive=True, deadline=time.monotonic() + 35),
                SessionReferenceGate(root, lease).exclusive(timeout=10),
            ):
                home_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    unlink_session_file(home_fd, _ARM_FILENAME)
                    unlink_session_file(home_fd, _STATE_FILENAME)
                finally:
                    os.close(home_fd)
                projection = retention_result(timestamp=now(), mode="disarmed", days=0)
                append_retention_history(root, lease, projection, authority_held=True)
                return RetentionAuthorityResult("retention disabled", projection, True)
        with (
            _try_authority(root, exclusive=True, deadline=time.monotonic() + 35),
            SessionReferenceGate(root, lease).exclusive(timeout=10),
        ):
                home_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    state = _load_state(home_fd)
                    preview = state.get("preview") if state else None
                    if not isinstance(preview, dict) or preview.get("days") != days:
                        message = "retention arm requires a completed preview for the current days"
                        projection = retention_result(timestamp=now(), mode="none", days=days, errors=(message,))
                        append_retention_history(root, lease, projection, authority_held=True)
                        return RetentionAuthorityResult(message, projection, False)
                    arm = {
                        "schema_version": 1, "gate_version": 1,
                        "armed_by_zeta_version": zeta_version or package_version("zeta"),
                        "armed_at": now(), "preview_id": preview["preview_id"],
                    }
                    validate_retention_arm(arm)
                    atomic_publish_file(home_fd, _ARM_FILENAME, (json.dumps(arm, separators=(",", ":"), sort_keys=True) + "\n").encode())
                    projection = retention_result(timestamp=now(), mode="armed", days=days, root_ids=tuple(preview.get("root_ids", ())), count=preview.get("count"), bytes=preview.get("bytes", 0), preview_id=preview.get("preview_id"), cutoff=preview.get("cutoff"), armed_by_zeta_version=arm["armed_by_zeta_version"])
                    append_retention_history(root, lease, projection, authority_held=True)
                    return RetentionAuthorityResult(_ARM_WARNING, projection, True)
                finally:
                    os.close(home_fd)
    except (TimeoutError, SessionReferenceUnavailable):
        message = "retention busy, try again" if days else "retention busy; disarm not applied, try again"
        projection = retention_result(
            timestamp=now(), mode="disabled" if days == 0 else "none", days=days,
            errors=(message,), disable_pending=days == 0,
        )
        if days == 0:
            try:
                append_retention_history(root, lease, projection, authority_held=True)
            except (OSError, RetentionPlannerError):
                pass
        return RetentionAuthorityResult(message, projection, False)


def disarm_retention(home: str | Path, lease: WriterLease) -> RetentionAuthorityResult:
    return retention_authority(home, lease, days=0)

def read_retention_history(home: str | Path) -> tuple[dict[str, object], ...]:
    """Read and validate the bounded history projection."""
    try:
        lines = (Path(home) / _HISTORY_FILENAME).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ()
    if len(lines) > _MAX_HISTORY:
        raise RetentionPlannerError("retention history exceeds bound")
    output = []
    for line in lines:
        value = json.loads(line)
        if type(value) is not dict:
            raise RetentionPlannerError("malformed retention history")
        validate_retention_history(value)
        output.append(value)
    return tuple(output)


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
    yield_check: Callable[[], bool] | None = None

    def charge_entry(self, size: int = 0) -> None:
        self.entries += 1
        if self.yield_check is not None and self.yield_check():
            raise _Deferred
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
        """Release all live cursors and leases owned by this planner."""
        self._close_flight()
        if self._sweep is not None and self._sweep.root_scan is not None:
            self._sweep.root_scan.close()
        self._sweep = None

    def _close_flight(self) -> None:
        flight, self._flight = self._flight, None
        if flight is not None:
            if flight.walk.listing is not None:
                flight.walk.listing[1].close()
            flight.stack.close()

    def progress_marker(self) -> tuple[object, ...]:
        """Return the in-memory cursor state used to detect forward progress."""
        sweep = self._sweep
        flight = self._flight
        if sweep is None:
            return ()
        walk = flight.walk if flight is not None else None
        return (
            sweep.days, sweep.current_root, sweep.discovery_done,
            sweep.discovery_blocked, tuple(sweep.selected), sweep.bytes,
            tuple(walk.frontier) if walk is not None else (),
            tuple(walk.fingerprints) if walk is not None else (),
            walk.entries if walk is not None else 0,
            walk.bytes if walk is not None else 0,
        )

    @property
    def enabled(self) -> bool:
        return feature_enabled(self.lease.observed_epoch, StorageFeature.SESSION_RETENTION)

    def run_pass(self, *, days: int) -> PassResult:
        """Advance one bounded sweep step and return what it proved."""
        if type(days) is not int or days <= 0 or days > 3650:
            raise RetentionPlannerError("retention days must be in 1..3650")
        if not self.enabled:
            return PassResult(enabled=False, complete=False)
        if planner_request_pending(self.home):
            self.close()
            return PassResult(enabled=True, complete=False)
        home_fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return self._run_pass(home_fd, days)
        finally:
            os.close(home_fd)
            if planner_request_pending(self.home):
                self.close()

    def _run_pass(self, home_fd: int, days: int) -> PassResult:
        if self._latest is _UNLOADED:
            state = _load_state(home_fd)
            self._latest = state["preview"] if state is not None else None
        if type(self._latest) is dict and self._latest.get("days") != days:
            self._latest = None
            self.close()
        transaction = _QuarantineTransaction(
            self.home,
            self.lease,
            entries=self.bounds.entries_total,
            worker_ns=self.bounds.worker_ns,
            clock=self._now_ns,
        )
        outcome = transaction.advance(
            days=days,
            preview=self._latest if type(self._latest) is dict else None,
            verify=self._verify_candidate,
            authority_held=False,
        )
        transaction_result = self._transaction_result(outcome)
        if transaction_result is not None:
            return transaction_result
        budget = _Budget(
            self.bounds, self._now_ns, self._now_ns() + self.bounds.worker_ns,
            yield_check=lambda: planner_request_pending(self.home),
        )
        sweep = self._sweep
        if sweep is None or sweep.days != days:
            self.close()
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
        with domain_authority(
            self.home, RETENTION_HOME_ID, _COORDINATOR_DOMAIN, exclusive=True
        ):
            _store_state(home_fd, self._state_payload())
        return PassResult(
            enabled=True, complete=complete, reasons=tuple(reasons),
            preview=_preview_snapshot(self._latest),
            in_progress_root=None if complete else sweep.current_root,
        )

    def _transaction_result(self, outcome: TransactionOutcome) -> PassResult | None:
        if not outcome.advanced and outcome.blocked is None:
            return None
        if outcome.completed:
            self._latest = None
        reasons = (
            (RootOutcome(outcome.session_id or "recovery", False, outcome.blocked),)
            if outcome.session_id is not None or outcome.blocked is not None
            else ()
        )
        return PassResult(
            enabled=True,
            complete=False,
            mode="deleted" if outcome.completed else ("error" if outcome.blocked else None),
            reasons=reasons,
            preview=_preview_snapshot(self._latest),
        )

    @contextmanager
    def _verify_candidate(self, root_id: str, days: int):
        """Repeat the full closure proof while the transaction owns authority."""
        budget = _Budget(
            self.bounds, self._now_ns, self._now_ns() + self.bounds.worker_ns,
            yield_check=lambda: planner_request_pending(self.home),
        )
        sweep = _Sweep(days, self._now_ns(), current_root=root_id)
        scan = _ClosureBudget(budget)
        home_fd = os.open(self.home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with child_directory(home_fd, "sessions") as sessions_fd:
                flight = self._hold(sessions_fd, root_id, scan)
                if not self._walk(flight, budget):
                    raise CandidateRejected("final candidate walk exceeded its bound")
                reason = self._closure_reason(home_fd, sessions_fd, flight, sweep, scan)
                if reason is not None:
                    raise CandidateRejected(reason)
                generation = read_reference_generation(self.home)
                descendants = tuple(
                    root_id + ":" + ":".join(part for part in comps if part != _AGENTS_DIRNAME)
                    for comps in flight.store_dirs
                    if comps
                )
                yield Candidate(
                    root_id,
                    flight.device,
                    flight.inode,
                    descendants,
                    flight.walk.bytes,
                    generation,
                )
        finally:
            self._close_flight()
            os.close(home_fd)

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
        self._close_flight()
        return RootOutcome(root_id, reason is None, reason, bytes_)

    def _hold(self, sessions_fd, root_id, scan) -> _InFlight:
        """Return the held lease set for ``root_id``, taking it when needed."""
        flight = self._flight
        if flight is not None and flight.root_id == root_id:
            info = os.stat(root_id, dir_fd=sessions_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (flight.device, flight.inode):
                raise RetentionPlannerError("session changed during walk")
            return flight
        self._close_flight()
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
                return True
            current = parent
        return True

    def _state_payload(self) -> dict:
        """Project memory state directly onto the registered shape."""
        payload = {
            "root_id": "pending", "device": 0, "inode": 0, "cutoff": "0",
            "walk_started_at": "0", "lease_epoch": self._lease_epoch,
            "frontier": [[]], "completed_directories": [], "max_activity": 0,
            "bytes": 0, "descendant_ids": [], "entry_count": 0,
            "reference_cursor": {}, "start_generation": 0, "references": {},
            "phase": "discover", "preview": self._latest,
        }
        if self._sweep is not None:
            payload.update(
                cutoff=str(self._sweep.cutoff_ns),
                walk_started_at=str(self._sweep.anchor),
            )
        if self._flight is not None:
            walk = self._flight.walk
            payload.update(
                root_id=self._flight.root_id,
                device=self._flight.device,
                inode=self._flight.inode,
                frontier=[list(c) for c in walk.frontier]
                + ([list(walk.listing[0])] if walk.listing is not None else []),
                completed_directories=[list(f) for f in walk.fingerprints],
                max_activity=walk.max_activity,
                bytes=walk.bytes,
                descendant_ids=["/".join(c) for c in self._flight.store_dirs if c],
                entry_count=walk.entries,
                phase="references" if walk.done else "walk",
            )
        return payload


class _ClosureAuthority:
    """One owner for candidate closure decisions and bounded references."""

    def __init__(self, planner, home_fd, sessions_fd, flight, sweep, scan):
        self.planner = planner
        self.home_fd = home_fd
        self.sessions_fd = sessions_fd
        self.flight = flight
        self.sweep = sweep
        self.scan = scan
        self._store_names = None

    def classify(self) -> str | None:
        root_id = self.flight.root_id
        generation = _generation_value(self.planner.home, self.scan)
        ids = {root_id}
        for comps in self.flight.store_dirs:
            if comps:
                child_parts = [part for part in comps if part != _AGENTS_DIRNAME]
                ids.add(root_id + ":" + ":".join(child_parts))
        checks = (
            lambda: "session is being deleted" if _reservation_present(self.home_fd, root_id, self.scan) else None,
            lambda: _authority_store_reason(self.flight, self.scan),
            lambda: _authority_transcript_recovery_reason(
                self.flight, self._names_by_store()
            ),
            lambda: _authority_memory_reconciliation_reason(
                self.flight, self._names_by_store()
            ),
            lambda: self.planner._authority_sent_reason(self.flight, self.scan),
            lambda: "pinned" if self.planner._pinned(self.sessions_fd, self.flight.root_fd, root_id, self.scan) else None,
            lambda: _authority_inbox_reason(self.home_fd, ids, self.scan),
            lambda: _authority_dependents(self.sessions_fd, root_id, self.scan),
        )
        if (reason := next((item for check in checks if (item := check()) is not None), None)) is not None:
            return reason
        if min(self.flight.walk.max_activity, self.sweep.anchor) >= self.sweep.cutoff_ns:
            return "not old enough"
        if _generation_value(self.planner.home, self.scan) != generation:
            return "references changed during scan"
        return None

    def _names_by_store(self):
        if self._store_names is None:
            stores = []
            for comps in self.flight.store_dirs:
                with (
                    _descend(self.flight.root_fd, comps) as dfd,
                    self.scan.iter_entries(dfd) as entries,
                ):
                    stores.append((comps, [entry.name for entry in entries]))
            self._store_names = stores
        return self._store_names


PlannerError = RetentionPlannerError

from ._helpers import (
    _authority_dependents,
    _authority_inbox_reason,
    _authority_memory_reconciliation_reason,
    _authority_parent_id,
    _authority_store_dirs,
    _authority_store_reason,
    _authority_transcript_recovery_reason,
    _descend,
    _EntryScan,
    _failure_reason,
    _fingerprints_intact,
    _flock_nb,
    _generation_value,
    _lease_file,
    _next_root,
    _preview_snapshot,
    _reservation_present,
    _sent_file_reason,
    _visit,
    _walk_directory,
)
