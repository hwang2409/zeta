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
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

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
from .storage_format import (
    StorageFeature,
    StorageFormatError,
    WriterLease,
    feature_enabled,
    validate_retention_state,
)

RETENTION_PIN_FILENAME = ".retention-pin.json"
_MAX_PIN_BYTES = 256
_PIN_DATA = b'{"pinned":true,"schema_version":1}\n'


def is_retention_pinned(directory_fd: int) -> bool:
    """Return pin state, treating every unsafe present sidecar as pinned."""
    try:
        fd = open_session_file(directory_fd, RETENTION_PIN_FILENAME, os.O_RDONLY)
    except FileNotFoundError:
        try:
            return any(
                name.startswith(f".{RETENTION_PIN_FILENAME}.")
                and name.endswith(".tmp")
                for name in os.listdir(directory_fd)
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
_SENT_TRACKING_FILE = "project-inbox-sent.json"
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


def _read_json(directory_fd: int, name: str) -> object | None:
    """Read one bounded regular JSON file, or None when it is absent.

    Any unsafe entry (symlink, nonregular, multi-link, oversized, malformed)
    raises so the caller can treat the root conservatively.
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
    import time

    return time.time_ns()


class RetentionPlanner:
    """A bounded, resumable, dormant preview planner for one home.

    One instance owns a single in-process sweep identity (``lease_epoch``). A new
    instance models a process restart: it cannot resume another instance's
    authoritative walk and resets that root to ``discover``.
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
        roots = self._discover(home_fd)
        state = _load_state(home_fd)
        state = self._resume_or_start(state, roots, days)
        if not roots:
            return PassResult(
                enabled=True,
                complete=True,
                preview=PreviewSnapshot(
                    preview_id=uuid.uuid4().hex, days=days,
                    cutoff=_cutoff_text(self._now_ns(), days),
                    root_ids=(), count=0, bytes=0,
                    created_at=str(self._now_ns()),
                ),
            )
        deadline = self._now_ns() + self.bounds.worker_ns
        reasons: list[RootOutcome] = []
        total_entries = 0
        advanced = 0
        cursor = state["root_id"]
        if cursor not in roots:
            cursor = roots[0]
            state = _reset_root(state, cursor)
        while True:
            outcome, state, total_entries, done = self._step_root(
                home_fd, roots, state, days, deadline, total_entries
            )
            if not done:
                _store_state(home_fd, state)
                return PassResult(
                    enabled=True, complete=False, in_progress_root=cursor,
                    reasons=tuple(reasons),
                )
            reasons.append(outcome)
            if outcome.closed:
                state = _accumulate(state, outcome)
            advanced += 1
            nxt = _next_root(roots, cursor)
            if nxt is None:
                _store_state(home_fd, state)
                return PassResult(
                    enabled=True, complete=True, reasons=tuple(reasons),
                    preview=_preview_from_state(state),
                )
            cursor = nxt
            state = _reset_root(state, cursor)
            if advanced >= self.bounds.roots_advanced or total_entries >= self.bounds.entries_total:
                _store_state(home_fd, state)
                return PassResult(
                    enabled=True, complete=False, in_progress_root=cursor,
                    reasons=tuple(reasons),
                )

    def _discover(self, home_fd: int) -> list[str]:
        """List bounded candidate root ids by sorted id (no eligibility granted)."""
        try:
            with child_directory(home_fd, "sessions") as sessions_fd:
                names = sorted(os.listdir(sessions_fd))
                roots: list[str] = []
                for name in names:
                    if name.startswith("."):
                        continue
                    try:
                        info = os.stat(name, dir_fd=sessions_fd, follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        roots.append(name)
                    if len(roots) >= self.bounds.metadata_candidates:
                        break
                return roots
        except FileNotFoundError:
            return []

    def _resume_or_start(self, state: dict | None, roots: list[str], days: int) -> dict:
        cutoff_anchor = self._now_ns()
        if (
            state is not None
            and state.get("lease_epoch") == self._lease_epoch
            and state.get("cutoff") == _cutoff_text(_walk_anchor(state), days)
        ):
            return state
        first = roots[0] if roots else ""
        return _fresh_state(first, cutoff_anchor, days, self._lease_epoch)

    def _step_root(
        self, home_fd, roots, state, days, deadline, total_entries
    ):
        root_id = state["root_id"]
        try:
            with (
                child_directory(home_fd, "sessions") as sessions_fd,
                self._lease_tree(sessions_fd, root_id) as (root_fd, store_dirs),
            ):
                return self._advance_root(
                    home_fd, sessions_fd, root_fd, store_dirs, roots,
                    state, days, deadline, total_entries,
                )
        except SessionInUseError:
            return (
                RootOutcome(root_id, False, "session is open or in use"),
                _reset_root(state, root_id), total_entries, True,
            )
        except SessionNotFoundError:
            return (
                RootOutcome(root_id, False, "session disappeared"),
                _reset_root(state, root_id), total_entries, True,
            )
        except RetentionPlannerError as exc:
            return (
                RootOutcome(root_id, False, _bounded_reason(str(exc))),
                _reset_root(state, root_id), total_entries, True,
            )
        except SessionError as exc:
            return (
                RootOutcome(root_id, False, _bounded_reason(str(exc))),
                _reset_root(state, root_id), total_entries, True,
            )

    @contextmanager
    def _lease_tree(self, sessions_fd, root_id):
        """Hold an exclusive lease on the root and every child store.

        A shared lease held by any open store (the root or a child) blocks the
        exclusive acquisition, so a process holding the tree is proven live.
        """
        with ExitStack() as stack:
            try:
                root_fd = stack.enter_context(child_directory(sessions_fd, root_id))
            except FileNotFoundError as exc:
                raise SessionNotFoundError(f"session {root_id} was not found") from exc
            store_dirs = [(), *_iter_store_dirs(root_fd)]
            for comps in store_dirs:
                dfd = stack.enter_context(_descend(root_fd, comps))
                try:
                    fcntl.flock(dfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise SessionInUseError("session is open or in use") from exc
                if os.fstat(dfd).st_nlink == 0:
                    raise SessionNotFoundError(f"session {root_id} was not found")
                self._lease_file(stack, dfd, _STORE_LOCK_NAME)
                self._lease_file(stack, dfd, _RUNTIME_LEASE_NAME)
            yield root_fd, store_dirs

    @staticmethod
    def _lease_file(stack, directory_fd, name):
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RetentionPlannerError(f"unsafe store lock: {name}") from exc
        stack.callback(os.close, fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SessionInUseError("session is open or in use") from exc

    def _advance_root(
        self, home_fd, sessions_fd, root_fd, store_dirs, roots,
        state, days, deadline, total_entries,
    ):
        root_id = state["root_id"]
        info = os.fstat(root_fd)
        anchor = _walk_anchor(state)
        cutoff_ns = anchor - days * _NS_PER_DAY
        if state["phase"] == "discover" or (state["device"], state["inode"]) not in (
            (0, 0), (info.st_dev, info.st_ino)
        ):
            state = _reset_root(state, root_id)
        state = {**state, "device": info.st_dev, "inode": info.st_ino, "phase": "walk"}
        if not self._fingerprints_intact(root_fd, state):
            state = {**state, "device": info.st_dev, "inode": info.st_ino,
                     "frontier": [[]], "completed_directories": [], "max_activity": 0,
                     "bytes": 0, "descendant_ids": [], "entry_count": 0, "phase": "walk"}
        state, total_entries, done = self._walk(root_fd, state, deadline, total_entries)
        if not done:
            return None, state, total_entries, False
        effective = min(state["max_activity"], anchor)
        generation = read_reference_generation(self.home)
        state = {**state, "start_generation": generation}
        reason = self._closure_reason(
            home_fd, sessions_fd, root_fd, store_dirs, root_id,
            state["descendant_ids"], effective, cutoff_ns,
        )
        if reason is None and read_reference_generation(self.home) != generation:
            # A reference transaction raced the scan; stay conservative and retry.
            reason = "references changed during scan"
        state = {**state, "phase": "ready"}
        return (
            RootOutcome(root_id, reason is None, reason, state["bytes"]),
            state, total_entries, True,
        )

    def _fingerprints_intact(self, root_fd, state) -> bool:
        for relpath, dev, ino, mtime in state["completed_directories"]:
            comps = tuple(p for p in relpath.split("/") if p)
            try:
                with _descend(root_fd, comps) as dfd:
                    cur = os.fstat(dfd)
            except (FileNotFoundError, SessionError, OSError):
                return False
            if (cur.st_dev, cur.st_ino, cur.st_mtime_ns) != (dev, ino, mtime):
                return False
        return True

    def _walk(self, root_fd, state, deadline, total_entries):
        frontier = [list(p) for p in state["frontier"]]
        completed = [list(e) for e in state["completed_directories"]]
        max_activity = state["max_activity"]
        total_bytes = state["bytes"]
        entry_count = state["entry_count"]
        descendants = list(state["descendant_ids"])
        pass_entries = 0
        while frontier:
            if (
                pass_entries >= self.bounds.entries_per_root
                or total_entries >= self.bounds.entries_total
                or self._now_ns() >= deadline
            ):
                return (
                    _with_walk(state, frontier, completed, max_activity,
                               total_bytes, descendants, entry_count),
                    total_entries, False,
                )
            comps = frontier.pop()
            if len(comps) > _MAX_TREE_DEPTH:
                raise RetentionPlannerError("tree is too deep")
            with _descend(root_fd, tuple(comps)) as dfd:
                dir_info = os.fstat(dfd)
                max_activity = max(max_activity, dir_info.st_mtime_ns)
                if comps[:1] == [_AGENTS_DIRNAME] and len(comps) == 2:
                    descendants.append("/".join(comps))
                for name in sorted(os.listdir(dfd)):
                    child = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                    mode = child.st_mode
                    if stat.S_ISDIR(mode):
                        frontier.append(comps + [name])
                    elif stat.S_ISREG(mode):
                        if child.st_nlink != 1:
                            raise RetentionPlannerError("hard-linked file in tree")
                        total_bytes += child.st_size
                        max_activity = max(max_activity, child.st_mtime_ns)
                        entry_count += 1
                        total_entries += 1
                        pass_entries += 1
                    else:
                        raise RetentionPlannerError("unsafe entry in tree")
                completed.append(["/".join(comps), dir_info.st_dev,
                                  dir_info.st_ino, dir_info.st_mtime_ns])
        return (
            _with_walk(state, [], completed, max_activity, total_bytes,
                       descendants, entry_count),
            total_entries, True,
        )

    def _closure_reason(
        self, home_fd, sessions_fd, root_fd, store_dirs, root_id,
        descendant_ids, effective_activity, cutoff_ns,
    ):
        """Return the first conservative blocker, or None when the root is closed."""
        ids = {root_id, *descendant_ids}
        if _reservation_present(home_fd, root_id):
            return "session is being deleted"
        for comps in store_dirs:
            with _descend(root_fd, tuple(comps)) as dfd:
                if _read_json(dfd, _PENDING_LINK_NAME) is not None:
                    return "pending project link"
                if _directory_nonempty(dfd, _PENDING_CHILD_LINKS_DIRNAME):
                    return "pending child link"
                if comps and not _child_lifecycle_terminal(dfd):
                    return "live or recoverable child"
        if _sent_tracker_nonterminal(root_fd):
            return "unacknowledged sent message"
        if self._pinned(sessions_fd, root_fd, root_id):
            return "pinned"
        inbox_reason = self._inbox_reference(home_fd, ids)
        if inbox_reason is not None:
            return inbox_reason
        if self._published_dependent(sessions_fd, root_id):
            return "published top-level dependent"
        if effective_activity >= cutoff_ns:
            return "not old enough"
        return None

    def _pinned(self, sessions_fd, root_fd, root_id) -> bool:
        if is_retention_pinned(root_fd):
            return True
        seen = {root_id}
        current = root_id
        for _ in range(self.bounds.metadata_candidates):
            meta = _read_meta(sessions_fd, current)
            parent = meta.get("parent_session_id") if meta else None
            if not isinstance(parent, str) or parent in seen:
                return False
            seen.add(parent)
            try:
                with child_directory(sessions_fd, parent) as parent_fd:
                    if is_retention_pinned(parent_fd):
                        return True
            except (FileNotFoundError, SessionError, OSError):
                return False
            current = parent
        return False

    def _inbox_reference(self, home_fd, ids) -> str | None:
        records = 0
        try:
            projects_fd = os.open(
                "projects", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RetentionPlannerError("unsafe projects directory") from exc
        try:
            for project in sorted(os.listdir(projects_fd)):
                try:
                    inbox_fd = os.open(
                        f"{project}/inbox",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=projects_fd,
                    )
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except OSError as exc:
                    raise RetentionPlannerError("unsafe inbox directory") from exc
                try:
                    for status in ("new", "claimed"):
                        try:
                            status_fd = os.open(
                                status, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=inbox_fd,
                            )
                        except (FileNotFoundError, NotADirectoryError):
                            continue
                        try:
                            for name in sorted(os.listdir(status_fd)):
                                if not name.endswith(".json"):
                                    continue
                                records += 1
                                if records > self.bounds.reference_records_total:
                                    raise RetentionPlannerError("too many inbox records")
                                record = _read_json(status_fd, name)
                                if _inbox_names(record, ids):
                                    return "pending inbox reference"
                        finally:
                            os.close(status_fd)
                finally:
                    os.close(inbox_fd)
        finally:
            os.close(projects_fd)
        return None

    def _published_dependent(self, sessions_fd, root_id) -> bool:
        scanned = 0
        for other in sorted(os.listdir(sessions_fd)):
            if other == root_id or other.startswith("."):
                continue
            scanned += 1
            if scanned > self.bounds.metadata_candidates:
                break
            meta = _read_meta(sessions_fd, other)
            if meta and meta.get("parent_session_id") == root_id:
                return True
            try:
                with child_directory(sessions_fd, other) as other_fd:
                    fork = _read_json(other_fd, _ATTENTION_FORK_NAME)
            except (FileNotFoundError, SessionError, OSError):
                continue
            if isinstance(fork, dict) and fork.get("forked_from_session") == root_id:
                return True
        return False


_descend = descend_directory


def _iter_store_dirs(root_fd, prefix=()):
    """Yield component paths for every child store directory under a root."""
    if len(prefix) > _MAX_TREE_DEPTH:
        return
    try:
        with _descend(root_fd, (*prefix, _AGENTS_DIRNAME)) as agents_fd:
            entries = []
            for name in sorted(os.listdir(agents_fd)):
                try:
                    info = os.stat(name, dir_fd=agents_fd, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    entries.append(name)
    except (FileNotFoundError, SessionError, NotADirectoryError):
        return
    except OSError as exc:
        raise RetentionPlannerError("unsafe agents directory") from exc
    for name in entries:
        comps = (*prefix, _AGENTS_DIRNAME, name)
        yield comps
        yield from _iter_store_dirs(root_fd, comps)


def _read_meta(sessions_fd, session_id):
    try:
        with child_directory(sessions_fd, session_id) as session_fd:
            value = _read_json(session_fd, _METADATA_NAME)
    except (FileNotFoundError, SessionError, OSError):
        return None
    except RetentionPlannerError:
        return None
    return value if isinstance(value, dict) else None


def _reservation_present(home_fd, root_id) -> bool:
    for relative in (
        f"session-trash/reservations/{root_id}.json",
        f"session-trash/{root_id}",
    ):
        try:
            os.stat(relative, dir_fd=home_fd, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            raise RetentionPlannerError("unsafe trash entry") from exc
        return True
    return False


def _directory_nonempty(directory_fd, name) -> bool:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise RetentionPlannerError(f"unsafe directory: {name}") from exc
    try:
        return any(not entry.startswith(".") for entry in os.listdir(fd))
    finally:
        os.close(fd)


def _child_lifecycle_terminal(directory_fd) -> bool:
    value = _read_json(directory_fd, _LIFECYCLE_NAME)
    if value is None or type(value) is not dict:
        return False
    return value.get("state") in _TERMINAL_AGENT_STATES


def _sent_tracker_nonterminal(root_fd) -> bool:
    value = _read_json(root_fd, _SENT_TRACKING_FILE)
    if value is None:
        return False
    if type(value) is not dict:
        raise RetentionPlannerError("malformed sent tracker")
    records = value.get("records")
    if not isinstance(records, list):
        return False
    return any(not _record_terminal(record) for record in records)


def _record_terminal(record) -> bool:
    if type(record) is not dict:
        return False
    if record.get("publication") != "published":
        return False
    reported = record.get("reported")
    if isinstance(reported, dict):
        return reported.get("status") == "done"
    return reported == "done"


def _inbox_names(record, ids) -> bool:
    if type(record) is not dict:
        return False
    sender = record.get("from")
    if isinstance(sender, dict) and sender.get("session") in ids:
        return True
    return record.get("to_session") in ids or record.get("claimer_session") in ids


def _walk_anchor(state) -> int:
    try:
        return int(state["walk_started_at"])
    except (KeyError, ValueError, TypeError):
        return 0


def _cutoff_text(anchor_ns: int, days: int) -> str:
    return str(anchor_ns - days * _NS_PER_DAY)


def _fresh_state(root_id: str, anchor_ns: int, days: int, lease_epoch: str) -> dict:
    return {
        "root_id": root_id or "pending",
        "device": 0,
        "inode": 0,
        "cutoff": _cutoff_text(anchor_ns, days),
        "walk_started_at": str(anchor_ns),
        "lease_epoch": lease_epoch,
        "frontier": [[]],
        "completed_directories": [],
        "max_activity": 0,
        "bytes": 0,
        "descendant_ids": [],
        "entry_count": 0,
        "reference_cursor": {},
        "start_generation": 0,
        "references": {},
        "phase": "discover",
        "preview": {
            "preview_id": uuid.uuid4().hex,
            "days": days,
            "cutoff": _cutoff_text(anchor_ns, days),
            "root_ids": [],
            "count": 0,
            "bytes": 0,
            "created_at": str(anchor_ns),
        },
    }


def _reset_root(state: dict, root_id: str) -> dict:
    return {
        **state,
        "root_id": root_id or "pending",
        "device": 0,
        "inode": 0,
        "frontier": [[]],
        "completed_directories": [],
        "max_activity": 0,
        "bytes": 0,
        "descendant_ids": [],
        "entry_count": 0,
        "reference_cursor": {},
        "references": {},
        "phase": "discover",
    }


def _with_walk(state, frontier, completed, max_activity, total_bytes, descendants, entry_count):
    return {
        **state,
        "frontier": [list(p) for p in frontier],
        "completed_directories": [list(e) for e in completed],
        "max_activity": max_activity,
        "bytes": total_bytes,
        "descendant_ids": list(descendants),
        "entry_count": entry_count,
        "phase": "walk",
    }


def _next_root(roots, cursor):
    try:
        index = roots.index(cursor)
    except ValueError:
        return roots[0] if roots else None
    return roots[index + 1] if index + 1 < len(roots) else None


def _accumulate(state: dict, outcome: RootOutcome) -> dict:
    preview = dict(state["preview"]) if state.get("preview") else None
    if preview is None:
        return state
    root_ids = list(preview["root_ids"])
    if outcome.root_id not in root_ids:
        root_ids.append(outcome.root_id)
    preview = {
        **preview,
        "root_ids": root_ids,
        "count": len(root_ids),
        "bytes": preview["bytes"] + outcome.bytes,
    }
    return {**state, "preview": preview}


def _preview_from_state(state: dict) -> PreviewSnapshot | None:
    preview = state.get("preview")
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
