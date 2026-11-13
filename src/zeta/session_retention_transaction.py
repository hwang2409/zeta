"""Dormant crash-recoverable session quarantine transaction.

The transaction is intentionally reachable only through the dormant planner.
It owns retention authority, durable reservation, cross-directory rename,
manifest replay, and bounded tree removal behind one ``advance`` interface.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .attention_forks import remove_attention_source_links
from .attention_records import remove_fork_session_links
from .core.session_files import (
    SessionError,
    SessionReferenceGate,
    atomic_publish_file,
    child_directory,
    read_reference_generation,
)
from .locking import RETENTION_HOME_ID, domain_authority
from .project_errors import ProjectRegistryError
from .project_inbox import InboxError, ProjectInbox
from .project_registry import ProjectRegistry
from .storage_format import (
    StorageFormatError,
    WriterLease,
    mapping_mutations,
    validate_retention_arm,
    validate_retention_manifest,
    validate_retention_reservation,
    validate_retention_state,
)
from .transcript_search.index import TranscriptIndexError, remove_indexed_sessions

_ARM_NAME = "retention-arm.json"
_STATE_NAME = "retention-state.json"
_TRASH_NAME = "session-trash"
_RESERVATIONS_NAME = "reservations"
_TRANSACTIONS_NAME = "transactions"
_MANIFEST_NAME = "manifest.json"
_TREE_NAME = "tree"
_COORDINATOR_DOMAIN = "coordinator"
_MAX_JSON_BYTES = 1 << 20
_MAX_DEPTH = 64
_MAX_TRANSACTIONS = 32


class TransactionError(Exception):
    """The transaction blocked without changing later durable stages."""


class CandidateRejected(TransactionError):
    """Final candidate evidence no longer authorizes quarantine."""


@dataclass(frozen=True)
class Candidate:
    """Final candidate evidence held under its complete shape lease set."""

    session_id: str
    device: int
    inode: int
    descendants: tuple[str, ...]
    bytes: int
    generation: int


class CandidateVerifier(Protocol):
    def __call__(
        self, session_id: str, days: int
    ) -> AbstractContextManager[Candidate]: ...


@dataclass(frozen=True)
class TransactionOutcome:
    """One bounded transaction step observed by the planner."""

    advanced: bool
    blocked: str | None = None
    session_id: str | None = None


class _ChargedScan:
    """The only transaction directory enumerator; never materializes entries."""

    def __init__(self, directory_fd: int, charge: Callable[[int], None]):
        try:
            self._fd = os.open(
                ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            self._entries = os.scandir(self._fd)
        except OSError as exc:
            raise TransactionError("unsafe retention directory") from exc
        self._charge = charge
        self._pending = None

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        return self

    def __next__(self) -> os.DirEntry[str]:
        if self._pending is None:
            try:
                entry = next(self._entries)
                info = entry.stat(follow_symlinks=False)
            except StopIteration:
                raise
            except OSError as exc:
                raise TransactionError("unsafe retention entry") from exc
            self._pending = (entry, info.st_size if stat.S_ISREG(info.st_mode) else 0)
        entry, size = self._pending
        self._charge(size)
        self._pending = None
        return entry

    def close(self) -> None:
        self._entries.close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class _Budget:
    def __init__(self, *, entries: int, deadline_ns: int, clock: Callable[[], int]):
        self.remaining = entries
        self.deadline_ns = deadline_ns
        self.clock = clock

    def charge(self, _size: int = 0) -> None:
        if self.remaining <= 0 or self.clock() >= self.deadline_ns:
            raise TransactionError("retention transaction bound reached")
        self.remaining -= 1


class _QuarantineTransaction:
    """Own one complete recovery-or-quarantine step per planner pass."""

    def __init__(
        self,
        home: Path,
        lease: WriterLease,
        *,
        entries: int,
        worker_ns: int,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        self.home = Path(home)
        self.lease = lease
        self.entries = entries
        self.worker_ns = worker_ns
        self.clock = clock
        self.project_ids: tuple[str, ...] = ()

    def advance(
        self,
        *,
        days: int,
        preview: dict | None,
        verify: CandidateVerifier,
    ) -> TransactionOutcome:
        """Recover first, otherwise quarantine one arm-authorized candidate."""
        try:
            import asyncio

            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise SessionError("retention transaction requested on an asyncio event-loop thread")
        budget = _Budget(
            entries=self.entries,
            deadline_ns=self.clock() + self.worker_ns,
            clock=self.clock,
        )
        try:
            self.project_ids = ()
            if _entry_exists(self.home / _TRASH_NAME, _RESERVATIONS_NAME):
                registry = ProjectRegistry(
                    self.home / "projects", _storage_lease=self.lease
                )
                self.project_ids = tuple(
                    project.project_id for project in registry.list_projects()
                )
            with domain_authority(
                self.home, RETENTION_HOME_ID, _COORDINATOR_DOMAIN, exclusive=True
            ), SessionReferenceGate(self.home, self.lease).exclusive():
                recovered = self._recover_one(budget)
                if recovered is not None:
                    return recovered
                return self._start_one(days, preview, verify)
        except (
            OSError, ProjectRegistryError, StorageFormatError, TransactionError
        ) as exc:
            return TransactionOutcome(False, _bounded(str(exc)))

    def _recover_one(self, budget: _Budget) -> TransactionOutcome | None:
        trash = self.home / _TRASH_NAME
        if not _entry_exists(self.home, _TRASH_NAME) or not _entry_exists(
            trash, _RESERVATIONS_NAME
        ):
            return None
        with (
                self._trash_directories(create=False) as (
                    _trash_fd, reservations_fd, txns_fd
                ),
                _ChargedScan(reservations_fd, budget.charge) as entries,
            ):
                    for entry in entries:
                        if entry.name.startswith("."):
                            continue
                        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                            raise TransactionError("unsafe retention reservation")
                        reservation = _read_validated_json(
                            reservations_fd, entry.name,
                            validate_retention_reservation,
                        )
                        return self._recover_reservation(
                            reservations_fd, txns_fd, entry.name, reservation, budget
                        )
        return None

    def _recover_reservation(
        self, reservations_fd: int, txns_fd: int, name: str,
        reservation: dict, budget: _Budget,
    ) -> TransactionOutcome:
        expected_name = f"{reservation['session_id']}.json"
        if name != expected_name:
            raise TransactionError("retention reservation name mismatch")
        txid = reservation["transaction_id"]
        _hex_id(txid, "transaction id")
        _component(reservation["session_id"], "session id")
        if reservation["source_path"] != f"sessions/{reservation['session_id']}":
            raise TransactionError("retention reservation source mismatch")
        if reservation["trash_path"] != f"session-trash/transactions/{txid}/tree":
            raise TransactionError("retention reservation trash mismatch")
        if txns_fd is None:
            self.lease.delete(reservations_fd, name)
            return TransactionOutcome(True, session_id=reservation["session_id"])
        try:
            txn_fd = os.open(
                txid, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=txns_fd
            )
        except FileNotFoundError:
            self.lease.delete(reservations_fd, name)
            return TransactionOutcome(True, session_id=reservation["session_id"])
        try:
            try:
                tree_fd = os.open(
                    _TREE_NAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=txn_fd,
                )
            except FileNotFoundError:
                manifest = _read_optional_validated_json(
                    txn_fd, _MANIFEST_NAME, validate_retention_manifest
                )
                if manifest is None:
                    if _remove_empty_transaction(txns_fd, txn_fd, txid):
                        self.lease.delete(reservations_fd, name)
                    return TransactionOutcome(True, session_id=reservation["session_id"])
                _match_manifest(reservation, manifest)
                if manifest["stage"] == "removing":
                    self._publish(
                        txn_fd, _MANIFEST_NAME,
                        {**manifest, "stage": "removed"},
                        validate_retention_manifest,
                    )
                    return TransactionOutcome(True, session_id=reservation["session_id"])
                if manifest["stage"] != "removed":
                    raise TransactionError("retention tree disappeared before removal")
                self.lease.delete(txn_fd, _MANIFEST_NAME)
                if _remove_empty_transaction(txns_fd, txn_fd, txid):
                    self.lease.delete(reservations_fd, name)
                return TransactionOutcome(True, session_id=reservation["session_id"])
            try:
                info = os.fstat(tree_fd)
                if (info.st_dev, info.st_ino) != (
                    reservation["device"], reservation["inode"]
                ):
                    raise TransactionError("retention trash identity mismatch")
                manifest = _read_optional_validated_json(
                    txn_fd, _MANIFEST_NAME, validate_retention_manifest
                )
                manifest_missing = manifest is None
                if manifest_missing:
                    descendants, retained_bytes = _scan_quarantined_tree(
                        tree_fd, reservation["session_id"], budget
                    )
                    manifest = {
                        **_manifest_from_reservation(reservation),
                        "descendants": list(descendants),
                        "bytes": retained_bytes,
                        "generation": read_reference_generation(self.home),
                    }
            finally:
                os.close(tree_fd)
            if manifest_missing:
                self._publish(txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest)
                return TransactionOutcome(True, session_id=reservation["session_id"])
            _match_manifest(reservation, manifest)
            return self._advance_manifest(
                reservations_fd, txns_fd, txn_fd, name, manifest, budget
            )
        finally:
            os.close(txn_fd)

    def _start_one(
        self, days: int, preview: dict | None, verify: CandidateVerifier
    ) -> TransactionOutcome:
        arm = self._read_arm()
        if arm is None or preview is None or arm["preview_id"] != preview.get("preview_id"):
            return TransactionOutcome(False)
        if preview.get("days") != days or not preview.get("root_ids"):
            return TransactionOutcome(False, "retention arm does not match current preview")
        session_id = preview["root_ids"][0]
        with verify(session_id, days) as candidate:
            txid = uuid.uuid4().hex
            reservation = {
                "schema_version": 1,
                "session_id": session_id,
                "transaction_id": txid,
                "source_path": f"sessions/{session_id}",
                "trash_path": f"session-trash/transactions/{txid}/tree",
                "device": candidate.device,
                "inode": candidate.inode,
            }
            validate_retention_reservation(reservation)
            with self._trash_directories(create=True) as (trash_fd, reservations_fd, txns_fd):
                os.mkdir(txid, mode=0o700, dir_fd=txns_fd)
                os.fsync(txns_fd)
                txn_fd = os.open(
                    txid, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=txns_fd,
                )
                try:
                    self._publish(
                        reservations_fd, f"{session_id}.json", reservation,
                        validate_retention_reservation,
                    )
                    home_fd = _open_home(self.home)
                    try:
                        sessions_fd = os.open(
                            "sessions", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=home_fd,
                        )
                        try:
                            os.rename(
                                session_id, _TREE_NAME,
                                src_dir_fd=sessions_fd, dst_dir_fd=txn_fd,
                            )
                            os.fsync(sessions_fd)
                            os.fsync(txn_fd)
                        finally:
                            os.close(sessions_fd)
                    finally:
                        os.close(home_fd)
                    manifest = {
                        **_manifest_from_reservation(reservation),
                        "descendants": list(candidate.descendants),
                        "bytes": candidate.bytes,
                        "generation": candidate.generation,
                    }
                    self._publish(
                        txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest
                    )
                finally:
                    os.close(txn_fd)
                os.fsync(trash_fd)
        return TransactionOutcome(True, session_id=session_id)

    def _advance_manifest(
        self, reservations_fd, txns_fd, txn_fd, reservation_name,
        manifest, budget,
    ) -> TransactionOutcome:
        session_id = manifest["session_id"]
        if _entry_exists(self.home / "sessions", session_id):
            if manifest["stage"] != "id_republished":
                manifest = {**manifest, "stage": "id_republished"}
                self._publish(txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest)
            return TransactionOutcome(False, "id_republished", session_id)
        stages = ("quarantined", "projects", "inbox", "attention", "indexes")
        if manifest["stage"] in stages:
            self._cleanup_stage(manifest["stage"], manifest, budget)
            following = (
                stages[stages.index(manifest["stage"]) + 1]
                if manifest["stage"] != "indexes"
                else "references-clean"
            )
            manifest = {**manifest, "stage": following}
            self._publish(
                txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest
            )
            return TransactionOutcome(True, session_id=session_id)
        if manifest["stage"] == "references-clean":
            manifest = {**manifest, "stage": "removing"}
            self._publish(txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest)
            return TransactionOutcome(True, session_id=session_id)
        if manifest["stage"] == "removing":
            if not self._remove_tree_step(txn_fd, manifest, budget):
                return TransactionOutcome(True, session_id=session_id)
            manifest = {**manifest, "stage": "removed"}
            self._publish(txn_fd, _MANIFEST_NAME, manifest, validate_retention_manifest)
            return TransactionOutcome(True, session_id=session_id)
        if manifest["stage"] == "removed":
            self.lease.delete(txn_fd, _MANIFEST_NAME)
            if not _remove_empty_transaction(
                txns_fd, txn_fd, manifest["transaction_id"]
            ):
                raise TransactionError("retention transaction directory is not empty")
            self.lease.delete(reservations_fd, reservation_name)
            return TransactionOutcome(True, session_id=session_id)
        return TransactionOutcome(False, "retention transaction is blocked", session_id)

    def _cleanup_stage(self, stage: str, manifest: dict, budget: _Budget) -> None:
        candidate_ids = (manifest["session_id"], *manifest["descendants"])
        budget.charge()
        registry = ProjectRegistry(
            self.home / "projects", _storage_lease=self.lease
        )
        try:
            if stage == "quarantined":
                registry.remove_session_links(
                    self.project_ids, candidate_ids, charge=budget.charge
                )
            elif stage == "projects":
                ProjectInbox(
                    registry, sessions_root=self.home / "sessions"
                ).remove_terminal_session_references(
                    self.project_ids, candidate_ids, charge=budget.charge
                )
            elif stage == "inbox":
                remove_fork_session_links(
                    self.home / "sessions",
                    candidate_ids,
                    writer_epoch=self.lease.observed_epoch,
                    charge=budget.charge,
                )
                remove_attention_source_links(
                    self.home / "sessions", candidate_ids, charge=budget.charge
                )
            elif stage == "attention":
                remove_indexed_sessions(
                    registry.root,
                    self.project_ids,
                    candidate_ids,
                    charge=budget.charge,
                )
            elif stage != "indexes":
                raise TransactionError("unknown retention cleanup stage")
        except (
            InboxError,
            ProjectRegistryError,
            TranscriptIndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise TransactionError(str(exc)) from exc

    def _remove_tree_step(self, txn_fd: int, manifest: dict, budget: _Budget) -> bool:
        home_fd = _open_home(self.home)
        try:
            state = _read_optional_validated_json(
                home_fd, _STATE_NAME, validate_retention_state
            )
        finally:
            os.close(home_fd)
        if state is None or state.get("phase") != "removing" or state.get("root_id") != manifest["session_id"]:
            state = _removal_state(manifest)
        frontier = [tuple(path) for path in state["frontier"]]
        if not frontier:
            return True
        path = frontier[-1]
        try:
            directory_fd = _descend(txn_fd, (_TREE_NAME, *path))
        except FileNotFoundError:
            frontier.pop()
            self._publish_state({**state, "frontier": [list(p) for p in frontier]})
            return not frontier
        try:
            first = None
            with _ChargedScan(directory_fd, budget.charge) as entries:
                for first in entries:
                    break
            if first is not None:
                info = first.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if len(path) >= _MAX_DEPTH:
                        raise TransactionError("retention trash is too deep")
                    frontier.append((*path, first.name))
                    self._publish_state({**state, "frontier": [list(p) for p in frontier]})
                    return False
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise TransactionError("unsafe retention trash entry")
                self._publish_state(state)
                os.unlink(first.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                return False
        finally:
            os.close(directory_fd)
        if path:
            parent_fd = _descend(txn_fd, (_TREE_NAME, *path[:-1]))
            try:
                os.rmdir(path[-1], dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        else:
            os.rmdir(_TREE_NAME, dir_fd=txn_fd)
            os.fsync(txn_fd)
        frontier.pop()
        self._publish_state({**state, "frontier": [list(p) for p in frontier]})
        return not frontier

    def _read_arm(self) -> dict | None:
        home_fd = _open_home(self.home)
        try:
            return _read_optional_validated_json(home_fd, _ARM_NAME, validate_retention_arm)
        finally:
            os.close(home_fd)

    def _publish(self, directory_fd, name, payload, validate) -> None:
        validate(payload)
        self.lease.publish_json(
            directory_fd, name, mapping_mutations(payload),
            lambda fd, filename, data: atomic_publish_file(fd, filename, data),
            initial={},
        )

    def _publish_state(self, state: dict) -> None:
        validate_retention_state(state)
        home_fd = _open_home(self.home)
        try:
            self._publish(home_fd, _STATE_NAME, state, validate_retention_state)
        finally:
            os.close(home_fd)

    def _trash_directories(self, *, create: bool):
        stack = ExitStack()
        try:
            home_fd = _open_home(self.home)
            stack.callback(os.close, home_fd)
            trash_fd = stack.enter_context(
                _durable_child_directory(home_fd, _TRASH_NAME, create=create)
            )
            reservations_fd = stack.enter_context(
                _durable_child_directory(trash_fd, _RESERVATIONS_NAME, create=create)
            )
            try:
                txns_fd = stack.enter_context(
                    _durable_child_directory(
                        trash_fd, _TRANSACTIONS_NAME, create=create
                    )
                )
            except FileNotFoundError:
                txns_fd = None
            return _StackedDirectories(stack, trash_fd, reservations_fd, txns_fd)
        except BaseException:
            stack.close()
            raise


class _StackedDirectories:
    def __init__(self, stack, trash_fd, reservations_fd, txns_fd):
        self.stack = stack
        self.value = (trash_fd, reservations_fd, txns_fd)

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        self.stack.close()


def _open_home(home: Path) -> int:
    return os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


@contextmanager
def _durable_child_directory(parent_fd: int, name: str, *, create: bool):
    existed = True
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        else:
            existed = False
    with child_directory(parent_fd, name) as directory_fd:
        if not existed:
            os.fsync(parent_fd)
        yield directory_fd


def _entry_exists(parent: Path, name: str) -> bool:
    try:
        parent_fd = _open_home(parent)
    except FileNotFoundError:
        return False
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent_fd)


def _component(value: str, what: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\x00" in value:
        raise TransactionError(f"invalid retention {what}")


def _hex_id(value: str, what: str) -> None:
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise TransactionError(f"invalid retention {what}")


def _read_validated_json(directory_fd, name, validate) -> dict:
    value = _read_optional_validated_json(directory_fd, name, validate)
    if value is None:
        raise TransactionError(f"missing retention record: {name}")
    return value


def _read_optional_validated_json(directory_fd, name, validate) -> dict | None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > _MAX_JSON_BYTES
        ):
            raise TransactionError(f"unsafe retention record: {name}")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise TransactionError(f"malformed retention record: {name}") from exc
    finally:
        if fd != -1:
            os.close(fd)
    if type(value) is not dict:
        raise TransactionError(f"malformed retention record: {name}")
    try:
        validate(value)
    except StorageFormatError as exc:
        raise TransactionError(f"malformed retention record: {name}") from exc
    return value


def _manifest_from_reservation(reservation: dict) -> dict:
    return {
        "schema_version": 1,
        "session_id": reservation["session_id"],
        "transaction_id": reservation["transaction_id"],
        "device": reservation["device"],
        "inode": reservation["inode"],
        "descendants": [],
        "bytes": 0,
        "stage": "quarantined",
        "generation": 0,
    }


def _scan_quarantined_tree(
    tree_fd: int,
    root_id: str,
    budget: _Budget,
    path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], int]:
    """Rebuild the manifest evidence lost by a crash before publication."""
    if len(path) > _MAX_DEPTH:
        raise TransactionError("retention trash is too deep")
    descendants = []
    retained_bytes = 0
    with _ChargedScan(tree_fd, budget.charge) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            child_path = (*path, entry.name)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=tree_fd,
                )
                try:
                    nested, nested_bytes = _scan_quarantined_tree(
                        child_fd, root_id, budget, child_path
                    )
                finally:
                    os.close(child_fd)
                descendants.extend(nested)
                retained_bytes += nested_bytes
                if path and path[-1] == "agents":
                    child_id = ":".join(part for part in child_path if part != "agents")
                    descendants.append(f"{root_id}:{child_id}")
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                retained_bytes += info.st_size
            else:
                raise TransactionError("unsafe retention trash entry")
    return tuple(sorted(descendants)), retained_bytes


def _match_manifest(reservation: dict, manifest: dict) -> None:
    for name in ("session_id", "transaction_id", "device", "inode"):
        if reservation[name] != manifest[name]:
            raise TransactionError("retention manifest does not match reservation")


def _removal_state(manifest: dict) -> dict:
    return {
        "root_id": manifest["session_id"], "device": manifest["device"],
        "inode": manifest["inode"], "cutoff": "0", "walk_started_at": "0",
        "lease_epoch": manifest["transaction_id"], "frontier": [[]],
        "completed_directories": [], "max_activity": 0, "bytes": manifest["bytes"],
        "descendant_ids": list(manifest["descendants"]), "entry_count": 0,
        "reference_cursor": {}, "start_generation": manifest["generation"],
        "references": {}, "phase": "removing", "preview": None,
    }


def _descend(parent_fd: int, components: tuple[str, ...]) -> int:
    fd = os.dup(parent_fd)
    try:
        for component in components:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=fd,
            )
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _remove_empty_transaction(txns_fd: int, txn_fd: int, txid: str) -> bool:
    try:
        os.rmdir(txid, dir_fd=txns_fd)
    except OSError as exc:
        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            return False
        raise
    os.fsync(txns_fd)
    return True


def _bounded(message: str) -> str:
    return message[:120]
