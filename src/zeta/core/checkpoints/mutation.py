"""Typed, generation-checked conversation mutations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from ..session_execution import SessionOwnedError
from ..session_files import read_session_file, session_directory
from ..store._incremental_validation import TASK_EXITED_NOTIFICATION_KIND
from . import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from .entry_types import sent_message_receipts

if TYPE_CHECKING:
    from ..store._store import ConversationStore


@dataclass(frozen=True, slots=True)
class ConversationAddress:
    project_id: str
    root_session_id: str
    agent_handle: str


@dataclass(frozen=True, slots=True)
class ConversationRow:
    entry_type: str
    data: Mapping[str, object]
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationDedupeKey:
    namespace: str
    id: str


class MutationCancellationPolicy(Enum):
    ABORT_BEFORE_WRITE = "abort_before_write"
    FINALIZE_ONCE_SUBMITTED = "finalize_once_submitted"


class ConversationMutation[ResultT](Protocol):
    """Immutable command accepted by the conversation mutation seam."""

    cancellation_policy: ClassVar[MutationCancellationPolicy]


class _ConditionalStoreMutation[ResultT](Protocol):
    baseline_count: int | None
    baseline_digest: bytes | None
    baseline_patch: Mapping[str, object] | None
    patch_fields: tuple[str, ...]
    accept_external_generation: bool

    def apply(self, store: ConversationStore) -> ResultT: ...


@dataclass(frozen=True, slots=True)
class AppendConversationRows[ResultT]:
    cancellation_policy: ClassVar[MutationCancellationPolicy] = (
        MutationCancellationPolicy.ABORT_BEFORE_WRITE
    )
    rows: tuple[ConversationRow, ...]
    dedupe_key: ConversationDedupeKey | None = None
    result: ResultT | None = None


@dataclass(frozen=True, slots=True)
class CommittedLogCursor:
    identity: tuple[int, int, int] | None
    offset: int
    mtime_ns: int | None
    prefix_fingerprint: bytes | None
    entry_prefix_digest: bytes


@dataclass(frozen=True, slots=True)
class _ResidentPatch:
    values: Mapping[str, object]
    expected: Mapping[str, object]
    replace_entries: bool = False
    raw_entries: bool = False


@dataclass(frozen=True, slots=True)
class CommittedDelta[ResultT]:
    generation_before: int
    generation_after: int
    committed_entries: tuple[ConversationEntry, ...]
    entries_to_install: tuple[ConversationEntry, ...]
    _resident_patch: _ResidentPatch
    result: ResultT
    log_cursor: CommittedLogCursor
    dedupe_key: ConversationDedupeKey | None
    duplicate: bool


class ConversationGenerationConflict(ConversationIntegrityError):
    def __init__(self, expected_generation: int, actual_generation: int) -> None:
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(
            "conversation generation changed "
            f"(expected {expected_generation}, actual {actual_generation})"
        )


class ConversationDedupeConflict(ConversationIntegrityError):
    def __init__(self, key: ConversationDedupeKey) -> None:
        self.key = key
        super().__init__(
            f"conversation dedupe key has different content: {key.namespace}:{key.id}"
        )


class ConversationTargetOwned(ConversationIntegrityError):
    def __init__(
        self,
        address: ConversationAddress,
        *,
        owner_surface: str | None = None,
        owner_pid: int | None = None,
        owner_epoch: str | None = None,
    ) -> None:
        self.address = address
        self.owner_surface = owner_surface
        self.owner_pid = owner_pid
        self.owner_epoch = owner_epoch
        super().__init__(f"conversation target is owned: {address.agent_handle}")


_TARGETS_LOCK = threading.RLock()
_TARGETS: weakref.WeakValueDictionary[str, ConversationStore] = (
    weakref.WeakValueDictionary()
)


@dataclass
class _TransientReservation:
    token: object
    waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]]


_TARGET_RESERVATIONS: dict[str, _TransientReservation] = {}


def _reservation_waiting(path: Path) -> None:
    """Test seam after a transient waiter joins an occupied reservation."""


def _target_key(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def register_conversation_target(store: ConversationStore) -> None:
    key = _target_key(store.session_dir)
    with _TARGETS_LOCK:
        if key in _TARGET_RESERVATIONS and not store._mutation_target:
            raise ConversationIntegrityError(
                "conversation target is reserved for mutation"
            )
        _TARGETS[key] = store


def unregister_conversation_target(store: ConversationStore) -> None:
    key = _target_key(store.session_dir)
    with _TARGETS_LOCK:
        if _TARGETS.get(key) is store:
            del _TARGETS[key]


def _conversation_path(address: ConversationAddress) -> tuple[Path, Path, str]:
    home = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
    sessions = home / "sessions"
    root = sessions / address.root_session_id
    if address.agent_handle == address.root_session_id:
        return root, sessions, address.root_session_id
    prefix = f"{address.root_session_id}:"
    if not address.agent_handle.startswith(prefix):
        raise ConversationIntegrityError(
            "conversation agent handle is outside its root"
        )
    parts = address.agent_handle[len(prefix) :].split(":")
    if not parts or any(not part.isdigit() or int(part) <= 0 for part in parts):
        raise ConversationIntegrityError("conversation agent handle is invalid")
    path = root
    for part in parts:
        path = path / "agents" / str(int(part))
    return path, path.parent, path.name


def _authenticate_address(address: ConversationAddress) -> tuple[Path, Path, str]:
    path, root_dir, session_id = _conversation_path(address)
    sessions = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta")) / "sessions"
    with session_directory(sessions, address.root_session_id) as (_, directory_fd):
        metadata = load_session_json(read_session_file(directory_fd, "meta.json"))
    if type(metadata) is not dict or metadata.get("project_id") != address.project_id:
        raise ConversationIntegrityError("conversation address project does not match")
    return path, root_dir, session_id


def _resident_target(path: Path) -> ConversationStore | None:
    key = _target_key(path)
    with _TARGETS_LOCK:
        resident = _TARGETS.get(key)
        return resident if resident is not None and not resident._closed else None


async def _acquire_transient_target(path: Path) -> object:
    key = _target_key(path)
    while True:
        with _TARGETS_LOCK:
            resident = _TARGETS.get(key)
            if resident is not None and not resident._closed:
                return resident
            reservation = _TARGET_RESERVATIONS.get(key)
            if reservation is None:
                token = object()
                _TARGET_RESERVATIONS[key] = _TransientReservation(token, [])
                return token
            loop = asyncio.get_running_loop()
            waiter: asyncio.Future[None] = loop.create_future()
            reservation.waiters.append((loop, waiter))
        _reservation_waiting(path)
        # test-contract: durability the caller owns the reservation wait deadline
        await asyncio.shield(waiter)


def _release_transient_target(path: Path, token: object) -> None:
    key = _target_key(path)
    with _TARGETS_LOCK:
        reservation = _TARGET_RESERVATIONS.get(key)
        if reservation is None or reservation.token is not token:
            return
        del _TARGET_RESERVATIONS[key]
        waiters = reservation.waiters
    for loop, waiter in waiters:
        loop.call_soon_threadsafe(_wake_waiter, waiter)


def _wake_waiter(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


def _cursor(store: ConversationStore) -> CommittedLogCursor:
    return CommittedLogCursor(
        identity=store._log_identity,
        offset=store._log_offset,
        mtime_ns=store._log_mtime_ns,
        prefix_fingerprint=store._log_prefix_fingerprint,
        entry_prefix_digest=store._entry_prefix_digest,
    )


def _dedupe_id(
    store: ConversationStore, key: ConversationDedupeKey, ordinal: int
) -> str:
    identity = _target_key(store.session_dir)
    material = f"{identity}\0{key.namespace}\0{key.id}\0{ordinal}".encode()
    return hashlib.sha256(material).hexdigest()


def _snapshot_patch(
    store: ConversationStore, fields: tuple[str, ...]
) -> dict[str, object]:
    return {field: copy.deepcopy(getattr(store, field)) for field in fields}


def _resolve_append_dedupe[ResultT](
    store: ConversationStore,
    command: AppendConversationRows[ResultT],
    generation_before: int,
    deterministic_ids: tuple[str, ...],
) -> CommittedDelta[ResultT | None] | None:
    key = command.dedupe_key
    if key is None:
        return None
    by_id = {entry.id: entry for entry in store._entries}
    existing = tuple(by_id.get(entry_id) for entry_id in deterministic_ids)
    next_exists = _dedupe_id(store, key, len(command.rows)) in by_id
    if not any(entry is not None for entry in existing) and not next_exists:
        return None
    if any(entry is None for entry in existing) or next_exists:
        raise ConversationDedupeConflict(key)
    committed = tuple(entry for entry in existing if entry is not None)
    expected_content = tuple((row.entry_type, dict(row.data)) for row in command.rows)
    actual_content = tuple((entry.type, entry.data) for entry in committed)
    if actual_content != expected_content:
        raise ConversationDedupeConflict(key)
    return CommittedDelta(
        generation_before=generation_before,
        generation_after=generation_before,
        committed_entries=copy.deepcopy(committed),
        entries_to_install=(),
        _resident_patch=_ResidentPatch({}, {}),
        result=command.result,
        log_cursor=_cursor(store),
        dedupe_key=key,
        duplicate=True,
    )


def _build_append_entries(
    store: ConversationStore,
    rows: tuple[ConversationRow, ...],
    deterministic_ids: tuple[str, ...],
) -> list[ConversationEntry]:
    prior_ids = set(store._entry_ids)
    parent_id = store._entries[-1].id if store._entries else None
    next_seq = store._entries[-1].seq + 1 if store._entries else 1
    entries: list[ConversationEntry] = []
    for index, row in enumerate(rows):
        if index == 0 and row.parent_id is not None:
            if row.parent_id not in prior_ids:
                raise ConversationIntegrityError(
                    f"missing prior parent {row.parent_id}"
                )
            parent_id = row.parent_id
        elif index > 0 and row.parent_id is not None and row.parent_id != parent_id:
            raise ConversationIntegrityError(
                "conversation row batch has conflicting parent"
            )
        entry_id = (
            deterministic_ids[index] if deterministic_ids else os.urandom(16).hex()
        )
        if entry_id in prior_ids:
            raise ConversationIntegrityError(f"duplicate conversation id: {entry_id}")
        entry = ConversationEntry(
            seq=next_seq + index,
            id=entry_id,
            parent_id=parent_id,
            lane="main",
            type=row.entry_type,
            data=copy.deepcopy(dict(row.data)),
        )
        store._validate_notification_write(entry.type, entry.data)
        store._validate_entry_payload(entry)
        entries.append(entry)
        prior_ids.add(entry_id)
        parent_id = entry_id
    store._entries.extend(entries)
    try:
        store._validate_entries()
    finally:
        del store._entries[-len(entries) :]
    return entries


def _install_appended_entries(
    store: ConversationStore, entries: list[ConversationEntry]
) -> None:
    linear = not store._entries or entries[0].parent_id == store._entries[-1].id
    for entry in entries:
        store._entries.append(entry)
        store._record_entry_prefix(entry)
        store._entry_ids.add(entry.id)
        store._sent_message_receipts.extend(sent_message_receipts(entry))
        if (
            entry.type == "notification"
            and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND
        ):
            task_id = entry.data.get("task_id")
            if type(task_id) is str and task_id:
                store._task_notification_ids.add(task_id)
    if linear:
        for entry in entries:
            store._record_active_entry(entry)
    else:
        store._validate_entries()
        store._rebuild_incremental_validation_state()
    store._record_client_delivery_entries(entries)


def _execute_append[ResultT](
    store: ConversationStore,
    command: AppendConversationRows[ResultT],
    expected_generation: int,
) -> CommittedDelta[ResultT | None]:
    if not command.rows:
        raise ConversationIntegrityError("conversation mutation rows must not be empty")
    with store._append_lock():
        generation_before = store.conversation_generation
        key = command.dedupe_key
        deterministic_ids = (
            tuple(_dedupe_id(store, key, index) for index in range(len(command.rows)))
            if key is not None
            else ()
        )
        duplicate = _resolve_append_dedupe(
            store, command, generation_before, deterministic_ids
        )
        if duplicate is not None:
            return duplicate
        if expected_generation != generation_before:
            raise ConversationGenerationConflict(expected_generation, generation_before)
        entries = _build_append_entries(store, command.rows, deterministic_ids)
        store._write_bytes(
            b"".join(store._encoded_entry_line(entry) for entry in entries)
        )
        _install_appended_entries(store, entries)
        generation_after = store.conversation_generation
        committed = tuple(store._snapshot_entry(entry) for entry in entries)
        if store.on_persisted_activity is not None:
            store.on_persisted_activity(entries[-1].seq)
        return CommittedDelta(
            generation_before=generation_before,
            generation_after=generation_after,
            committed_entries=committed,
            entries_to_install=committed,
            _resident_patch=_ResidentPatch({}, {}),
            result=command.result,
            log_cursor=_cursor(store),
            dedupe_key=key,
            duplicate=False,
        )


def _execute_store_command[ResultT](
    store: ConversationStore,
    command: _ConditionalStoreMutation[ResultT],
    expected_generation: int,
) -> CommittedDelta[ResultT]:
    if (
        not command.accept_external_generation
        and store.conversation_generation != expected_generation
    ):
        raise ConversationGenerationConflict(
            expected_generation, store.conversation_generation
        )
    baseline_count = (
        len(store._entries)
        if command.baseline_count is None
        else command.baseline_count
    )
    baseline_digest = (
        store._entry_prefix_digest
        if command.baseline_digest is None
        else command.baseline_digest
    )
    expected_patch = (
        _snapshot_patch(store, command.patch_fields)
        if command.baseline_patch is None
        else command.baseline_patch
    )
    baseline_patch = _snapshot_patch(store, command.patch_fields)
    writer_baseline_matches = (
        len(store._entries) == baseline_count
        and store._entry_prefix_digest == baseline_digest
    )
    full_reload_count = store._full_reload_count
    result = command.apply(store)
    entries = store._owned_installed_delta(tuple(store._entries[baseline_count:]))
    baseline_parent_id = (
        store._entries[baseline_count - 1].id
        if 0 < baseline_count <= len(store._entries)
        else None
    )
    replace_entries = (
        command.accept_external_generation
        or not writer_baseline_matches
        or store._full_reload_count != full_reload_count
        or bool(entries and entries[0].parent_id != baseline_parent_id)
    )
    if replace_entries:
        entries = store._owned_installed_delta(tuple(store._entries))
    changed_patch = {
        field: copy.deepcopy(getattr(store, field))
        for field, old in baseline_patch.items()
        if getattr(store, field) != old
    }
    expected_patch = {field: expected_patch[field] for field in changed_patch}
    return CommittedDelta(
        generation_before=expected_generation,
        generation_after=store.conversation_generation,
        committed_entries=copy.deepcopy(entries),
        entries_to_install=copy.deepcopy(entries),
        _resident_patch=_ResidentPatch(changed_patch, expected_patch, replace_entries),
        result=result,
        log_cursor=_cursor(store),
        dedupe_key=None,
        duplicate=False,
    )


def execute_mutation[ResultT](
    store: ConversationStore,
    command: ConversationMutation[ResultT],
    expected_generation: int,
) -> CommittedDelta[ResultT]:
    installed_callback = store.on_entries_installed
    store.on_entries_installed = None
    try:
        if isinstance(command, AppendConversationRows):
            return _execute_append(store, command, expected_generation)  # type: ignore[return-value]
        return _execute_store_command(store, command, expected_generation)  # type: ignore[arg-type]
    finally:
        store.on_entries_installed = installed_callback


async def commit_conversation_async[ResultT](
    address: ConversationAddress,
    command: ConversationMutation[ResultT],
) -> CommittedDelta[ResultT]:
    loop = asyncio.get_running_loop()
    path, root_dir, session_id = await loop.run_in_executor(
        None, _authenticate_address, address
    )
    target = await _acquire_transient_target(path)
    if hasattr(target, "_commit_address_async"):
        return await target._commit_address_async(command)
    token = target
    try:

        def commit_transient(cancellation: object) -> object:
            from ..store._async_writes import (
                _MutationCancellation,
                _SubmittedMutationOutcome,
            )
            from ..store._store import ConversationStore

            assert isinstance(cancellation, _MutationCancellation)
            try:
                target = ConversationStore(
                    root_dir,
                    session_id=session_id,
                    _must_exist=True,
                    _owner_surface="recovery",
                    _recovery_owner=True,
                    _mutation_target=True,
                )
            except SessionOwnedError as exc:
                raise ConversationTargetOwned(
                    address,
                    owner_surface=exc.owner_surface,
                    owner_pid=exc.owner_pid,
                    owner_epoch=exc.owner_epoch,
                ) from exc
            try:
                last_conflict: ConversationGenerationConflict | None = None
                try:
                    for _attempt in range(4):
                        expected = target.conversation_generation
                        try:
                            delta = target.commit(command, expected)
                            return _SubmittedMutationOutcome(
                                None, delta, None, target.take_persisted_appends()
                            )
                        except ConversationGenerationConflict as exc:
                            last_conflict = exc
                    assert last_conflict is not None
                    raise last_conflict
                except BaseException as error:  # noqa: BLE001 - finish owned recovery
                    return _SubmittedMutationOutcome(
                        None, None, error, target.take_persisted_appends()
                    )
            finally:
                target.close()

        from ..store._async_writes import (
            _submit_durable_mutation,
            _SubmittedMutationOutcome,
        )

        async def finish_transient(
            outcome: _SubmittedMutationOutcome,
        ) -> CommittedDelta[ResultT]:
            if outcome.error is not None:
                raise outcome.error
            assert outcome.delta is not None
            return outcome.delta

        return await _submit_durable_mutation(
            command, commit_transient, finish_transient
        )
    finally:
        _release_transient_target(path, token)
