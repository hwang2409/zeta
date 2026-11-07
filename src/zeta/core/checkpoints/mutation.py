"""Typed, generation-checked conversation mutations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..session_execution import SessionOwnedError
from ..session_files import read_session_file, session_directory
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


class ConversationMutation[ResultT](Protocol):
    """Immutable command accepted by the conversation mutation seam."""


@dataclass(frozen=True, slots=True)
class AppendConversationRows[ResultT]:
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


@dataclass(frozen=True, slots=True)
class _MethodMutation[ResultT]:
    method: Callable[..., ResultT]
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    method_accepts_store: bool = True
    patch_fields: tuple[str, ...] = ()
    persisted_callback: Callable[[], None] | None = None
    accept_external_generation: bool = False


_TARGETS_LOCK = threading.RLock()
_TARGETS: weakref.WeakValueDictionary[str, ConversationStore] = (
    weakref.WeakValueDictionary()
)
_TARGET_RESERVATIONS: set[str] = set()


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


def _reserve_target(path: Path) -> ConversationStore | None:
    key = _target_key(path)
    with _TARGETS_LOCK:
        if key in _TARGET_RESERVATIONS:
            raise ConversationIntegrityError(
                "conversation target already has a commit slot"
            )
        _TARGET_RESERVATIONS.add(key)
        resident = _TARGETS.get(key)
        return resident if resident is not None and not resident._closed else None


def _release_target(path: Path) -> None:
    with _TARGETS_LOCK:
        _TARGET_RESERVATIONS.discard(_target_key(path))


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
        if key is not None:
            by_id = {entry.id: entry for entry in store._entries}
            existing = tuple(by_id.get(entry_id) for entry_id in deterministic_ids)
            next_exists = _dedupe_id(store, key, len(command.rows)) in by_id
            if any(entry is not None for entry in existing) or next_exists:
                if any(entry is None for entry in existing) or next_exists:
                    raise ConversationDedupeConflict(key)
                committed = tuple(entry for entry in existing if entry is not None)
                expected_content = tuple(
                    (row.entry_type, dict(row.data)) for row in command.rows
                )
                actual_content = tuple((entry.type, entry.data) for entry in committed)
                if actual_content != expected_content:
                    raise ConversationDedupeConflict(key)
                return CommittedDelta(
                    generation_before=generation_before,
                    generation_after=generation_before,
                    committed_entries=copy.deepcopy(committed),
                    entries_to_install=(),
                    _resident_patch=_ResidentPatch({}),
                    result=command.result,
                    log_cursor=_cursor(store),
                    dedupe_key=key,
                    duplicate=True,
                )
        if expected_generation != generation_before:
            raise ConversationGenerationConflict(expected_generation, generation_before)
        prior_ids = set(store._entry_ids)
        parent_id = store._entries[-1].id if store._entries else None
        next_seq = store._entries[-1].seq + 1 if store._entries else 1
        entries: list[ConversationEntry] = []
        for index, row in enumerate(command.rows):
            if index == 0:
                if row.parent_id is not None:
                    if row.parent_id not in prior_ids:
                        raise ConversationIntegrityError(
                            f"missing prior parent {row.parent_id}"
                        )
                    parent_id = row.parent_id
            elif row.parent_id is not None and row.parent_id != parent_id:
                raise ConversationIntegrityError(
                    "conversation row batch has conflicting parent"
                )
            entry_id = (
                deterministic_ids[index] if key is not None else os.urandom(16).hex()
            )
            if entry_id in prior_ids:
                raise ConversationIntegrityError(
                    f"duplicate conversation id: {entry_id}"
                )
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
        encoded = b"".join(store._encoded_entry_line(entry) for entry in entries)
        store._write_bytes(encoded)
        for entry in entries:
            store._entries.append(entry)
            store._record_entry_prefix(entry)
            store._entry_ids.add(entry.id)
            store._record_active_entry(entry)
            store._sent_message_receipts.extend(sent_message_receipts(entry))
        store._record_client_delivery_entries(entries)
        generation_after = store.conversation_generation
        committed = tuple(store._snapshot_entry(entry) for entry in entries)
        if store.on_persisted_activity is not None:
            store.on_persisted_activity(entries[-1].seq)
        return CommittedDelta(
            generation_before=generation_before,
            generation_after=generation_after,
            committed_entries=committed,
            entries_to_install=committed,
            _resident_patch=_ResidentPatch({}),
            result=command.result,
            log_cursor=_cursor(store),
            dedupe_key=key,
            duplicate=False,
        )


def _execute_method[ResultT](
    store: ConversationStore,
    command: _MethodMutation[ResultT],
    expected_generation: int,
) -> CommittedDelta[ResultT]:
    if (
        not command.accept_external_generation
        and store.conversation_generation != expected_generation
    ):
        raise ConversationGenerationConflict(
            expected_generation, store.conversation_generation
        )
    baseline_count = len(store._entries)
    baseline_digest = store._entry_prefix_digest
    before_patch = _snapshot_patch(store, command.patch_fields)
    result = (
        command.method(store, *command.args, **dict(command.kwargs))
        if command.method_accepts_store
        else command.method(*command.args, **dict(command.kwargs))
    )
    if command.persisted_callback is not None:
        command.persisted_callback()
    entries = (
        store._installed_delta_from_baseline(
            store._entries, (baseline_count, baseline_digest)
        )
        if command.accept_external_generation
        else store._owned_installed_delta(tuple(store._entries[baseline_count:]))
    )
    if entries == "reset":
        raise ConversationIntegrityError(
            "typed mutation produced a nonlinear transcript"
        )
    changed_patch = {
        field: copy.deepcopy(getattr(store, field))
        for field, old in before_patch.items()
        if command.accept_external_generation or getattr(store, field) != old
    }
    return CommittedDelta(
        generation_before=expected_generation,
        generation_after=store.conversation_generation,
        committed_entries=copy.deepcopy(entries),
        entries_to_install=copy.deepcopy(entries),
        _resident_patch=_ResidentPatch(changed_patch),
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
        if isinstance(command, _MethodMutation):
            return _execute_method(store, command, expected_generation)
        raise TypeError(f"unsupported conversation mutation: {type(command).__name__}")
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
    resident = _reserve_target(path)
    try:
        if resident is not None:
            last_conflict: ConversationGenerationConflict | None = None
            for _attempt in range(4):
                expected = resident.conversation_generation
                try:
                    return await resident.commit_async(command, expected)
                except ConversationGenerationConflict as exc:
                    last_conflict = exc
            assert last_conflict is not None
            raise last_conflict

        def commit_transient() -> CommittedDelta[ResultT]:
            from ..store._store import ConversationStore

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
                for _attempt in range(4):
                    expected = target.conversation_generation
                    try:
                        return target.commit(command, expected)
                    except ConversationGenerationConflict as exc:
                        last_conflict = exc
                assert last_conflict is not None
                raise last_conflict
            finally:
                target.close()

        worker = loop.run_in_executor(None, commit_transient)
        result, cancelled = await resident_await_started(worker)
        if cancelled:
            raise asyncio.CancelledError
        return result
    finally:
        _release_target(path)


async def resident_await_started[ResultT](
    worker: asyncio.Future[ResultT],
) -> tuple[ResultT, bool]:
    cancelled = False
    while True:
        try:
            return await asyncio.shield(worker), cancelled
        except asyncio.CancelledError:
            cancelled = True
