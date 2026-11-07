"""Home-wide stored-format gate and lossless JSON publication."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

WRITER_EPOCH: Final = 3

# These are the authoritative stored roots. An empty sessions/ directory is safe:
# startup can create it before a session exists. The lock, logs, caches, and other
# derived or ephemeral output are intentionally not gated.
GATED_HOME_PATHS: Final[tuple[tuple[str, str], ...]] = (
    ("sessions/*", "zeta.core.store and zeta.session_retention"),
    (
        "projects",
        "zeta.project_registry projects and legacy lanes, zeta.memory, and zeta.project_inbox",
    ),
    ("accounts.json", "zeta.accounts.store"),
    ("accounts", "zeta.accounts.store OAuth credentials"),
    ("anthropic-oauth.json", "zeta.accounts.store legacy OAuth credentials"),
    ("codex-oauth.json", "zeta.accounts.store legacy OAuth credentials"),
    ("mcp-tokens", "zeta.mcp.oauth_store"),
)

# Persisted mappings that reject unknown keys.  Names are wire-shape names, not
# Python type names, so a conditional variant can be compared independently.
ACCEPTED_KEY_SETS: Final[dict[str, frozenset[str]]] = {
    "hook_config": frozenset({"hook"}),
    "hook_record": frozenset({"event", "command", "timeout_seconds", "tools", "on_error"}),
    "todo_item": frozenset({"content", "status"}),
    "notification_marker": frozenset({"notification_id"}),
    "turn_history": frozenset({"turn_id", "reason"}),
    "prompt_component": frozenset({"offset", "length", "digest"}),
    "approval_fact": frozenset({"action"}),
    "approval_display": frozenset({"effective_cwd", "resolved_path", "utf8_bytes"}),
    "child_run": frozenset({"run_id", "parent_run_id", "name", "task", "started_at"}),
    "response_action": frozenset({"action"}),
    "response_action_report": frozenset({"action", "report"}),
    "response_report": frozenset({"status", "summary", "head_sha", "verdict"}),
    "child_idle_notification": frozenset(
        {"origin", "kind", "child_instance_id", "child_session_path", "description", "status", "text", "ask_parent_question_id", "display_only"}
    ),
    "project_registry_update": frozenset({"description", "repo", "repo_agents"}),
    "project_registry_record": frozenset({"schema_version", "project_id", "name", "scope", "created_at", "updated_at", "canonical_integration_root", "lanes", "repo_agents"}),
    "memory_sync_digest": frozenset({"kind", "digests"}),
    "memory_sync_entry": frozenset({"kind", "entry_ids", "digests"}),
    "memory_reconciliation": frozenset({"changes"}),
    "memory_reconciliation_change": frozenset({"path", "content", "action"}),
    "memory_reconciliation_source": frozenset({"project_id", "session_id", "entry_id"}),
    "memory_operations": frozenset({"operations"}),
    "account_credential": frozenset({"provider", "name", "sha256"}),
    "account_publication_marker": frozenset({"provider", "name", "sha256"}),
    "project_session_reference": frozenset({"session_id", "transcript_path", "role", "parent_session_id", "recorded_at"}),
    "memory_state_root": frozenset({"format", "project_id", "generation", "schema", "entries", "compacted_through_version"}),
    "memory_schema": frozenset({"version", "profile", "kinds"}),
    "memory_kind": frozenset({"key", "name", "description", "prompt_mode", "prompt_priority", "prompt_max_entries", "default_expiry_days"}),
    "memory_entry": frozenset({"id", "project_id", "kind", "text", "representation", "status", "created_at", "updated_at", "seen_at", "expires_at", "expiry_is_default", "valid_from", "valid_until", "supersedes", "superseded_by", "sources", "automatic", "accepted_at", "accepted_by", "last_operation_id", "section", "migration_order", "migration_source"}),
    "memory_missing_entry": frozenset({"id", "missing"}),
    "memory_source": frozenset({"session_id", "seq_start", "seq_end", "origins", "observed_at", "evidence_rank"}),
    "memory_migration_source": frozenset({"source_digest", "source_version"}),
    "memory_operation_receipt": frozenset({"operation_id", "type", "target_ids", "result_ids", "reason", "reconciliation_key", "automatic"}),
    "remote_memory_sync_state": frozenset({"schema", "project_id", "files", "conflicts"}),
    "remote_memory_sync_conflict": frozenset({"id", "local", "remote"}),
    "entry_sync_state": frozenset({"version", "entries", "conflicts", "transitions"}),
    "entry_sync_schema": frozenset({"version", "kind", "fields"}),
    "entry_sync_conflict": frozenset({"id", "local", "remote", "resolution"}),
    "entry_sync_transition": frozenset({"id", "from", "to", "at"}),
    "entry_sync_resolution_candidate": frozenset({"id", "value", "source"}),
    "background_output_journal_commit": frozenset({"task_id", "offset", "length"}),
    "client_delivery_send_acceptance": frozenset(
        {"delivery_id", "method", "status", "outcome"}
    ),
    "client_delivery_steer_acceptance": frozenset(
        {"delivery_id", "method", "status", "outcome"}
    ),
    "client_delivery_queued_steer_acceptance_message": frozenset(
        {"delivery_id", "method", "status", "outcome", "accepted_message"}
    ),
    "client_delivery_delivered_transition": frozenset(
        {"delivery_id", "method", "status"}
    ),
    "client_delivery_dropped_transition": frozenset(
        {"delivery_id", "method", "status", "reason"}
    ),
    "client_delivery_accepted_outcome": frozenset({"accepted", "session_id"}),
    "client_delivery_steer_accepted_outcome": frozenset({"accepted"}),
    "pending_queue_closed": frozenset(),
    "fact_floor_root": frozenset({"version", "oldest", "newest"}),
    "fact_floor_aggregate": frozenset({"fact", "first_seq", "last_seq", "value_rank"}),
    "fact_pr": frozenset({"type", "pr"}),
    "fact_pr_verdict": frozenset({"type", "pr", "verdict"}),
    "fact_pr_head": frozenset({"type", "pr", "sha"}),
    "fact_child_status": frozenset({"type", "handle", "status"}),
    "fact_child_head": frozenset({"type", "handle", "sha"}),
    "tool_result_text": frozenset({"type", "text", "truncated", "full_size", "full_size_chars", "next_offset", "spill_path", "annotations"}),
    "tool_result_image": frozenset({"type", "data", "mimeType", "annotations", "caption", "width", "height", "path", "size"}),
    "tool_result_resource": frozenset({"type", "resource", "annotations"}),
    "tool_result_resource_contents": frozenset({"uri", "mimeType", "text", "blob"}),
    "tool_result_annotations": frozenset({"audience", "priority", "lastModified"}),
    "attention_record": frozenset({"id", "created_at", "session_id", "project_id", "entry_id", "entry_seq", "lane", "title", "why", "options", "recommendation", "status", "resolved_at", "fork_session_id", "decision"}),
    "attention_fork_meta": frozenset({"forked_from_session", "forked_at_entry", "attention_id"}),
    "storage_manifest": frozenset({"writer_epoch"}),
}


BROAD_OWNED_SUBTREES: Final[dict[str, frozenset[tuple[str, ...]]]] = {
    "memory-current.json": frozenset({("history",)}),
    "meta.json": frozenset({("prompt_components",)}),
    "workspace.json": frozenset({("snapshots",)}),
}

CLOSED_VARIANTS: Final[dict[str, frozenset[object]]] = {
    "account.provider": frozenset({"anthropic", "codex"}),
    "account_registry.version": frozenset({1}),
    "conversation.entry_type": frozenset(
        {
            "message",
            "child_run_started",
            "compaction",
            "client_delivery",
            "warning",
            "checkpoint",
            "fork",
            "approval_resolution",
            "approval_history",
            "notification",
            "notification_ack",
            "notification_tui_presented",
            "pending_prompt",
            "pending_prompt_ack",
            "pending_queue_closed",
            "terminal_transition",
        }
    ),
    "inbox.kind": frozenset(
        {"bug_report", "change_request", "question", "info", "reply"}
    ),
    "inbox.schema_version": frozenset({1}),
    "memory.format": frozenset({1, 2}),
    "project.repo_agents": frozenset({"include", "ignore"}),
    "project.schema_version": frozenset({1}),
    "project.session_role": frozenset({"session", "orchestrator", "worker"}),
    "sent_message_ledger.reported_status": frozenset({"claimed", "done"}),
    "sent_message_ledger.schema_version": frozenset({1}),
}
_MANIFEST = "storage-format.json"
_LOCK = ".storage-format.lock"


class StorageFormatError(ValueError):
    """Raised when a home cannot be accessed by this stored-format version."""


def reject_unknown_keys(payload: Mapping[str, object], shape: str) -> None:
    """Enforce the one authoritative accepted-key declaration for a payload."""

    accepted = ACCEPTED_KEY_SETS.get(shape)
    if accepted is None:
        raise StorageFormatError(f"stored key shape is not registered: {shape}")
    unknown = set(payload) - accepted
    if unknown:
        raise StorageFormatError(
            f"unknown keys for {shape}: {sorted(unknown)!r}"
        )


def require_closed_variant(shape: str, value: object) -> object:
    """Return a registered stored variant or fail closed."""

    accepted = CLOSED_VARIANTS.get(shape)
    if accepted is None:
        raise StorageFormatError(f"stored variant shape is not registered: {shape}")
    if value not in accepted:
        raise StorageFormatError(f"unsupported stored variant for {shape}: {value!r}")
    return value


def registry_dump() -> dict[str, object]:
    """Return the deterministic compatibility registry for this release."""

    return {
        "writer_epoch": WRITER_EPOCH,
        "closed_variants": {
            name: sorted(values, key=lambda value: (type(value).__name__, repr(value)))
            for name, values in sorted(CLOSED_VARIANTS.items())
        },
        "broad_owned_subtrees": {
            name: sorted(".".join(path) for path in paths)
            for name, paths in sorted(BROAD_OWNED_SUBTREES.items())
        },
        "accepted_key_sets": {
            name: sorted(values) for name, values in sorted(ACCEPTED_KEY_SETS.items())
        },
    }


def compare_registry_dumps(
    predecessor: Mapping[str, object], candidate: Mapping[str, object]
) -> None:
    """Reject registry changes unless the candidate epoch is strictly newer."""

    previous_epoch = predecessor.get("writer_epoch")
    candidate_epoch = candidate.get("writer_epoch")
    if type(previous_epoch) is not int or type(candidate_epoch) is not int:
        raise StorageFormatError("registry dump has an invalid writer_epoch")
    changes: list[str] = []
    for registry_name, label in (
        ("closed_variants", "registry value"),
        ("broad_owned_subtrees", "broad-owned subtree"),
        ("accepted_key_sets", "accepted key"),
    ):
        before = predecessor.get(registry_name, {})
        after = candidate.get(registry_name, {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise StorageFormatError(f"registry dump has invalid {registry_name}")
        for shape in sorted(set(before) | set(after)):
            old = set(before.get(shape, ()))
            new = set(after.get(shape, ()))
            for value in sorted(old ^ new, key=repr):
                changes.append(f"shape {shape}: {label} {value!r} changed")
    if changes and candidate_epoch <= previous_epoch:
        raise StorageFormatError(
            "; ".join(changes)
            + "; registry delta requires WRITER_EPOCH = latest origin/main + 1 "
            "(candidate must be greater than the predecessor's); "
            "bump the epoch in src/zeta/storage_format.py"
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    writer_epoch: int


@dataclass(frozen=True, slots=True)
class SetValue:
    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True, slots=True)
class DeleteValue:
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplaceSubtree:
    """Replace a subtree that this format explicitly owns in full."""

    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True, slots=True)
class PatchRecords:
    """Upsert and explicitly delete object records at one owned path."""

    path: tuple[str, ...]
    upserts: tuple[Mapping[str, Any], ...]
    identity: str | tuple[str, ...]
    delete_ids: tuple[str, ...] = ()


type Mutation = SetValue | DeleteValue | ReplaceSubtree | PatchRecords
PublishBytes = Callable[[int, str, bytes], None]


@dataclass(slots=True)
class _RegistryEntry:
    fd: int
    references: int
    epoch: int
    shape_locks: dict[tuple[int, int, str], tuple[threading.Lock, int]]


_REGISTRY: dict[Path, _RegistryEntry] = {}
_REGISTRY_LOCK = threading.RLock()


@contextmanager
def _shape_lock(
    entry: _RegistryEntry, home: Path, directory_fd: int, name: str
) -> Iterator[None]:
    info = os.fstat(directory_fd)
    shape = (info.st_dev, info.st_ino, name)
    digest = hashlib.sha256(f"{info.st_dev}:{info.st_ino}:{name}".encode()).hexdigest()
    with _REGISTRY_LOCK:
        thread_lock, users = entry.shape_locks.get(shape, (threading.Lock(), 0))
        entry.shape_locks[shape] = (thread_lock, users + 1)
    try:
        with thread_lock:
            lock_root = home / ".storage-format-shape-locks"
            lock_root.mkdir(mode=0o700, exist_ok=True)
            root_fd = os.open(
                lock_root,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                fd = os.open(
                    digest,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=root_fd,
                )
            finally:
                os.close(root_fd)
            try:
                os.set_inheritable(fd, False)
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
    finally:
        with _REGISTRY_LOCK:
            _, users = entry.shape_locks[shape]
            if users == 1:
                del entry.shape_locks[shape]
            else:
                entry.shape_locks[shape] = (thread_lock, users - 1)


def _canonical_home(home: str | Path) -> Path:
    return Path(home).expanduser().resolve()


def _open_lock(home: Path) -> int:
    home.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(home / _LOCK, flags, 0o600)
    os.set_inheritable(fd, False)
    if os.get_inheritable(fd):
        os.close(fd)
        raise StorageFormatError("stored-format lock descriptor is inheritable")
    return fd


def _missing_manifest_error(home: Path) -> StorageFormatError:
    return StorageFormatError(
        f"stored-format manifest is missing at {home}; run "
        "zeta storage-format advance --expected-epoch 0 "
        f"--target-epoch {WRITER_EPOCH}"
    )


def _read_manifest(home: Path, *, absent_epoch: int | None = None) -> int:
    try:
        raw = (home / _MANIFEST).read_bytes()
    except FileNotFoundError:
        if absent_epoch is not None:
            return absent_epoch
        raise _missing_manifest_error(home) from None
    try:
        decoder = json.JSONDecoder()
        text = raw.decode()
        value, end = decoder.raw_decode(text)
        if (
            text[end:].strip()
            or type(value) is not dict
            or set(value) != {"writer_epoch"}
        ):
            raise ValueError
        epoch = value["writer_epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageFormatError(
            f"stored-format manifest is invalid at {home}"
        ) from exc
    return epoch


def _home_has_gated_state(home: Path) -> bool:
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for pattern, _owner in GATED_HOME_PATHS:
            name = pattern.removesuffix("/*")
            try:
                info = os.stat(name, dir_fd=home_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not pattern.endswith("/*") or not stat.S_ISDIR(info.st_mode):
                return True
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=home_fd,
                )
            except OSError:
                return True
            try:
                if os.listdir(child_fd):
                    return True
            finally:
                os.close(child_fd)
        return False
    finally:
        os.close(home_fd)


def _bootstrap_empty_home(fd: int, home: Path) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise _missing_manifest_error(home) from None
    if not (home / _MANIFEST).exists():
        if _home_has_gated_state(home):
            _read_manifest(home)
        _publish_manifest(home, WRITER_EPOCH)
    fcntl.flock(fd, fcntl.LOCK_SH)


def open_writer(home: str | Path, *, capability: int = WRITER_EPOCH) -> WriterLease:
    """Retain a process-shared home lease after checking writer compatibility."""

    canonical = _canonical_home(home)
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(canonical)
        if registered is not None:
            if registered.epoch > capability:
                raise StorageFormatError(
                    f"stored format {registered.epoch} is newer than writer capability {capability}"
                )
            registered.references += 1
            return WriterLease(canonical, registered)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise StorageFormatError(
                "stored-format lock acquisition must run off the event loop"
            )
        fd = _open_lock(canonical)
        try:
            if (canonical / _MANIFEST).exists():
                fcntl.flock(fd, fcntl.LOCK_SH)
            else:
                _bootstrap_empty_home(fd, canonical)
            epoch = _read_manifest(canonical)
            if epoch > capability:
                raise StorageFormatError(
                    f"stored format {epoch} is newer than writer capability {capability}"
                )
            registered = _RegistryEntry(
                fd=fd, references=1, epoch=epoch, shape_locks={}
            )
            _REGISTRY[canonical] = registered
            return WriterLease(canonical, registered)
        except BaseException:
            os.close(fd)
            raise


def _publish_manifest(home: Path, epoch: int) -> None:
    temporary = home / f".{_MANIFEST}.{secrets.token_hex(16)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(f'{{"writer_epoch":{epoch}}}\n'.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, home / _MANIFEST)
        directory_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def advance(
    home: str | Path, *, expected_epoch: int, target_epoch: int = WRITER_EPOCH
) -> None:
    """Advance an idle home to a strictly newer supported writer epoch."""

    if target_epoch > WRITER_EPOCH or target_epoch <= expected_epoch:
        raise StorageFormatError("stored-format target must be newer and supported")
    canonical = _canonical_home(home)
    fd = _open_lock(canonical)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StorageFormatError("stored-format home has active writers") from exc
        observed = _read_manifest(canonical, absent_epoch=0)
        if observed != expected_epoch:
            raise StorageFormatError(
                f"stored-format epoch changed: expected {expected_epoch}, found {observed}"
            )
        _publish_manifest(canonical, target_epoch)
    finally:
        os.close(fd)


def verify(home: str | Path, *, capability: int = WRITER_EPOCH) -> VerificationReport:
    canonical = _canonical_home(home)
    epoch = _read_manifest(canonical)
    if epoch > capability:
        raise StorageFormatError(
            f"stored format {epoch} is newer than writer capability {capability}"
        )
    return VerificationReport(writer_epoch=epoch)


def _mapping_at(
    root: dict[str, Any], path: tuple[str, ...], *, create: bool
) -> tuple[dict[str, Any], str]:
    if not path:
        raise StorageFormatError("a mutation path cannot be empty")
    parent = root
    for key in path[:-1]:
        child = parent.get(key)
        if child is None and create:
            child = {}
            parent[key] = child
        if type(child) is not dict:
            raise StorageFormatError(
                f"stored JSON path is not an object: {'.'.join(path)}"
            )
        parent = child
    return parent, path[-1]


def _merge_mapping(existing: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key, child in value.items():
        if type(key) is not str:
            raise StorageFormatError("stored JSON object keys must be strings")
        current = existing.get(key)
        if type(child) is dict and type(current) is dict:
            _merge_mapping(current, child)
        else:
            existing[key] = copy.deepcopy(child)


def apply_mutations(raw: object, mutations: Iterable[Mutation]) -> dict[str, Any]:
    """Apply exact declared deltas while retaining every unowned JSON key."""

    if type(raw) is not dict:
        raise StorageFormatError("stored JSON root must be an object")
    output = copy.deepcopy(raw)
    for mutation in mutations:
        parent, key = _mapping_at(
            output, mutation.path, create=not isinstance(mutation, DeleteValue)
        )
        if isinstance(mutation, DeleteValue):
            parent.pop(key, None)
        elif isinstance(mutation, ReplaceSubtree):
            parent[key] = copy.deepcopy(mutation.value)
        elif isinstance(mutation, PatchRecords):
            parent[key] = _apply_record_mutations(
                parent.get(key, []),
                mutation.upserts,
                identity=mutation.identity,
                delete_ids=mutation.delete_ids,
                name=".".join(mutation.path),
            )
        else:
            current = parent.get(key)
            if type(mutation.value) is dict and type(current) is dict:
                _merge_mapping(current, mutation.value)
            else:
                parent[key] = copy.deepcopy(mutation.value)
    return output


def mapping_mutations(
    value: Mapping[str, Any],
    *,
    delete: Iterable[str] = (),
    broad: Iterable[str] = (),
) -> tuple[Mutation, ...]:
    """Describe a normalized mapping as leaf patches and explicit broad sequences."""

    broad_paths = frozenset(broad)
    mutations: list[Mutation] = [DeleteValue((key,)) for key in delete]
    for key, item in value.items():
        mutation_type = (
            ReplaceSubtree if isinstance(item, list) or key in broad_paths else SetValue
        )
        mutations.append(mutation_type((key,), item))
    return tuple(mutations)


def _require_record_id(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise StorageFormatError(f"stored JSON record identity is invalid: {name}")
    return value


def _record_id(
    record: Mapping[str, Any], identity: str | tuple[str, ...], name: str
) -> str:
    if isinstance(identity, str):
        return _require_record_id(record.get(identity), name)
    values = tuple(record.get(field) for field in identity)
    if not identity or any(
        not isinstance(value, (str, int)) or isinstance(value, bool) for value in values
    ):
        raise StorageFormatError(f"stored JSON record identity is invalid: {name}")
    return json.dumps(values, separators=(",", ":"))


def _apply_record_mutations(
    raw: object,
    upserts: Iterable[Mapping[str, Any]],
    *,
    identity: str | tuple[str, ...],
    delete_ids: Iterable[str],
    name: str,
) -> list[dict[str, Any]]:
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise StorageFormatError(f"stored JSON record collection is invalid: {name}")

    output = copy.deepcopy(raw)
    positions: dict[str, int] = {}
    for index, record in enumerate(output):
        record_id = _record_id(record, identity, name)
        if record_id in positions:
            raise StorageFormatError(f"stored JSON record identity is invalid: {name}")
        positions[record_id] = index

    deleted = {_require_record_id(record_id, name) for record_id in delete_ids}
    seen: set[str] = set()
    for record in upserts:
        record_id = _record_id(record, identity, name)
        if record_id in seen or record_id in deleted:
            raise StorageFormatError(f"stored JSON record identity is invalid: {name}")
        seen.add(record_id)
        if record_id in positions:
            _merge_mapping(output[positions[record_id]], record)
        else:
            positions[record_id] = len(output)
            current: dict[str, Any] = {}
            _merge_mapping(current, record)
            output.append(current)

    return [
        record for record in output if _record_id(record, identity, name) not in deleted
    ]


class WriterLease:
    """A retained shared lock that is the sole authoritative publication seam."""

    __slots__ = ("_closed", "_entry", "home")

    def __init__(self, home: Path, entry: _RegistryEntry) -> None:
        self.home = home
        self._entry = entry
        self._closed = False

    def retain(self) -> WriterLease:
        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        with _REGISTRY_LOCK:
            self._entry.references += 1
        return WriterLease(self.home, self._entry)

    def publish_json(
        self,
        directory_fd: int,
        name: str,
        mutations: Iterable[Mutation],
        publish: PublishBytes,
        *,
        initial: object | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        with _shape_lock(self._entry, self.home, directory_fd, name):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                raw = {} if initial is None else initial
            else:
                try:
                    with os.fdopen(fd, "rb") as stream:
                        raw = json.load(stream)
                except (OSError, ValueError) as exc:
                    raise StorageFormatError(f"stored JSON is invalid: {name}") from exc
            output = apply_mutations(raw, mutations)
            encoded = (
                json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            publish(directory_fd, name, encoded)
            return output

    def publish_records(
        self,
        directory_fd: int,
        name: str,
        upserts: Iterable[Mapping[str, Any]],
        *,
        identity: str,
        delete_ids: Iterable[str],
        publish: PublishBytes,
        path: tuple[str, ...] = (),
        initial: object | None = None,
    ) -> list[dict[str, Any]]:
        """Patch and explicitly delete object records by identity."""

        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        with _shape_lock(self._entry, self.home, directory_fd, name):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                raw: object = ([] if not path else {}) if initial is None else initial
            else:
                try:
                    with os.fdopen(fd, "rb") as stream:
                        raw = json.load(stream)
                except (OSError, ValueError) as exc:
                    raise StorageFormatError(f"stored JSON is invalid: {name}") from exc
            records = raw
            if path:
                if type(raw) is not dict:
                    raise StorageFormatError("stored JSON root must be an object")
                parent, key = _mapping_at(raw, path, create=True)
                records = parent.get(key, [])
            output = _apply_record_mutations(
                records,
                upserts,
                identity=identity,
                delete_ids=delete_ids,
                name=name,
            )
            published: object = output
            if path:
                published = apply_mutations(raw, (ReplaceSubtree(path, output),))
            encoded = (
                json.dumps(published, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            publish(directory_fd, name, encoded)
            return output

    def delete(self, directory_fd: int, name: str) -> None:
        """Delete one authoritative shape while excluding all of its publishers."""

        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        with _shape_lock(self._entry, self.home, directory_fd, name):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                return
            os.fsync(directory_fd)

    def publish_append(
        self,
        shape: str,
        variants: Iterable[object],
        publish: Callable[[], None],
    ) -> None:
        """Validate closed variants before an authoritative append."""
        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        for variant in variants:
            require_closed_variant(shape, variant)
        publish()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _REGISTRY_LOCK:
            self._entry.references -= 1
            if self._entry.references:
                return
            _REGISTRY.pop(self.home, None)
            fd, self._entry.fd = self._entry.fd, -1
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
