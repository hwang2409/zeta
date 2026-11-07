"""Dormant, descriptor-safe transcript recovery transaction and records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .checkpoints import ConversationEntry, load_session_json
from .session_files import (
    child_directory,
    open_session_file,
    session_directory,
    write_json_file,
)

_BUFFER_SIZE = 1024 * 1024
_MAX_RECORD_BYTES = 64 * 1024
_MAX_RECORDS = 1024
_MAX_TEXT_BYTES = 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_EVIDENCE_SCHEMA = "zeta.transcript-quarantine.v1"
_STATE_SCHEMA = "zeta.transcript-recovery.v1"
_EVIDENCE_PREFIX = "conversation.quarantine."
_STATE_PREFIX = "conversation.recovery."

RepairState = Literal["pending", "committed", "abandoned"]
MemoryStep = Literal["not_required", "review_pending", "resumed"]
FaultStep = Literal[
    "quarantine_synced",
    "staged_active_synced",
    "evidence_published",
    "state_pending",
    "active_replaced",
    "strict_reopened",
    "state_committed",
]


class TranscriptRecoveryError(ValueError):
    """A recovery refusal that does not expose transcript content or paths."""


class TranscriptSourceChanged(TranscriptRecoveryError):
    """The diagnosed transcript changed before recovery acquired ownership."""


class TranscriptRecoveryIntegrityError(TranscriptRecoveryError):
    """Durable recovery files disagree with the active transcript."""


@dataclass(frozen=True, slots=True)
class RepairTarget:
    root_session_id: str
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        if not _safe_component(self.root_session_id):
            raise ValueError("invalid root session id")
        if self.child_run_id is not None:
            _child_run_segments(self.root_session_id, self.child_run_id)


@dataclass(frozen=True, slots=True)
class ValidatedPrefix:
    end_offset: int
    last_sequence: int
    last_entry_id: str | None
    last_lane: str

    def __post_init__(self) -> None:
        if (
            type(self.end_offset) is not int
            or self.end_offset < 0
            or type(self.last_sequence) is not int
            or self.last_sequence < 0
            or (self.last_entry_id is not None and not self.last_entry_id)
            or type(self.last_lane) is not str
            or not self.last_lane
        ):
            raise ValueError("invalid validated transcript prefix")


@dataclass(frozen=True, slots=True)
class TranscriptSourceFacts:
    device: int
    inode: int
    size: int
    source_digest: str
    prefix_digest: str
    first_bad_row: int
    first_bad_offset: int
    prefix: ValidatedPrefix
    diagnosed_at: str

    def __post_init__(self) -> None:
        if (
            type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode < 0
            or type(self.size) is not int
            or self.size < 0
            or _SHA256.fullmatch(self.source_digest) is None
            or _SHA256.fullmatch(self.prefix_digest) is None
            or type(self.first_bad_row) is not int
            or self.first_bad_row < 2
            or type(self.first_bad_offset) is not int
            or self.first_bad_offset != self.prefix.end_offset
            or self.size < self.prefix.end_offset
            or not _bounded_text(self.diagnosed_at)
        ):
            raise ValueError("invalid transcript source facts")


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    quarantine_id: str
    target: RepairTarget
    source_digest: str
    source_size: int
    first_bad_row: int
    first_bad_offset: int
    valid_prefix_end: int
    retained_last_sequence: int
    repaired_digest: str
    diagnosed_at: str
    payload_digest: str
    payload_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _EVIDENCE_SCHEMA,
            "quarantine_id": self.quarantine_id,
            "target": _target_dict(self.target),
            "source_digest": self.source_digest,
            "source_size": self.source_size,
            "first_bad_row": self.first_bad_row,
            "first_bad_offset": self.first_bad_offset,
            "valid_prefix_end": self.valid_prefix_end,
            "retained_last_sequence": self.retained_last_sequence,
            "repaired_digest": self.repaired_digest,
            "diagnosed_at": self.diagnosed_at,
            "payload_digest": self.payload_digest,
            "payload_size": self.payload_size,
        }


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    quarantine_id: str
    target: RepairTarget
    state: RepairState
    expected_source_digest: str
    repaired_digest: str
    cutoff_sequence: int
    diagnosis_timestamp: str
    memory_step: MemoryStep

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _STATE_SCHEMA,
            "quarantine_id": self.quarantine_id,
            "target": _target_dict(self.target),
            "state": self.state,
            "expected_source_digest": self.expected_source_digest,
            "repaired_digest": self.repaired_digest,
            "cutoff_sequence": self.cutoff_sequence,
            "diagnosis_timestamp": self.diagnosis_timestamp,
            "memory_step": self.memory_step,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    quarantine_id: str
    target: RepairTarget
    retained_last_sequence: int
    source_digest: str
    repaired_digest: str
    memory_review_required: bool


@dataclass(frozen=True, slots=True)
class RecoveryStatus:
    latest_committed: RecoveryRecord | None
    quarantine_count: int
    quarantine_bytes: int
    abandoned_evidence_count: int


StrictReopen = Callable[[], None]
_ALLOWED_STATE_MEMORY_STEPS: frozenset[tuple[RepairState, MemoryStep]] = frozenset(
    {
        ("pending", "not_required"),
        ("pending", "review_pending"),
        ("committed", "not_required"),
        ("committed", "review_pending"),
        ("committed", "resumed"),
        ("abandoned", "not_required"),
    }
)


def _after_recovery_step(_step: FaultStep) -> None:
    """Private fault boundary for transaction tests."""


def parse_evidence_manifest(value: Mapping[str, Any]) -> EvidenceManifest:
    """Strictly parse one immutable quarantine manifest."""
    keys = {
        "schema",
        "quarantine_id",
        "target",
        "source_digest",
        "source_size",
        "first_bad_row",
        "first_bad_offset",
        "valid_prefix_end",
        "retained_last_sequence",
        "repaired_digest",
        "diagnosed_at",
        "payload_digest",
        "payload_size",
    }
    if set(value) != keys or value.get("schema") != _EVIDENCE_SCHEMA:
        raise TranscriptRecoveryIntegrityError("invalid quarantine manifest shape")
    manifest = EvidenceManifest(
        quarantine_id=_record_id(value.get("quarantine_id")),
        target=_parse_target(value.get("target")),
        source_digest=_digest(value.get("source_digest")),
        source_size=_nonnegative(value.get("source_size"), "source size"),
        first_bad_row=_positive(value.get("first_bad_row"), "first bad row"),
        first_bad_offset=_nonnegative(
            value.get("first_bad_offset"), "first bad offset"
        ),
        valid_prefix_end=_nonnegative(
            value.get("valid_prefix_end"), "valid prefix end"
        ),
        retained_last_sequence=_nonnegative(
            value.get("retained_last_sequence"), "retained sequence"
        ),
        repaired_digest=_digest(value.get("repaired_digest")),
        diagnosed_at=_text(value.get("diagnosed_at"), "diagnosis timestamp"),
        payload_digest=_digest(value.get("payload_digest")),
        payload_size=_nonnegative(value.get("payload_size"), "payload size"),
    )
    if (
        manifest.first_bad_row < 2
        or manifest.first_bad_offset != manifest.valid_prefix_end
        or manifest.source_size < manifest.valid_prefix_end
        or manifest.payload_size != manifest.source_size
        or manifest.payload_digest != manifest.source_digest
    ):
        raise TranscriptRecoveryIntegrityError("inconsistent quarantine manifest")
    return manifest


def parse_recovery_record(value: Mapping[str, Any]) -> RecoveryRecord:
    """Strictly parse one mutable recovery-state record."""
    keys = {
        "schema",
        "quarantine_id",
        "target",
        "state",
        "expected_source_digest",
        "repaired_digest",
        "cutoff_sequence",
        "diagnosis_timestamp",
        "memory_step",
    }
    if set(value) != keys or value.get("schema") != _STATE_SCHEMA:
        raise TranscriptRecoveryIntegrityError("invalid recovery record shape")
    state = value.get("state")
    memory_step = value.get("memory_step")
    if state not in {"pending", "committed", "abandoned"}:
        raise TranscriptRecoveryIntegrityError("invalid recovery state")
    if memory_step not in {"not_required", "review_pending", "resumed"}:
        raise TranscriptRecoveryIntegrityError("invalid recovery memory step")
    if (state, memory_step) not in _ALLOWED_STATE_MEMORY_STEPS:
        raise TranscriptRecoveryIntegrityError("invalid recovery state and memory step")
    return RecoveryRecord(
        quarantine_id=_record_id(value.get("quarantine_id")),
        target=_parse_target(value.get("target")),
        state=state,
        expected_source_digest=_digest(value.get("expected_source_digest")),
        repaired_digest=_digest(value.get("repaired_digest")),
        cutoff_sequence=_nonnegative(value.get("cutoff_sequence"), "cutoff sequence"),
        diagnosis_timestamp=_text(
            value.get("diagnosis_timestamp"), "diagnosis timestamp"
        ),
        memory_step=memory_step,
    )


@contextmanager
def repair_target_directory(
    sessions_root: os.PathLike[str], target: RepairTarget
) -> Iterator[int]:
    """Pin exactly one validated root or nested child transcript directory."""
    with ExitStack() as stack:
        _, directory_fd = stack.enter_context(
            session_directory(Path(sessions_root), target.root_session_id)
        )
        current_run_id = target.root_session_id
        for segment in _child_run_segments(target.root_session_id, target.child_run_id):
            from .store._child_runs import verify_child_run_directory

            directory_fd = stack.enter_context(child_directory(directory_fd, "agents"))
            directory_fd = stack.enter_context(
                child_directory(directory_fd, str(segment))
            )
            current_run_id = f"{current_run_id}:{segment}"
            verify_child_run_directory(directory_fd, current_run_id)
        yield directory_fd


def repair_transcript(
    directory_fd: int,
    target: RepairTarget,
    source: TranscriptSourceFacts,
    *,
    strict_reopen: StrictReopen,
    memory_review_required: bool,
) -> RepairResult:
    """Run one already-owned repair transaction; no production caller exists yet."""
    quarantine_id = uuid.uuid4().hex
    payload_name = _payload_name(quarantine_id)
    evidence_name = _evidence_name(quarantine_id)
    state_name = _state_name(quarantine_id)
    staged_name = f".conversation.repair.{quarantine_id}.tmp"
    source_fd = open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY)
    evidence_published = False
    active_replaced = False
    try:
        info = os.fstat(source_fd)
        if (info.st_dev, info.st_ino, info.st_size) != (
            source.device,
            source.inode,
            source.size,
        ) or _hash_prefix(source_fd, source.prefix.end_offset) != source.prefix_digest:
            raise TranscriptSourceChanged("transcript changed after diagnosis")

        payload_fd = open_session_file(
            directory_fd, payload_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        try:
            source_digest, source_size = _copy_and_hash(
                source_fd, payload_fd, source.size
            )
            if source_digest != source.source_digest:
                raise TranscriptSourceChanged(
                    "transcript bytes changed after diagnosis"
                )
            os.fsync(payload_fd)
        finally:
            os.close(payload_fd)
        os.fsync(directory_fd)
        _after_recovery_step("quarantine_synced")

        staged_fd = open_session_file(
            directory_fd, staged_name, os.O_RDWR | os.O_CREAT | os.O_EXCL
        )
        try:
            os.lseek(source_fd, 0, os.SEEK_SET)
            _copy_exact(source_fd, staged_fd, source.prefix.end_offset)
            warning = _warning_entry(target, quarantine_id, source, source_digest)
            _write_all(staged_fd, _encoded_line(warning.to_dict()))
            os.fsync(staged_fd)
            repaired_digest = _hash_fd(staged_fd)
        finally:
            os.close(staged_fd)
        _after_recovery_step("staged_active_synced")

        manifest = EvidenceManifest(
            quarantine_id=quarantine_id,
            target=target,
            source_digest=source_digest,
            source_size=source_size,
            first_bad_row=source.first_bad_row,
            first_bad_offset=source.first_bad_offset,
            valid_prefix_end=source.prefix.end_offset,
            retained_last_sequence=source.prefix.last_sequence,
            repaired_digest=repaired_digest,
            diagnosed_at=source.diagnosed_at,
            payload_digest=source_digest,
            payload_size=source_size,
        )
        _publish_immutable_json(directory_fd, evidence_name, manifest.to_dict())
        evidence_published = True
        _after_recovery_step("evidence_published")

        record = RecoveryRecord(
            quarantine_id=quarantine_id,
            target=target,
            state="pending",
            expected_source_digest=source_digest,
            repaired_digest=repaired_digest,
            cutoff_sequence=source.prefix.last_sequence,
            diagnosis_timestamp=source.diagnosed_at,
            memory_step="review_pending" if memory_review_required else "not_required",
        )
        write_json_file(directory_fd, state_name, record.to_dict())
        _after_recovery_step("state_pending")

        os.replace(
            staged_name,
            "conversation.jsonl",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        active_replaced = True
        os.fsync(directory_fd)
        _after_recovery_step("active_replaced")
        strict_reopen()
        _after_recovery_step("strict_reopened")
        write_json_file(
            directory_fd, state_name, replace(record, state="committed").to_dict()
        )
        _after_recovery_step("state_committed")
        return RepairResult(
            quarantine_id=quarantine_id,
            target=target,
            retained_last_sequence=source.prefix.last_sequence,
            source_digest=source_digest,
            repaired_digest=repaired_digest,
            memory_review_required=memory_review_required,
        )
    except BaseException:
        if evidence_published and not active_replaced:
            _abandon_recovery(directory_fd, quarantine_id, state_name)
        raise
    finally:
        os.close(source_fd)
        _unlink_if_present(directory_fd, staged_name)
        if not evidence_published:
            _unlink_if_present(directory_fd, payload_name)


def _abandon_recovery(directory_fd: int, quarantine_id: str, state_name: str) -> None:
    try:
        pending = read_recovery_record(directory_fd, quarantine_id)
        if pending is not None:
            abandoned = replace(pending, state="abandoned", memory_step="not_required")
            write_json_file(directory_fd, state_name, abandoned.to_dict())
    except (OSError, TranscriptRecoveryError):
        pass


def read_evidence_manifest(directory_fd: int, quarantine_id: str) -> EvidenceManifest:
    value = _read_record(directory_fd, _evidence_name(_record_id(quarantine_id)))
    return parse_evidence_manifest(_mapping(value, "quarantine manifest"))


def read_recovery_record(
    directory_fd: int, quarantine_id: str
) -> RecoveryRecord | None:
    try:
        value = _read_record(directory_fd, _state_name(_record_id(quarantine_id)))
    except FileNotFoundError:
        return None
    return parse_recovery_record(_mapping(value, "recovery record"))


def reconcile_pending(
    directory_fd: int,
    target: RepairTarget,
    *,
    strict_reopen: StrictReopen,
) -> tuple[RecoveryRecord, ...]:
    """Resolve pending and manifest-only evidence from durable active bytes."""
    manifests, records = _read_all_records(directory_fd, target)
    active_digest = _hash_named_file(directory_fd, "conversation.jsonl")
    resolved: list[RecoveryRecord] = []
    for quarantine_id, manifest in manifests.items():
        record = records.get(quarantine_id)
        if record is None:
            record = RecoveryRecord(
                quarantine_id=quarantine_id,
                target=target,
                state="abandoned",
                expected_source_digest=manifest.source_digest,
                repaired_digest=manifest.repaired_digest,
                cutoff_sequence=manifest.retained_last_sequence,
                diagnosis_timestamp=manifest.diagnosed_at,
                memory_step="not_required",
            )
            write_json_file(directory_fd, _state_name(quarantine_id), record.to_dict())
        elif record.state == "pending":
            if active_digest == record.repaired_digest:
                strict_reopen()
                record = replace(record, state="committed")
            elif active_digest == record.expected_source_digest:
                record = replace(record, state="abandoned", memory_step="not_required")
            else:
                raise TranscriptRecoveryIntegrityError(
                    "active transcript does not match pending recovery"
                )
            write_json_file(directory_fd, _state_name(quarantine_id), record.to_dict())
        resolved.append(record)
    orphaned = set(records) - set(manifests)
    if orphaned:
        raise TranscriptRecoveryIntegrityError(
            "recovery state has no evidence manifest"
        )
    return tuple(resolved)


def recovery_status(directory_fd: int, target: RepairTarget) -> RecoveryStatus:
    manifests, records = _read_all_records(directory_fd, target)
    latest = max(
        (record for record in records.values() if record.state == "committed"),
        key=lambda record: (record.diagnosis_timestamp, record.quarantine_id),
        default=None,
    )
    abandoned = sum(
        records.get(quarantine_id) is None
        or records[quarantine_id].state == "abandoned"
        for quarantine_id in manifests
    )
    return RecoveryStatus(
        latest_committed=latest,
        quarantine_count=len(manifests),
        quarantine_bytes=sum(item.payload_size for item in manifests.values()),
        abandoned_evidence_count=abandoned,
    )


def _read_all_records(
    directory_fd: int, target: RepairTarget
) -> tuple[dict[str, EvidenceManifest], dict[str, RecoveryRecord]]:
    names = os.listdir(directory_fd)
    evidence_names = _recovery_names(names, _EVIDENCE_PREFIX)
    state_names = _recovery_names(names, _STATE_PREFIX)
    if len(evidence_names) + len(state_names) > _MAX_RECORDS:
        raise TranscriptRecoveryIntegrityError("too many transcript recovery records")
    manifests = _read_manifests(directory_fd, target, evidence_names)
    records = _read_recovery_states(directory_fd, target, state_names, manifests)
    return manifests, records


def _recovery_names(names: list[str], prefix: str) -> list[str]:
    return sorted(
        name for name in names if name.startswith(prefix) and name.endswith(".json")
    )


def _read_manifests(
    directory_fd: int, target: RepairTarget, names: list[str]
) -> dict[str, EvidenceManifest]:
    manifests: dict[str, EvidenceManifest] = {}
    for name in names:
        item = parse_evidence_manifest(
            _mapping(_read_record(directory_fd, name), "quarantine manifest")
        )
        if item.target != target or name != _evidence_name(item.quarantine_id):
            raise TranscriptRecoveryIntegrityError(
                "quarantine manifest target mismatch"
            )
        _validate_payload(directory_fd, item)
        if item.quarantine_id in manifests:
            raise TranscriptRecoveryIntegrityError("duplicate quarantine manifest")
        manifests[item.quarantine_id] = item
    return manifests


def _read_recovery_states(
    directory_fd: int,
    target: RepairTarget,
    names: list[str],
    manifests: Mapping[str, EvidenceManifest],
) -> dict[str, RecoveryRecord]:
    records: dict[str, RecoveryRecord] = {}
    for name in names:
        item = parse_recovery_record(
            _mapping(_read_record(directory_fd, name), "recovery record")
        )
        if item.target != target or name != _state_name(item.quarantine_id):
            raise TranscriptRecoveryIntegrityError("recovery record target mismatch")
        _validate_record_manifest(item, manifests.get(item.quarantine_id))
        if item.quarantine_id in records:
            raise TranscriptRecoveryIntegrityError("duplicate recovery record")
        records[item.quarantine_id] = item
    return records


def _validate_record_manifest(
    record: RecoveryRecord, manifest: EvidenceManifest | None
) -> None:
    if manifest is not None and (
        record.expected_source_digest != manifest.source_digest
        or record.repaired_digest != manifest.repaired_digest
        or record.cutoff_sequence != manifest.retained_last_sequence
    ):
        raise TranscriptRecoveryIntegrityError("recovery record evidence mismatch")


def _validate_payload(directory_fd: int, manifest: EvidenceManifest) -> None:
    fd = open_session_file(
        directory_fd, _payload_name(manifest.quarantine_id), os.O_RDONLY
    )
    try:
        info = os.fstat(fd)
        digest = _hash_fd(fd)
    finally:
        os.close(fd)
    if info.st_size != manifest.payload_size or digest != manifest.payload_digest:
        raise TranscriptRecoveryIntegrityError(
            "quarantine payload does not match manifest"
        )


def _warning_entry(
    target: RepairTarget,
    quarantine_id: str,
    source: TranscriptSourceFacts,
    source_digest: str,
) -> ConversationEntry:
    return ConversationEntry(
        seq=source.prefix.last_sequence + 1,
        id=uuid.uuid4().hex,
        parent_id=source.prefix.last_entry_id,
        lane=source.prefix.last_lane,
        type="warning",
        data={
            "message": "recovered transcript prefix after corrupt persisted data",
            "kind": "transcript_recovery",
            "visible": True,
            "quarantine_id": quarantine_id,
            "source_digest": source_digest,
            "first_bad_row": source.first_bad_row,
            "retained_last_sequence": source.prefix.last_sequence,
            "session_id": target.root_session_id,
            **(
                {"child_run_id": target.child_run_id}
                if target.child_run_id is not None
                else {}
            ),
        },
    )


def _copy_and_hash(source_fd: int, destination_fd: int, length: int) -> tuple[str, int]:
    os.lseek(source_fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    copied = 0
    while copied < length:
        chunk = os.read(source_fd, min(_BUFFER_SIZE, length - copied))
        if not chunk:
            raise TranscriptSourceChanged("transcript shortened during recovery")
        _write_all(destination_fd, chunk)
        digest.update(chunk)
        copied += len(chunk)
    if os.read(source_fd, 1):
        raise TranscriptSourceChanged("transcript grew during recovery")
    return digest.hexdigest(), copied


def _copy_exact(source_fd: int, destination_fd: int, length: int) -> None:
    copied = 0
    while copied < length:
        chunk = os.read(source_fd, min(_BUFFER_SIZE, length - copied))
        if not chunk:
            raise TranscriptSourceChanged("validated prefix became incomplete")
        _write_all(destination_fd, chunk)
        copied += len(chunk)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("could not write transcript recovery file")
        view = view[written:]


def _hash_prefix(fd: int, length: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        chunk = os.pread(fd, min(_BUFFER_SIZE, length - offset), offset)
        if not chunk:
            raise TranscriptSourceChanged("validated prefix became incomplete")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _hash_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, _BUFFER_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_named_file(directory_fd: int, name: str) -> str:
    fd = open_session_file(directory_fd, name, os.O_RDONLY)
    try:
        return _hash_fd(fd)
    finally:
        os.close(fd)


def _publish_immutable_json(directory_fd: int, name: str, value: object) -> None:
    data = _encoded_line(value)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = open_session_file(
        directory_fd, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        _unlink_if_present(directory_fd, temporary)


def _read_record(directory_fd: int, name: str) -> Any:
    fd = open_session_file(directory_fd, name, os.O_RDONLY)
    try:
        raw = os.read(fd, _MAX_RECORD_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _MAX_RECORD_BYTES:
        raise TranscriptRecoveryIntegrityError(
            "transcript recovery record is too large"
        )
    try:
        return load_session_json(raw)
    except ValueError as exc:
        raise TranscriptRecoveryIntegrityError(
            "invalid transcript recovery JSON"
        ) from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TranscriptRecoveryIntegrityError(f"{label} is not an object")
    return value


def _parse_target(value: Any) -> RepairTarget:
    if not isinstance(value, Mapping) or set(value) != {
        "root_session_id",
        "child_run_id",
    }:
        raise TranscriptRecoveryIntegrityError("invalid recovery target")
    root = value.get("root_session_id")
    child = value.get("child_run_id")
    if type(root) is not str or type(child) not in {str, type(None)}:
        raise TranscriptRecoveryIntegrityError("invalid recovery target")
    try:
        return RepairTarget(root, child)
    except ValueError as exc:
        raise TranscriptRecoveryIntegrityError("invalid recovery target") from exc


def _target_dict(target: RepairTarget) -> dict[str, str | None]:
    return {
        "root_session_id": target.root_session_id,
        "child_run_id": target.child_run_id,
    }


def _record_id(value: Any) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise TranscriptRecoveryIntegrityError("invalid quarantine id")
    return value


def _digest(value: Any) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TranscriptRecoveryIntegrityError("invalid recovery digest")
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TranscriptRecoveryIntegrityError(f"invalid {label}")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TranscriptRecoveryIntegrityError(f"invalid {label}")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not _bounded_text(value):
        raise TranscriptRecoveryIntegrityError(f"invalid {label}")
    return value


def _bounded_text(value: str) -> bool:
    return bool(value) and len(value.encode("utf-8")) <= _MAX_TEXT_BYTES


def _child_run_segments(
    root_session_id: str, child_run_id: str | None
) -> tuple[int, ...]:
    if child_run_id is None:
        return ()
    # Import lazily so core recovery does not initialize the store package.
    from .store._child_runs import child_run_segments

    return child_run_segments(root_session_id, child_run_id)


def _safe_component(value: Any) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\x00" not in value
    )


def _encoded_line(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _payload_name(quarantine_id: str) -> str:
    return f"{_EVIDENCE_PREFIX}{quarantine_id}.jsonl"


def _evidence_name(quarantine_id: str) -> str:
    return f"{_EVIDENCE_PREFIX}{quarantine_id}.json"


def _state_name(quarantine_id: str) -> str:
    return f"{_STATE_PREFIX}{quarantine_id}.json"


def _unlink_if_present(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
