"""Atomic session-local storage for user attention requests."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .storage_format import (
    FEATURE_EPOCHS,
    StorageFeature,
    feature_enabled,
    require_closed_variant,
)

# merge-resolution: src/zeta/attention_records.py constant ATTENTION_KIND_EPOCH -> owned by FEATURE_EPOCHS[StorageFeature.ATTENTION_KINDS]
AttentionKind = Literal["decision", "answer", "decided", "deferred"]
AttentionReason = Literal[
    "unreadable",
    "invalid_name",
    "invalid_json",
    "invalid_schema",
    "id_mismatch",
    "too_large",
]

from .core.session_files import (
    child_directory,
    open_session_file,
    session_directory,
    write_session_file,
)
from .storage_format import reject_unknown_keys, verify

_MAX_RECORD_BYTES = 256 * 1024
_MAX_BROKEN_ID_CHARS = 160


def observed_epoch_for_session(session_dir: Path) -> int:
    return verify(Path(session_dir).parent.parent).writer_epoch


def _safe_broken_id(filename: str) -> str:
    rendered = "".join(
        character if character.isprintable() else "�" for character in filename
    )
    if len(rendered) <= _MAX_BROKEN_ID_CHARS:
        return rendered
    return rendered[: _MAX_BROKEN_ID_CHARS - 3] + "..."


@dataclass(frozen=True, slots=True)
class Missing:
    pass


@dataclass(frozen=True, slots=True)
class Loaded:
    value: AttentionRecord


@dataclass(frozen=True, slots=True)
class Broken:
    id: str
    path: str
    reason: AttentionReason
    recoverability: str
    bytes: bytes


RecordState = Missing | Loaded | Broken


def broken_record_refusal(record: Broken) -> str:
    return (
        f"attention record {record.id} is broken ({record.reason}); "
        f"repair {record.path} before replying"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class AttentionRecord:
    id: str
    created_at: str
    session_id: str
    project_id: str | None
    entry_id: str | None
    entry_seq: int | None
    lane: str
    title: str
    why: str
    options: tuple[str, ...]
    recommendation: str | None
    status: str
    resolved_at: str | None = None
    fork_session_id: str | None = None
    decision: str | None = None
    kind: AttentionKind = "decision"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttentionRecord:
        reject_unknown_keys(value, "attention_record")
        allowed = set(cls.__dataclass_fields__)
        if not set(value).issubset(allowed):
            raise ValueError("unknown attention record fields")
        normalized = dict(value)
        normalized["kind"] = value.get("kind", "decision")
        require_closed_variant("attention.kind", normalized["kind"])
        normalized["options"] = tuple(value.get("options", ()))
        record = cls(**normalized)
        if (
            not isinstance(record.id, str)
            or len(record.id) != 32
            or any(ch not in "0123456789abcdef" for ch in record.id)
            or record.status not in {"open", "resolved"}
            or not isinstance(record.title, str)
            or not record.title
            or not isinstance(record.why, str)
            or not record.why
            or any(
                not isinstance(option, str) or not option for option in record.options
            )
        ):
            raise ValueError("invalid attention record")
        return record

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "entry_id": self.entry_id,
            "entry_seq": self.entry_seq,
            "lane": self.lane,
            "title": self.title,
            "why": self.why,
            "options": list(self.options),
            "recommendation": self.recommendation,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "fork_session_id": self.fork_session_id,
            "decision": self.decision,
        }
        if self.kind != "decision":
            result["kind"] = self.kind
        return result


class AttentionStore:
    """Own attention records behind one small session-local interface."""

    def __init__(self, session_dir: Path, *, writer_epoch: int):
        self.session_dir = Path(session_dir)
        self.writer_epoch = writer_epoch
        self.directory = self.session_dir / "attention"

    def request(
        self,
        *,
        session_id: str,
        project_id: str | None,
        entry_id: str | None,
        entry_seq: int | None,
        title: str,
        why: str,
        options: list[str] | tuple[str, ...] = (),
        recommendation: str | None = None,
        lane: str = "orchestrator",
        kind: AttentionKind = "decision",
    ) -> AttentionRecord:
        require_closed_variant("attention.kind", kind)
        if kind != "decision" and not feature_enabled(
            self.writer_epoch, StorageFeature.ATTENTION_KINDS
        ):
            required_epoch = FEATURE_EPOCHS[StorageFeature.ATTENTION_KINDS]
            raise ValueError(
                "non-decision attention kinds require the one-way shared cutover: "
                f"zeta storage-format advance --expected-epoch {self.writer_epoch} "
                f"--target-epoch {required_epoch}"
            )
        if kind != "decision" and (options or recommendation is not None):
            raise ValueError(
                "non-decision attention kinds cannot have options or recommendation"
            )
        if (
            not isinstance(title, str)
            or not title.strip()
            or len(title) > 160
            or not isinstance(why, str)
            or not why.strip()
            or len(why) > 32_000
            or len(options) > 20
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 2_000
                for item in options
            )
            or (
                recommendation is not None
                and (
                    not isinstance(recommendation, str)
                    or not recommendation.strip()
                    or len(recommendation) > 8_000
                )
            )
        ):
            raise ValueError("attention request fields are invalid or too large")
        record = AttentionRecord(
            id=uuid.uuid4().hex,
            created_at=_now(),
            session_id=session_id,
            project_id=project_id,
            entry_id=entry_id,
            entry_seq=entry_seq,
            lane=lane,
            title=title.strip(),
            why=why.strip(),
            options=tuple(item.strip() for item in options),
            recommendation=recommendation.strip() if recommendation else None,
            status="open",
            kind=kind,
        )
        self.replace(record, create_directory=True)
        return record

    def _broken(
        self,
        path: Path,
        reason: AttentionReason,
        raw: bytes = b"",
        *,
        record_id: str | None = None,
    ) -> Broken:
        return Broken(
            path.stem if record_id is None else record_id,
            self._relative_path(path),
            reason,
            "manual repair",
            raw,
        )

    def record_state(self, attention_id: str) -> RecordState:
        if len(attention_id) != 32 or any(
            character not in "0123456789abcdef" for character in attention_id
        ):
            raise ValueError("invalid attention id")
        path = self.directory / f"{attention_id}.json"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return Missing()
        except OSError:
            return self._broken(path, "unreadable")
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(fd)
                return self._broken(path, "unreadable")
            with os.fdopen(fd, "rb") as stream:
                raw = stream.read(_MAX_RECORD_BYTES + 1)
        except OSError:
            return self._broken(path, "unreadable")
        try:
            if len(raw) > _MAX_RECORD_BYTES:
                raise ValueError("too large")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError("not an object")
            record = AttentionRecord.from_dict(value)
            if record.id != attention_id:
                raise LookupError("id mismatch")
            return Loaded(record)
        except json.JSONDecodeError:
            reason = "invalid_json"
        except TypeError:
            reason = "invalid_schema"
        except LookupError:
            reason = "id_mismatch"
        except ValueError as exc:
            reason = "too_large" if str(exc) == "too large" else "invalid_schema"
        return self._broken(path, reason, raw)

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.session_dir).as_posix()

    def get(self, attention_id: str) -> AttentionRecord:
        state = self.record_state(attention_id)
        if isinstance(state, Loaded):
            return state.value
        if isinstance(state, Missing):
            raise FileNotFoundError(attention_id)
        raise ValueError(f"broken attention record: {state.reason}")

    def release_fork(self, attention_id: str, fork_session_id: str) -> None:
        """clear an open record's binding to `fork_session_id`, if it still has it."""
        with self._resolution_lock() as attention_fd:
            try:
                record = self._read_at(attention_fd, attention_id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return
            if record.status != "resolved" and record.fork_session_id == fork_session_id:
                self._write_at(attention_fd, replace(record, fork_session_id=None))

    @contextmanager
    def _resolution_lock(self) -> Iterator[int]:
        with (
            session_directory(self.session_dir.parent, self.session_dir.name) as (
                _,
                session_fd,
            ),
            child_directory(session_fd, "attention") as attention_fd,
        ):
            fcntl.flock(attention_fd, fcntl.LOCK_EX)
            try:
                lock_fd = open_session_file(
                    attention_fd, ".resolution.lock", os.O_RDWR | os.O_CREAT
                )
            finally:
                fcntl.flock(attention_fd, fcntl.LOCK_UN)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield attention_fd
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def list(self) -> tuple[AttentionRecord | Broken, ...]:
        try:
            paths = tuple(self.directory.iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return ()
        records: list[AttentionRecord | Broken] = []
        for path in paths:
            if path.suffix != ".json":
                continue
            if len(path.stem) != 32 or any(
                character not in "0123456789abcdef" for character in path.stem
            ):
                records.append(
                    self._broken(
                        path,
                        "invalid_name",
                        record_id=_safe_broken_id(path.name),
                    )
                )
                continue
            state = self.record_state(path.stem)
            if isinstance(state, Loaded):
                records.append(state.value)
            elif isinstance(state, Broken):
                records.append(state)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    item.created_at if isinstance(item, AttentionRecord) else item.path
                ),
            )
        )

    def replace(
        self, record: AttentionRecord, *, create_directory: bool = False
    ) -> None:
        with (
            session_directory(self.session_dir.parent, record.session_id) as (
                _,
                session_fd,
            ),
            child_directory(
                session_fd, "attention", create=create_directory
            ) as attention_fd,
        ):
            self._write_at(attention_fd, record)

    @staticmethod
    def _read_at(attention_fd: int, attention_id: str) -> AttentionRecord:
        _check_id(attention_id)
        value = read_bounded_session_json(attention_fd, f"{attention_id}.json")
        if not isinstance(value, dict):
            raise TypeError("attention record must be a JSON object")
        return AttentionRecord.from_dict(value)

    @staticmethod
    def _write_at(attention_fd: int, record: AttentionRecord) -> None:
        write_session_file(
            attention_fd,
            f"{record.id}.json",
            (
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode(),
        )


def remove_fork_session_links(
    sessions_root: Path,
    session_ids: tuple[str, ...],
    *,
    writer_epoch: int,
    charge: Callable[[], None],
) -> None:
    """Durably clear attention records bound to quarantined fork sessions."""
    remove_ids = set(session_ids)
    if not remove_ids:
        return
    try:
        entries = sorted(Path(sessions_root).iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return
    for session_dir in entries:
        charge()
        if (
            len(session_dir.name) != 32
            or any(character not in "0123456789abcdef" for character in session_dir.name)
        ):
            continue
        info = session_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("unsafe session directory during attention cleanup")
        store = AttentionStore(session_dir, writer_epoch=writer_epoch)
        _remove_store_fork_links(store, remove_ids, charge)


def _remove_store_fork_links(store, remove_ids, charge) -> None:
    for record in store.list():
        charge()
        if isinstance(record, Broken):
            raise TypeError(f"broken attention record: {record.reason}")
        if record.fork_session_id in remove_ids:
            store.replace(replace(record, fork_session_id=None))


def _check_id(attention_id: str) -> None:
    if len(attention_id) != 32 or any(
        character not in "0123456789abcdef" for character in attention_id
    ):
        raise ValueError("invalid attention id")


def _read_bounded_json_fd(fd: int) -> dict[str, Any] | list[Any]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("attention record must be a regular, unshared file")
    with os.fdopen(os.dup(fd), "rb") as stream:
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("attention record is too large")
    value = json.loads(data)
    if not isinstance(value, (dict, list)):
        raise TypeError("attention record must be a JSON object or array")
    return value


def _bounded_json(path: Path) -> dict[str, Any] | list[Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        return _read_bounded_json_fd(fd)
    finally:
        os.close(fd)


def read_bounded_session_json(
    directory_fd: int, name: str
) -> dict[str, Any] | list[Any]:
    """Read one bounded session JSON file through an already-verified directory."""
    fd = open_session_file(directory_fd, name, os.O_RDONLY | os.O_NONBLOCK)
    try:
        return _read_bounded_json_fd(fd)
    finally:
        os.close(fd)
