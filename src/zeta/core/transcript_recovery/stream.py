"""Bounded physical-row streaming for full transcript validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..checkpoints import ConversationIntegrityError, load_session_json

if TYPE_CHECKING:
    from ..store._log import TranscriptLoadError
    from ..store._store import ConversationStore
    from . import ValidatedPrefix


type TranscriptFailureKind = Literal[
    "torn_tail",
    "invalid_row",
    "broken_links",
    "unsupported_format",
    "unrecoverable_header",
]


@dataclass(frozen=True, slots=True)
class PhysicalRow:
    row_number: int
    byte_offset: int
    prefix_digest: str
    value: dict[str, Any] | None
    terminated: bool
    torn: bool = False


class StreamingTranscript:
    """Read and fingerprint one transcript generation without retaining its bytes."""

    def __init__(self, store: ConversationStore, fd: int, stat: os.stat_result) -> None:
        self.store = store
        self.fd = fd
        self.stat = stat
        self._handle = os.fdopen(fd, "rb", buffering=64 * 1024, closefd=False)
        self._digest = hashlib.sha256()
        self._offset = 0
        self._complete = False

    def finish_digest(self) -> str:
        while not self._complete:
            remaining = self.stat.st_size - self._offset
            if remaining <= 0:
                self._complete = True
                break
            chunk = self._handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                self._complete = True
                break
            self._digest.update(chunk)
            self._offset += len(chunk)
        current = os.fstat(self.fd)
        changed_identity = (
            current.st_dev != self.stat.st_dev or current.st_ino != self.stat.st_ino
        )
        changed_prefix = current.st_size < self.stat.st_size or (
            current.st_size == self.stat.st_size
            and (
                current.st_mtime_ns != self.stat.st_mtime_ns
                or current.st_ctime_ns != self.stat.st_ctime_ns
            )
        )
        if changed_identity or changed_prefix:
            raise ConversationIntegrityError(
                f"transcript changed during validation: {self.store.path}"
            )
        return self._digest.hexdigest()

    def error(
        self,
        row: PhysicalRow | None,
        *,
        kind: TranscriptFailureKind,
        reason_code: str,
        repairable: bool,
        validated_prefix: ValidatedPrefix | None = None,
    ) -> TranscriptLoadError:
        source_digest = self.finish_digest()
        return self.store._transcript_error(
            kind=kind,
            row_number=None if row is None else row.row_number,
            byte_offset=None if row is None else row.byte_offset,
            valid_prefix_end=(
                None if row is None or row.row_number == 1 else row.byte_offset
            ),
            reason_code=reason_code,
            repairable=repairable,
            stat=self.stat,
            prefix_digest=(
                hashlib.sha256().hexdigest() if row is None else row.prefix_digest
            ),
            source_digest=source_digest,
            validated_prefix=validated_prefix,
        )

    def rows(
        self,
        validated_prefix: Callable[[int], ValidatedPrefix],
    ) -> Iterator[PhysicalRow]:
        row_number = 0
        while self._offset < self.stat.st_size:
            remaining = self.stat.st_size - self._offset
            line = self._handle.readline(remaining)
            if not line:
                self._complete = True
                return
            row_number += 1
            if row_number > 1 and (row_number - 1) % 64 == 0:
                time.sleep(0.0001)
            row_offset = self._offset
            prefix_digest = self._digest.hexdigest()
            self._digest.update(line)
            self._offset += len(line)
            terminated = line.endswith(b"\n")
            if not terminated:
                self._complete = True
            try:
                value = load_session_json(line)
            except ConversationIntegrityError as exc:
                torn = not terminated and isinstance(
                    exc.__cause__, (json.JSONDecodeError, UnicodeError)
                )
                row = PhysicalRow(
                    row_number,
                    row_offset,
                    prefix_digest,
                    None,
                    terminated,
                    torn=torn,
                )
                if torn and not self.store._read_only:
                    yield row
                    return
                repairable = row_number > 1
                raise self.error(
                    row,
                    kind=(
                        "unrecoverable_header"
                        if row_number == 1
                        else "torn_tail"
                        if torn
                        else "invalid_row"
                    ),
                    reason_code="invalid_json",
                    repairable=repairable,
                    validated_prefix=(
                        validated_prefix(row_offset) if repairable else None
                    ),
                ) from exc
            row = PhysicalRow(
                row_number,
                row_offset,
                prefix_digest,
                value if isinstance(value, dict) else None,
                terminated,
            )
            if not isinstance(value, dict):
                repairable = row_number > 1
                raise self.error(
                    row,
                    kind=("unrecoverable_header" if row_number == 1 else "invalid_row"),
                    reason_code="row_not_object",
                    repairable=repairable,
                    validated_prefix=(
                        validated_prefix(row_offset) if repairable else None
                    ),
                )
            yield row
            if not terminated:
                return
        self._complete = True
