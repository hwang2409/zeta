"""Bounded physical-row streaming for transcript validation."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ..checkpoints import ConversationIntegrityError

_READ_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class PhysicalRow:
    row_number: int
    byte_offset: int
    prefix_digest: str
    data: bytes
    terminated: bool
    final: bool


class TranscriptStream:
    """Read one transcript generation without retaining the whole file."""

    def __init__(
        self,
        fd: int,
        stat: os.stat_result,
        *,
        read: Callable[[int, int], bytes] = os.read,
    ) -> None:
        self.fd = fd
        self.stat = stat
        self._read = read
        self._buffer = bytearray()
        self._read_offset = 0
        self._row_offset = 0
        self._source_digest = hashlib.sha256()
        self._prefix_digest = hashlib.sha256()
        self._exhausted = False

    def _fill(self) -> bool:
        remaining = self.stat.st_size - self._read_offset
        if remaining <= 0 or self._exhausted:
            return False
        chunk = self._read(self.fd, min(remaining, _READ_SIZE))
        if not chunk:
            self._exhausted = True
            return False
        self._buffer.extend(chunk)
        self._source_digest.update(chunk)
        self._read_offset += len(chunk)
        return True

    def _row_end(self) -> int | None:
        search_from = 0
        while True:
            lf_index = self._buffer.find(b"\n", search_from)
            cr_index = self._buffer.find(b"\r", search_from)
            if lf_index >= 0 and (cr_index < 0 or lf_index < cr_index):
                return lf_index + 1
            if cr_index >= 0:
                if cr_index + 1 < len(self._buffer):
                    return cr_index + 2 if self._buffer[cr_index + 1] == 10 else cr_index + 1
                if self._read_offset >= self.stat.st_size or self._exhausted:
                    return cr_index + 1
                search_from = cr_index
            else:
                search_from = len(self._buffer)
            if not self._fill():
                return len(self._buffer) or None

    def rows(self) -> Iterator[PhysicalRow]:
        row_number = 0
        while self._buffer or self._read_offset < self.stat.st_size:
            row_end = self._row_end()
            if row_end is None:
                return
            row_number += 1
            data = bytes(self._buffer[:row_end])
            del self._buffer[:row_end]
            offset = self._row_offset
            prefix_digest = self._prefix_digest.hexdigest()
            self._prefix_digest.update(data)
            self._row_offset += len(data)
            yield PhysicalRow(
                row_number=row_number,
                byte_offset=offset,
                prefix_digest=prefix_digest,
                data=data,
                terminated=data.endswith(b"\n"),
                final=(
                    not self._buffer
                    and self._row_offset == self.stat.st_size
                    and self._read_offset == self.stat.st_size
                ),
            )

    def finish_digest(self) -> str:
        while self._read_offset < self.stat.st_size and self._fill():
            pass
        current = os.fstat(self.fd)
        if current.st_dev != self.stat.st_dev or current.st_ino != self.stat.st_ino:
            reason = "identity changed"
        elif current.st_size < self.stat.st_size:
            reason = "file shrank"
        elif self._read_offset < self.stat.st_size:
            reason = "short read"
        else:
            return self._source_digest.hexdigest()
        raise ConversationIntegrityError(
            f"transcript changed during validation: {reason}"
        )
