"""observer snapshot mixin."""

from __future__ import annotations

import fcntl
import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..checkpoints import ConversationIntegrityError
from ..session_files import open_session_file
from ..transcript_recovery import ObserverSnapshotChanged
from ..transcript_recovery.stream import TranscriptStream
from ._log import (
    PREFIX_FINGERPRINT_BYTES,
    _FullLoadValidator,
)

if TYPE_CHECKING:
    from ._store import ConversationStore


def _resize_stat(stat: os.stat_result, size: int) -> os.stat_result:
    """Return ``stat`` with ``st_size`` set to ``size``, identity fields intact."""
    values = list(stat)
    values[6] = size
    return os.stat_result(
        values,
        {
            "st_atime_ns": stat.st_atime_ns,
            "st_mtime_ns": stat.st_mtime_ns,
            "st_ctime_ns": stat.st_ctime_ns,
        },
    )


@dataclass(frozen=True, slots=True)
class _ObserverValidation:
    """One validated transcript range: accepted state plus any integrity error."""

    state: ConversationStore
    error: ConversationIntegrityError | None


class ObserverSnapshotMixin:
    def _sync_observer_snapshot(
        self: ConversationStore, *, retry_source_change: bool = True
    ) -> None:
        """Install only a validated prefix from one authoritative source view."""
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            g0 = os.fstat(fd)
            raw = self._read_fd(fd, g0.st_size)
            complete_end = raw.rfind(b"\n") + 1
            if (
                complete_end == g0.st_size
                and complete_end == getattr(self, "_log_offset", 0)
                and complete_end
                and self._known_prefix_matches_bytes(raw, complete_end)
            ):
                self._check_observer_source(g0, retry_source_change)
                return
            candidate = raw[:complete_end]
            baseline = (len(self._entries), self._entry_prefix_digest)
            validation = self._validate_observer_range(fd, g0, candidate)
            has_suffix = complete_end < g0.st_size
            full_validation: _ObserverValidation | None = None
            if has_suffix:
                full_validation = self._validate_observer_range(fd, g0, raw)
            append_active = has_suffix and self._observer_append_active()
            if self._check_observer_source(g0, retry_source_change):
                return
            if validation.error is not None:
                raise validation.error
            if has_suffix and not append_active:
                assert full_validation is not None
                if full_validation.error is not None:
                    raise full_validation.error
                validation = full_validation
                candidate = raw
            for attribute in _FullLoadValidator._STATE_ATTRIBUTES:
                setattr(self, attribute, getattr(validation.state, attribute))
            self._resident_rows = validation.state._resident_rows
            delta = (
                self._installed_delta_from_baseline(self._entries, baseline)
                if self.on_entries_installed is not None
                else ()
            )
            self._set_observer_log_stat(g0, len(candidate), raw)
            self._publish_entries_installed(delta)
        finally:
            os.close(fd)

    def _validate_observer_range(
        self: ConversationStore, fd: int, g0: os.stat_result, data: bytes
    ) -> _ObserverValidation:
        """Validate one in-memory byte range through the shared streaming validator.

        The authoritative bytes are read once (``data``); feeding them to
        ``TranscriptStream`` reuses the loader's validator and physical-row
        iteration without a second parse. ``finish_digest`` re-stats ``fd`` only
        on the error path, so a resized stat keeps the digest bound to ``data``.
        """
        stat = g0 if len(data) == g0.st_size else _resize_stat(g0, len(data))
        cursor = 0

        def read(_fd: int, size: int) -> bytes:
            nonlocal cursor
            chunk = data[cursor : cursor + size]
            cursor += len(chunk)
            return chunk

        source = TranscriptStream(fd, stat, read=read)
        validator = _FullLoadValidator(self, source)
        try:
            for physical_row in self._iter_physical_rows(
                source, validator.validated_prefix
            ):
                validator.accept(physical_row)
            validator.finish(install=False)
        except ConversationIntegrityError as exc:
            return _ObserverValidation(validator.state, exc)
        return _ObserverValidation(validator.state, None)

    def _check_observer_source(
        self: ConversationStore,
        g0: os.stat_result,
        retry_source_change: bool,
    ) -> bool:
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            g1 = os.fstat(fd)
        finally:
            os.close(fd)
        if (g0.st_dev, g0.st_ino, g0.st_ctime_ns, g0.st_size, g0.st_mtime_ns) == (
            g1.st_dev,
            g1.st_ino,
            g1.st_ctime_ns,
            g1.st_size,
            g1.st_mtime_ns,
        ):
            return False
        replaced = (g0.st_dev, g0.st_ino) != (g1.st_dev, g1.st_ino)
        if retry_source_change and (not self._entries or not replaced):
            self._sync_observer_snapshot(retry_source_change=False)
            return True
        raise ObserverSnapshotChanged("observer transcript changed during read")

    def _observer_append_active(self: ConversationStore) -> bool:
        fd = open_session_file(self.directory_fd, ".lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
        finally:
            os.close(fd)

    def _known_prefix_matches_bytes(self: ConversationStore, raw: bytes, offset: int) -> bool:
        known = getattr(self, "_log_prefix_fingerprint", None)
        if known is None:
            return False
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        return hashlib.blake2b(raw[offset - length : offset], digest_size=16).digest() == known

    def _set_observer_log_stat(
        self: ConversationStore, stat: os.stat_result, offset: int, raw: bytes
    ) -> None:
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        self._log_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._log_offset = offset
        self._log_mtime_ns = stat.st_mtime_ns
        self._log_prefix_fingerprint = hashlib.blake2b(
            raw[offset - length : offset], digest_size=16
        ).digest()
        self._transcript_generation += 1

