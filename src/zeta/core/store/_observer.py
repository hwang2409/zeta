"""observer snapshot loading helpers."""

from __future__ import annotations

import fcntl
import hashlib
import os
from typing import TYPE_CHECKING

from ..checkpoints import ConversationIntegrityError
from ..session_files import open_session_file

if TYPE_CHECKING:
    from ._store import ConversationStore


def sync_observer_snapshot(
    self: ConversationStore, *, retry_source_change: bool = True
) -> None:
    """Install only a validated prefix from one authoritative source view."""
    from ._log import _FullLoadValidator
    fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
    try:
        g0 = os.fstat(fd)
        raw = self._read_fd(fd, g0.st_size)
    finally:
        os.close(fd)
    complete_end = raw.rfind(b"\n") + 1
    if (
        complete_end == g0.st_size
        and complete_end == getattr(self, "_log_offset", 0)
        and complete_end
        and known_prefix_matches_bytes(self, raw, complete_end)
    ):
        check_observer_source(self, g0, retry_source_change)
        return
    candidate = raw[:complete_end]
    baseline = (len(self._entries), self._entry_prefix_digest)
    validator = _FullLoadValidator(self, candidate, g0)
    validation_error: ConversationIntegrityError | None = None
    try:
        for physical_row in self._iter_physical_rows(
            candidate, g0, validator.validated_prefix
        ):
            validator.accept(physical_row)
        validator.finish(install=False)
    except ConversationIntegrityError as exc:
        validation_error = exc
    has_suffix = complete_end < g0.st_size
    full_validator: _FullLoadValidator | None = None
    full_validation_error: ConversationIntegrityError | None = None
    if has_suffix:
        full_validator = _FullLoadValidator(self, raw, g0)
        try:
            for physical_row in self._iter_physical_rows(
                raw, g0, full_validator.validated_prefix
            ):
                full_validator.accept(physical_row)
            full_validator.finish(install=False)
        except ConversationIntegrityError as exc:
            full_validation_error = exc
    append_active = has_suffix and self._observer_append_active()
    if check_observer_source(self, g0, retry_source_change):
        return
    if validation_error is not None:
        raise validation_error
    if has_suffix and not append_active:
        if full_validation_error is not None:
            raise full_validation_error
        assert full_validator is not None
        validator = full_validator
        candidate = raw
    for attribute in _FullLoadValidator._STATE_ATTRIBUTES:
        setattr(self, attribute, getattr(validator.state, attribute))
    delta = (
        self._installed_delta_from_baseline(self._entries, baseline)
        if self.on_entries_installed is not None
        else ()
    )
    set_observer_log_stat(self, g0, len(candidate), raw)
    self._publish_entries_installed(delta)

def check_observer_source(
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
    from ._log import ObserverSnapshotChanged
    raise ObserverSnapshotChanged("observer transcript changed during read")

def observer_append_active(self: ConversationStore) -> bool:
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

def known_prefix_matches_bytes(self: ConversationStore, raw: bytes, offset: int) -> bool:
    from ._log import PREFIX_FINGERPRINT_BYTES
    known = getattr(self, "_log_prefix_fingerprint", None)
    if known is None:
        return False
    length = min(offset, PREFIX_FINGERPRINT_BYTES)
    return hashlib.blake2b(raw[offset - length : offset], digest_size=16).digest() == known

def set_observer_log_stat(
    self: ConversationStore, stat: os.stat_result, offset: int, raw: bytes
) -> None:
    from ._log import PREFIX_FINGERPRINT_BYTES
    length = min(offset, PREFIX_FINGERPRINT_BYTES)
    self._log_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
    self._log_offset = offset
    self._log_mtime_ns = stat.st_mtime_ns
    self._log_prefix_fingerprint = hashlib.blake2b(
        raw[offset - length : offset], digest_size=16
    ).digest()
    self._transcript_generation += 1
