from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.support.branch_reads import resident_entries
from zeta.core.checkpoints import ConversationIntegrityError, load_session_json
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent

if TYPE_CHECKING:
    from zeta.core.transcript_recovery.stream import PhysicalRow, TranscriptStream


def _message(text: str) -> Message:
    return Message(MessageRole.ASSISTANT, [TextContent(text)])


def test_full_load_never_requests_one_transcript_sized_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="streamed-full-load")
    try:
        store.append_message(_message("x" * (128 * 1024)))
        store.append_message(_message("final"))
        transcript_size = store.path.stat().st_size
    finally:
        store.close()

    requested_reads: list[int] = []
    original = ConversationStore._read_fd

    def record_read(fd: int, length: int) -> bytes:
        requested_reads.append(length)
        return original(fd, length)

    # test-contract: seam catches whole-file full-load reads on main
    monkeypatch.setattr(ConversationStore, "_read_fd", staticmethod(record_read))
    reopened = ConversationStore(
        tmp_path, session_id="streamed-full-load", _read_only=True
    )
    try:
        assert len(resident_entries(reopened)) == 2
        assert transcript_size not in requested_reads, (
            "full load requested one transcript-sized read"
        )
    finally:
        reopened.close()


def _legacy_classification(raw: bytes) -> tuple[object, ...]:
    lines = raw.splitlines(keepends=True)
    offsets: list[tuple[int, int, bool]] = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            value = load_session_json(line)
        except ConversationIntegrityError as exc:
            torn = (
                index == len(lines) - 1
                and not line.endswith(b"\n")
                and isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError))
            )
            return (
                "error",
                "torn_tail" if torn else "invalid_row",
                index + 1,
                offset,
                hashlib.sha256(raw[:offset]).hexdigest(),
                hashlib.sha256(raw).hexdigest(),
            )
        if not isinstance(value, dict):
            return (
                "error",
                "invalid_row",
                index + 1,
                offset,
                hashlib.sha256(raw[:offset]).hexdigest(),
                hashlib.sha256(raw).hexdigest(),
            )
        offsets.append((offset, len(line), line.endswith(b"\n")))
        offset += len(line)
    return ("valid", tuple(offsets), hashlib.sha256(raw).hexdigest())


def _stream_classification(
    rows: Iterable[PhysicalRow], source: TranscriptStream
) -> tuple[object, ...]:
    offsets: list[tuple[int, int, bool]] = []
    for row in rows:
        try:
            value = load_session_json(row.data)
        except ConversationIntegrityError as exc:
            torn = (
                row.final
                and not row.terminated
                and isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError))
            )
            return (
                "error",
                "torn_tail" if torn else "invalid_row",
                row.row_number,
                row.byte_offset,
                row.prefix_digest,
                source.finish_digest(),
            )
        if not isinstance(value, dict):
            return (
                "error",
                "invalid_row",
                row.row_number,
                row.byte_offset,
                row.prefix_digest,
                source.finish_digest(),
            )
        offsets.append((row.byte_offset, len(row.data), row.terminated))
    return ("valid", tuple(offsets), source.finish_digest())


def _json_line(value: object, ending: bytes = b"\n") -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + ending


def _boundary_row() -> bytes:
    prefix = b'{"text":"'
    suffix = b'"}\n'
    return prefix + (b"x" * (64 * 1024 - len(prefix) - len(suffix))) + suffix


def _split_boundary_crlf_row() -> bytes:
    prefix = b'{"text":"'
    suffix = b'"}'
    return prefix + (b"x" * (64 * 1024 - len(prefix) - len(suffix) - 1)) + suffix + b"\r\n"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(_json_line({"a": 1}) + _json_line({"b": 2}), id="valid"),
        pytest.param(_json_line({"a": 1}) + b"{\n" + _json_line({"b": 2}), id="corrupt-middle"),
        pytest.param(_json_line({"a": 1}) + b'{"b":', id="torn-tail"),
        pytest.param(_json_line({"a": 1}, b""), id="missing-final-newline"),
        pytest.param(_json_line({"a": 1}, b"\r\n") + _json_line({"b": 2}, b"\r\n"), id="crlf"),
        pytest.param(_json_line({"a": 1}, b"\r") + _json_line({"b": 2}, b"\r"), id="cr"),
        pytest.param(_json_line({"a": 1}) + b"\n" + _json_line({"b": 2}), id="empty-line"),
        pytest.param(_json_line({"text": "carriage\rline\u2028next\u0085nel"}), id="embedded-line-characters"),
        pytest.param(_json_line({"text": "x" * (128 * 1024)}), id="larger-than-chunk"),
        pytest.param(b"", id="empty"),
        pytest.param(_boundary_row() + _json_line({"after": True}), id="exact-chunk-boundary"),
        pytest.param(_split_boundary_crlf_row() + _json_line({"after": True}), id="split-boundary-crlf"),
    ],
)
def test_stream_classification_matches_whole_file_splitlines(
    tmp_path: Path, raw: bytes
) -> None:
    from zeta.core.transcript_recovery.stream import TranscriptStream

    path = tmp_path / "conversation.jsonl"
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY)
    try:
        source = TranscriptStream(fd, os.fstat(fd))
        actual = _stream_classification(source.rows(), source)
    finally:
        os.close(fd)

    assert actual == _legacy_classification(raw)


def test_stream_accepts_append_metadata_race_but_rejects_shrink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.transcript_recovery import stream as stream_module
    from zeta.core.transcript_recovery.stream import TranscriptStream

    initial = _json_line({"before": True})
    appended = _json_line({"after": True})
    path = tmp_path / "conversation.jsonl"
    path.write_bytes(initial)
    fd = os.open(path, os.O_RDONLY)
    initial_stat = os.fstat(fd)
    appended_during_read = False

    def read_then_append(read_fd: int, length: int) -> bytes:
        nonlocal appended_during_read
        chunk = os.read(read_fd, length)
        with path.open("ab") as transcript:
            transcript.write(appended)
        appended_during_read = True
        return chunk

    metadata_before_size = SimpleNamespace(
        st_dev=initial_stat.st_dev,
        st_ino=initial_stat.st_ino,
        st_size=initial_stat.st_size,
        st_mtime_ns=initial_stat.st_mtime_ns + 1,
        st_ctime_ns=initial_stat.st_ctime_ns + 1,
    )
    try:
        source = TranscriptStream(fd, initial_stat, read=read_then_append)
        rows = list(source.rows())
        with monkeypatch.context() as patch:
            patch.setattr(
                stream_module.os,
                "fstat",
                lambda _fd: metadata_before_size,
            )
            try:
                digest = source.finish_digest()
            except ConversationIntegrityError as exc:
                pytest.fail(f"append-only prefix validation failed: {exc}")
    finally:
        os.close(fd)

    assert appended_during_read
    assert [row.data for row in rows] == [initial]
    assert digest == hashlib.sha256(initial).hexdigest()
    assert path.read_bytes() == initial + appended

    fd = os.open(path, os.O_RDONLY)
    try:
        source = TranscriptStream(fd, os.fstat(fd))
        list(source.rows())
        path.write_bytes(b"")
        with pytest.raises(ConversationIntegrityError, match="file shrank"):
            source.finish_digest()
    finally:
        os.close(fd)


def test_transcript_stream_bounds_each_read(
    tmp_path: Path,
) -> None:
    from zeta.core.transcript_recovery.stream import TranscriptStream

    raw = _json_line({"text": "x" * (256 * 1024)})
    path = tmp_path / "conversation.jsonl"
    path.write_bytes(raw)
    requests: list[int] = []

    def counted_read(fd: int, length: int) -> bytes:
        requests.append(length)
        return os.read(fd, length)

    fd = os.open(path, os.O_RDONLY)
    try:
        source = TranscriptStream(fd, os.fstat(fd), read=counted_read)
        rows = list(source.rows())
        digest = source.finish_digest()
    finally:
        os.close(fd)

    assert [row.data for row in rows] == raw.splitlines(keepends=True)
    assert requests
    assert max(requests) <= 64 * 1024
    assert digest == hashlib.sha256(raw).hexdigest()
