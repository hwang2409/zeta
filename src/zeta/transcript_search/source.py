"""Bounded, sanitized transcript source snapshots for the search index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zeta.core.checkpoints import ConversationEntry
from zeta.core.store import PersistedAppend
from zeta.memory.safety import redact_secrets

MAX_DIAGNOSTIC_CHARS = 240
MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROW_BYTES = 32 * 1024 * 1024
_SAFE_ARGUMENTS = ("command", "path", "url", "query", "name", "cwd")


@dataclass(frozen=True, slots=True)
class _Cursor:
    byte_offset: int
    last_seq: int
    source_device: int
    source_inode: int
    source_mtime_ns: int
    source_ctime_ns: int
    source_size: int
    incremental_ready: bool
    last_entry_id: str | None
    active_tail: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ReadResult:
    rows: tuple[dict[str, Any], ...]
    byte_offset: int
    last_seq: int
    source_device: int
    source_inode: int
    source_mtime_ns: int
    source_ctime_ns: int
    source_size: int
    last_entry_id: str | None
    full: bool
    bytes_read: int


def _bounded_text(value: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean if len(clean) <= max_chars else clean[: max_chars - 1] + "…"


def _project_entry(entry: ConversationEntry) -> dict[str, Any]:
    """Retain only sanitized evidence and branch fields needed for later rendering."""

    data: dict[str, Any] = {}
    for key in ("created_at", "timestamp"):
        value = entry.data.get(key)
        if isinstance(value, str):
            data[key] = redact_secrets(value)
    if entry.type == "message":
        message = entry.data.get("message")
        if isinstance(message, Mapping):
            projected = _project_message(message)
            if projected is not None:
                data["message"] = projected
    elif entry.type == "notification" and entry.data.get("kind") == "agent_completion":
        data.update(
            {
                "kind": "agent_completion",
                "description": redact_secrets(
                    str(entry.data.get("description") or "child agent")
                ),
                "status": redact_secrets(str(entry.data.get("status") or "unknown")),
                "text": redact_secrets(str(entry.data.get("text") or "")),
            }
        )
    return ConversationEntry(
        seq=entry.seq,
        id=entry.id,
        parent_id=entry.parent_id,
        lane=entry.lane,
        type=entry.type,
        data=data,
    ).to_dict()


def _project_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if not isinstance(role, str):
        return None
    metadata = message.get("metadata")
    projected_metadata = {
        key: value
        for key in ("zeta.origin", "response_state", "turn_failed")
        if isinstance(metadata, Mapping)
        and isinstance((value := metadata.get(key)), (str, bool))
    }
    content: list[dict[str, Any]] = []
    blocks = message.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                content.append(
                    {"type": "text", "text": redact_secrets(block["text"])}
                )
            elif block.get("type") == "tool_use":
                call = block.get("tool_call")
                if isinstance(call, Mapping):
                    content.append(
                        {"type": "tool_use", "tool_call": _project_tool_call(call)}
                    )
    projected: dict[str, Any] = {
        "role": role,
        "content": content,
        "metadata": projected_metadata,
    }
    result = message.get("tool_result")
    if isinstance(result, Mapping):
        raw = result.get("content")
        text = raw if isinstance(raw, str) else ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        selected = lines if len(lines) <= 1 else [lines[0], lines[-1]]
        projected["tool_result"] = {
            "tool_call_id": result.get("tool_call_id"),
            "content": "\n".join(
                _bounded_text(redact_secrets(line), MAX_DIAGNOSTIC_CHARS)
                for line in selected
            ),
            "source_bytes": len(text.encode("utf-8")),
            "is_error": result.get("is_error") is True,
        }
    return projected


def _project_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    projected_arguments: dict[str, str] = {}
    if isinstance(arguments, Mapping):
        for key in _SAFE_ARGUMENTS:
            value = arguments.get(key)
            if isinstance(value, str):
                projected_arguments[key] = _bounded_text(redact_secrets(value), 320)
    return {
        "id": call.get("id"),
        "name": call.get("name"),
        "arguments": projected_arguments,
    }


def _read_transcript(
    path: Path,
    cursor: _Cursor | None,
    append_receipts: tuple[PersistedAppend, ...] | None = None,
) -> _ReadResult:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript is not a bounded regular file")
        unchanged = (
            cursor is not None
            and cursor.incremental_ready
            and _source_identity(info) == _cursor_identity(cursor)
        )
        verified = (
            cursor is not None
            and cursor.incremental_ready
            and append_receipts is not None
            and _receipts_extend_cursor(fd, info, cursor, append_receipts)
        )
        full = cursor is None or not (unchanged or verified)
        offset = 0 if full else cursor.byte_offset
        os.lseek(fd, offset, os.SEEK_SET)
        rows: list[dict[str, Any]] = []
        last_seq = 0 if full else cursor.last_seq
        last_entry_id = None if full else cursor.last_entry_id
        bytes_read = 0
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while True:
                start = handle.tell()
                remaining = info.st_size - start
                if remaining <= 0:
                    offset = start
                    break
                line = handle.readline(min(MAX_ROW_BYTES + 1, remaining))
                if not line:
                    offset = start
                    break
                bytes_read += len(line)
                if len(line) > MAX_ROW_BYTES:
                    raise ValueError("transcript row exceeds size limit")
                if not line.endswith(b"\n"):
                    offset = start
                    break
                offset = handle.tell()
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("transcript row is not an object")
                if value.get("type") == "header":
                    if not full or rows:
                        if not full:
                            return _read_transcript(path, None)
                        raise ValueError("unexpected transcript header")
                    continue
                entry = ConversationEntry.from_dict(value)
                if entry.seq != last_seq + 1:
                    if not full:
                        return _read_transcript(path, None)
                    raise ValueError("transcript sequence is not contiguous")
                rows.append(_project_entry(entry))
                last_seq = entry.seq
                last_entry_id = entry.id
        return _ReadResult(
            tuple(rows),
            offset,
            last_seq,
            info.st_dev,
            info.st_ino,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_size,
            last_entry_id,
            full,
            bytes_read,
        )
    finally:
        os.close(fd)


def _source_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _cursor_identity(cursor: _Cursor) -> tuple[int, int, int, int, int]:
    return (
        cursor.source_device,
        cursor.source_inode,
        cursor.source_size,
        cursor.source_mtime_ns,
        cursor.source_ctime_ns,
    )


def _receipts_extend_cursor(
    fd: int,
    info: os.stat_result,
    cursor: _Cursor,
    receipts: tuple[PersistedAppend, ...],
) -> bool:
    if not receipts or cursor.byte_offset != cursor.source_size:
        return False
    expected_offset = cursor.source_size
    expected_identity = (
        cursor.source_device,
        cursor.source_inode,
        cursor.source_mtime_ns,
        cursor.source_ctime_ns,
    )
    for receipt in receipts:
        before_identity = (
            receipt.source_device,
            receipt.source_inode,
            receipt.before_mtime_ns,
            receipt.before_ctime_ns,
        )
        if receipt.start_offset != expected_offset or before_identity != expected_identity:
            return False
        length = receipt.end_offset - receipt.start_offset
        if length <= 0:
            return False
        appended = os.pread(fd, length, receipt.start_offset)
        if len(appended) != length or hashlib.sha256(appended).hexdigest() != receipt.digest:
            return False
        expected_offset = receipt.end_offset
        expected_identity = (
            receipt.source_device,
            receipt.source_inode,
            receipt.after_mtime_ns,
            receipt.after_ctime_ns,
        )
    current_identity = (
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    return expected_offset == info.st_size and expected_identity == current_identity

