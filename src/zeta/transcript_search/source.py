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
from zeta.memory.safety import redact_secrets

MAX_DIAGNOSTIC_CHARS = 240
MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROW_BYTES = 32 * 1024 * 1024
PREFIX_FINGERPRINT_BYTES = 64 * 1024
_SAFE_ARGUMENTS = ("command", "path", "url", "query", "name", "cwd")


@dataclass(frozen=True, slots=True)
class _Cursor:
    byte_offset: int
    last_seq: int
    source_device: int
    source_inode: int
    source_mtime_ns: int
    source_size: int
    prefix_fingerprint: str
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
    source_size: int
    prefix_fingerprint: str
    last_entry_id: str | None
    full: bool


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


def _read_transcript(path: Path, cursor: _Cursor | None) -> _ReadResult:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript is not a bounded regular file")
        full = cursor is None or not _cursor_matches_source(fd, info, cursor)
        offset = 0 if full else cursor.byte_offset
        os.lseek(fd, offset, os.SEEK_SET)
        rows: list[dict[str, Any]] = []
        last_seq = 0 if full else cursor.last_seq
        last_entry_id = None if full else cursor.last_entry_id
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while True:
                start = handle.tell()
                line = handle.readline(MAX_ROW_BYTES + 1)
                if not line:
                    offset = start
                    break
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
            info.st_size,
            _prefix_fingerprint(fd, offset),
            last_entry_id,
            full,
        )
    finally:
        os.close(fd)


def _cursor_matches_source(fd: int, info: os.stat_result, cursor: _Cursor) -> bool:
    if (
        cursor.source_device != info.st_dev
        or cursor.source_inode != info.st_ino
        or cursor.byte_offset > info.st_size
        or cursor.source_size > info.st_size
        or not cursor.prefix_fingerprint
    ):
        return False
    if info.st_size == cursor.source_size and info.st_mtime_ns != cursor.source_mtime_ns:
        return False
    if cursor.byte_offset and os.pread(fd, 1, cursor.byte_offset - 1) != b"\n":
        return False
    return _prefix_fingerprint(fd, cursor.byte_offset) == cursor.prefix_fingerprint


def _prefix_fingerprint(fd: int, offset: int) -> str:
    digest = hashlib.sha256()
    first_size = min(offset, PREFIX_FINGERPRINT_BYTES)
    digest.update(os.pread(fd, first_size, 0))
    tail_start = max(first_size, offset - PREFIX_FINGERPRINT_BYTES)
    digest.update(os.pread(fd, offset - tail_start, tail_start))
    digest.update(str(offset).encode("ascii"))
    return digest.hexdigest()


