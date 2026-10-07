"""Stable, sanitized evidence units rendered from durable transcript rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from zeta.memory.safety import redact_secrets

MAX_UNIT_BYTES = 12 * 1024
MAX_TOOL_DIGEST_BYTES = 4 * 1024
MAX_DIAGNOSTIC_CHARS = 240
_UNIT_SCHEMA_VERSION = 1
_SAFE_ARGUMENTS = ("command", "path", "url", "query", "name", "cwd")


@dataclass(frozen=True, slots=True)
class TranscriptUnit:
    """One bounded search document with a trusted source range."""

    schema_version: int
    unit_id: str
    turn_id: str
    project_id: str
    session_id: str
    seq_start: int
    seq_end: int
    started_at: str | None
    ended_at: str | None
    origin: str
    kind: Literal["turn", "child_report"]
    chunk_index: int
    chunk_count: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranscriptUnit:
        return cls(**dict(value))


@dataclass(slots=True)
class _Turn:
    rows: list[Mapping[str, Any]]
    human: list[str]
    inputs: list[str]
    assistant: list[str]
    tools: list[str]
    completed: bool = False


def render_transcript_units(
    project_id: str,
    session_id: str,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[TranscriptUnit, ...]:
    """Render completed turns and parent-visible child reports deterministically."""

    ordered = sorted(
        (row for row in rows if type(row.get("seq")) is int),
        key=lambda row: int(row["seq"]),
    )
    units: list[TranscriptUnit] = []
    turn: _Turn | None = None
    call_names: dict[str, str] = {}
    for row in ordered:
        report = _child_report(row)
        if report is not None:
            units.extend(_report_units(project_id, session_id, row, report))
            continue
        message = _message(row)
        if message is None:
            continue
        role = message.get("role")
        if role == "user":
            if turn is not None and turn.completed:
                units.extend(_turn_units(project_id, session_id, turn))
                turn = None
            if turn is None:
                turn = _Turn([], [], [], [], [])
            turn.rows.append(row)
            text = _message_text(message)
            if text:
                if _is_human(message, row):
                    turn.human.append(text)
                else:
                    turn.inputs.append(text)
            continue
        if turn is None:
            continue
        turn.rows.append(row)
        if role == "assistant":
            text = _message_text(message)
            if text and message.get("metadata", {}).get("response_state") != "synthetic":
                turn.assistant.append(text)
            for call in _tool_calls(message):
                call_id = call.get("id")
                name = call.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    call_names[call_id] = name
                turn.tools.append(_tool_call_digest(call))
            turn.completed = _completed_assistant(message)
        elif role == "tool_result":
            result = message.get("tool_result")
            if isinstance(result, Mapping):
                turn.tools.append(_tool_result_digest(result, call_names))
    if turn is not None and turn.completed:
        units.extend(_turn_units(project_id, session_id, turn))
    return tuple(sorted(units, key=lambda unit: (unit.seq_start, unit.kind, unit.chunk_index)))


def _message(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if row.get("type") != "message":
        return None
    data = row.get("data")
    message = data.get("message") if isinstance(data, Mapping) else None
    return message if isinstance(message, Mapping) else None


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    values = [
        redact_secrets(block["text"])
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(value.strip() for value in values if value.strip())


def _is_human(message: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    origin = metadata.get("origin") if isinstance(metadata, Mapping) else None
    if origin is None:
        data = row.get("data")
        origin = data.get("origin") if isinstance(data, Mapping) else None
    return origin in {"human", "user"}


def _completed_assistant(message: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    state = metadata.get("response_state")
    if state is not None:
        return state == "completed"
    return not metadata.get("turn_failed")


def _tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            call = block.get("tool_call")
            if isinstance(call, Mapping):
                calls.append(call)
    return calls


def _tool_call_digest(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or "unknown")
    arguments = call.get("arguments")
    fields: list[str] = []
    if isinstance(arguments, Mapping):
        keys = sorted(str(key) for key in arguments)
        fields.append("arguments=" + ",".join(keys[:24]))
        for key in _SAFE_ARGUMENTS:
            value = arguments.get(key)
            if isinstance(value, str) and value:
                fields.append(f"{key}={_bounded_text(redact_secrets(value), 320)}")
    return _bounded_text(f"Tool {name} call: {'; '.join(fields)}", 720)


def _tool_result_digest(result: Mapping[str, Any], names: Mapping[str, str]) -> str:
    call_id = result.get("tool_call_id")
    name = names.get(call_id, "unknown") if isinstance(call_id, str) else "unknown"
    status = "error" if result.get("is_error") else "success"
    content = result.get("content")
    text = redact_secrets(content) if isinstance(content, str) else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    diagnostic = ""
    if lines:
        selected = [lines[0]] if len(lines) == 1 else [lines[0], lines[-1]]
        diagnostic = "; diagnostics=" + " | ".join(
            _bounded_text(line, MAX_DIAGNOSTIC_CHARS) for line in selected
        )
    size = len(content.encode("utf-8")) if isinstance(content, str) else 0
    return _bounded_text(
        f"Tool {name} result: {status}; bytes={size}{diagnostic}", 720
    )


def _child_report(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if row.get("type") != "notification":
        return None
    data = row.get("data")
    if not isinstance(data, Mapping) or data.get("kind") != "agent_completion":
        return None
    text = data.get("text")
    if not isinstance(text, str) or not text:
        return None
    return (
        str(data.get("description") or "child agent"),
        str(data.get("status") or "unknown"),
        redact_secrets(text),
    )


def _turn_units(project_id: str, session_id: str, turn: _Turn) -> list[TranscriptUnit]:
    sections: list[str] = []
    sections.extend(f"Human: {text}" for text in turn.human)
    sections.extend(f"Input: {text}" for text in turn.inputs)
    sections.extend(f"Agent: {text}" for text in turn.assistant)
    if turn.tools:
        digest = _bounded_utf8("\n".join(turn.tools), MAX_TOOL_DIGEST_BYTES)
        sections.append("Tool activity:\n" + digest)
    origin = "human" if turn.human else "unverified_input"
    return _make_units(project_id, session_id, turn.rows, origin, "turn", "\n\n".join(sections))


def _report_units(
    project_id: str,
    session_id: str,
    row: Mapping[str, Any],
    report: tuple[str, str, str],
) -> list[TranscriptUnit]:
    description, status, text = report
    rendered = f"Child report ({description}, {status}):\n{text}"
    return _make_units(
        project_id, session_id, [row], "parent_visible_child", "child_report", rendered
    )


def _make_units(
    project_id: str,
    session_id: str,
    rows: list[Mapping[str, Any]],
    origin: str,
    kind: Literal["turn", "child_report"],
    text: str,
) -> list[TranscriptUnit]:
    if not text.strip():
        return []
    seq_start = int(rows[0]["seq"])
    seq_end = int(rows[-1]["seq"])
    turn_id = _stable_id(session_id, kind, seq_start, seq_end)
    chunks = _split_utf8(text, MAX_UNIT_BYTES)
    timestamps = [_timestamp(row) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    base = TranscriptUnit(
        schema_version=_UNIT_SCHEMA_VERSION,
        unit_id="",
        turn_id=turn_id,
        project_id=project_id,
        session_id=session_id,
        seq_start=seq_start,
        seq_end=seq_end,
        started_at=timestamps[0] if timestamps else None,
        ended_at=timestamps[-1] if timestamps else None,
        origin=origin,
        kind=kind,
        chunk_index=0,
        chunk_count=len(chunks),
        text="",
    )
    return [
        replace(
            base,
            unit_id=_stable_id(turn_id, str(index)),
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def _timestamp(row: Mapping[str, Any]) -> str | None:
    data = row.get("data")
    if not isinstance(data, Mapping):
        return None
    value = data.get("created_at") or data.get("timestamp")
    return value if isinstance(value, str) else None


def _stable_id(*parts: str | int) -> str:
    source = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(source).hexdigest()[:32]


def _bounded_text(value: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean if len(clean) <= max_chars else clean[: max_chars - 1] + "…"


def _bounded_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[: max_bytes - 3].decode("utf-8", errors="ignore") + "…"


def _split_utf8(value: str, max_bytes: int) -> list[str]:
    remaining = value.strip()
    chunks: list[str] = []
    while len(remaining.encode("utf-8")) > max_bytes:
        encoded = remaining.encode("utf-8")
        prefix = encoded[:max_bytes].decode("utf-8", errors="ignore")
        split = max(prefix.rfind("\n\n"), prefix.rfind("\n"), prefix.rfind(" "))
        if split < max_bytes // 2:
            split = len(prefix)
        chunk = prefix[:split].rstrip()
        chunks.append(chunk)
        remaining = remaining[len(prefix[:split]) :].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks or [""]


def dump_units(units: Iterable[TranscriptUnit]) -> str:
    """Serialize units as stable JSONL for fixtures and diagnostics."""

    return "".join(json.dumps(unit.to_dict(), sort_keys=True) + "\n" for unit in units)
