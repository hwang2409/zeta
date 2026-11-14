from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from zeta.storage_format import open_writer

ROOT_BYTES = 64 * 1024 * 1024
ROOT_ROWS = 17_527
CHILD_BYTES = 23_086_500
CHILD_ROWS = 1_154
CHILD_ID = "651"
VISIBLE_PROMPT = "Henry's task, verbatim"
_CHILD_PROMPT_PREFIX = VISIBLE_PROMPT + ". Do the focused child fixture task. "
CHILD_PROMPT = _CHILD_PROMPT_PREFIX + "p" * (2_000 - len(_CHILD_PROMPT_PREFIX))
ROOT_MARKER = "complex-root-anchor-ux1309"


@dataclass(frozen=True)
class ChildViewHome:
    home: Path
    session_id: str
    child_path: Path
    generation_seconds: float


def _row(seq: int, row_type: str, data: dict[str, Any], parent_id: str | None) -> dict[str, Any]:
    return {
        "data": data,
        "id": f"{seq:032x}",
        "lane": "main",
        "parent_id": parent_id,
        "seq": seq,
        "type": row_type,
    }


def _message(role: str, content: list[dict[str, Any]], *, origin: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content, "role": role}
    if origin is not None:
        message["metadata"] = {"zeta.origin": origin}
    return {"message": message}


def _encoded(row: dict[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _write_sized_log(path: Path, header: dict[str, Any], rows: list[dict[str, Any]], total_bytes: int) -> None:
    encoded = [_encoded(header), *(_encoded(row) for row in rows)]
    missing = total_bytes - sum(map(len, encoded))
    text_rows = [
        row
        for row in rows
        if row["type"] == "message"
        and row["data"]["message"]["content"]
        and row["data"]["message"]["content"][0].get("type") == "text"
        and not row["data"]["message"]["content"][0].get("text", "").startswith((VISIBLE_PROMPT, ROOT_MARKER))
    ]
    if missing < len(text_rows):
        raise AssertionError("fixture rows exceed measured byte budget")
    share, remainder = divmod(missing, len(text_rows))
    for index, row in enumerate(text_rows):
        row["data"]["message"]["content"][0]["text"] += "x" * (
            share + (1 if index < remainder else 0)
        )
    with path.open("wb") as handle:
        handle.write(_encoded(header))
        for row in rows:
            handle.write(_encoded(row))
    assert path.stat().st_size == total_bytes


def _root_rows(session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parent: str | None = None

    def add(row_type: str, data: dict[str, Any]) -> str:
        nonlocal parent
        seq = len(rows) + 1
        row = _row(seq, row_type, data, parent)
        rows.append(row)
        parent = row["id"]
        return parent

    roles = ["assistant"] * 5_664 + ["tool_result"] * 4_761 + ["system"] * 1_347 + ["user"] * 560
    for index, role in enumerate(roles):
        if index == 5_000:
            continue
        origin = "user" if role == "user" else None
        if role == "tool_result":
            data = _message(role, [{"type": "text", "text": f"root result {index} "}])
            data["message"]["tool_result"] = {
                "tool_call_id": f"root-call-{index}",
                "content": f"root result {index}",
                "is_error": False,
            }
        else:
            data = _message(role, [{"type": "text", "text": f"root row {index} "}], origin=origin)
        add("message", data)

    notification_ids: list[str] = []
    for index in range(1_510):
        notification_ids.append(
            add(
                "notification",
                {"kind": "task_exited", "origin": "notification", "task_id": f"task-{index}", "headline": "fixture task", "output_tail": "done", "exit_code": 0},
            )
        )
    for notification_id in notification_ids[:1_508]:
        add("notification_ack", {"notification_id": notification_id})
    for notification_id in notification_ids[:1_494]:
        add("notification_tui_presented", {"notification_id": notification_id})
    for index in range(613):
        child_number = 651 if index == 0 else index
        add(
            "child_run_started",
            {
                "run_id": f"{session_id}:{child_number}",
                "parent_run_id": None,
                "name": "worker",
                "task": "fixture child",
                "started_at": "2026-01-01T00:00:00+00:00",
            },
        )
    for index in range(68):
        add(
            "compaction",
            {
                "kind": "summary",
                "replaces": [],
                "source_seq_start": 1,
                "source_seq_end": 1,
                "summary": f"fixture compaction {index}",
            },
        )
    add(
        "message",
        _message(
            "assistant",
            [
                {"type": "text", "text": "\n".join(f"{ROOT_MARKER} line {line}" for line in range(369))},
                {"type": "tool_use", "tool_call": {"id": "root-tool-a", "name": "read", "arguments": {"path": "alpha"}}},
                {"type": "tool_use", "tool_call": {"id": "root-tool-b", "name": "read", "arguments": {"path": "beta"}}},
            ],
        ),
    )
    add("pending_queue_closed", {})
    assert len(rows) + 1 == ROOT_ROWS
    return rows


def _child_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parent: str | None = None

    def add(row_type: str, data: dict[str, Any]) -> str:
        nonlocal parent
        seq = len(rows) + 1
        row = _row(seq, row_type, data, parent)
        rows.append(row)
        parent = row["id"]
        return parent

    prompt = _message(
        "user",
        [{"type": "text", "text": CHILD_PROMPT}],
        origin="agent_prompt",
    )
    prompt["message"]["metadata"]["fixture_padding"] = "q" * 1_723
    add("message", prompt)
    add(
        "message",
        _message(
            "assistant",
            [
                {"type": "thinking", "text": "inspect the fixture"},
                {"type": "text", "text": "working"},
                {"type": "tool_use", "tool_call": {"id": "child-tool-1", "name": "read", "arguments": {"path": "fixture"}}},
            ],
        ),
    )
    for index in range(1_089):
        role = "tool_result" if index % 2 == 0 else "assistant"
        hidden = "Henry" if index < 919 else "ordinary"
        if role == "tool_result":
            data = _message(role, [{"type": "text", "text": f"visible result {index} "}])
            data["message"]["tool_result"] = {
                "tool_call_id": f"child-call-{index}",
                "content": f"visible result {index}",
                "structured_content": {"hidden_owner": hidden},
                "is_error": False,
            }
        else:
            data = _message(
                role,
                [
                    {"type": "thinking", "text": f"thought {index}"},
                    {"type": "text", "text": f"visible assistant {index} "},
                    {"type": "tool_use", "tool_call": {"id": f"child-call-{index}", "name": "read", "arguments": {"hidden_owner": hidden}}},
                ],
            )
        add("message", data)
    notification_ids: list[str] = []
    for index in range(23):
        notification_ids.append(add("notification", {"kind": "task_exited", "origin": "notification", "task_id": f"child-task-{index}", "headline": "fixture", "output_tail": "done", "exit_code": 0}))
    for notification_id in notification_ids:
        add("notification_ack", {"notification_id": notification_id})
    for index in range(10):
        add("compaction", {"kind": "summary", "replaces": [], "source_seq_start": 1, "source_seq_end": 1, "summary": f"child compaction {index}"})
    prompt_ids = [add("pending_prompt", {"text": f"queued prompt {index}"}) for index in range(2)]
    for prompt_id in prompt_ids:
        add("pending_prompt_ack", {"prompt_id": prompt_id})
    add("pending_queue_closed", {})
    add("terminal_transition", {"action": "cancel"})
    assert len(rows) + 1 == CHILD_ROWS
    return rows


def generate_child_view_home(base: Path) -> ChildViewHome:
    started = perf_counter()
    home = base / "home"
    session_id = "root-fixture"
    with open_writer(home):
        pass
    session_dir = home / "sessions" / session_id
    child_path = session_dir / "agents" / CHILD_ID
    child_path.mkdir(parents=True)
    header = {
        "data": {
            "created_at": "2026-01-01T00:00:00+00:00",
            "cwd": str(base),
            "schema": "zeta.conversation.v1",
            "session_id": session_id,
        },
        "type": "header",
    }
    child_header = {
        "data": {
            "created_at": "2026-01-01T00:00:00+00:00",
            "cwd": str(base),
            "schema": "zeta.conversation.v1",
            "session_id": CHILD_ID,
        },
        "type": "header",
    }
    _write_sized_log(session_dir / "conversation.jsonl", header, _root_rows(session_id), ROOT_BYTES)
    _write_sized_log(child_path / "conversation.jsonl", child_header, _child_rows(), CHILD_BYTES)
    (child_path / "agent_lifecycle.json").write_text(
        json.dumps(
            {
                "agent_type": "worker",
                "cwd": "/fixture",
                "depth": 1,
                "description": "fixture child 651",
                "elapsed": 1.0,
                "final_result": "finished fixture child",
                "finished_at": "2026-01-01T00:00:01+00:00",
                "handle": f"{session_id}:{CHILD_ID}",
                "started_at": "2026-01-01T00:00:00+00:00",
                "state": "running",
                "tool_calls": 556,
                "turns_used": 508,
            },
            separators=(",", ":"),
        )
    )
    assert not (home / "accounts").exists()
    assert not (home / "secrets").exists()
    assert sum(1 for _ in (session_dir / "conversation.jsonl").open("rb")) == ROOT_ROWS
    assert sum(1 for _ in (child_path / "conversation.jsonl").open("rb")) == CHILD_ROWS
    return ChildViewHome(home, session_id, child_path, perf_counter() - started)
