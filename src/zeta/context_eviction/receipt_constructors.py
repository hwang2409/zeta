"""Deterministic field-based constructors for generated eviction receipts."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ..protocol.types import Message, MessageRole, TextContent, ToolResult

DIGEST_LIMIT = 440
WORKFLOW_MESSAGE_LIMIT = 4
RECEIPT_KIND_METADATA = "eviction_receipt"
RECEIPT_FIELDS_METADATA = "eviction_receipt_fields"


def _assistant_receipt(
    seq: int,
    kind: str,
    *,
    role: MessageRole = MessageRole.ASSISTANT,
    metadata: Mapping[str, object] | None = None,
) -> Message:
    return Message(
        role,
        [TextContent(f"[{kind} evicted · seq {seq}]")],
        metadata={
            **(metadata or {}),
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "assistant",
            RECEIPT_FIELDS_METADATA: {"kind": kind},
            "source_seq": seq,
        },
    )


def _collapsed_assistant_receipt(
    seq: int, collapsed_into_seq: int, *, older_read: bool
) -> Message:
    label = "older duplicate read" if older_read else "duplicate result"
    return Message(
        MessageRole.ASSISTANT,
        [TextContent(f"[{label} collapsed into seq {collapsed_into_seq}]")],
        metadata={
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "assistant",
            RECEIPT_FIELDS_METADATA: {
                "collapsed_into_seq": collapsed_into_seq,
                "older_read": older_read,
            },
            "source_seq": seq,
            "collapsed_into_seq": collapsed_into_seq,
        },
    )


def _semantic_result_receipt(
    *,
    role: MessageRole,
    tool_name: str,
    tool_call_id: str,
    seq: int,
    digest: str,
    content_digest: str,
    is_error: bool = False,
    is_canceled: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> Message:
    prefix = f"[semantic {tool_name} digest · seq {seq}] "
    hint = (
        f". recall_history seq_start={seq}, seq_end={seq} for exact output; "
        "re-read only if the source may have changed."
    )
    body_limit = DIGEST_LIMIT - len(prefix) - len(hint)
    body = (
        digest
        if len(digest) <= body_limit
        else digest[: body_limit - 3].rstrip() + "..."
    )
    return Message(
        role,
        tool_result=ToolResult(
            tool_call_id,
            f"{prefix}{body}{hint}",
            is_error=is_error,
            is_canceled=is_canceled,
        ),
        metadata={
            **(metadata or {}),
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "tool_result",
            RECEIPT_FIELDS_METADATA: {"tool_name": tool_name, "digest": body},
            "source_seq": seq,
            "eviction_content_digest": content_digest,
        },
    )


def _structured_result_receipt(
    *,
    role: MessageRole,
    receipt_kind: str,
    tool_name: str,
    tool_call_id: str,
    seq: int,
    payload: Mapping[str, object],
    content_digest: str,
    is_error: bool = False,
    is_canceled: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> Message:
    canonical = _canonical_receipt_payload(
        receipt_kind, tool_name, tool_call_id, payload
    )
    receipt = _structured_receipt(
        f"{receipt_kind} result receipt", canonical, seq, "result"
    )
    return Message(
        role,
        tool_result=ToolResult(
            tool_call_id, receipt, is_error=is_error, is_canceled=is_canceled
        ),
        metadata={
            **(metadata or {}),
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "tool_result",
            RECEIPT_FIELDS_METADATA: {
                "receipt_kind": receipt_kind,
                "tool_name": tool_name,
                "payload": canonical,
            },
            "source_seq": seq,
            "eviction_content_digest": content_digest,
        },
    )


def _canonical_receipt_payload(
    receipt_kind: str,
    tool_name: str,
    tool_call_id: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if receipt_kind == "orchestration":
        canonical: dict[str, object] = {
            "tool": tool_name,
            "call": tool_call_id,
            "status": _one_line(payload.get("status")),
        }
        for key in ("child_instance_id", "task_id", "handle", "description"):
            if value := _one_line(payload.get(key)):
                canonical[key] = value
        if type(original_chars := payload.get("original_chars")) is int:
            canonical["original_chars"] = original_chars
        if sha256 := _one_line(payload.get("sha256"), limit=16):
            canonical["sha256"] = sha256
        return canonical
    if tool_name == "inbox":
        return _canonical_inbox_payload(payload)
    if tool_name == "project":
        canonical = {"action": _one_line(payload.get("action")) or "inspect"}
        for key in ("project_id", "project_name"):
            if value := _one_line(payload.get(key)):
                canonical[key] = value
        sections = payload.get("sections")
        canonical["sections"] = (
            sorted(_one_line(item) for item in sections)
            if isinstance(sections, list)
            else []
        )
        files = payload.get("files")
        if isinstance(files, list):
            canonical["files"] = sorted(_one_line(item) for item in files)[:16]
        return canonical
    if tool_name == "recall_history":
        canonical = {
            "action": _one_line(payload.get("action")),
            "result_chars": payload.get("result_chars"),
        }
        for key in ("seq_start", "seq_end", "offset", "max_chars"):
            if type(value := payload.get(key)) is int:
                canonical[key] = value
        if query := _one_line(payload.get("query")):
            canonical["query"] = query
        return canonical
    canonical = {
        "command": _one_line(payload.get("command"), limit=120),
        "status": _one_line(payload.get("status")),
    }
    if task_id := _one_line(payload.get("task_id")):
        canonical["task_id"] = task_id
    return canonical


def _canonical_inbox_payload(payload: Mapping[str, object]) -> dict[str, object]:
    action = _one_line(payload.get("action")) or "unknown"
    canonical: dict[str, object] = {"action": action}
    if action == "send":
        for key in ("target_project", "id"):
            if value := _one_line(payload.get(key)):
                canonical[key] = value
        return canonical
    if action == "projects":
        projects = payload.get("projects")
        canonical["projects"] = (
            [
                {
                    key: value
                    for key in ("id", "name", "scope")
                    if (value := _one_line(item.get(key)))
                }
                for item in projects[:WORKFLOW_MESSAGE_LIMIT]
                if isinstance(item, Mapping)
            ]
            if isinstance(projects, list)
            else []
        )
        if type(omitted := payload.get("omitted_projects")) is int:
            canonical["omitted_projects"] = omitted
        return canonical
    messages = payload.get("messages")
    canonical["messages"] = (
        [
            {
                "id": _one_line(item.get("id")),
                "kind": _one_line(item.get("kind")),
                "title": _one_line(item.get("title"), limit=80),
                "status": _one_line(item.get("status")),
                "claimed": item.get("claimed") is True,
                "done": item.get("done") is True,
                **{
                    key: value
                    for key in ("from_project", "from_session")
                    if (value := _one_line(item.get(key)))
                },
                **(
                    {"outcome": _one_line(item.get("outcome"), limit=120)}
                    if "outcome" in item
                    else {}
                ),
            }
            for item in messages[:WORKFLOW_MESSAGE_LIMIT]
            if isinstance(item, Mapping)
        ]
        if isinstance(messages, list)
        else []
    )
    if type(omitted := payload.get("omitted_messages")) is int:
        canonical["omitted_messages"] = omitted
    return canonical


def _notification_receipt_from_fields(
    *,
    role: MessageRole,
    seq: int,
    payload: Mapping[str, object],
    content_digest: str,
) -> Message:
    canonical = _canonical_notification_payload(payload)
    receipt = _structured_receipt(
        "notification receipt", canonical, seq, "notification"
    )
    return Message(
        role,
        [TextContent(receipt)],
        metadata={
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "notification",
            RECEIPT_FIELDS_METADATA: {"payload": canonical},
            "source_seq": seq,
            "eviction_content_digest": content_digest,
        },
    )


def _canonical_notification_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    notifications = payload.get("notifications")
    canonical_notifications = []
    if isinstance(notifications, list):
        for item in notifications:
            if not isinstance(item, Mapping):
                continue
            summary: dict[str, object] = {"kind": _one_line(item.get("kind"))}
            for key in ("child_instance_id", "task_id", "status", "description"):
                if value := _one_line(item.get(key)):
                    summary[key] = value
            if type(exit_code := item.get("exit_code")) is int:
                summary["exit_code"] = exit_code
            canonical_notifications.append(summary)
    return {
        "notifications": canonical_notifications,
        "original_chars": payload.get("original_chars"),
        "sha256": _one_line(payload.get("sha256"), limit=16),
    }


def _structured_receipt(
    prefix: str, payload: Mapping[str, object], seq: int, exact_kind: str
) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return (
        f"[{prefix}] {encoded}; recall_history "
        f"seq_start={seq}, seq_end={seq} for exact {exact_kind}"
    )


def _one_line(value: object, *, limit: int = 160) -> str:
    if value is None:
        return ""
    compact = " ".join(str(value).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."
