"""Serve history projection: row building, grouping, and the per-store cache."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ...agent.turn_limit import project_notification_data, project_tool_result_data
from ...core.store._client_delivery import CLIENT_DELIVERY_METADATA
from ...protocol.types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    Message,
    is_passive_harness_message,
    is_visible_harness_notice,
)
from .. import event_contract
from ..protocol import MAX_FRAME_BYTES, MAX_REQUEST_ID_BYTES, FrameCodec, bounded

if TYPE_CHECKING:
    from ...core.store import ConversationStore

ENTITLEMENT_ERROR_CODES = {"auth_error", "model_not_found", "permission_denied"}


def _failed_turn_history(message: Mapping[str, object]) -> dict[str, object] | None:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict) or metadata.get(FAILED_TURN_MARKER) is not True:
        return None
    error = metadata.get(FAILED_TURN_ERROR)
    if not isinstance(error, dict):
        return {"code": "backend_error", "message": "turn failed"}
    code = error.get("code")
    text = error.get("message")
    if not isinstance(code, str) or not code or not isinstance(text, str) or not text:
        return {"code": "backend_error", "message": "turn failed"}
    provider_error = error.get("provider_error") is True
    status_code = error.get("status_code")
    entitlement_failure = (
        code != "auth_error"
        and provider_error
        and (code in ENTITLEMENT_ERROR_CODES or status_code in {400, 401, 403, 404})
    )
    return {
        "code": "model_access_error" if entitlement_failure else bounded(code, 100),
        "message": bounded(text, 1000),
        "provider_error": provider_error,
    }


Generation = tuple[int, bytes]
HISTORY_GROUP_TRUNCATED = "[history group too large; truncated]"


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    entry_groups: tuple[tuple[Any, ...], ...]
    approval_calls: Mapping[str, object]
    entry_index: Mapping[str, Any] = MappingProxyType({})
    groups: tuple[tuple[dict[str, object], ...], ...] = ()
    body_sizes: tuple[int, ...] = ()
    # seq of the newest active-branch row this projection includes
    frontier: int = 0


@dataclass
class _CachedProjection:
    generation: Generation
    projection: HistoryProjection


_cache_creation_lock = threading.Lock()
_HISTORY_ROW_FEATURES = (
    "include_delivery_id",
    "fidelity",
    "content_read",
    "served_turn_end",
)


def _history_feature_key(
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
) -> tuple[bool, ...]:
    values = locals()
    return tuple(values[name] for name in _HISTORY_ROW_FEATURES)


def _generation(store: ConversationStore) -> Generation:
    return (
        getattr(store, "_log_offset", 0),
        getattr(store, "_entry_prefix_digest", b""),
    )


def _cache(
    store: ConversationStore,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
) -> _CachedProjection | None:
    cache = getattr(store, "_serve_history_projections", {})
    return cache.get(
        _history_feature_key(
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
        )
    )


def _publish_cache(
    store: ConversationStore,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    served_turn_end: bool,
    cached: _CachedProjection,
) -> None:
    cache = getattr(store, "_serve_history_projections", None)
    if cache is None:
        cache = {}
        store._serve_history_projections = cache
    cache[
        _history_feature_key(
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
        )
    ] = cached


def _bounded_history_value(value: object) -> object:
    if isinstance(value, str):
        return bounded(value)
    if isinstance(value, dict):
        return {key: _bounded_history_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded_history_value(item) for item in value]
    return value


def _history_frame_size(
    codec: FrameCodec,
    request_id: str,
    *,
    messages_body_size: int,
    next_offset: int | None = None,
    next_before: int | None = None,
    history_validator: dict[str, object] | None = None,
) -> int:
    reply = {"messages": [], "next_offset": next_offset}
    if next_before is not None or history_validator is not None:
        reply = {
            "messages": [],
            "next_before": next_before,
            "history_validator": history_validator,
        }
    empty_page_size = codec.response_size(request_id, reply)
    return empty_page_size + messages_body_size


def _history_group_body_size(
    codec: FrameCodec, group: tuple[dict[str, object], ...]
) -> int:
    return codec.serialized_size(group) - len(b"[]\n")


def _project_history_group(
    rows: list[dict[str, object]],
    envelope: dict[str, object],
) -> tuple[tuple[dict[str, object], ...], int]:
    """Return the group raw, bounded, or as a marker, whichever fits alone in a
    page with this cursor envelope and the largest legal request id."""
    codec = FrameCodec()
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    raw_group = tuple(rows)
    raw_body_size = _history_group_body_size(codec, raw_group)
    if (
        _history_frame_size(
            codec,
            request_id,
            messages_body_size=raw_body_size,
            **envelope,
        )
        <= MAX_FRAME_BYTES
    ):
        return raw_group, raw_body_size

    group = tuple(
        _bounded_history_value(row)
        if row.get("event") in {"approval_request", "approval_end"}
        or row.get("notification") is not None
        else row
        for row in rows
    )
    body_size = _history_group_body_size(codec, group)
    if (
        _history_frame_size(
            codec,
            request_id,
            messages_body_size=body_size,
            **envelope,
        )
        <= MAX_FRAME_BYTES
    ):
        return group, body_size
    row_id = rows[0].get("id", "history") if rows else "history"
    group = (
        {
            "id": row_id,
            "role": "system",
            "content": [{"type": "text", "text": HISTORY_GROUP_TRUNCATED}],
            "tool_result": None,
        },
    )
    return group, _history_group_body_size(codec, group)


def history_projection(
    store: ConversationStore,
    *,
    include_delivery_id: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
    served_turn_end: bool = False,
) -> HistoryProjection:
    """Return one immutable wire projection for the store's generation."""
    cached = _cache(
        store,
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
        served_turn_end=served_turn_end,
    )
    generation = _generation(store)
    if cached is not None and cached.generation == generation:
        return cached.projection

    # The store is loop-owned, but its validated snapshots are read here by the
    # serve worker. Retry if an append publishes a newer generation mid-copy.
    while True:
        generation = _generation(store)
        branch = store.active_branch_snapshot()
        if generation == _generation(store):
            break

    approval_calls: dict[str, object] = {}
    accepted_messages: dict[str, dict[str, object]] = {}
    entries: list[list[Any]] = [[]]
    for entry in branch:
        if entry.type == "client_delivery" and isinstance(
            entry.data.get("accepted_message"), dict
        ):
            accepted_messages[entry.data["delivery_id"]] = entry.data[
                "accepted_message"
            ]
        elif entry.type == "client_delivery" and entry.data.get("status") in {
            "delivered",
            "dropped",
        }:
            accepted = accepted_messages.get(entry.data["delivery_id"])
            if accepted is not None:
                entry = replace(
                    entry, data={**entry.data, "_accepted_message": accepted}
                )
        if entry.type == "message":
            approval_calls.update(
                {
                    request["request_id"]: request["tool_call"]
                    for request in entry.data.get("approval_requests", [])
                }
            )
        if (
            entry.type == "message"
            and is_passive_harness_message(entry.data["message"])
            and not is_visible_harness_notice(entry.data["message"])
        ):
            continue
        if entry.type in {"message", "notification"} or (
            entry.type == "warning" and entry.data.get("kind") == "transcript_recovery"
        ):
            entries[-1].append(entry)
            entries.append([])
        elif entry.type in {
            "client_delivery",
            "approval_resolution",
            "approval_history",
        } or (entry.type == "turn_end" and served_turn_end):
            entries[-1].append(entry)

    entry_index = {
        entry.id: entry for group in entries for entry in group if hasattr(entry, "id")
    }
    built = HistoryProjection(
        tuple(tuple(group) for group in entries),
        MappingProxyType(approval_calls),
        MappingProxyType(entry_index),
        frontier=branch[-1].seq if branch else 0,
    )
    with _cache_creation_lock:
        cached = _cache(
            store,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
            served_turn_end=served_turn_end,
        )
        if cached is None or cached.generation != generation:
            _publish_cache(
                store,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
                served_turn_end=served_turn_end,
                cached=_CachedProjection(generation, built),
            )
            return built
        return cached.projection


def _attach_row_handle(
    block: dict[str, object],
    entry_id: str,
    field: str,
    index: int,
    *,
    fidelity: bool,
    content_read: bool,
) -> dict[str, object]:
    if fidelity and content_read:
        from ..content import row_handle

        block["handle"] = row_handle(entry_id, field, index)
    return block


def _project_tool_call_arguments(
    call: dict[str, object], *, fidelity: bool
) -> dict[str, object]:
    arguments = call.get("arguments", {})
    if not fidelity:
        return {"arguments": {}}
    projected, truncated = event_contract.project_tool_value(arguments)
    return {"arguments": projected, "arguments_truncated": truncated}


def _transcript_recovery_history_row(entry: object) -> dict[str, object]:
    data = entry.data
    recovery_keys = (
        "quarantine_id",
        "source_digest",
        "first_bad_row",
        "retained_last_sequence",
        "session_id",
        "child_run_id",
    )
    return {
        "id": entry.id,
        "role": "system",
        "content": [{"type": "text", "text": bounded(data["message"], 8000)}],
        "tool_result": None,
        "transcript_recovery": {
            key: data[key] for key in recovery_keys if key in data
        },
    }


def _history_entry_rows(
    entry: object,
    approval_calls: Mapping[str, object],
    *,
    include_delivery_id: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
) -> list[dict[str, object]]:
    """Project one durable entry to one atomic group of history rows."""

    entry_type = entry.type
    if entry_type == "client_delivery":
        if not fidelity:
            return []
        data = entry.data
        if (
            data.get("status") not in {"delivered", "dropped"}
            or data.get("method") != "steer"
        ):
            return []
        accepted = data.get("_accepted_message")
        message = Message.from_dict(accepted) if isinstance(accepted, dict) else None
        return [event_contract.delivery_status_row(entry, message)]
    if entry_type == "turn_end":
        return [event_contract.served_turn_end_row(entry)]
    if entry_type == "approval_history":
        return [{"id": entry.id, "role": "system", **entry.data}]
    if entry_type == "approval_resolution":
        data = {
            "decision": entry.data["decision"],
            **({"reason": entry.data["reason"]} if "reason" in entry.data else {}),
        }
        payload = event_contract.approval_end(
            {
                "request_id": entry.data["request_id"],
                "tool_call": approval_calls.get(entry.data["request_id"]),
                "data": data,
                **({"reason": entry.data["reason"]} if "reason" in entry.data else {}),
            }
        )
        return [{"id": entry.id, "role": "system", "event": "approval_end", **payload}]
    if entry_type == "warning":
        return [_transcript_recovery_history_row(entry)]
    if entry_type == "notification":
        data = project_notification_data(entry.data)
        kind = data.get("kind", "agent_completion")
        if kind == "task_exited":
            text = (
                f"background task {data.get('task_id', '?')} exited "
                f"({data.get('exit_code')}); {data.get('headline', '')}"
            )
            note = data.get("note")
            if isinstance(note, str) and note:
                text += f"; {note}"
        else:
            text = data.get("text", "background agent notification")
        return [
            {
                "id": entry.id,
                "role": "system",
                "content": [{"type": "text", "text": bounded(text, 8000)}],
                "tool_result": None,
                "notification": dict(data),
            }
        ]

    message = entry.data["message"]
    content = []
    for index, block in enumerate(message.get("content", [])):
        if block.get("type") == "image":
            attachment = {
                "type": "attachment",
                "name": Path(block.get("path") or "image").name,
                "size": block.get("size") or 0,
            }
            content.append(
                _attach_row_handle(
                    attachment,
                    entry.id,
                    "message.content",
                    index,
                    fidelity=fidelity,
                    content_read=content_read,
                )
            )
        elif block.get("type") == "text":
            content.append({"type": "text", "text": bounded(block["text"], 8000)})
        elif block.get("type") == "tool_use":
            call = block["tool_call"]
            content.append(
                {
                    "type": "tool_use",
                    "tool_call": {
                        "id": call["id"],
                        "name": call["name"],
                        **_project_tool_call_arguments(call, fidelity=fidelity),
                    },
                }
            )
        elif block.get("type") == "thinking":
            body = block.get("body")
            if type(body) is str and body:
                body = bounded(body, 8000)
                content.append({"type": "thinking", "text": body, "body": body})
            else:
                content.append({"type": "thinking", "text": ""})
    result = message.get("tool_result")
    if result:
        result = project_tool_result_data(result)
        blocks = []
        if fidelity:
            for index, block in enumerate(result.get("content_blocks") or []):
                if isinstance(block, dict) and block.get("type") == "image":
                    stub = event_contract.image_stub(block)
                    blocks.append(
                        _attach_row_handle(
                            stub,
                            entry.id,
                            "message.tool_result.content_blocks",
                            index,
                            fidelity=fidelity,
                            content_read=content_read,
                        )
                    )
        result = {
            **result,
            "content": bounded(result.get("content", ""), 8000),
            "content_blocks": blocks,
        }
        structured = result.get("structured_content")
        if fidelity and structured is not None:
            projected, wire_truncated = event_contract.project_tool_value(structured)
            result["structured_content"] = projected
            result["structured_content_truncated"] = (
                event_contract.structured_content_truncated(
                    structured, wire_truncated=wire_truncated
                )
            )
        else:
            result["structured_content"] = None
    row = {
        "id": entry.id,
        "role": message["role"],
        "content": content,
        "tool_result": result,
    }
    failed_turn = _failed_turn_history(message)
    if failed_turn is not None:
        row["failed_turn"] = failed_turn
    metadata = message.get("metadata")
    if (
        include_delivery_id
        and message["role"] == "user"
        and isinstance(metadata, dict)
        and isinstance(metadata.get(CLIENT_DELIVERY_METADATA), str)
    ):
        row["delivery_id"] = metadata[CLIENT_DELIVERY_METADATA]
    rows = [row]
    for approval in entry.data.get("approval_requests", []):
        payload = {
            "request_id": approval["request_id"],
            "tool_call": approval["tool_call"],
            "delegated": False,
        }
        approval_facts = approval.get("approval_facts", {})
        if approval_facts.get("action") is not None:
            payload["approval_action"] = approval_facts["action"]
        if approval.get("approval_display"):
            payload["approval_display"] = approval["approval_display"]
        rows.append(
            {
                "id": entry.id,
                "role": "system",
                "event": "approval_request",
                **event_contract.approval_request(payload),
            }
        )
    return rows


def _history_endpoint(projection: HistoryProjection, end: int) -> str | None:
    if end == 0:
        return None
    for entry in reversed(projection.entry_groups[end - 1]):
        if entry.type in {"message", "notification", "warning"}:
            return entry.id
    return None


def _history_group(
    projection: HistoryProjection,
    index: int,
    *,
    end: int,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    backward: bool,
    validator: dict[str, object] | None,
) -> tuple[tuple[dict[str, object], ...], int, tuple[dict[str, object], ...], int]:
    raw = [
        row
        for entry in projection.entry_groups[index]
        for row in _history_entry_rows(
            entry,
            projection.approval_calls,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
        )
    ]
    forward, forward_size = _project_history_group(raw, {"next_offset": end})
    if not backward:
        return forward, forward_size, forward, forward_size
    response, response_size = _project_history_group(
        raw,
        {"next_before": end, "history_validator": validator},
    )
    return response, response_size, forward, forward_size


def _history_group_fits(
    codec: FrameCodec,
    request_id: str,
    group: tuple[dict[str, object], ...],
    size: int,
    *,
    body_size: int,
    rows: list[dict[str, object]],
    cursor: int,
    end: int,
    backward: bool,
    validator: dict[str, object] | None,
    raw_group: tuple[object, ...],
    projection: HistoryProjection,
    include_delivery_id: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
) -> tuple[bool, tuple[dict[str, object], ...], int]:
    candidate = body_size + int(bool(rows and group)) + size
    kwargs = (
        {"next_before": cursor or None, "history_validator": validator}
        if backward
        else {
            "next_offset": cursor if cursor < end else None,
        }
    )
    if (
        _history_frame_size(codec, request_id, messages_body_size=candidate, **kwargs)
        <= MAX_FRAME_BYTES
    ):
        return True, group, size
    if rows:
        return False, group, size
    # Alone on its page, the group must still fit this page's real envelope.
    raw = [
        row
        for entry in raw_group
        for row in _history_entry_rows(
            entry,
            projection.approval_calls,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
        )
    ]
    group, size = _project_history_group(raw, kwargs)
    candidate = body_size + int(bool(rows and group)) + size
    return (
        _history_frame_size(codec, request_id, messages_body_size=candidate, **kwargs)
        <= MAX_FRAME_BYTES,
        group,
        size,
    )
