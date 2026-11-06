"""Protocol 1.1 session views and bounded image inputs over shared core seams."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import shutil
import threading
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.checkpoints import list_branches_from_entries
from ..core.session_files import child_directory, write_session_file
from ..core.store._client_delivery import (
    CLIENT_DELIVERY_METADATA,
)
from ..core.store._client_delivery import (
    image_payload_digest as _semantic_image_payload_digest,
)
from ..media.image_normalization import normalize_image_bytes
from ..media.image_policy import image_policy_for_provider
from ..media.images import image_signature_matches
from ..models.catalog import PROVIDER_MODELS, known_model_names
from ..protocol.types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    ImageContent,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    is_passive_harness_message,
    with_message_origin,
)
from . import event_contract
from .protocol import (
    MAX_FRAME_BYTES,
    MAX_REQUEST_ID_BYTES,
    FrameCodec,
    ProtocolError,
    bounded,
)
from .runtime import ServerRuntime

if TYPE_CHECKING:
    from ..core.store import ConversationStore

EXTENSION_REQUESTS = [
    "rename_session",
    "delete_session",
    "session_tree",
    "session_history",
    "content_read",
    "switch_branch",
    "fork_message",
    "model_catalog",
    "session_settings",
    "set_settings",
    "set_compaction",
    "send_images",
    "slash_list",
    "slash_run",
    "account_list",
    "account_logout",
    "account_refresh",
    "account_switch",
]
MAX_IMAGE_BYTES = 512 * 1024
IMAGE_EXTENSIONS = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
}
DIRECTION_CONTROLS = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
ENTITLEMENT_ERROR_CODES = {"auth_error", "model_not_found", "permission_denied"}


async def mutate_session(
    runtime: ServerRuntime,
    method: str,
    session_id: str,
    name: object,
    require_idle: Callable[[], Awaitable[None]],
) -> dict[str, object]:
    session_id = await asyncio.to_thread(runtime.manager.resolve_id, session_id)
    if method == "delete_session":
        await require_idle()
        if runtime.opened is not None and session_id == runtime.session_id:
            raise ProtocolError(
                -32005,
                "select another session before deleting this session",
                {"code": "active_session"},
            )
        await asyncio.to_thread(runtime.manager.delete, session_id)
        return {"session_id": session_id}
    if not isinstance(name, str):
        raise ProtocolError(-32602, "name must be a string")
    metadata = await asyncio.to_thread(runtime.manager.rename, session_id, name)
    if runtime.opened is not None and session_id == runtime.session_id:
        runtime.manager._copy_metadata(runtime.metadata, metadata)
    return {"session": metadata.to_dict()}


def active(runtime: ServerRuntime, params: dict):
    session_id = params.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError(-32602, "session_id must be a nonempty string")
    if runtime.opened is None or session_id != runtime.session_id:
        raise ProtocolError(-32003, "session is not active")
    return runtime.opened.store


def require_mutable(runtime: ServerRuntime, has_pending_approvals: bool = False) -> None:
    loop = runtime.loop
    if loop is not None and (
        loop.background_children_running
        or loop.store.has_outstanding_tool_calls(loop.store.replay())
        or has_pending_approvals
    ):
        raise ProtocolError(-32004, "finish pending tools and background agents first")


def tree(runtime: ServerRuntime) -> dict:
    store = runtime.opened.store
    entries = store.entries
    children: dict[str, int] = {}
    for entry in entries:
        if entry.parent_id:
            children[entry.parent_id] = children.get(entry.parent_id, 0) + 1
    depths: dict[str, int] = {}
    for entry in entries:
        depths[entry.id] = depths.get(entry.parent_id, 0) + (
            children.get(entry.parent_id, 0) > 1
        )
    return {
        "branches": [
            {
                "id": branch.head.id,
                "label": branch.preview,
                "depth": depths[branch.head.id],
                "current": branch.is_current,
            }
            for branch in list_branches_from_entries(entries)
        ]
    }


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
    entitlement_failure = code != "auth_error" and provider_error and (
        code in ENTITLEMENT_ERROR_CODES
        or status_code in {400, 401, 403, 404}
    )
    return {
        "code": "model_access_error" if entitlement_failure else bounded(code, 100),
        "message": bounded(text, 1000),
        "provider_error": provider_error,
    }


Generation = tuple[int, bytes]
HISTORY_FIRST_PAGE_ENTRIES = 128
HISTORY_GROUP_TRUNCATED = "[history group too large; truncated]"


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    entry_groups: tuple[tuple[Any, ...], ...]
    approval_calls: Mapping[str, object]
    entry_index: Mapping[str, Any] = MappingProxyType({})
    groups: tuple[tuple[dict[str, object], ...], ...] = ()
    body_sizes: tuple[int, ...] = ()


@dataclass
class _CachedProjection:
    generation: Generation
    projection: HistoryProjection


_cache_creation_lock = threading.Lock()
_HISTORY_ROW_FEATURES = ("include_delivery_id", "fidelity", "content_read")


def _history_feature_key(
    *, include_delivery_id: bool, fidelity: bool, content_read: bool
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
) -> _CachedProjection | None:
    cache = getattr(store, "_serve_history_projections", {})
    return cache.get(
        _history_feature_key(
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
        )
    )


def _publish_cache(
    store: ConversationStore,
    *,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
    cached: _CachedProjection,
) -> None:
    cache = getattr(store, "_serve_history_projections", None)
    if cache is None:
        cache = {}
        store._serve_history_projections = cache
    cache[_history_feature_key(
        include_delivery_id=include_delivery_id,
        fidelity=fidelity,
        content_read=content_read,
    )] = cached


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
    rows: list[dict[str, object]], envelope: dict[str, object],
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
    store: ConversationStore, *, include_delivery_id: bool = False,
    fidelity: bool = False, content_read: bool = False,
) -> HistoryProjection:
    """Return one immutable wire projection for the store's generation."""
    cached = _cache(
        store, include_delivery_id=include_delivery_id, fidelity=fidelity,
        content_read=content_read,
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
    entries: list[list[Any]] = [[]]
    for entry in branch:
        if entry.type == "message":
            approval_calls.update(
                {
                    request["request_id"]: request["tool_call"]
                    for request in entry.data.get("approval_requests", [])
                }
            )
        if entry.type == "message" and is_passive_harness_message(
            entry.data["message"]
        ):
            continue
        if entry.type in {"message", "notification"}:
            entries[-1].append(entry)
            entries.append([])
        elif entry.type in {"approval_resolution", "approval_history"}:
            entries[-1].append(entry)

    entry_index = {entry.id: entry for group in entries for entry in group if hasattr(entry, "id")}
    built = HistoryProjection(
        tuple(tuple(group) for group in entries), MappingProxyType(approval_calls), MappingProxyType(entry_index)
    )
    with _cache_creation_lock:
        cached = _cache(
            store, include_delivery_id=include_delivery_id, fidelity=fidelity,
            content_read=content_read,
        )
        if cached is None or cached.generation != generation:
            _publish_cache(
                store,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
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
        from .content import row_handle

        block["handle"] = row_handle(entry_id, field, index)
    return block


def _history_entry_rows(
    entry: object,
    approval_calls: Mapping[str, object],
    *,
    include_delivery_id: bool = False, fidelity: bool = False, content_read: bool = False,
) -> list[dict[str, object]]:
    """Project one durable entry to one atomic group of history rows."""

    entry_type = entry.type
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
    if entry_type == "notification":
        data = entry.data
        kind = data.get("kind", "agent_completion")
        if kind == "task_exited":
            text = (
                f"background task {data.get('task_id', '?')} exited "
                f"({data.get('exit_code')}); {data.get('headline', '')}"
            )
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
            attachment = {"type": "attachment", "name": Path(block.get("path") or "image").name, "size": block.get("size") or 0}
            content.append(_attach_row_handle(
                attachment, entry.id, "message.content", index,
                fidelity=fidelity, content_read=content_read,
            ))
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
                        "arguments": {},
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
        blocks = []
        if fidelity:
            for index, block in enumerate(result.get("content_blocks") or []):
                if isinstance(block, dict) and block.get("type") == "image":
                    stub = {key: block[key] for key in ("type", "mimeType", "caption", "width", "height", "size") if key in block}
                    stub["size"] = stub.get("size", 0)
                    stub["data_omitted"] = True
                    blocks.append(_attach_row_handle(
                        stub, entry.id, "message.tool_result.content_blocks", index,
                        fidelity=fidelity, content_read=content_read,
                    ))
        result = {**result, "content": bounded(result.get("content", ""), 8000), "content_blocks": blocks, "structured_content": None}
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
        if entry.type in {"message", "notification"}:
            return entry.id
    return None


def _history_group(
    projection: HistoryProjection,
    index: int,
    *,
    end: int,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
    backward: bool,
    validator: dict[str, object] | None,
) -> tuple[tuple[dict[str, object], ...], int, tuple[dict[str, object], ...], int]:
    raw = [
        row
        for entry in projection.entry_groups[index]
        for row in _history_entry_rows(
            entry, projection.approval_calls,
            include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read,
        )
    ]
    forward, forward_size = _project_history_group(raw, {"next_offset": end})
    if not backward:
        return forward, forward_size, forward, forward_size
    response, response_size = _project_history_group(
        raw, {"next_before": end, "history_validator": validator},
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
    include_delivery_id: bool = False, fidelity: bool = False, content_read: bool = False,
) -> tuple[bool, tuple[dict[str, object], ...], int]:
    candidate = body_size + int(bool(rows and group)) + size
    kwargs = {"next_before": cursor or None, "history_validator": validator} if backward else {
        "next_offset": cursor if cursor < end else None,
    }
    if _history_frame_size(codec, request_id, messages_body_size=candidate, **kwargs) <= MAX_FRAME_BYTES:
        return True, group, size
    if rows:
        return False, group, size
    # Alone on its page, the group must still fit this page's real envelope.
    raw = [
        row
        for entry in raw_group
        for row in _history_entry_rows(
            entry, projection.approval_calls,
            include_delivery_id=include_delivery_id,
            fidelity=fidelity,
            content_read=content_read,
        )
    ]
    group, size = _project_history_group(raw, kwargs)
    candidate = body_size + int(bool(rows and group)) + size
    return _history_frame_size(codec, request_id, messages_body_size=candidate, **kwargs) <= MAX_FRAME_BYTES, group, size


def _fit_history_groups(
    projection: HistoryProjection,
    indexes: range,
    *,
    end: int,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
    backward: bool,
    validator: dict[str, object] | None = None,
) -> tuple[
    list[dict[str, object]], list[int], list[tuple[dict[str, object], ...]], HistoryProjection
]:
    groups = list(projection.groups)
    sizes = list(projection.body_sizes)
    cache_groups = list(projection.groups)
    cache_sizes = list(projection.body_sizes)
    rows: list[dict[str, object]] = []
    selected: list[int] = []
    response_groups: list[tuple[dict[str, object], ...]] = []
    body_size = 0
    codec = FrameCodec()
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    for index in indexes:
        while index >= len(groups):
            next_index = len(groups)
            group, size, forward, forward_size = _history_group(
                projection, next_index, end=end,
                include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read,
                backward=backward, validator=validator,
            )
            cache_groups.append(forward)
            cache_sizes.append(forward_size)
            groups.append(group)
            sizes.append(size)
        cursor = index if backward else min(index + 1, end)
        raw_group = projection.entry_groups[index]
        fits, group, size = _history_group_fits(
            codec, request_id, groups[index], sizes[index],
            body_size=body_size, rows=rows, cursor=cursor, end=end,
            backward=backward, validator=validator, raw_group=raw_group,
            projection=projection, include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read,
        )
        if not fits:
            break
        groups[index], sizes[index] = group, size
        rows.extend(group)
        body_size += int(bool(rows[:-len(group)] and group)) + size if group else 0
        selected.append(index)
        response_groups.append(group)
    return rows, selected, response_groups, HistoryProjection(
        projection.entry_groups, projection.approval_calls, projection.entry_index,
        tuple(cache_groups), tuple(cache_sizes),
    )

def _forward_page_history(
    projection: HistoryProjection,
    offset: int,
    *,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
) -> tuple[dict[str, object], HistoryProjection]:
    entry_count = len(projection.entry_groups) - 1
    if offset > entry_count:
        return {"messages": [], "next_offset": None}, projection
    stop = min(HISTORY_FIRST_PAGE_ENTRIES, entry_count) if offset == 0 else entry_count
    if stop == entry_count:
        stop += 1
    rows, selected, _response_groups, expanded = _fit_history_groups(
        projection, range(offset, stop), end=entry_count,
        include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read, backward=False,
    )
    next_offset = min(selected[-1] + 1, entry_count) if selected else offset
    return {"messages": rows, "next_offset": next_offset if next_offset < entry_count else None}, expanded


def _backward_page_history(
    projection: HistoryProjection,
    before: int,
    validator: dict[str, object],
    *,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
    first_page: bool = False,
) -> tuple[dict[str, object], HistoryProjection]:
    end = len(projection.entry_groups) - 1
    stable_end = validator["end"]
    if (
        not isinstance(stable_end, int)
        or stable_end > end
        or validator.get("entry_id") != _history_endpoint(projection, stable_end)
    ):
        raise ProtocolError(
            -32009,
            "session history changed; restart from the newest page",
            {"code": "stale_history_snapshot"},
        )
    if before < 0 or before > stable_end + int(first_page):
        raise ProtocolError(-32602, "before must not exceed history end")
    rows, selected, response_groups, expanded = _fit_history_groups(
        projection, range(before - 1, -1, -1), end=end,
        include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read, backward=True,
        validator=validator,
    )
    selected.reverse()
    response_groups.reverse()
    rows = [row for group in response_groups for row in group]
    boundary = selected[0] if selected else 0
    return {
        "messages": rows,
        "next_before": boundary if boundary > 0 else None,
        "history_validator": validator,
    }, expanded


def page_history(
    projection: HistoryProjection,
    cursor: int | None,
    *,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
    backward: bool = False,
    validator: dict[str, object] | None = None,
    first_page: bool = False,
) -> tuple[dict[str, object], HistoryProjection]:
    assert cursor is not None
    if backward:
        assert validator is not None
        return _backward_page_history(
            projection, cursor, validator,
            include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read, first_page=first_page,
        )
    return _forward_page_history(projection, cursor, include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read)

def _publish_history_page(
    store: ConversationStore,
    generation: tuple[int, bytes],
    result: dict,
    expanded: HistoryProjection,
    include_delivery_id: bool,
    fidelity: bool,
    content_read: bool,
) -> dict:
    with _cache_creation_lock:
        cached = _cache(
            store, include_delivery_id=include_delivery_id, fidelity=fidelity,
            content_read=content_read,
        )
        if cached is not None and cached.generation == generation and len(cached.projection.groups) < len(expanded.groups):
            _publish_cache(
                store,
                include_delivery_id=include_delivery_id,
                fidelity=fidelity,
                content_read=content_read,
                cached=_CachedProjection(generation, expanded),
            )
    return result


def _backward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
) -> dict:
    before = params["before"]
    validator = params.get("history_validator")
    if before is None:
        validator = {"end": 0, "entry_id": None}
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store, include_delivery_id=include_delivery_id, fidelity=fidelity,
        content_read=content_read,
    )
    end = len(projection.entry_groups) - 1
    if before is None:
        validator = {"end": end, "entry_id": _history_endpoint(projection, end)}
        cursor, first_page = end + 1, True
    else:
        cursor, first_page = before, False
    result, expanded = page_history(
        projection, cursor, include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read,
        backward=True, validator=validator, first_page=first_page,
    )
    return _publish_history_page(
        store, generation, result, expanded, include_delivery_id, fidelity, content_read
    )


def _forward_history_request(
    runtime: ServerRuntime,
    params: dict,
    *,
    include_delivery_id: bool, fidelity: bool, content_read: bool,
) -> dict:
    store = runtime.opened.store
    generation = _generation(store)
    projection = history_projection(
        store, include_delivery_id=include_delivery_id, fidelity=fidelity,
        content_read=content_read,
    )
    result, expanded = page_history(
        projection, params.get("offset", 0), include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read,
    )
    return _publish_history_page(
        store, generation, result, expanded, include_delivery_id, fidelity, content_read
    )


def _validate_history_params(params: dict) -> None:
    if set(params) - {"session_id", "before", "history_validator"}:
        raise ProtocolError(-32602, "invalid backward session history parameters")
    before = params["before"]
    if before is not None and (type(before) is not int or before < 0):
        raise ProtocolError(-32602, "before must be null or a nonnegative integer")
    validator = params.get("history_validator")
    if before is None and "history_validator" in params:
        raise ProtocolError(-32602, "history_validator is only valid for continuations")
    if before is not None and (
        not isinstance(validator, dict)
        or set(validator) != {"end", "entry_id"}
        or type(validator.get("end")) is not int
        or validator.get("end") < 0
        or (validator.get("entry_id") is not None and not isinstance(validator.get("entry_id"), str))
    ):
        raise ProtocolError(-32602, "history_validator is invalid")


def history(
    runtime: ServerRuntime, params: dict, *, include_delivery_id: bool = False,
    backward_enabled: bool = False, fidelity: bool = False, content_read: bool = False,
) -> dict:
    backward_keys = {"before", "history_validator"}
    if backward_keys.intersection(params) and not backward_enabled:
        raise ProtocolError(-32602, "backward session history requires negotiation")
    if "before" in params:
        _validate_history_params(params)
        return _backward_history_request(runtime, params, include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read)
    if backward_keys.intersection(params):
        raise ProtocolError(-32602, "invalid session history parameters")
    offset = params.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ProtocolError(-32602, "offset must be a nonnegative integer")
    return _forward_history_request(runtime, params, include_delivery_id=include_delivery_id, fidelity=fidelity, content_read=content_read)

def settings(runtime: ServerRuntime) -> dict:
    return {"model": runtime.model, "approval_mode": runtime.policy.default.value}


def catalog(runtime: ServerRuntime) -> dict:
    return {
        "models": known_model_names(),
        "providers": {
            model: provider
            for provider, models in PROVIDER_MODELS.items()
            for model in models
        },
    }


def _validated_image_payload(
    params: dict,
) -> tuple[str, list[tuple[str, str, str, bytes]]]:
    text = params.get("text", "")
    images = params.get("images")
    if (
        not isinstance(text, str)
        or not isinstance(images, list)
        or not 1 <= len(images) <= 4
    ):
        raise ProtocolError(-32602, "send_images needs text and 1 to 4 images")
    decoded = []
    total = 0
    for item in images:
        if not isinstance(item, dict):
            raise ProtocolError(-32602, "image must be an object")
        name, mime, data = item.get("name"), item.get("mime_type"), item.get("data")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or any(unicodedata.category(c) == "Cc" or c in DIRECTION_CONTROLS for c in name)
        ):
            raise ProtocolError(
                -32602, "image name must be a filename of at most 128 characters"
            )
        if (
            not isinstance(mime, str)
            or mime not in IMAGE_EXTENSIONS
            or not isinstance(data, str)
        ):
            raise ProtocolError(-32602, "unsupported image type or data")
        compact_data = "".join(data.split())
        if len(compact_data) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ProtocolError(-32602, "images exceed 512 KiB")
        try:
            raw = base64.b64decode(compact_data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ProtocolError(-32602, "invalid image base64") from exc
        total += len(raw)
        if total > MAX_IMAGE_BYTES:
            raise ProtocolError(-32602, "images exceed 512 KiB")
        if not raw or not image_signature_matches(mime, raw):
            raise ProtocolError(-32602, "image signature does not match its type")
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS[mime]:
            raise ProtocolError(-32602, "image extension does not match its type")
        decoded.append((name, mime, base64.b64encode(raw).decode(), raw))
    return text, decoded


def image_payload_digest(params: dict) -> str:
    text, decoded = _validated_image_payload(params)
    return _semantic_image_payload_digest(
        text, [(name, mime, raw) for name, mime, _data, raw in decoded]
    )


def image_message(runtime: ServerRuntime, params: dict) -> Message:
    """Decode, normalize, and publish images from an off-loop worker."""

    text, decoded = _validated_image_payload(params)
    provider = runtime.provider
    if provider is None:
        raise ProtocolError(-32003, "no active provider")
    policy = image_policy_for_provider(provider)
    blocks = [TextContent(text)]
    store = runtime.opened.store
    attachments_existed = "attachments" in os.listdir(store.directory_fd)
    with child_directory(store.directory_fd, "attachments", create=True) as attachments_fd:
        directories = []
        try:
            for name, _mime, _data, raw in decoded:
                try:
                    prepared = asyncio.run(normalize_image_bytes(raw, policy))
                except ValueError as exc:
                    raise ProtocolError(-32602, f"image cannot be prepared: {exc}") from exc
                raw = prepared.data or b""
                mime = prepared.media_type
                data = base64.b64encode(raw).decode("ascii")
                directory = uuid4().hex
                os.mkdir(directory, mode=0o700, dir_fd=attachments_fd)
                directories.append(directory)
                with child_directory(attachments_fd, directory) as directory_fd:
                    write_session_file(directory_fd, name, raw)
                path = store.session_dir / "attachments" / directory / name
                blocks.append(ImageContent(data, mime, str(path), len(raw)))
        except BaseException:
            for directory in directories:
                shutil.rmtree(directory, dir_fd=attachments_fd)
            if not attachments_existed:
                os.rmdir("attachments", dir_fd=store.directory_fd)
            raise
    return with_message_origin(
        Message(MessageRole.USER, blocks), MessageOrigin.USER
    )


def attachment_summary(message: Message) -> list[dict[str, object]]:
    """Describe image attachments for a ``user_message`` event without base64."""

    return [
        {
            "name": Path(block.path).name if block.path else "",
            "mime_type": block.mime_type,
            "size": block.size,
        }
        for block in message.content
        if isinstance(block, ImageContent)
    ]
