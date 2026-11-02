"""Protocol 1.1 session views and bounded image inputs over shared core seams."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import os
import shutil
import threading
import unicodedata
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.session_files import child_directory, write_session_file
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
from .protocol import MAX_REQUEST_ID_BYTES, FrameCodec, ProtocolError, bounded
from .runtime import ServerRuntime

EXTENSION_REQUESTS = [
    "rename_session",
    "delete_session",
    "session_tree",
    "session_history",
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


def bind_observer_store(server: Any) -> None:
    if server._bound_store is not None:
        server._bound_store.on_entries_installed = None
        server._bound_store = None
    state = server.runtime.state
    if state is None or server.connections is None:
        return
    store = state.opened.store
    session_id = state.session_id
    store.on_entries_installed = lambda delta: server.connections.offer_installed_rows(session_id, delta)
    server._bound_store = store


async def observer_snapshot(runtime: ServerRuntime, registry: ConnectionRegistry, observer: Any, params: dict) -> dict[str, object]:
    limit = params.get("limit", 128)
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ProtocolError(-32602, "limit must be an integer from 1 to 128")
    if runtime.opened is None:
        raise ProtocolError(-32003, "session is not active")
    session_id = runtime.session_id
    registry.begin_snapshot(observer, session_id)

    def read_page() -> tuple[list[Any], bool]:
        entries = list(runtime.opened.store.replay())
        return entries[-limit:], len(entries) > limit

    entries, has_more = await asyncio.to_thread(read_page)
    rows = [ConnectionRegistry._row_projection(entry) for entry in entries]
    anchor = {"seq": entries[-1].seq, "entry_id": entries[-1].id} if entries else {"seq": -1, "entry_id": None}
    result: dict[str, object] = {"session_id": session_id, "rows": rows, "anchor": anchor, "has_more_before": has_more, "partial_output_available": False}
    if not registry.activate_snapshot(observer, session_id, result):
        raise ProtocolError(-32009, "snapshot is stale", {"code": "stale_snapshot"})
    return result


def active(runtime: ServerRuntime, params: dict):
    session_id = params.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError(-32602, "session_id must be a nonempty string")
    if runtime.opened is None or session_id != runtime.session_id:
        raise ProtocolError(-32003, "session is not active")
    return runtime.opened.store


def require_mutable(runtime: ServerRuntime) -> None:
    loop = runtime.loop
    if loop is not None and (
        loop.background_children_running
        or loop.store.has_outstanding_tool_calls(loop.store.replay())
        or (runtime.policy is not None and runtime.policy.pending_requests())
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
            for branch in store.list_branches()
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


def history(runtime: ServerRuntime, params: dict) -> dict:
    offset = params.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ProtocolError(-32602, "offset must be a nonnegative integer")
    entries = [
        entry
        for entry in runtime.opened.store.replay()
        if entry.type in {"message", "notification"}
        and not (
            entry.type == "message"
            and is_passive_harness_message(entry.data["message"])
        )
    ]
    rows = []
    codec = FrameCodec()
    # Reserve the largest legal JSON-escaped request id, including the envelope.
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    for entry in entries[offset : offset + 8]:
        if entry.type == "notification":
            data = entry.data
            kind = data.get("kind", "agent_completion")
            if kind == "task_exited":
                text = (
                    f"background task {data.get('task_id', '?')} exited "
                    f"({data.get('exit_code')}); {data.get('headline', '')}"
                )
            else:
                text = data.get("text", "background agent notification")
            row = {
                "id": entry.id,
                "role": "system",
                "content": [{"type": "text", "text": bounded(text, 8000)}],
                "tool_result": None,
                "notification": dict(data),
            }
        else:
            message = entry.data["message"]
            content = []
            for block in message.get("content", []):
                if block.get("type") == "image":
                    content.append(
                        {
                            "type": "attachment",
                            "name": Path(block.get("path") or "image").name,
                            "size": block.get("size") or 0,
                        }
                    )
                elif block.get("type") == "text":
                    content.append({"type": "text", "text": bounded(block["text"], 8000)})
                elif block.get("type") == "tool_use":
                    call = block["tool_call"]
                    content.append({
                        "type": "tool_use",
                        "tool_call": {"id": call["id"], "name": call["name"], "arguments": {}},
                    })
                elif block.get("type") == "thinking":
                    body = block.get("body")
                    if type(body) is str and body:
                        body = bounded(body, 8000)
                        content.append({"type": "thinking", "text": body, "body": body})
                    else:
                        # Preserve the old header-only row without copying the
                        # legacy block's potentially private text.
                        content.append({"type": "thinking", "text": ""})
            result = message.get("tool_result")
            if result:
                result = {
                    **result,
                    "content": bounded(result.get("content", ""), 8000),
                    "content_blocks": [],
                    "structured_content": None,
                }
            row = {
                "id": entry.id,
                "role": message["role"],
                "content": content,
                "tool_result": result,
            }
            failed_turn = _failed_turn_history(message)
            if failed_turn is not None:
                row["failed_turn"] = failed_turn
        next_offset = offset + len(rows) + 1
        candidate = {
            "messages": [*rows, row],
            "next_offset": next_offset if next_offset < len(entries) else None,
        }
        if not codec.response_fits(request_id, candidate):
            if rows:
                break
            # Persisted messages can exceed a frame even after per-block bounds.
            # Keep their identity and advance the cursor instead of wedging refresh.
            row["content"] = [{"type": "text", "text": "[history row too large; truncated]"}]
            row["tool_result"] = None
            if entry.type == "notification":
                row["notification"] = None
        rows.append(row)
    next_offset = offset + len(rows)
    return {
        "messages": rows,
        "next_offset": next_offset if next_offset < len(entries) else None,
    }


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


def image_message(runtime: ServerRuntime, params: dict) -> Message:
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
        if len(data) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ProtocolError(-32602, "images exceed 512 KiB")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ProtocolError(-32602, "invalid image base64") from exc
        total += len(raw)
        if total > MAX_IMAGE_BYTES:
            raise ProtocolError(-32602, "images exceed 512 KiB")
        if not raw or not image_signature_matches(mime, raw):
            raise ProtocolError(-32602, "image signature does not match its type")
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS[mime]:
            raise ProtocolError(-32602, "image extension does not match its type")
        decoded.append((name, mime, data, raw))
    blocks = [TextContent(text)]
    store = runtime.opened.store
    attachments_existed = "attachments" in os.listdir(store.directory_fd)
    with child_directory(store.directory_fd, "attachments", create=True) as attachments_fd:
        directories = []
        try:
            for name, mime, data, raw in decoded:
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

OBSERVER_REQUESTS = frozenset({
    "list_sessions", "status", "session_tree", "session_history",
    "model_catalog", "session_settings", "ping", "delivery_status",
    "session_snapshot",
})
OBSERVER_READ_ONLY_CODE = "observer_read_only"

@dataclass
class _Connection:
    connection: Any
    role: str | None = None
    features: frozenset[str] = frozenset()
    queue: asyncio.Queue[tuple[bytes, asyncio.Future[bool]]] | None = None
    writer_task: asyncio.Task[None] | None = None
    snapshot_required: bool = True
    snapshot_session_id: str | None = None
    snapshot_seq: int = -1

class ConnectionRegistry:
    """Owns serve connection claims and the single output fan-out seam."""
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._connections: dict[Any, _Connection] = {}
        self._controller: _Connection | None = None
        self._observers: list[_Connection] = []
        self._closed = False
        self._ingress_lock = threading.Lock()
        self._ingress: deque[tuple[str, Any]] = deque()
        self._ingress_rows = 0
        self._drain_scheduled = False
        self._ingress_overflow = False

    def register(self, connection: Any) -> None:
        if self._closed:
            raise RuntimeError("connection registry is closed")
        self._connections[connection] = _Connection(connection)

    def claim(self, connection: Any, role: str, features: frozenset[str]) -> None:
        state = self._connections[connection]
        if state.role is not None:
            raise ValueError("connection already claimed")
        if role == "controller":
            if self._controller is not None:
                raise RuntimeError("another client is already connected")
            self._controller = state
        elif role == "observer":
            if len(self._observers) >= 8:
                raise RuntimeError("observer limit reached")
            state.queue = asyncio.Queue(maxsize=256)
            state.writer_task = asyncio.create_task(self._observer_writer(state))
            self._observers.append(state)
        else:
            raise ValueError("invalid role")
        state.role = role
        state.features = features
        if role == "observer":
            state.snapshot_required = True

    def release(self, connection: Any) -> None:
        state = self._connections.pop(connection, None)
        if state is None:
            return
        if state is self._controller:
            self._controller = None
        if state in self._observers:
            self._observers.remove(state)
        if state.writer_task is not None:
            state.writer_task.cancel()
        for item in list(state.queue._queue) if state.queue is not None else ():
            item[1].cancel()

    async def send(self, connection: Any, frame: bytes) -> bool:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            return await connection._write_physical(frame)
        assert state.queue is not None
        if state.queue.full():
            await self._overflow(state)
            return False
        future: asyncio.Future[bool] = self.loop.create_future()
        state.queue.put_nowait((frame, future))
        return await future

    async def offer_live(self, frame: bytes, *, required_feature: str | None = None) -> bool:
        controller_result = True
        if self._controller is not None and (
            required_feature is None or required_feature in self._controller.features
        ):
            controller_result = await self.send(self._controller.connection, frame)
        for state in tuple(self._observers):
            if required_feature is not None and required_feature not in state.features:
                continue
            if state.queue is None or state.queue.full():
                await self._overflow(state)
                continue
            future: asyncio.Future[bool] = self.loop.create_future()
            state.queue.put_nowait((frame, future))
        return controller_result

    async def _observer_writer(self, state: _Connection) -> None:
        assert state.queue is not None
        try:
            while True:
                frame, future = await state.queue.get()
                if future.cancelled():
                    continue
                try:
                    result = await state.connection._write_physical(frame)
                except (ConnectionError, OSError, RuntimeError) as exc:
                    if not future.done():
                        future.set_exception(exc)
                    return
                if not future.done():
                    future.set_result(result)
        except asyncio.CancelledError:
            return

    async def _overflow(self, state: _Connection) -> None:
        if state.connection not in self._connections:
            return
        if state.queue is not None:
            while not state.queue.empty():
                _, future = state.queue.get_nowait()
                future.cancel()
        if state.writer_task is not None:
            state.writer_task.cancel()
            await asyncio.gather(state.writer_task, return_exceptions=True)
            state.writer_task = None
        error = json.dumps(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32011, "message": "observer output overflow"}},
            separators=(",", ":"),
        ).encode() + b"\n"
        with contextlib.suppress(ConnectionError, OSError, RuntimeError):
            await state.connection._write_physical(error)
        await state.connection.close()

    def begin_snapshot(self, observer: Any, session_id: str) -> None:
        state = self._connections.get(observer)
        if state is None or state.role != "observer":
            raise RuntimeError("observer is not connected")
        state.snapshot_required = True
        state.snapshot_session_id = session_id

    def activate_snapshot(
        self, observer: Any, session_id: str, page: dict[str, Any]
    ) -> bool:
        state = self._connections.get(observer)
        if state is None or state.role != "observer":
            return False
        if state.snapshot_session_id != session_id:
            return False
        anchor = page.get("anchor")
        state.snapshot_seq = anchor.get("seq", -1) if isinstance(anchor, dict) else -1
        state.snapshot_required = False
        return True

    @staticmethod
    def _row_projection(entry: Any) -> dict[str, Any]:
        data = getattr(entry, "data", {})
        display = None
        if getattr(entry, "type", None) == "message":
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = []
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    content.append({"type": "text", "text": str(block.get("text", ""))[:8000]})
                elif kind == "thinking":
                    content.append({"type": "thinking", "text": ""})
                elif kind == "image":
                    content.append({"type": "attachment", "name": str(block.get("path", "image")).split("/")[-1], "size": block.get("size", 0) or 0})
                elif kind == "tool_use":
                    call = block.get("tool_call", {})
                    content.append({"type": "tool_use", "tool_call": {"id": call.get("id", ""), "name": call.get("name", ""), "arguments": {}}})
            display = {"role": message.get("role"), "content": content, "tool_result": None}
        elif getattr(entry, "type", None) == "notification":
            display = {"role": "system", "content": [{"type": "text", "text": str(data.get("text", "background agent notification"))[:8000]}], "tool_result": None}
        return {"seq": entry.seq, "entry_id": entry.id, "parent_entry_id": entry.parent_id, "lane": entry.lane, "type": entry.type, "display": display}

    def _enqueue_projection(self, session_id: str, entry: Any) -> None:
        for state in tuple(self._observers):
            if state.snapshot_required or state.snapshot_session_id not in {None, session_id}:
                continue
            if entry.seq <= state.snapshot_seq:
                continue
            payload = (json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"event": "transcript_row", "session_id": session_id, "row": self._row_projection(entry)}}, separators=(",", ":")) + "\n").encode()
            if state.queue is None or state.queue.full():
                asyncio.create_task(self._overflow(state))
                continue
            state.queue.put_nowait((payload, self.loop.create_future()))

    def offer_installed_rows(self, session_id: str, owned_delta: Any) -> None:
        try:
            count = len(owned_delta) if owned_delta != "reset" else 0
            with self._ingress_lock:
                if self._closed:
                    return
                if len(self._ingress) >= 256 or self._ingress_rows + count > 4096:
                    self._ingress_overflow = True
                    if not self._drain_scheduled:
                        self._drain_scheduled = True
                        self.loop.call_soon_threadsafe(self._drain_installed_ingress)
                    return
                self._ingress.append((session_id, owned_delta))
                self._ingress_rows += count
                if self._drain_scheduled:
                    return
                self._drain_scheduled = True
            self.loop.call_soon_threadsafe(self._drain_installed_ingress)
        except RuntimeError:
            return

    def _drain_installed_ingress(self) -> None:
        with self._ingress_lock:
            items = list(self._ingress)
            self._ingress.clear()
            self._ingress_rows = 0
            overflow = self._ingress_overflow
            self._ingress_overflow = False
            self._drain_scheduled = False
        if overflow:
            for state in tuple(self._observers):
                asyncio.create_task(self._overflow(state))
            items.clear()
        for session_id, delta in items:
            if delta == "reset":
                for state in tuple(self._observers):
                    if state.snapshot_session_id == session_id:
                        state.snapshot_required = True
                continue
            for entry in delta:
                self._enqueue_projection(session_id, entry)

    def close(self) -> None:
        self._closed = True
        for state in tuple(self._connections.values()):
            if state.writer_task is not None:
                state.writer_task.cancel()
            asyncio.create_task(state.connection.close())
        self._connections.clear()
        self._observers.clear()
        self._controller = None
