"""Protocol 1.1 session views and bounded image inputs over shared core seams."""

from __future__ import annotations

import base64
import binascii
import os
import shutil
import unicodedata
from pathlib import Path
from uuid import uuid4

from ..core.session_files import child_directory, write_session_file
from ..model_catalog import PROVIDER_MODELS, known_model_names
from ..types import (
    ImageContent,
    Message,
    MessageRole,
    TextContent,
    image_signature_matches,
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
    "send_images",
]
MAX_IMAGE_BYTES = 512 * 1024
IMAGE_EXTENSIONS = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
}
DIRECTION_CONTROLS = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"


def active(runtime: ServerRuntime, params: dict):
    if runtime.opened is None or params.get("session_id") != runtime.session_id:
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


def history(runtime: ServerRuntime, params: dict) -> dict:
    offset = params.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ProtocolError(-32602, "offset must be a nonnegative integer")
    entries = [
        entry for entry in runtime.opened.store.replay() if entry.type == "message"
    ]
    rows = []
    codec = FrameCodec()
    # Reserve the largest legal JSON-escaped request id, including the envelope.
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    for entry in entries[offset : offset + 8]:
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
            row["content"] = [{"type": "text", "text": "[message too large for history; truncated]"}]
            row["tool_result"] = None
        rows.append(row)
    next_offset = offset + len(rows)
    return {
        "messages": rows,
        "next_offset": next_offset if next_offset < len(entries) else None,
    }


def settings(runtime: ServerRuntime) -> dict:
    return {"model": runtime.model, "approval_mode": runtime.policy.default.value}


def catalog(runtime: ServerRuntime) -> dict:
    if runtime.fake_catalog:
        return {"models": ["faster", "offline"]}
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
    return Message(MessageRole.USER, blocks)
