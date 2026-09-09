"""Protocol 1.1 session views and bounded image inputs over shared core seams."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from uuid import uuid4

from ..model_catalog import PROVIDER_MODELS
from ..types import (
    ImageContent,
    Message,
    MessageRole,
    TextContent,
    image_signature_matches,
)
from .protocol import ProtocolError, bounded
from .runtime import ServerRuntime

EXTENSION_REQUESTS = [
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
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


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
    # Page history work; the frame codec also bounds the encoded response.
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
                content.append(block)
        result = message.get("tool_result")
        if result:
            result = {
                **result,
                "content": bounded(result.get("content", ""), 8000),
                "content_blocks": [],
                "structured_content": None,
            }
        rows.append(
            {
                "id": entry.id,
                "role": message["role"],
                "content": content,
                "tool_result": result,
            }
        )
    next_offset = offset + len(rows)
    return {
        "messages": rows,
        "next_offset": next_offset if next_offset < len(entries) else None,
    }


def settings(runtime: ServerRuntime) -> dict:
    return {"model": runtime.model, "approval_mode": runtime.policy.default.value}


def catalog(runtime: ServerRuntime) -> dict:
    models = PROVIDER_MODELS.get(
        runtime.provider,
        frozenset({"offline", "faster"}) if runtime.provider == "fake" else frozenset(),
    )
    return {"models": sorted(models | {runtime.model})}


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
            or Path(name).name != name
            or name in {".", ".."}
            or any(ord(c) < 32 for c in name)
        ):
            raise ProtocolError(
                -32602, "image name must be a filename of at most 128 characters"
            )
        if (
            not isinstance(mime, str)
            or mime not in IMAGE_TYPES
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
        decoded.append((name, mime, data, raw))
    blocks = [TextContent(text)]
    written = []
    try:
        for name, mime, data, raw in decoded:
            directory = runtime.opened.store.session_dir / "attachments" / uuid4().hex
            directory.mkdir(parents=True, mode=0o700)
            path = directory / name
            path.write_bytes(raw)
            written.append(path)
            blocks.append(ImageContent(data, mime, str(path), len(raw)))
    except OSError:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return Message(MessageRole.USER, blocks)
