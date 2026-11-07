"""Validation and staged-file lifecycle for serve image deliveries."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ..core.session_files import child_directory, write_session_file
from ..core.store._client_delivery import (
    image_payload_digest as _semantic_image_payload_digest,
)
from ..media.image_normalization import normalize_image_bytes
from ..media.image_policy import image_policy_for_provider
from ..media.images import image_signature_matches
from ..protocol.types import (
    ImageContent,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from .protocol import ErrorDescriptor, ProtocolError
from .runtime import ServerRuntime

MAX_IMAGE_BYTES = 512 * 1024
IMAGE_EXTENSIONS = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
}
DIRECTION_CONTROLS = (
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


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
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "send_images needs text and 1 to 4 images"
        )
    decoded = []
    total = 0
    for item in images:
        if not isinstance(item, dict):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "image must be an object"
            )
        name, mime, data = item.get("name"), item.get("mime_type"), item.get("data")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or any(
                unicodedata.category(c) == "Cc" or c in DIRECTION_CONTROLS for c in name
            )
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "image name must be a filename of at most 128 characters",
            )
        if (
            not isinstance(mime, str)
            or mime not in IMAGE_EXTENSIONS
            or not isinstance(data, str)
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "unsupported image type or data"
            )
        compact_data = "".join(data.split())
        if len(compact_data) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "images exceed 512 KiB")
        try:
            raw = base64.b64decode(compact_data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "invalid image base64"
            ) from exc
        total += len(raw)
        if total > MAX_IMAGE_BYTES:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "images exceed 512 KiB")
        if not raw or not image_signature_matches(mime, raw):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "image signature does not match its type",
            )
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS[mime]:
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "image extension does not match its type",
            )
        decoded.append((name, mime, base64.b64encode(raw).decode(), raw))
    return text, decoded


@dataclass(frozen=True, slots=True)
class PreparedImageDelivery:
    payload_digest: str
    message: Message
    attachments: list[dict[str, object]]
    created_attachments_root: bool


def prepare_image_delivery(
    runtime: ServerRuntime, params: dict
) -> PreparedImageDelivery:
    """Decode, normalize, and publish images from an off-loop worker."""

    text, decoded = _validated_image_payload(params)
    payload_digest = _semantic_image_payload_digest(
        text, [(name, mime, raw) for name, mime, _data, raw in decoded]
    )
    provider = runtime.provider
    if provider is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active provider")
    policy = image_policy_for_provider(provider)
    blocks = [TextContent(text)]
    store = runtime.opened.store
    attachments_existed = "attachments" in os.listdir(store.directory_fd)
    with child_directory(
        store.directory_fd, "attachments", create=True
    ) as attachments_fd:
        directories = []
        try:
            for name, _mime, _data, raw in decoded:
                try:
                    prepared = asyncio.run(normalize_image_bytes(raw, policy))
                except ValueError as exc:
                    raise ProtocolError(
                        ErrorDescriptor.INVALID_PARAMS,
                        f"image cannot be prepared: {exc}",
                    ) from exc
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
    message = with_message_origin(Message(MessageRole.USER, blocks), MessageOrigin.USER)
    return PreparedImageDelivery(
        payload_digest,
        message,
        attachment_summary(message),
        not attachments_existed,
    )


def cleanup_image_delivery(
    runtime: ServerRuntime, prepared: PreparedImageDelivery
) -> None:
    """Remove files produced by a prepared delivery that will not publish."""

    store = runtime.opened.store
    with child_directory(store.directory_fd, "attachments") as attachments_fd:
        for block in prepared.message.content:
            if isinstance(block, ImageContent) and block.path is not None:
                shutil.rmtree(Path(block.path).parent.name, dir_fd=attachments_fd)
    if prepared.created_attachments_root:
        os.rmdir("attachments", dir_fd=store.directory_fd)


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
