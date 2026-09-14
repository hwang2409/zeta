"""Image detection, validation, and presentation helpers."""

from __future__ import annotations

import base64
import binascii
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .types import ToolImageBlock


SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)
ImageValidation = Literal["valid", "incomplete", "invalid"]
IMAGE_DEGRADATION_WARNING = "[image block unavailable: invalid stored image data]"


def detect_image_media_type(data: bytes, *, complete: bool = False) -> str | None:
    """Detect a supported image, optionally requiring a complete payload."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime_type = "image/gif"
    elif (
        len(data) >= 4
        and data[:4] == b"RIFF"
        and (len(data) < 12 or data[8:12] == b"WEBP")
    ):
        mime_type = "image/webp"
    else:
        return None
    if complete and image_validation_status(mime_type, data) != "valid":
        return None
    return mime_type


def image_signature_matches(mime_type: str, data: bytes) -> bool:
    if mime_type == "image/png":
        return (
            len(data) >= 24
            and data[:8] == b"\x89PNG\r\n\x1a\n"
            and data[12:16] == b"IHDR"
            and all(struct.unpack(">II", data[16:24]))
        )
    if mime_type == "image/jpeg":
        return data[:3] == b"\xff\xd8\xff"
    if mime_type == "image/gif":
        return (
            len(data) >= 10
            and data[:6] in {b"GIF87a", b"GIF89a"}
            and all(struct.unpack("<HH", data[6:10]))
        )
    if mime_type == "image/webp":
        return _webp_dimensions(data) is not None
    return True


def image_validation_status(
    mime_type: str, data: bytes, *, total_size: int | None = None
) -> ImageValidation:
    """Classify a complete or capped image payload."""

    total_size = len(data) if total_size is None else total_size
    if total_size < len(data):
        return "invalid"
    validator = _IMAGE_VALIDATORS.get(mime_type)
    return validator(data, total_size) if validator is not None else "invalid"


def _png_validation_status(data: bytes, total_size: int) -> ImageValidation:
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return "invalid"
    offset = 8
    saw_header = False
    while True:
        if offset + 8 > len(data):
            return "incomplete" if len(data) < total_size else "invalid"
        chunk_size = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_size
        if chunk_end > total_size:
            return "invalid"
        if chunk_end > len(data):
            return "incomplete"
        if not saw_header and chunk_type != b"IHDR":
            return "invalid"
        if chunk_type == b"IHDR":
            if (
                saw_header
                or chunk_size != 13
                or not all(struct.unpack(">II", data[offset + 8 : offset + 16]))
            ):
                return "invalid"
            saw_header = True
        if chunk_type == b"IEND":
            return "valid" if saw_header and chunk_size == 0 else "invalid"
        offset = chunk_end


def _jpeg_validation_status(data: bytes, total_size: int) -> ImageValidation:
    if len(data) < 2:
        return "incomplete" if len(data) < total_size else "invalid"
    if data[:2] != b"\xff\xd8":
        return "invalid"
    offset = 2
    while True:
        if offset >= len(data):
            return "incomplete" if len(data) < total_size else "invalid"
        if data[offset] != 0xFF:
            return "invalid"
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return "incomplete" if len(data) < total_size else "invalid"
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            return "valid"
        if marker == 0x00:
            return "invalid"
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > total_size:
            return "invalid"
        if offset + 2 > len(data):
            return "incomplete"
        segment_size = int.from_bytes(data[offset : offset + 2], "big")
        if segment_size < 2:
            return "invalid"
        segment_end = offset + segment_size
        if segment_end > total_size:
            return "invalid"
        if segment_end > len(data):
            return "incomplete"
        offset = segment_end
        if marker != 0xDA:
            continue
        while True:
            if offset >= len(data):
                return "incomplete" if len(data) < total_size else "invalid"
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker_start = offset
            offset += 1
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                return "incomplete" if len(data) < total_size else "invalid"
            scan_marker = data[offset]
            if scan_marker == 0x00 or 0xD0 <= scan_marker <= 0xD7:
                offset += 1
                continue
            if scan_marker == 0xD9:
                return "valid"
            offset = marker_start
            break


def _gif_validation_status(data: bytes, total_size: int) -> ImageValidation:
    if len(data) < 6:
        return "incomplete" if len(data) < total_size else "invalid"
    if data[:6] not in {b"GIF87a", b"GIF89a"}:
        return "invalid"
    if len(data) < 13:
        return "incomplete" if len(data) < total_size else "invalid"
    offset = 13
    packed = data[10]
    if packed & 0x80:
        color_table_end = offset + 3 * (1 << ((packed & 0x07) + 1))
        if color_table_end > total_size:
            return "invalid"
        if color_table_end > len(data):
            return "incomplete"
        offset = color_table_end
    while True:
        if offset >= len(data):
            return "incomplete" if len(data) < total_size else "invalid"
        block_type = data[offset]
        offset += 1
        if block_type == 0x3B:
            return "valid"
        if block_type == 0x21:
            if offset >= len(data):
                return "incomplete" if len(data) < total_size else "invalid"
            offset += 1
        elif block_type == 0x2C:
            descriptor_end = offset + 9
            if descriptor_end > total_size:
                return "invalid"
            if descriptor_end > len(data):
                return "incomplete"
            packed = data[offset + 8]
            offset = descriptor_end
            if packed & 0x80:
                color_table_end = offset + 3 * (1 << ((packed & 0x07) + 1))
                if color_table_end > total_size:
                    return "invalid"
                if color_table_end > len(data):
                    return "incomplete"
                offset = color_table_end
            if offset >= len(data):
                return "incomplete" if len(data) < total_size else "invalid"
            offset += 1
        else:
            return "invalid"
        while True:
            if offset >= len(data):
                return "incomplete" if len(data) < total_size else "invalid"
            block_size = data[offset]
            offset += 1
            block_end = offset + block_size
            if block_end > total_size:
                return "invalid"
            if block_end > len(data):
                return "incomplete"
            offset = block_end
            if block_size == 0:
                break


def _webp_chunk_dimensions(
    data: bytes, chunk_type: bytes, offset: int, chunk_size: int
) -> tuple[int, int] | None:
    if chunk_type == b"VP8X" and chunk_size >= 10:
        dimensions = (
            1 + int.from_bytes(data[offset + 4 : offset + 7], "little"),
            1 + int.from_bytes(data[offset + 7 : offset + 10], "little"),
        )
        return dimensions if all(dimensions) else None
    if chunk_type == b"VP8L" and chunk_size >= 5 and data[offset] == 0x2F:
        packed = int.from_bytes(data[offset + 1 : offset + 5], "little")
        dimensions = 1 + (packed & 0x3FFF), 1 + ((packed >> 14) & 0x3FFF)
        return dimensions if all(dimensions) else None
    if (
        chunk_type == b"VP8 "
        and chunk_size >= 10
        and data[offset + 3 : offset + 6] == b"\x9d\x01\x2a"
    ):
        dimensions = (
            int.from_bytes(data[offset + 6 : offset + 8], "little") & 0x3FFF,
            int.from_bytes(data[offset + 8 : offset + 10], "little") & 0x3FFF,
        )
        return dimensions if all(dimensions) else None
    return None


def _parse_webp(
    data: bytes, total_size: int
) -> tuple[ImageValidation, tuple[int, int] | None]:
    if len(data) < 12:
        return ("incomplete" if len(data) < total_size else "invalid", None)
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return "invalid", None
    riff_end = int.from_bytes(data[4:8], "little") + 8
    if riff_end > total_size:
        return "invalid", None
    if len(data) < riff_end:
        return "incomplete", None
    offset = 12
    image_dimensions: tuple[int, int] | None = None
    while offset < riff_end:
        if offset + 8 > riff_end:
            return "invalid", None
        chunk_type = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload_offset = offset + 8
        chunk_end = payload_offset + chunk_size
        padded_end = chunk_end + (chunk_size & 1)
        if padded_end > riff_end:
            return "invalid", None
        if padded_end > len(data):
            return "incomplete", None
        if chunk_type in {b"VP8 ", b"VP8L", b"VP8X"}:
            chunk_dimensions = _webp_chunk_dimensions(
                data, chunk_type, payload_offset, chunk_size
            )
            if chunk_dimensions is None:
                return "invalid", None
            if image_dimensions is None:
                image_dimensions = chunk_dimensions
        offset = padded_end
    if image_dimensions is None:
        return "invalid", None
    return "valid", image_dimensions


def _webp_validation_status(data: bytes, total_size: int) -> ImageValidation:
    return _parse_webp(data, total_size)[0]


_IMAGE_VALIDATORS: dict[str, Callable[[bytes, int], ImageValidation]] = {
    "image/png": _png_validation_status,
    "image/jpeg": _jpeg_validation_status,
    "image/gif": _gif_validation_status,
    "image/webp": _webp_validation_status,
}


def decoded_image_bytes(block: ToolImageBlock) -> bytes | None:
    try:
        return base64.b64decode(block["data"], validate=True)
    except (binascii.Error, ValueError):
        return None


def image_dimensions(
    block: ToolImageBlock, data: bytes | None = None
) -> tuple[int, int] | None:
    """Return cheap raster dimensions when the image header exposes them."""

    data = decoded_image_bytes(block) if data is None else data
    mime_type = block["mimeType"]
    if (
        data is not None
        and mime_type == "image/png"
        and image_signature_matches(mime_type, data)
    ):
        dimensions = struct.unpack(">II", data[16:24])
        if all(dimensions):
            return dimensions
    if (
        data is not None
        and mime_type == "image/gif"
        and image_signature_matches(mime_type, data)
    ):
        dimensions = struct.unpack("<HH", data[6:10])
        if all(dimensions):
            return dimensions
    if data is not None and mime_type == "image/webp":
        dimensions = _webp_dimensions(data)
        if dimensions is not None:
            return dimensions
    if (
        data is not None
        and mime_type == "image/jpeg"
        and image_signature_matches(mime_type, data)
    ):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9}:
                continue
            if offset + 2 > len(data):
                break
            segment_size = int.from_bytes(data[offset : offset + 2], "big")
            if segment_size < 2 or offset + segment_size > len(data):
                break
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(
                range(0xC9, 0xCC)
            ) | set(range(0xCD, 0xD0)):
                dimensions = (
                    int.from_bytes(data[offset + 5 : offset + 7], "big"),
                    int.from_bytes(data[offset + 3 : offset + 5], "big"),
                )
                if all(dimensions):
                    return dimensions
            offset += segment_size
    width = block.get("width")
    height = block.get("height")
    if type(width) is int and type(height) is int:
        return width, height
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    return _parse_webp(data, len(data))[1]


def image_description(
    block: ToolImageBlock,
    *,
    detailed: bool = False,
    reason: str | None = None,
    tool_name: str | None = None,
) -> str:
    """Describe an image without exposing its base64 payload."""

    data = decoded_image_bytes(block)
    dimensions = image_dimensions(block, data)
    if not detailed and dimensions is None and reason is None:
        return "[image block]"
    parts = ["[image block]"]
    if tool_name:
        parts.append(f"tool={tool_name}")
    parts.append(f"media_type={block['mimeType']}")
    if dimensions:
        parts.append(f"dimensions={dimensions[0]}x{dimensions[1]}")
    parts.append(f"bytes={len(data) if data is not None else 'unknown'}")
    path = block.get("path")
    if path:
        parts.append(f"path={path}")
    caption = block.get("caption")
    if caption:
        parts.append(f"caption={caption}")
    if reason:
        parts.append(f"fallback={reason}")
    return " ".join(parts)
