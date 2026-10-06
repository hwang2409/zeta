"""Provider-neutral image input limits and image normalization."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from .images import detect_image_media_type, image_dimensions


@dataclass(frozen=True)
class ImageLimits:
    """Limits for one image sent to a completion provider."""

    max_bytes: int | None
    max_dimension: int | None


# Anthropic documents 5 MB and 8,000 pixels per dimension. OpenAI documents
# 20 MB per image and no source-dimension cap. Ollama's adapter is text-only,
# so it has no image wire limit. Keep this table provider-neutral so tools do
# not import provider modules.
ANTHROPIC_IMAGE_LIMITS = ImageLimits(5 * 1024 * 1024, 8_000)
CODEX_IMAGE_LIMITS = ImageLimits(20 * 1024 * 1024, None)
OLLAMA_IMAGE_LIMITS = ImageLimits(None, None)

_PROVIDER_IMAGE_LIMITS = {
    "anthropic": ANTHROPIC_IMAGE_LIMITS,
    "claude": ANTHROPIC_IMAGE_LIMITS,
    "codex": CODEX_IMAGE_LIMITS,
    "fake": ANTHROPIC_IMAGE_LIMITS,
    "openai": CODEX_IMAGE_LIMITS,
    "ollama": OLLAMA_IMAGE_LIMITS,
}

# Pillow's default decompression-bomb threshold rejects legitimate images over
# about 178 MP before JPEG draft decoding can reduce them. Normalization below
# requests decoder subsampling before load and bounds the provider-facing image.
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    media_type: str
    original_bytes: int
    original_width: int | None
    original_height: int | None
    original_format: str
    sent_width: int | None
    sent_height: int | None
    sent_format: str
    original_sha256: str


_FORMAT_NAMES = {
    "GIF": "gif",
    "JPEG": "jpeg",
    "PNG": "png",
    "WEBP": "webp",
}
_MEDIA_TYPES = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
_QUALITY_STEPS = (90, 80, 70, 60, 50, 40, 30, 20)


def image_limits_for_provider(provider: str) -> ImageLimits:
    """Return image limits for a configured provider."""

    try:
        return _PROVIDER_IMAGE_LIMITS[provider]
    except KeyError as exc:
        raise ValueError(f"unknown image provider: {provider}") from exc


def prepare_image(
    file_descriptor: int,
    *,
    file_size: int,
    limits: ImageLimits,
) -> PreparedImage:
    """Validate and, when needed, normalize one open image file.

    The function owns and closes ``file_descriptor``. Callers should run it in
    a worker thread because image decode and encoding are blocking operations.
    """

    with os.fdopen(file_descriptor, "rb") as handle:
        try:
            if limits.max_bytes is None or file_size <= limits.max_bytes:
                data = handle.read()
                media_type = detect_image_media_type(data, complete=True)
                if media_type is not None:
                    format_name = media_type.removeprefix("image/")
                    dimensions = image_dimensions(
                        {"type": "image", "data": "", "mimeType": media_type},
                        data,
                    )
                    if _fits_limits(file_size, dimensions, limits):
                        width, height = dimensions or (None, None)
                        return PreparedImage(
                            data=data,
                            media_type=media_type,
                            original_bytes=file_size,
                            original_width=width,
                            original_height=height,
                            original_format=format_name,
                            sent_width=width,
                            sent_height=height,
                            sent_format=format_name,
                            original_sha256=hashlib.sha256(data).hexdigest(),
                        )
                handle.seek(0)
            with Image.open(handle) as image:
                format_name = _FORMAT_NAMES.get(image.format or "")
                if format_name is None:
                    raise ValueError(f"unsupported image format: {image.format}")
                original_width, original_height = image.size
                if _fits_limits(file_size, image.size, limits):
                    image.verify()
                    handle.seek(0)
                    data = handle.read()
                    return PreparedImage(
                        data=data,
                        media_type=_MEDIA_TYPES[format_name],
                        original_bytes=file_size,
                        original_width=original_width,
                        original_height=original_height,
                        original_format=format_name,
                        sent_width=original_width,
                        sent_height=original_height,
                        sent_format=format_name,
                        original_sha256=hashlib.sha256(data).hexdigest(),
                    )

                target = _initial_target(image.size, file_size, limits)
                has_transparency = "A" in image.getbands() or "transparency" in image.info
                image.draft("RGB", target)
                image.load()
                if image.size != target:
                    image.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
                data, sent_format, sent_size = _encode_to_limit(
                    image,
                    original_format=format_name,
                    has_transparency=has_transparency,
                    max_bytes=limits.max_bytes,
                )
                sent_width, sent_height = sent_size
                handle.seek(0)
                original_sha256 = _sha256(handle)
                return PreparedImage(
                    data=data,
                    media_type=_MEDIA_TYPES[sent_format],
                    original_bytes=file_size,
                    original_width=original_width,
                    original_height=original_height,
                    original_format=format_name,
                    sent_width=sent_width,
                    sent_height=sent_height,
                    sent_format=sent_format,
                    original_sha256=original_sha256,
                )
        except (OSError, UnidentifiedImageError, SyntaxError) as exc:
            raise ValueError(f"could not decode image: {exc}") from exc


def _sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _fits_limits(
    file_size: int,
    dimensions: tuple[int, int] | None,
    limits: ImageLimits,
) -> bool:
    return (
        (limits.max_bytes is None or file_size <= limits.max_bytes)
        and (
            limits.max_dimension is None
            or (
                dimensions is not None
                and max(dimensions) <= limits.max_dimension
            )
        )
    )


def _initial_target(
    dimensions: tuple[int, int], file_size: int, limits: ImageLimits
) -> tuple[int, int]:
    scale = 1.0
    if limits.max_dimension is not None:
        scale = min(scale, limits.max_dimension / max(dimensions))
    if limits.max_bytes is not None and file_size > limits.max_bytes:
        scale = min(scale, math.sqrt(limits.max_bytes / file_size) * 0.9)
    return tuple(max(1, round(dimension * scale)) for dimension in dimensions)


def _encode_to_limit(
    image: Image.Image,
    *,
    original_format: str,
    has_transparency: bool,
    max_bytes: int | None,
) -> tuple[bytes, str, tuple[int, int]]:
    if has_transparency:
        working = image.convert("RGBA")
        formats = ("png", "webp")
    else:
        working = image.convert("RGB")
        if original_format == "webp":
            formats = ("webp", "jpeg")
        elif original_format == "png":
            formats = ("png", "jpeg")
        else:
            formats = ("jpeg", "webp")

    while True:
        for format_name in formats:
            qualities = (None,) if format_name == "png" else _QUALITY_STEPS
            for quality in qualities:
                data = _encode(working, format_name, quality)
                if max_bytes is None or len(data) <= max_bytes:
                    return data, format_name, working.size
        next_size = tuple(max(1, round(dimension * 0.8)) for dimension in working.size)
        if next_size == working.size:
            # A 1x1 encoded image is always well below documented provider limits.
            return data, format_name, working.size
        working.thumbnail(next_size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def _encode(image: Image.Image, format_name: str, quality: int | None) -> bytes:
    output = BytesIO()
    if format_name == "png":
        image.save(output, format="PNG", optimize=True)
    elif format_name == "webp":
        image.save(output, format="WEBP", quality=quality, method=4)
    else:
        image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()
