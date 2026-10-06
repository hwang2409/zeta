"""Private executable for memory-bounded image normalization."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import pickle
import resource
import sys
from io import BytesIO
from typing import Any, BinaryIO

from .image_policy import AnimationPolicy, ImagePolicy, WireLimitUnit
from .images import detect_image_media_type, image_dimensions

_FORMAT_NAMES = {"GIF": "gif", "JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}
_MEDIA_TYPES = {name: f"image/{name}" for name in ("gif", "jpeg", "png", "webp")}
_QUALITY_STEPS = (90, 80, 70, 60, 50, 40, 30, 20)


def _policy(raw: str) -> ImagePolicy:
    value = json.loads(raw)
    return ImagePolicy(
        value["max_wire_size"],
        WireLimitUnit(value["wire_limit_unit"]),
        value["max_dimension"],
        frozenset(value["accepted_formats"]),
        AnimationPolicy(value["animation"]),
        value["decoded_memory_budget"],
    )


def _sha256(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _stream_is_utf8(handle: BinaryIO) -> bool:
    handle.seek(0)
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        while chunk := handle.read(1024 * 1024):
            decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _fits(size: int, dimensions: tuple[int, int], media_type: str, policy: ImagePolicy) -> bool:
    raw_limit = policy.max_raw_bytes(media_type)
    return (
        (raw_limit is None or size <= raw_limit)
        and (policy.max_dimension is None or max(dimensions) <= policy.max_dimension)
    )


def _target(dimensions: tuple[int, int], size: int, media_type: str, policy: ImagePolicy) -> tuple[int, int]:
    scale = 1.0
    if policy.max_dimension is not None:
        scale = min(scale, policy.max_dimension / max(dimensions))
    raw_limit = policy.max_raw_bytes(media_type)
    if raw_limit is not None and size > raw_limit:
        scale = min(scale, math.sqrt(raw_limit / size) * 0.9)
    return tuple(max(1, round(value * scale)) for value in dimensions)


def _encode(image: Any, format_name: str, quality: int | None, icc_profile: bytes | None) -> bytes:
    output = BytesIO()
    common = {"icc_profile": icc_profile} if icc_profile else {}
    if format_name == "png":
        image.save(output, format="PNG", optimize=True, **common)
    elif format_name == "webp":
        image.save(output, format="WEBP", quality=quality, method=4, **common)
    else:
        image.save(output, format="JPEG", quality=quality, optimize=True, **common)
    return output.getvalue()


def _encode_to_limit(image: Any, original_format: str, policy: ImagePolicy, icc_profile: bytes | None) -> tuple[bytes, str]:
    has_transparency = "A" in image.getbands() or "transparency" in image.info
    if has_transparency:
        working = image.convert("RGBA")
        formats = ("png", "webp")
    else:
        working = image.convert("RGB")
        formats = ("webp", "jpeg") if original_format == "webp" else ("jpeg", "webp")
    formats = tuple(name for name in formats if name in policy.accepted_formats)
    while True:
        for format_name in formats:
            qualities = (None,) if format_name == "png" else _QUALITY_STEPS
            for quality in qualities:
                data = _encode(working, format_name, quality, icc_profile)
                raw_limit = policy.max_raw_bytes(_MEDIA_TYPES[format_name])
                if raw_limit is None or len(data) <= raw_limit:
                    return data, format_name
        next_size = tuple(max(1, round(value * 0.8)) for value in working.size)
        if next_size == working.size:
            return data, format_name
        from PIL import Image

        working.thumbnail(next_size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def _omitted(handle: BinaryIO, *, file_size: int, media_type: str, dimensions: tuple[int, int], format_name: str, note: str) -> dict[str, object]:
    width, height = dimensions
    return {
        "kind": "image",
        "image": {
            "data": None,
            "media_type": media_type,
            "original_bytes": file_size,
            "original_width": width,
            "original_height": height,
            "original_format": format_name,
            "sent_width": None,
            "sent_height": None,
            "sent_format": None,
            "original_sha256": _sha256(handle),
            "note": note,
        },
    }


def _run(handle: BinaryIO, file_size: int, policy: ImagePolicy) -> dict[str, object]:
    header = handle.read(64 * 1024)
    media_type = detect_image_media_type(header)
    dimensions = image_dimensions(
        {"type": "image", "data": "", "mimeType": media_type or ""}, header
    ) if media_type else None
    format_name = media_type.removeprefix("image/") if media_type else "unknown"
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError

        handle.seek(0)
        with Image.open(handle) as image:
            detected_format = _FORMAT_NAMES.get(image.format or "")
            if detected_format is None or detected_format not in policy.accepted_formats:
                raise ValueError(f"unsupported image format: {image.format}")
            original_dimensions = image.size
            media_type = _MEDIA_TYPES[detected_format]
            estimated_decode_bytes = original_dimensions[0] * original_dimensions[1] * 4
            if (
                detected_format != "jpeg"
                and not _fits(file_size, original_dimensions, media_type, policy)
                and estimated_decode_bytes > policy.decoded_memory_budget
            ):
                return _omitted(
                    handle,
                    file_size=file_size,
                    media_type=media_type,
                    dimensions=original_dimensions,
                    format_name=detected_format,
                    note=(
                        "Pixels could not be sent because decoding this raster would exceed "
                        f"the {policy.decoded_memory_budget // (1024 * 1024)} MiB safety budget; "
                        "the original remains available at the reported path."
                    ),
                )
            animated = getattr(image, "n_frames", 1) > 1
            orientation = image.getexif().get(0x0112, 1)
            normalize = (
                not _fits(file_size, original_dimensions, media_type, policy)
                or animated
                or orientation != 1
            )
            if not normalize:
                handle.seek(0)
                data = handle.read()
                if detect_image_media_type(data, complete=True) is None:
                    raise ValueError("could not decode image: invalid image container")
                width, height = original_dimensions
                return {"kind": "image", "image": {
                    "data": data, "media_type": media_type, "original_bytes": file_size,
                    "original_width": width, "original_height": height,
                    "original_format": detected_format, "sent_width": width,
                    "sent_height": height, "sent_format": detected_format,
                    "original_sha256": hashlib.sha256(data).hexdigest(), "note": None,
                }}

            notes: list[str] = []
            if detected_format == "jpeg" and not _fits(
                file_size, original_dimensions, media_type, policy
            ):
                image.draft(
                    "RGB", _target(original_dimensions, file_size, media_type, policy)
                )
            if animated:
                image.seek(0)
                notes.append("Animation was flattened explicitly to its first frame.")
            displayed = ImageOps.exif_transpose(image)
            displayed.load()
            if orientation != 1:
                notes.append("EXIF orientation was applied before sizing.")
            icc_profile = displayed.info.get("icc_profile")
            if displayed.mode == "CMYK":
                if icc_profile:
                    try:
                        from PIL import ImageCms
                        displayed = ImageCms.profileToProfile(
                            displayed,
                            BytesIO(icc_profile),
                            ImageCms.createProfile("sRGB"),
                            outputMode="RGB",
                        )
                        icc_profile = None
                    except (OSError, ValueError):
                        displayed = displayed.convert("RGB")
                        notes.append("The embedded color profile could not be converted; color may shift.")
                else:
                    displayed = displayed.convert("RGB")
                    notes.append("No embedded color profile was present; CMYK color may shift.")
            target = _target(displayed.size, file_size, media_type, policy)
            displayed.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
            data, sent_format = _encode_to_limit(
                displayed, detected_format, policy, icc_profile
            )
            return {"kind": "image", "image": {
                "data": data, "media_type": _MEDIA_TYPES[sent_format],
                "original_bytes": file_size, "original_width": original_dimensions[0],
                "original_height": original_dimensions[1], "original_format": detected_format,
                "sent_width": displayed.width, "sent_height": displayed.height,
                "sent_format": sent_format, "original_sha256": _sha256(handle),
                "note": " ".join(notes) or None,
            }}
    except (
        Image.DecompressionBombError,
        OSError,
        RuntimeError,
        SyntaxError,
        ValueError,
        UnidentifiedImageError,
    ) as exc:
        if (
            isinstance(exc, Image.DecompressionBombError)
            and media_type is not None
            and dimensions is not None
        ):
            return _omitted(
                handle,
                file_size=file_size,
                media_type=media_type,
                dimensions=dimensions,
                format_name=format_name,
                note=(
                    "Pixels could not be sent because decoding this raster would exceed "
                    f"the {policy.decoded_memory_budget // (1024 * 1024)} MiB safety budget; "
                    "the original remains available at the reported path."
                ),
            )
        handle.seek(0)
        data = handle.read() if file_size <= 32 * 1024 * 1024 else b""
        complete_type = detect_image_media_type(data, complete=True)
        if complete_type is not None:
            fallback_dimensions = image_dimensions(
                {"type": "image", "data": "", "mimeType": complete_type}, data
            ) or dimensions
            if fallback_dimensions is not None and _fits(
                file_size, fallback_dimensions, complete_type, policy
            ):
                fallback_format = complete_type.removeprefix("image/")
                return {"kind": "image", "image": {
                    "data": data, "media_type": complete_type,
                    "original_bytes": file_size,
                    "original_width": fallback_dimensions[0],
                    "original_height": fallback_dimensions[1],
                    "original_format": fallback_format,
                    "sent_width": fallback_dimensions[0],
                    "sent_height": fallback_dimensions[1],
                    "sent_format": fallback_format,
                    "original_sha256": hashlib.sha256(data).hexdigest(),
                    "note": None,
                }}
        if _stream_is_utf8(handle):
            return {"kind": "text"}
        return {"kind": "error", "error": f"could not decode image: {exc}"}
    except MemoryError:
        if media_type and dimensions:
            return _omitted(
                handle, file_size=file_size, media_type=media_type,
                dimensions=dimensions, format_name=format_name,
                note="Pixels could not be sent because decoding exceeded the memory safety budget; the original remains available at the reported path.",
            )
        return {"kind": "error", "error": "could not decode image within the memory safety budget"}


def main() -> None:
    input_fd, output_fd, file_size, raw_policy = sys.argv[1:]
    policy = _policy(raw_policy)
    if sys.platform.startswith("linux"):
        address_limit = policy.decoded_memory_budget + 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
    with os.fdopen(int(input_fd), "rb") as handle:
        result = _run(handle, int(file_size), policy)
    with os.fdopen(int(output_fd), "wb") as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
