"""Private executable for provider-safe image normalization."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import pickle
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


def _fits(
    size: int,
    dimensions: tuple[int, int],
    media_type: str,
    policy: ImagePolicy,
) -> bool:
    raw_limit = policy.max_raw_bytes(media_type)
    return (
        (raw_limit is None or size <= raw_limit)
        and (policy.max_dimension is None or max(dimensions) <= policy.max_dimension)
    )


def _target(
    dimensions: tuple[int, int],
    size: int,
    media_type: str,
    policy: ImagePolicy,
) -> tuple[int, int]:
    scale = 1.0
    if policy.max_dimension is not None:
        scale = min(scale, policy.max_dimension / max(dimensions))
    raw_limit = policy.max_raw_bytes(media_type)
    if raw_limit is not None and size > raw_limit:
        scale = min(scale, math.sqrt(raw_limit / size) * 0.9)
    return tuple(max(1, round(value * scale)) for value in dimensions)


def _encode(
    image: Any,
    format_name: str,
    quality: int | None,
    icc_profile: bytes | None,
) -> bytes:
    output = BytesIO()
    common = {"icc_profile": icc_profile} if icc_profile else {}
    if format_name == "png":
        image.save(output, format="PNG", optimize=True, **common)
    elif format_name == "webp":
        image.save(output, format="WEBP", quality=quality, method=4, **common)
    else:
        image.save(output, format="JPEG", quality=quality, optimize=True, **common)
    return output.getvalue()


def _encode_to_limit(
    image: Any,
    original_format: str,
    policy: ImagePolicy,
    icc_profile: bytes | None,
) -> tuple[bytes, str]:
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


def _image_payload(
    handle: BinaryIO,
    *,
    data: bytes,
    media_type: str,
    file_size: int,
    original_dimensions: tuple[int, int],
    original_format: str,
    sent_dimensions: tuple[int, int],
    sent_format: str,
    note: str | None,
) -> dict[str, object]:
    return {
        "kind": "image",
        "image": {
            "data": data,
            "media_type": media_type,
            "original_bytes": file_size,
            "original_width": original_dimensions[0],
            "original_height": original_dimensions[1],
            "original_format": original_format,
            "sent_width": sent_dimensions[0],
            "sent_height": sent_dimensions[1],
            "sent_format": sent_format,
            "original_sha256": _sha256(handle),
            "note": note,
        },
    }


def _normalize_vips(
    handle: BinaryIO,
    file_size: int,
    policy: ImagePolicy,
    original_dimensions: tuple[int, int],
    original_format: str,
    *,
    animated: bool,
    orientation: int,
) -> dict[str, object]:
    import pyvips

    media_type = _MEDIA_TYPES[original_format]
    displayed_dimensions = (
        original_dimensions[::-1] if orientation in {5, 6, 7, 8} else original_dimensions
    )
    target = _target(displayed_dimensions, file_size, media_type, policy)
    source_path = f"/dev/fd/{handle.fileno()}"
    notes: list[str] = []
    if animated:
        notes.append("Animation was flattened explicitly to its first frame.")
    if orientation != 1:
        notes.append("EXIF orientation was applied before sizing.")

    while True:
        handle.seek(0)
        source = pyvips.Image.new_from_file(source_path, access="sequential")
        image = source.thumbnail_image(
            target[0], height=target[1], size="down", auto_rotate=True
        )
        if image.hasalpha():
            format_name = "png" if "png" in policy.accepted_formats else "webp"
            suffix = ".png" if format_name == "png" else ".webp"
            options: dict[str, object] = {"compression": 6} if format_name == "png" else {"Q": 90}
        else:
            format_name = "jpeg" if "jpeg" in policy.accepted_formats else "webp"
            suffix = ".jpg" if format_name == "jpeg" else ".webp"
            options = {"Q": 90, "strip": True}
        data = image.write_to_buffer(suffix, **options)
        raw_limit = policy.max_raw_bytes(_MEDIA_TYPES[format_name])
        if raw_limit is None or len(data) <= raw_limit:
            return _image_payload(
                handle,
                data=data,
                media_type=_MEDIA_TYPES[format_name],
                file_size=file_size,
                original_dimensions=original_dimensions,
                original_format=original_format,
                sent_dimensions=(image.width, image.height),
                sent_format=format_name,
                note=" ".join(notes) or None,
            )
        next_target = tuple(max(1, round(value * 0.8)) for value in target)
        if next_target == target:
            raise ValueError("could not encode image within provider wire limit")
        target = next_target


def _normalize_pillow(
    handle: BinaryIO,
    file_size: int,
    policy: ImagePolicy,
    *,
    extra_note: str | None = None,
) -> dict[str, object]:
    from PIL import Image, ImageOps

    handle.seek(0)
    with Image.open(handle) as image:
        detected_format = _FORMAT_NAMES.get(image.format or "")
        if detected_format is None or detected_format not in policy.accepted_formats:
            raise ValueError(f"unsupported image format: {image.format}")
        original_dimensions = image.size
        media_type = _MEDIA_TYPES[detected_format]
        animated = getattr(image, "n_frames", 1) > 1
        orientation = 1
        if raw_exif := image.info.get("exif"):
            exif = Image.Exif()
            exif.load(raw_exif)
            orientation = exif.get(0x0112, 1)
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
            return _image_payload(
                handle,
                data=data,
                media_type=media_type,
                file_size=file_size,
                original_dimensions=original_dimensions,
                original_format=detected_format,
                sent_dimensions=original_dimensions,
                sent_format=detected_format,
                note=extra_note,
            )
        if detected_format != "jpeg":
            try:
                return _normalize_vips(
                    handle,
                    file_size,
                    policy,
                    original_dimensions,
                    detected_format,
                    animated=animated,
                    orientation=orientation,
                )
            except ImportError:
                extra_note = (
                    "libvips is unavailable on this platform; Pillow fallback was used "
                    "under the worker watchdog."
                )

        notes = [extra_note] if extra_note else []
        if detected_format == "jpeg" and not _fits(
            file_size, original_dimensions, media_type, policy
        ):
            image.draft("RGB", _target(original_dimensions, file_size, media_type, policy))
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
                    notes.append(
                        "The embedded color profile could not be converted; color may shift."
                    )
            else:
                displayed = displayed.convert("RGB")
                notes.append("No embedded color profile was present; CMYK color may shift.")
        target = _target(displayed.size, file_size, media_type, policy)
        displayed.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
        data, sent_format = _encode_to_limit(
            displayed, detected_format, policy, icc_profile
        )
        return _image_payload(
            handle,
            data=data,
            media_type=_MEDIA_TYPES[sent_format],
            file_size=file_size,
            original_dimensions=original_dimensions,
            original_format=detected_format,
            sent_dimensions=displayed.size,
            sent_format=sent_format,
            note=" ".join(notes) or None,
        )


def _run(handle: BinaryIO, file_size: int, policy: ImagePolicy) -> dict[str, object]:
    header = handle.read(64 * 1024)
    media_type = detect_image_media_type(header)
    dimensions = (
        image_dimensions(
            {"type": "image", "data": "", "mimeType": media_type or ""}, header
        )
        if media_type
        else None
    )
    try:
        return _normalize_pillow(handle, file_size, policy)
    except MemoryError:
        return {
            "kind": "error",
            "error": "could not decode image within the memory safety budget",
        }
    except Exception as exc:
        from PIL import Image

        if (
            isinstance(exc, Image.DecompressionBombError)
            and media_type is not None
            and media_type != "image/jpeg"
            and dimensions is not None
        ):
            try:
                return _normalize_vips(
                    handle,
                    file_size,
                    policy,
                    dimensions,
                    media_type.removeprefix("image/"),
                    animated=False,
                    orientation=1,
                )
            except ImportError:
                Image.MAX_IMAGE_PIXELS = None
                return _normalize_pillow(
                    handle,
                    file_size,
                    policy,
                    extra_note=(
                        "libvips is unavailable on this platform; Pillow fallback was "
                        "used under the worker watchdog."
                    ),
                )
        if not isinstance(exc, (OSError, RuntimeError, SyntaxError, ValueError)):
            raise
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
                return _image_payload(
                    handle,
                    data=data,
                    media_type=complete_type,
                    file_size=file_size,
                    original_dimensions=fallback_dimensions,
                    original_format=fallback_format,
                    sent_dimensions=fallback_dimensions,
                    sent_format=fallback_format,
                    note=None,
                )
        if _stream_is_utf8(handle):
            return {"kind": "text"}
        return {"kind": "error", "error": f"could not decode image: {exc}"}


def main() -> None:
    input_fd, output_fd, file_size, raw_policy = sys.argv[1:]
    policy = _policy(raw_policy)
    with os.fdopen(int(input_fd), "rb") as handle:
        result = _run(handle, int(file_size), policy)
    with os.fdopen(int(output_fd), "wb") as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
