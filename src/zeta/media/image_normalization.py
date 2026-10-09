"""Provider-safe image normalization behind one cancellable interface.

Small unchanged images stay in-process. Transformations run in a worker which
is limited by a parent-owned 1 GiB RSS watchdog and a 30 second deadline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

from .image_policy import ImagePolicy
from .images import detect_image_media_type, image_dimensions

WORKER_RSS_BUDGET_BYTES = 1024 * 1024 * 1024
WORKER_DEADLINE_SECONDS = 30.0
MAX_NORMALIZABLE_PIXELS = 1_000_000_000
_RSS_POLL_SECONDS = 0.2 if sys.platform == "darwin" else 0.05
_MAX_JPEG_HEADER_SCAN = 8 * 1024 * 1024
_FORMAT_NAMES = {"GIF": "gif", "JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}


@dataclass(frozen=True)
class PreparedImage:
    data: bytes | None
    media_type: str
    original_bytes: int
    original_width: int | None
    original_height: int | None
    original_format: str
    sent_width: int | None
    sent_height: int | None
    sent_format: str | None
    original_sha256: str
    note: str | None = None


@dataclass(frozen=True)
class NormalizationResult:
    image: PreparedImage | None = None
    utf8_text: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _ImageHeader:
    media_type: str
    format_name: str
    dimensions: tuple[int, int]
    animated: bool | None
    orientation: int | None


def _pread_all(file_descriptor: int, file_size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < file_size:
        chunk = os.pread(file_descriptor, min(1024 * 1024, file_size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _sha256(file_descriptor: int, file_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < file_size:
        chunk = os.pread(file_descriptor, min(1024 * 1024, file_size - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _fits(
    file_size: int,
    dimensions: tuple[int, int],
    media_type: str,
    policy: ImagePolicy,
) -> bool:
    raw_limit = policy.max_raw_bytes(media_type)
    return (
        (raw_limit is None or file_size <= raw_limit)
        and (policy.max_dimension is None or max(dimensions) <= policy.max_dimension)
    )


def _jpeg_dimensions(file_descriptor: int, file_size: int) -> tuple[int, int] | None:
    """Find a JPEG SOF marker without decoding pixels, within a bounded prefix."""

    data = os.pread(file_descriptor, min(file_size, _MAX_JPEG_HEADER_SCAN), 0)
    if len(data) < 2 or data[:2] != b"\xff\xd8":
        return None
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            return None
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return None
        marker = data[offset]
        offset += 1
        if marker == 0xDA or marker == 0xD9:
            return None
        if marker == 0x00 or marker == 0xD8 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            return None
        segment_size = int.from_bytes(data[offset : offset + 2], "big")
        if segment_size < 2 or offset + segment_size > len(data):
            return None
        if 0xC0 <= marker <= 0xC3 or 0xC5 <= marker <= 0xC7 or 0xC9 <= marker <= 0xCB or 0xCD <= marker <= 0xCF:
            if segment_size < 7:
                return None
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return (width, height) if width and height else None
        offset += segment_size
    return None


def _header(file_descriptor: int, file_size: int) -> _ImageHeader | None:
    prefix = os.pread(file_descriptor, min(file_size, 64 * 1024), 0)
    media_type = detect_image_media_type(prefix)
    if media_type is None:
        return None
    dimensions = image_dimensions(
        {"type": "image", "data": "", "mimeType": media_type}, prefix
    )
    if dimensions is None and media_type == "image/jpeg":
        dimensions = _jpeg_dimensions(file_descriptor, file_size)
    if dimensions is None:
        return None
    fallback = _ImageHeader(
        media_type,
        media_type.removeprefix("image/"),
        dimensions,
        None,
        None,
    )
    if dimensions[0] * dimensions[1] > MAX_NORMALIZABLE_PIXELS:
        return fallback
    try:
        from PIL import Image

        with os.fdopen(os.dup(file_descriptor), "rb") as handle, Image.open(handle) as image:
            format_name = _FORMAT_NAMES.get(image.format or "")
            if format_name is None:
                return fallback
            orientation = 1
            if raw_exif := image.info.get("exif"):
                exif = Image.Exif()
                exif.load(raw_exif)
                orientation = exif.get(0x0112, 1)
            return _ImageHeader(
                f"image/{format_name}",
                format_name,
                image.size,
                getattr(image, "n_frames", 1) > 1,
                orientation,
            )
    except Image.DecompressionBombError:
        return fallback
    except (OSError, RuntimeError, SyntaxError, ValueError):
        return fallback


def _omitted(
    file_descriptor: int,
    file_size: int,
    header: _ImageHeader,
    note: str,
) -> NormalizationResult:
    return NormalizationResult(
        image=PreparedImage(
            data=None,
            media_type=header.media_type,
            original_bytes=file_size,
            original_width=header.dimensions[0],
            original_height=header.dimensions[1],
            original_format=header.format_name,
            sent_width=None,
            sent_height=None,
            sent_format=None,
            original_sha256=_sha256(file_descriptor, file_size),
            note=note,
        )
    )


def _fast_path(
    file_descriptor: int,
    file_size: int,
    policy: ImagePolicy,
    header: _ImageHeader,
) -> NormalizationResult | None:
    if (
        header.animated is not False
        or header.orientation != 1
        or header.format_name not in policy.accepted_formats
        or not _fits(file_size, header.dimensions, header.media_type, policy)
    ):
        return None
    data = _pread_all(file_descriptor, file_size)
    if detect_image_media_type(data, complete=True) is None:
        return None
    width, height = header.dimensions
    return NormalizationResult(
        image=PreparedImage(
            data=data,
            media_type=header.media_type,
            original_bytes=file_size,
            original_width=width,
            original_height=height,
            original_format=header.format_name,
            sent_width=width,
            sent_height=height,
            sent_format=header.format_name,
            original_sha256=hashlib.sha256(data).hexdigest(),
        )
    )


def _process_rss_bytes(process_id: int) -> int:
    """Return current resident bytes for a worker on Linux or macOS."""

    proc_status = f"/proc/{process_id}/status"
    try:
        with open(proc_status, encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(process_id)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(value) * 1024 if value else 0
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0


def _kill_worker(process: asyncio.subprocess.Process) -> bool:
    try:
        process.kill()
    except ProcessLookupError:
        return False
    return True


async def _watch_worker(process: asyncio.subprocess.Process) -> str | None:
    started = time.monotonic()
    while process.returncode is None:
        rss = await asyncio.to_thread(_process_rss_bytes, process.pid)
        if rss > WORKER_RSS_BUDGET_BYTES:
            return "RSS safety budget" if _kill_worker(process) else None
        if time.monotonic() - started > WORKER_DEADLINE_SECONDS:
            return "time safety deadline" if _kill_worker(process) else None
        await asyncio.sleep(_RSS_POLL_SECONDS)
    return None


async def prepare_image(
    file_descriptor: int,
    *,
    file_size: int,
    policy: ImagePolicy,
) -> NormalizationResult:
    """Return a provider-safe image while owning and closing ``file_descriptor``.

    Valid images at or below one gigapixel return pixels unless the worker hits
    its hard RSS or time budget. Larger valid images return metadata and a hash.
    """

    try:
        header = _header(file_descriptor, file_size)
        if (
            header is not None
            and header.dimensions[0] * header.dimensions[1]
            > MAX_NORMALIZABLE_PIXELS
        ):
            return _omitted(
                file_descriptor,
                file_size,
                header,
                "Pixels could not be sent because the image exceeds 1 gigapixel; "
                "the original remains available at the reported path.",
            )
        if header is not None:
            fast = _fast_path(file_descriptor, file_size, policy, header)
            if fast is not None:
                return fast

        serialized = asdict(policy)
        serialized["accepted_formats"] = sorted(policy.accepted_formats)
        serialized["wire_limit_unit"] = policy.wire_limit_unit.value
        serialized["animation"] = policy.animation.value
        config = json.dumps(serialized)
        with tempfile.TemporaryFile() as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "zeta.media.image_normalization_worker",
                str(file_descriptor),
                str(output.fileno()),
                str(file_size),
                config,
                pass_fds=(file_descriptor, output.fileno()),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            communication = asyncio.create_task(process.communicate())
            watcher = asyncio.create_task(_watch_worker(process))
            try:
                _stdout, stderr = await communication
                limit = await watcher
            except asyncio.CancelledError:
                _kill_worker(process)
                await process.wait()
                communication.cancel()
                watcher.cancel()
                raise
            finally:
                if not watcher.done():
                    watcher.cancel()
            if limit is not None:
                await process.wait()
                if header is not None:
                    return _omitted(
                        file_descriptor,
                        file_size,
                        header,
                        f"Pixels were not sent because normalization hit the {limit}; "
                        "the original remains available at the reported path.",
                    )
                return NormalizationResult(
                    error=f"image normalization hit the {limit}"
                )
            output.seek(0)
            if process.returncode != 0:
                detail = stderr.decode("utf-8", "replace").strip()
                return NormalizationResult(
                    error=f"image worker failed: {detail or process.returncode}"
                )
            payload = pickle.load(output)
            if payload["kind"] == "image":
                return NormalizationResult(image=PreparedImage(**payload["image"]))
            if payload["kind"] == "text":
                return NormalizationResult(utf8_text=True)
            return NormalizationResult(error=payload["error"])
    finally:
        os.close(file_descriptor)
