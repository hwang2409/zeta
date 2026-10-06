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
_RSS_POLL_SECONDS = 0.05
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


def _header(file_descriptor: int, file_size: int) -> _ImageHeader | None:
    prefix = os.pread(file_descriptor, min(file_size, 64 * 1024), 0)
    media_type = detect_image_media_type(prefix)
    if media_type is None:
        return None
    dimensions = image_dimensions(
        {"type": "image", "data": "", "mimeType": media_type}, prefix
    )
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
            return _ImageHeader(
                f"image/{format_name}",
                format_name,
                image.size,
                getattr(image, "n_frames", 1) > 1,
                image.getexif().get(0x0112, 1),
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


async def _watch_worker(process: asyncio.subprocess.Process) -> str | None:
    started = time.monotonic()
    while process.returncode is None:
        rss = await asyncio.to_thread(_process_rss_bytes, process.pid)
        if rss > WORKER_RSS_BUDGET_BYTES:
            process.kill()
            return "RSS safety budget"
        if time.monotonic() - started > WORKER_DEADLINE_SECONDS:
            process.kill()
            return "time safety deadline"
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
                process.kill()
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
