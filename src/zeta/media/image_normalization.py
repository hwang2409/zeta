"""Cancellable image normalization in a memory-bounded worker process."""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import tempfile
from dataclasses import asdict, dataclass

from .image_policy import ImagePolicy


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


async def prepare_image(
    file_descriptor: int,
    *,
    file_size: int,
    policy: ImagePolicy,
) -> NormalizationResult:
    """Normalize an open image in a subprocess; cancellation kills all decode work.

    The function owns and closes ``file_descriptor``. The subprocess has a hard
    address-space limit equal to the policy's decoded-memory budget. Images that
    cannot fit return metadata without pixels rather than an error.
    """

    serialized = asdict(policy)
    serialized["accepted_formats"] = sorted(policy.accepted_formats)
    serialized["wire_limit_unit"] = policy.wire_limit_unit.value
    serialized["animation"] = policy.animation.value
    config = json.dumps(serialized)
    try:
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
            try:
                _stdout, stderr = await process.communicate()
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
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
