"""off-loop whisper.cpp invocation and immutable model cache."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Protocol

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.en.bin"
MODEL_SHA256 = "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"
MAX_MODEL_BYTES = 500_000_000


class Transcriber(Protocol):
    async def transcribe(self, wav: Path) -> str: ...
    async def close(self) -> None: ...


class WhisperCpp:
    def __init__(self, *, zeta_home: Path, model: str = "small.en") -> None:
        self._home = zeta_home
        self._model = model

    async def transcribe(self, wav: Path) -> str:
        model = self._home / "cache" / "voice" / "whisper.cpp" / f"ggml-{self._model}.bin"
        await self._ensure_model(model)
        return await asyncio.to_thread(self._transcribe_blocking, wav)

    async def _ensure_model(self, model: Path) -> None:
        if await verify_model(model):
            return
        await asyncio.to_thread(_download_model, model)
        if not await verify_model(model):
            raise RuntimeError("downloaded voice model failed sha256 verification")

    def _transcribe_blocking(self, wav: Path) -> str:
        model = self._home / "cache" / "voice" / "whisper.cpp" / f"ggml-{self._model}.bin"
        try:
            proc = subprocess.run(
                ["whisper-cli", "-m", str(model), "-l", "en", "-nt", "-np", str(wav)],
                check=True, capture_output=True, text=True, timeout=180,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("whisper-cli is missing; run brew install whisper-cpp ffmpeg") from exc
        return proc.stdout.strip()

    async def close(self) -> None:
        return None


def _download_model(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb", opener=lambda p, f: os.open(p, f, 0o600)) as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise RuntimeError("voice model download exceeds 500 MB")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not _verify_model(temporary):
            raise RuntimeError("downloaded voice model failed sha256 verification")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def verify_model(path: Path) -> bool:
    return await asyncio.to_thread(_verify_model, path)


def _verify_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256
