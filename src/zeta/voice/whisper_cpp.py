"""off-loop whisper.cpp invocation and immutable model cache."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path
from typing import Protocol

from .process import OwnedProcess, run_stoppable

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.en.bin"
MODEL_SHA256 = "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"
MAX_MODEL_BYTES = 500_000_000


class Transcriber(Protocol):
    async def prepare(self) -> None: ...
    async def transcribe(self, wav: Path) -> str: ...


class WhisperCpp:
    def __init__(self, *, zeta_home: Path, model: str = "small.en") -> None:
        self._model_path = zeta_home / "cache" / "voice" / "whisper.cpp" / f"ggml-{model}.bin"

    async def prepare(self) -> None:
        """Reuse a verified model or download, verify, and atomically rename it."""
        if await run_stoppable(lambda stop: verify_model(self._model_path, stop)):
            return
        await run_stoppable(lambda stop: download_model(self._model_path, stop))

    async def transcribe(self, wav: Path) -> str:
        argv = ["whisper-cli", "-m", str(self._model_path), "-l", "en", "-nt", "-np", str(wav)]
        try:
            process = await OwnedProcess.spawn(argv)
        except FileNotFoundError as exc:
            raise RuntimeError("whisper-cli is missing; run brew install whisper-cpp ffmpeg") from exc
        try:
            code = await process.wait()
        finally:
            await process.kill()  # reaps on cancel; a no-op after a normal exit
        if code != 0:
            raise RuntimeError(f"whisper-cli failed ({code}): {bytes(process.stderr).decode(errors='replace')[-300:]}")
        return bytes(process.stdout).decode(errors="replace").strip()


def download_model(path: Path, stop: threading.Event, url: str = MODEL_URL) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.urandom(6).hex()}.partial")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=30) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                if stop.is_set():
                    raise RuntimeError("voice model download cancelled")
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise RuntimeError("voice model download exceeds 500 MB")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not verify_model(temporary, stop):
            raise RuntimeError("downloaded voice model failed sha256 verification")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_model(path: Path, stop: threading.Event) -> bool:
    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if stop.is_set():
                return False
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256
