"""audio capture and bounded wav validation."""

from __future__ import annotations

import asyncio
import struct
import subprocess
from pathlib import Path
from typing import Protocol

MAX_SECONDS = 120
MAX_PCM_BYTES = 3_840_000


class AudioSource(Protocol):
    async def start(self, path: Path) -> None: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...


class MicrophoneSource:
    """Capture avfoundation audio in a private wav path."""

    def __init__(self, *, zeta_home: Path) -> None:
        self._root = zeta_home / "tmp" / "voice"
        self._process: subprocess.Popen[bytes] | None = None
        self._path: Path | None = None

    async def start(self, path: Path) -> None:
        self._path = path
        await asyncio.to_thread(self._start_blocking, path)

    def _start_blocking(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=False)
        self._process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f",
                "avfoundation", "-i", ":0", "-t", "125", "-ac", "1",
                "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_blocking)

    def _stop_blocking(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.write(b"q")
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    async def close(self) -> None:
        await self.stop()


def validate_wav(path: Path) -> None:
    """Reject anything except complete mono 16 kHz signed PCM wav."""
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("voice capture is not a wav file")
    _format, channels, rate = struct.unpack_from("<HHI", data, 20)
    bits = struct.unpack_from("<H", data, 34)[0]
    if channels != 1 or rate != 16_000 or bits != 16:
        raise ValueError("voice capture must be mono 16 kHz PCM")
    if len(data) - 44 > MAX_PCM_BYTES:
        raise ValueError("voice capture is longer than 120 seconds")
    if len(data) != struct.unpack_from("<I", data, 4)[0] + 8:
        raise ValueError("voice capture is incomplete")
