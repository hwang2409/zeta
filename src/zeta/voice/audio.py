"""ffmpeg capture into a private wav, bounded validation, and stale cleanup."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import signal
import stat
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .process import OwnedProcess

MAX_PCM_BYTES = 3_840_000  # 120 s of 16 kHz mono s16le; the writer stops storing past it
CAPTURE_NAME = "capture.wav"
PID_NAME = "ffmpeg.json"
LOCK_NAME = "lock"
_KNOWN_NAMES = (CAPTURE_NAME, PID_NAME, LOCK_NAME)
_STOP_GRACE = 3.0


class AudioSource(Protocol):
    async def start(self, path: Path) -> None: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...


class MicrophoneSource:
    """Capture avfoundation audio; ffmpeg streams PCM to a pipe we own.

    The reader thread writes the private 0600 wav, so if zeta dies while frames
    flow, ffmpeg gets EPIPE/SIGPIPE on its next write and exits. Accepted
    residual: if zeta is SIGKILLed while the device gives no frames (for example
    a pending TCC prompt), an idle ffmpeg that captured nothing stays until the
    next voice start, whose stale cleanup ends it through the pid record.
    """

    def __init__(self) -> None:
        self._process: OwnedProcess | None = None
        self._writer: WavWriter | None = None

    async def start(self, path: Path) -> None:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        writer = await asyncio.to_thread(WavWriter, path, lambda: loop.call_soon_threadsafe(_resolve, ready))
        self._writer = writer
        fixture = os.environ.get("ZETA_VOICE_INPUT_FILE")
        source = ["-re", "-i", fixture] if fixture else ["-f", "avfoundation", "-i", ":0"]
        argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", *source]
        # -t 125 caps media time (120 s accepted plus margin); it is not a wall clock
        argv += ["-t", "125", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
        try:
            self._process = await OwnedProcess.spawn(argv, stdout_sink=writer)
        except FileNotFoundError as exc:
            await asyncio.to_thread(writer.close)
            raise RuntimeError("ffmpeg is missing; run brew install whisper-cpp ffmpeg") from exc
        await asyncio.to_thread(write_pid_record, path.parent / PID_NAME, self._process.pid)
        await ready
        if not writer.received:
            await self.close()
            raise RuntimeError("voice capture ended before audio arrived: " + self._error())

    def _error(self) -> str:
        process = self._process
        tail = bytes(process.stderr).decode(errors="replace").strip() if process else ""
        return tail[-300:] or "no microphone input"

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        code = await process.request_stop(b"q", _STOP_GRACE)
        if code not in (0, 255, -signal.SIGTERM):
            raise RuntimeError(f"ffmpeg failed ({code}): {self._error()}")
        await self.close()

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            await process.kill()
        if self._writer is not None:
            await asyncio.to_thread(_remove, self._writer.path.parent / PID_NAME)
            self._writer = None


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class WavWriter:
    """Write streamed s16le PCM as a bounded wav; runs on the pipe reader thread."""

    def __init__(self, path: Path, on_ready: Callable[[], object]) -> None:
        self.path = path
        self.received = 0
        self._on_ready = on_ready
        self._lock = threading.Lock()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        self._fd: int | None = os.open(path, flags, 0o600)
        os.write(self._fd, _wav_header(0))

    def write(self, chunk: bytes) -> None:
        with self._lock:
            if self._fd is None:
                return
            first = self.received == 0
            room = MAX_PCM_BYTES - self.received
            if room > 0:
                os.write(self._fd, chunk[:room])
                self.received += min(room, len(chunk))
            if first:
                self._on_ready()

    def close(self) -> None:
        with self._lock:
            fd, self._fd = self._fd, None
            if fd is None:
                return
            try:
                os.pwrite(fd, _wav_header(self.received), 0)
            finally:
                os.close(fd)
                self._on_ready()


def _wav_header(pcm_bytes: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + pcm_bytes, b"WAVE", b"fmt ", 16, 1, 1, 16_000, 32_000, 2, 16, b"data", pcm_bytes,
    )


def validate_wav(path: Path) -> None:
    """Reject anything except complete mono 16 kHz signed PCM wav."""
    with path.open("rb") as stream:
        data = stream.read(44 + MAX_PCM_BYTES + 1)
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("voice capture is not a wav file")
    audio_format, channels, rate = struct.unpack_from("<HHI", data, 20)
    bits = struct.unpack_from("<H", data, 34)[0]
    if audio_format != 1 or channels != 1 or rate != 16_000 or bits != 16:
        raise ValueError("voice capture must be mono 16 kHz PCM")
    if len(data) - 44 > MAX_PCM_BYTES:
        raise ValueError("voice capture is longer than 120 seconds")
    if len(data) != struct.unpack_from("<I", data, 4)[0] + 8:
        raise ValueError("voice capture is incomplete")
    if len(data) == 44:
        raise ValueError("no speech was captured")


def process_identity(pid: int) -> str | None:
    """Start time and executable of a live pid, or None when it is gone."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=,comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def write_pid_record(path: Path, pid: int) -> None:
    identity = process_identity(pid)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as record:
        json.dump({"pid": pid, "identity": identity}, record)


def claim_directory(root: Path) -> tuple[Path, int]:
    """Create and lock this runtime's private directory after stale cleanup."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    sweep_stale(root)
    directory = Path(os.path.realpath(root)) / os.urandom(8).hex()
    directory.mkdir(mode=0o700)
    fd = os.open(directory / LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return directory, fd


def release_directory(directory: Path, fd: int) -> None:
    for name in _KNOWN_NAMES:
        _remove(directory / name)
    try:
        directory.rmdir()
    finally:
        os.close(fd)


def sweep_stale(root: Path) -> None:
    """Remove leftovers of runtimes whose lock is free; never follow symlinks."""
    for entry in os.scandir(root):
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            continue
        directory = Path(entry.path)
        try:
            fd = os.open(directory / LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                continue  # a live runtime owns it
            raise
        try:
            _end_recorded_process(directory / PID_NAME)
            for name in _KNOWN_NAMES:
                _remove(directory / name)
            try:
                directory.rmdir()
            except OSError:
                pass  # unknown entries stay untouched
        finally:
            os.close(fd)


def _end_recorded_process(record: Path) -> None:
    try:
        fd = os.open(record, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            data = json.load(stream)
        pid, identity = int(data["pid"]), data["identity"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    # signal only the same process: start time and executable must still match
    if identity is None or process_identity(pid) != identity:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if process_identity(pid) != identity:
                return
            time.sleep(0.05)  # not our child, so there is no wait() to block on


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
