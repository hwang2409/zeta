"""the small public seam for local voice capture."""

from __future__ import annotations

import asyncio
import enum
import os
import signal
import tempfile
from collections.abc import Callable
from pathlib import Path

from .audio import AudioSource, MicrophoneSource, validate_wav
from .whisper_cpp import Transcriber, WhisperCpp


class VoiceState(enum.Enum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    PREPARING = "preparing"
    TRANSCRIBING = "transcribing"
    CANCELLING = "cancelling"
    CLOSED = "closed"


class VoiceStateError(RuntimeError):
    """The requested voice action is not valid in the current state."""


class VoiceRuntime:
    """Own capture, temporary audio, inference, and cleanup for one tui."""

    def __init__(
        self,
        *,
        zeta_home: str | Path,
        progress: Callable[[VoiceState], object] | None = None,
        source: AudioSource | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._home = Path(zeta_home)
        self._progress = progress
        self._source = source or MicrophoneSource(zeta_home=self._home)
        self._transcriber = transcriber or WhisperCpp(zeta_home=self._home)
        self._state = VoiceState.IDLE
        self._lock = asyncio.Lock()
        self._path: Path | None = None
        self._operation: asyncio.Task[object] | None = None
        self._generation = 0

    @property
    def state(self) -> VoiceState:
        return self._state

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        if self._progress is not None:
            self._progress(state)

    async def start_recording(self) -> None:
        async with self._lock:
            if self._state is not VoiceState.IDLE:
                raise VoiceStateError(f"cannot start voice in {self._state.value}")
            self._generation += 1
            generation = self._generation
            self._set_state(VoiceState.STARTING)
            root = self._home / "tmp" / "voice"
            await asyncio.to_thread(self._cleanup_stale, root)
            await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True, mode=0o700)
            directory = Path(tempfile.mkdtemp(prefix="", dir=root))
            self._path = directory / "capture.wav"
            try:
                await self._source.start(self._path)
            except BaseException:
                await asyncio.shield(self._source.close())
                await asyncio.shield(self._remove_capture())
                self._set_state(VoiceState.IDLE)
                raise
            if generation != self._generation:
                raise asyncio.CancelledError
            self._set_state(VoiceState.RECORDING)

    async def stop_and_transcribe(self) -> str:
        async with self._lock:
            if self._state is not VoiceState.RECORDING or self._path is None:
                raise VoiceStateError(f"cannot stop voice in {self._state.value}")
            self._set_state(VoiceState.STOPPING)
            try:
                await self._source.stop()
                await asyncio.to_thread(validate_wav, self._path)
                self._set_state(VoiceState.PREPARING)
                self._set_state(VoiceState.TRANSCRIBING)
                return await self._transcriber.transcribe(self._path)
            except asyncio.CancelledError:
                await asyncio.shield(self._source.close())
                raise
            finally:
                await asyncio.shield(self._remove_capture())
                if self._state is not VoiceState.CLOSED:
                    self._set_state(VoiceState.IDLE)

    async def cancel(self) -> None:
        async with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._generation += 1
            self._set_state(VoiceState.CANCELLING)
            try:
                await self._source.close()
            finally:
                await asyncio.shield(self._remove_capture())
                self._set_state(VoiceState.IDLE)

    async def close(self) -> None:
        async with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._generation += 1
            self._set_state(VoiceState.CANCELLING)
            try:
                await self._source.close()
                await self._transcriber.close()
            finally:
                await asyncio.shield(self._remove_capture())
                self._set_state(VoiceState.CLOSED)

    async def _remove_capture(self) -> None:
        path = self._path
        self._path = None
        if path is None:
            return
        await asyncio.to_thread(self._remove_path, path)

    @staticmethod
    def _cleanup_stale(root: Path) -> None:
        if not root.is_dir():
            return
        for directory in root.iterdir():
            record = directory / "process.pid"
            try:
                pid = int(record.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (OSError, ValueError):
                pass
            record.unlink(missing_ok=True)
            for child in directory.iterdir():
                child.unlink(missing_ok=True)
            directory.rmdir()

    @staticmethod
    def _remove_path(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        except OSError:
            pass
