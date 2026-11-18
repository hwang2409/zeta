"""the small public seam for local voice capture."""

from __future__ import annotations

import asyncio
import contextlib
import enum
from collections.abc import Awaitable, Callable
from pathlib import Path

from .audio import (
    CAPTURE_NAME,
    AudioSource,
    MicrophoneSource,
    claim_directory,
    release_directory,
    remove_file,
    validate_wav,
)
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
    """Own capture, temporary audio, inference, and cleanup for one tui.

    Transitions happen synchronously on the loop, so they are serialized. Each
    start or stop runs as one runtime-owned task; cancelling its caller, or
    ``cancel``/``close``, cancels that task, then waits for shielded cleanup
    (children reaped, files removed) before cancellation propagates.
    """

    def __init__(
        self,
        *,
        zeta_home: str | Path,
        voice_engine: str = "whisper.cpp",
        voice_model: str = "small.en",
        progress: Callable[[VoiceState], object] | None = None,
        source: AudioSource | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._root = Path(zeta_home) / "tmp" / "voice"
        self._progress = progress
        self._source = source or MicrophoneSource()
        if voice_engine != "whisper.cpp":
            raise ValueError(f"unsupported voice engine: {voice_engine}")
        self._transcriber = transcriber or WhisperCpp(zeta_home=Path(zeta_home), model=voice_model)
        self._state = VoiceState.IDLE
        self._generation = 0
        self._operation: asyncio.Task[object] | None = None
        self._cleanup: asyncio.Task[None] | None = None
        self._directory: tuple[Path, int] | None = None

    @property
    def state(self) -> VoiceState:
        return self._state

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        if self._progress is not None:
            self._progress(state)

    def _begin(self, required: VoiceState, next_state: VoiceState, action: str) -> int:
        if self._state is not required:
            raise VoiceStateError(f"cannot {action} voice in {self._state.value}")
        self._generation += 1
        self._set_state(next_state)
        return self._generation

    async def start_recording(self) -> None:
        generation = self._begin(VoiceState.IDLE, VoiceState.STARTING, "start")
        await self._run(self._start(), generation)

    async def stop_and_transcribe(self) -> str:
        generation = self._begin(VoiceState.RECORDING, VoiceState.STOPPING, "stop")
        return await self._run(self._stop(), generation)

    async def cancel(self) -> None:
        if self._state is VoiceState.CLOSED:
            return
        self._generation += 1  # a finished transcript not yet handed back is dropped too
        if self._cleanup is None and self._state is not VoiceState.IDLE:
            self._cleanup = asyncio.create_task(self._discard(VoiceState.IDLE))
        if self._cleanup is not None:
            await _finish(self._cleanup)

    async def close(self) -> None:
        while self._cleanup is not None:
            with contextlib.suppress(Exception):  # a failed cancel must not stop close
                await _finish(self._cleanup)
        if self._state is VoiceState.CLOSED:
            return
        self._generation += 1
        if self._state is VoiceState.IDLE and self._directory is None:
            self._set_state(VoiceState.CLOSED)  # never used: close without yielding to the loop
            return
        self._cleanup = asyncio.create_task(self._discard(VoiceState.CLOSED))
        await _finish(self._cleanup)

    async def _run[T](self, operation: Awaitable[T], generation: int) -> T:
        task = asyncio.ensure_future(operation)
        self._operation = task  # type: ignore[assignment]
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                await self.cancel()  # our caller was cancelled
            raise
        finally:
            if self._operation is task:
                self._operation = None
        if generation != self._generation:
            raise asyncio.CancelledError  # cancelled or closed during the hand-back
        return result

    async def _start(self) -> None:
        try:
            if self._directory is None:
                claim = asyncio.ensure_future(asyncio.to_thread(claim_directory, self._root))
                try:
                    await _finish(claim)
                finally:  # keep a claimed directory even when cancelled, so cleanup owns it
                    if claim.exception() is None:
                        self._directory = claim.result()
            await self._source.start(self._directory[0] / CAPTURE_NAME)
        except Exception:
            await self._reset(VoiceState.IDLE)
            raise
        self._set_state(VoiceState.RECORDING)

    async def _stop(self) -> str:
        assert self._directory is not None
        capture = self._directory[0] / CAPTURE_NAME
        try:
            await self._source.stop()
            await asyncio.to_thread(validate_wav, capture)
            self._set_state(VoiceState.PREPARING)
            await self._transcriber.prepare()
            self._set_state(VoiceState.TRANSCRIBING)
            text = await self._transcriber.transcribe(capture)
        except Exception:
            await self._reset(VoiceState.IDLE)
            raise
        await self._reset(VoiceState.IDLE)
        return text

    async def _discard(self, final: VoiceState) -> None:
        try:
            self._set_state(VoiceState.CANCELLING)
            operation = self._operation
            if operation is not None:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
            await self._reset(final)
        finally:
            self._cleanup = None

    async def _reset(self, final: VoiceState) -> None:
        """Reap capture, delete audio, and publish ``final``; survives cancellation."""
        try:
            await _finish(asyncio.ensure_future(self._source.close()))
        finally:
            directory = self._directory
            if final is VoiceState.CLOSED:
                self._directory = None
            try:
                if directory is not None:
                    await _finish(asyncio.ensure_future(asyncio.to_thread(_remove_files, directory, final)))
            finally:
                self._set_state(final)


def _remove_files(directory: tuple[Path, int], final: VoiceState) -> None:
    if final is VoiceState.CLOSED:
        release_directory(*directory)
    else:
        remove_file(directory[0] / CAPTURE_NAME)


async def _finish[T](task: asyncio.Future[T]) -> T:
    """Await ``task`` to completion even if we are cancelled, then re-raise."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return task.result()
