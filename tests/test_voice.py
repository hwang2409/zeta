from __future__ import annotations

import wave
from pathlib import Path

import pytest

from zeta.voice import VoiceRuntime, VoiceState, VoiceStateError


class FakeSource:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.closed = False

    async def start(self, path: Path) -> None:
        self.path = path
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 160)

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeTranscriber:
    async def transcribe(self, wav: Path) -> str:
        assert wav.exists()
        return "hello local voice"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_transcribes_and_cleans_private_audio(tmp_path: Path) -> None:
    source = FakeSource()
    runtime = VoiceRuntime(
        zeta_home=tmp_path,
        source=source,
        transcriber=FakeTranscriber(),
    )
    await runtime.start_recording()
    assert runtime.state is VoiceState.RECORDING
    assert source.path is not None and source.path.exists()
    assert await runtime.stop_and_transcribe() == "hello local voice"
    assert runtime.state is VoiceState.IDLE
    assert not source.path.exists()
    await runtime.close()
    assert runtime.state is VoiceState.CLOSED


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_transitions(tmp_path: Path) -> None:
    runtime = VoiceRuntime(
        zeta_home=tmp_path,
        source=FakeSource(),
        transcriber=FakeTranscriber(),
    )
    with pytest.raises(VoiceStateError):
        await runtime.stop_and_transcribe()
    await runtime.close()
    with pytest.raises(VoiceStateError):
        await runtime.start_recording()


@pytest.mark.asyncio
async def test_cancel_removes_capture(tmp_path: Path) -> None:
    source = FakeSource()
    runtime = VoiceRuntime(
        zeta_home=tmp_path,
        source=source,
        transcriber=FakeTranscriber(),
    )
    await runtime.start_recording()
    await runtime.cancel()
    assert runtime.state is VoiceState.IDLE
    assert source.closed
    assert source.path is not None and not source.path.exists()
    await runtime.close()
