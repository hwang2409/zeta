from __future__ import annotations

import asyncio
import os
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit.application.current import create_app_session, set_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.voice import VoiceRuntime, VoiceState, VoiceStateError
from zeta.voice.process import OwnedProcess


class FakeSource:
    """Writes a valid capture; optionally blocks in start until cancelled."""

    def __init__(self, *, block_start: bool = False) -> None:
        self.block_start = block_start
        self.reached = asyncio.Event()
        self.path: Path | None = None
        self.live = False

    async def start(self, path: Path) -> None:
        self.path = path
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\1\0" * 160)
        self.live = True
        if self.block_start:
            self.block_start = False
            self.reached.set()
            await asyncio.Future()  # held until the runtime cancels capture startup

    async def stop(self) -> None:
        self.live = False

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.live = False


class FakeTranscriber:
    """Blocks once in prepare (with a partial download) or in a real child process."""

    def __init__(self, home: Path, *, block: str | None = None, text: str = "hello local voice") -> None:
        self.block = block
        self.text = text
        self.reached = asyncio.Event()
        self.partial = home / "cache" / "voice" / "ggml-small.en.bin.partial"
        self.child_pid: int | None = None

    async def prepare(self) -> None:
        if self.block != "prepare":
            return
        self.block = None
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self.partial.write_bytes(b"partial model")
        try:
            self.reached.set()
            await asyncio.Future()
        finally:
            self.partial.unlink()

    async def transcribe(self, wav: Path) -> str:
        assert wav.exists()
        if self.block != "transcribe":
            return self.text
        self.block = None
        process = await OwnedProcess.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
        self.child_pid = process.pid
        try:
            self.reached.set()
            await asyncio.wait_for(process.wait(), 30)  # held until the runtime cancels transcription
            return "late text"
        finally:
            await process.kill()


def _assert_reaped(pid: int | None) -> None:
    assert pid is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def _voice_files(home: Path) -> list[str]:
    root = home / "tmp" / "voice"
    names = [path.name for path in root.rglob("*")] if root.exists() else []
    return sorted(name for name in names if name != "lock" and len(name) != 16)


@pytest.mark.asyncio
async def test_runtime_transcribes_and_cleans_private_audio(tmp_path: Path) -> None:
    source = FakeSource()
    runtime = VoiceRuntime(zeta_home=tmp_path, source=source, transcriber=FakeTranscriber(tmp_path))
    try:
        await runtime.start_recording()
        assert runtime.state is VoiceState.RECORDING
        assert source.path is not None
        assert source.path.parent.stat().st_mode & 0o777 == 0o700
        assert await runtime.stop_and_transcribe() == "hello local voice"
        assert runtime.state is VoiceState.IDLE
        assert not source.path.exists()
    finally:
        await runtime.close()
    assert runtime.state is VoiceState.CLOSED
    assert list((tmp_path / "tmp" / "voice").iterdir()) == []


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_transitions(tmp_path: Path) -> None:
    runtime = VoiceRuntime(zeta_home=tmp_path, source=FakeSource(), transcriber=FakeTranscriber(tmp_path))
    with pytest.raises(VoiceStateError):
        await runtime.stop_and_transcribe()
    await runtime.close()
    with pytest.raises(VoiceStateError):
        await runtime.start_recording()


PHASES = ("capture_start", "recording", "model_prepare", "transcription", "insertion_handoff")
CASES = [
    (phase, trigger)
    for phase in PHASES
    for trigger in ("cancel", "caller_cancel", "close")
    if not (phase == "recording" and trigger == "caller_cancel")  # no caller is waiting while recording
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("phase", "trigger"), CASES)
async def test_cancellation_at_every_phase_cleans_up_before_it_propagates(
    tmp_path: Path, phase: str, trigger: str
) -> None:
    source = FakeSource(block_start=phase == "capture_start")
    block = {"model_prepare": "prepare", "transcription": "transcribe"}.get(phase)
    transcriber = FakeTranscriber(tmp_path, block=block)
    caller: asyncio.Task[object] | None = None
    triggered: list[asyncio.Future[object]] = []
    states: list[VoiceState] = []

    def fire() -> None:
        if trigger == "caller_cancel":
            assert caller is not None
            caller.cancel()
        else:
            triggered.append(asyncio.ensure_future(getattr(runtime, trigger)()))

    def progress(state: VoiceState) -> None:
        # insertion handoff: the transcript exists, but the caller has not taken it yet
        handoff = phase == "insertion_handoff" and state is VoiceState.IDLE and states[-1:] == [VoiceState.TRANSCRIBING]
        if handoff and not triggered and not (caller is not None and caller.cancelled()):
            fire()
        states.append(state)

    runtime = VoiceRuntime(zeta_home=tmp_path, progress=progress, source=source, transcriber=transcriber)
    try:
        if phase == "capture_start":
            caller = asyncio.create_task(runtime.start_recording())
            await await_or_fail(source.reached.wait(), what="capture startup")
        else:
            await runtime.start_recording()
            if phase != "recording":
                caller = asyncio.create_task(runtime.stop_and_transcribe())
            if block is not None:
                await await_or_fail(transcriber.reached.wait(), what=f"{phase} reached")
        if phase not in ("recording", "insertion_handoff"):
            fire()
        if caller is not None:
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(caller, what=f"{phase} caller")
        if triggered:
            await await_or_fail(triggered[0], what=f"{trigger} cleanup")
        elif phase == "recording":
            fire()
            await await_or_fail(triggered[0], what=f"{trigger} cleanup")

        # cleanup finished before cancellation reached the caller
        assert not source.live
        assert not transcriber.partial.exists()
        if phase == "transcription":
            _assert_reaped(transcriber.child_pid)
        assert _voice_files(tmp_path) == []
        assert runtime.state is (VoiceState.CLOSED if trigger == "close" else VoiceState.IDLE)
        if trigger != "close":
            transcriber.text = "fresh text"
            await runtime.start_recording()
            assert await await_or_fail(runtime.stop_and_transcribe(), what="fresh round") == "fresh text"
    finally:
        await runtime.close()
    assert list((tmp_path / "tmp" / "voice").iterdir()) == []


def _voice_app(tmp_path: Path, transcriber: FakeTranscriber, changed: asyncio.Event | None = None) -> TUIApp:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        zeta_home=tmp_path / "home",
        console=Console(file=open(os.devnull, "w"), force_terminal=False, width=120),  # noqa: SIM115
    )

    def progress(state: VoiceState) -> None:
        app._voice_progress(state)
        if changed is not None:
            changed.set()

    app._voice = VoiceRuntime(zeta_home=tmp_path / "home", progress=progress, source=FakeSource(), transcriber=transcriber)
    return app


async def _until(changed: asyncio.Event, predicate: Callable[[], bool], what: str) -> None:
    """Wait for ``predicate`` after each editor, voice-state, or submit change."""
    while True:
        changed.clear()
        if predicate():
            return
        await await_or_fail(changed.wait(), what=what)


@pytest.mark.asyncio
async def test_voice_toggle_inserts_editable_transcript_through_real_tui(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(tmp_path, text="spoken words")
    changed = asyncio.Event()
    app = _voice_app(tmp_path, transcriber, changed)
    submitted: list[str] = []

    def submit(value: str) -> None:
        submitted.append(value)
        changed.set()

    app._submit_input = submit  # type: ignore[method-assign]
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        session = app._make_session()
        input_ready = asyncio.Event()
        run_full_screen = app._run_full_screen

        async def run_full_screen_after_ready(*args: object) -> None:
            input_ready.set()
            await run_full_screen(*args)

        app._run_full_screen = run_full_screen_after_ready  # type: ignore[method-assign]
        running = asyncio.create_task(app.run(session))
        try:
            await await_or_fail(input_ready.wait(), what="tui input ready")
            buffer = session.default_buffer
            buffer.on_text_changed += lambda _buffer: changed.set()
            pipe.send_text("say: ")
            await _until(changed, lambda: buffer.text == "say: ", "draft typed")

            def toolbar() -> str:
                return "".join(text for _style, text in app._status_toolbar())

            pipe.send_text("\x18\x16")  # c-x c-v starts
            await _until(changed, lambda: app._voice.state is VoiceState.RECORDING, "recording")
            assert "voice: recording 0:00" in toolbar()
            pipe.send_text("\x18\x16")  # c-x c-v stops and transcribes
            await _until(changed, lambda: buffer.text == "say: spoken words", "transcript inserted at cursor")
            assert app._voice.state is VoiceState.IDLE
            assert "voice:" not in toolbar()
            assert submitted == []  # nothing is sent until enter

            # a selection is replaced by the next transcript
            transcriber.text = "voice"
            buffer.cursor_position = len("say: ")
            buffer.start_selection()
            buffer.cursor_position = len(buffer.text)
            pipe.send_text("\x18\x16")
            await _until(changed, lambda: app._voice.state is VoiceState.RECORDING, "second recording")
            pipe.send_text("\x18\x16")
            await _until(changed, lambda: buffer.text == "say: voice", "selection replaced")

            # a session change during transcription drops the late transcript
            transcriber.block = "transcribe"
            pipe.send_text("\x18\x16")
            await _until(changed, lambda: app._voice.state is VoiceState.RECORDING, "third recording")
            pipe.send_text("\x18\x16")
            await await_or_fail(transcriber.reached.wait(), what="transcription started")
            assert "voice: transcribing" in toolbar()
            app._agent_navigation.transcript_control.view_path = tmp_path / "other-session"
            app._viewed_session_changed()
            await _until(changed, lambda: app._voice.state is VoiceState.IDLE, "cancelled by session change")
            _assert_reaped(transcriber.child_pid)
            assert buffer.text == "say: voice"

            # escape cancels a recording without touching the editor
            pipe.send_text("\x18\x16")
            await _until(changed, lambda: app._voice.state is VoiceState.RECORDING, "fourth recording")
            pipe.send_bytes(b"\x1b")
            await _until(changed, lambda: app._voice.state is VoiceState.IDLE, "escape cancelled")
            assert buffer.text == "say: voice"

            pipe.send_text(" edited")
            await _until(changed, lambda: buffer.text == "say: voice edited", "edit typed")
            assert submitted == []
            pipe.send_text("\r")
            await _until(changed, lambda: submitted == ["say: voice edited"], "enter submits")
        finally:
            app.request_exit()
            await await_or_fail(running, what="tui shutdown")
            await app.close()
    assert app._voice.state is VoiceState.CLOSED
    assert _voice_files(tmp_path / "home") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [EditingMode.EMACS, EditingMode.VI])
async def test_voice_chord_resolves_only_to_voice_toggle(tmp_path: Path, mode: EditingMode) -> None:
    app = _voice_app(tmp_path, FakeTranscriber(tmp_path))
    app.vim_mode = mode is EditingMode.VI
    try:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            session = app._make_session()
            assert session.app.editing_mode is mode
            bindings = session.app.key_bindings
            assert bindings is not None
            chord = (Keys.ControlX, Keys.ControlV)

            def active(keys: tuple[Keys, ...]) -> list[str]:
                with set_app(session.app):
                    return [
                        binding.handler.__name__
                        for binding in bindings.get_bindings_for_keys(keys)
                        if binding.filter()
                    ]

            assert active(chord) == ["toggle_voice"]
            # no binding on the lone prefix can swallow the chord
            assert active((Keys.ControlX,)) == []
    finally:
        await app.close()
