"""TUI wiring for local voice input: toggle, status, and composer insertion."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from ...voice import VoiceRuntime, VoiceState, VoiceStateError
from ..composer import FullScreenPromptSession

_LABELS = {
    VoiceState.STARTING: "starting",
    VoiceState.STOPPING: "stopping",
    VoiceState.PREPARING: "preparing model",
    VoiceState.TRANSCRIBING: "transcribing",
    VoiceState.CANCELLING: "cancelling",
}


class VoiceInputMixin:
    """Voice capture edits only the local composer; it never submits."""

    _active_session: Any

    def _init_voice(self, zeta_home: Path) -> None:
        app = weakref.proxy(self)  # no cycle: a closed TUI must be freed without gc
        self._voice = VoiceRuntime(zeta_home=zeta_home, progress=lambda state: app._voice_progress(state))
        self._voice_tasks: set[asyncio.Task[None]] = set()
        self._voice_error: str | None = None
        self._voice_started_at = 0.0
        self._voice_tick: asyncio.TimerHandle | None = None

    @property
    def voice_active(self) -> bool:
        return self._voice.state not in (VoiceState.IDLE, VoiceState.CLOSED)

    def voice_status(self) -> str | None:
        state = self._voice.state
        if state is VoiceState.RECORDING:
            seconds = int(time.monotonic() - self._voice_started_at)
            return f"voice: recording {seconds // 60}:{seconds % 60:02d} · c-x c-v stop · esc cancel"
        if state in _LABELS:
            return f"voice: {_LABELS[state]} · esc cancel"
        return f"voice: {self._voice_error}" if self._voice_error else None

    def _voice_progress(self, state: VoiceState) -> None:
        if state is VoiceState.RECORDING:
            self._voice_started_at = time.monotonic()
            self._schedule_voice_tick()
        self._invalidate_prompt()  # type: ignore[attr-defined]

    def _schedule_voice_tick(self) -> None:
        if self._voice_tick is not None:
            self._voice_tick.cancel()
        self._voice_tick = asyncio.get_running_loop().call_later(1, self._on_voice_tick)

    def _on_voice_tick(self) -> None:
        self._voice_tick = None
        if self._voice.state is VoiceState.RECORDING:
            self._invalidate_prompt()  # type: ignore[attr-defined]
            self._schedule_voice_tick()

    def _toggle_voice(self) -> None:
        state = self._voice.state
        if state is VoiceState.IDLE:
            self._voice_error = None
            self._spawn_voice(self._voice.start_recording())
        elif state is VoiceState.RECORDING:
            self._spawn_voice(self._transcribe_into_composer())

    def _cancel_voice(self) -> None:
        if self.voice_active or any(not task.done() for task in self._voice_tasks):
            self._spawn_voice(self._voice.cancel())

    def _spawn_voice(self, action: Awaitable[None]) -> None:
        task = asyncio.create_task(self._run_voice(action))
        self._voice_tasks.add(task)
        task.add_done_callback(self._voice_tasks.discard)

    async def _run_voice(self, action: Awaitable[None]) -> None:
        try:
            await action
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except VoiceStateError:
            pass
        except Exception as exc:  # noqa: BLE001 - shown as an actionable status
            self._voice_error = str(exc) or type(exc).__name__
        self._invalidate_prompt()  # type: ignore[attr-defined]

    async def _transcribe_into_composer(self) -> None:
        target = self._viewed_session_path()  # type: ignore[attr-defined]
        session = self._active_session
        text = await self._voice.stop_and_transcribe()
        if (
            not text
            or session is not self._active_session
            or target != self._viewed_session_path()  # type: ignore[attr-defined]
            or not isinstance(session, FullScreenPromptSession)
        ):
            return
        buffer = session.default_buffer
        if buffer.selection_state is not None:
            buffer.cut_selection()
        buffer.insert_text(text)

    async def _close_voice(self) -> None:
        try:
            await self._voice.close()
        finally:
            if self._voice_tick is not None:
                self._voice_tick.cancel()
                self._voice_tick = None
            await asyncio.gather(*self._voice_tasks, return_exceptions=True)
