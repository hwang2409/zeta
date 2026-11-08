"""Prompt-toolkit lifecycle for the in-TUI decisions popup.

``TUIApp`` mixes this in. The mixin owns everything that touches live state: it
reads open attention decisions for the shown session (off the event loop), feeds
the pure :class:`DecisionsPanel` view model, delivers a quick answer to the
asking orchestrator's inbox, and starts a discussion fork and switches the TUI
to it. The read also drives the status-bar decisions count, which a bounded
poller keeps fresh while the app is open.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit.document import Document

from ...protocol.types import StreamEvent, StreamEventType
from ...runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)
from ..cards.decisions import DecisionsPanel
from ..composer import FullScreenPromptSession
from ..overlay import OverlayControl

# The scan touches the filesystem (a directory flock per session plus its
# attention records); a few-second cadence keeps the status-bar count fresh
# without busy-polling.
_POLL_INTERVAL = 2.0
_SCAN_TIMEOUT = 10.0
_SCAN_RETRY_LIMIT = 3
_SCAN_RETRY_BASE = 0.5


def _registry_state(app: object) -> ProjectRegistryState:
    loop = app.loop
    tool_registry = getattr(loop, "tool_registry", None)
    state = getattr(tool_registry, "project_registry_state", None)
    if state is not None:
        return state
    registry = getattr(loop, "project_registry", None)
    return (
        ProjectRegistryState(registry)
        if registry is not None
        else available_project_registry(app._home / "projects")
    )


class DecisionsMixin:
    """Serve the interactive decisions popup and its status-bar count."""

    def _init_decisions_panel(self) -> None:
        self._decisions_panel = DecisionsPanel()
        self._decisions_panel_control = OverlayControl()
        self._decisions_panel_open = False
        self._decisions_count = 0
        self._decisions_error: str | None = None
        self._decisions_scan_task: asyncio.Task[None] | None = None
        self._decisions_scan_work: asyncio.Task[tuple] | None = None
        self._decisions_scan_requested = asyncio.Event()
        self._decisions_stopping = False
        self._decisions_restore_text = ""
        self._decisions_restore_cursor = 0
        self._decisions_answer_buffer = ""
        self._decisions_switching = False

    @property
    def _home(self) -> Path:
        # home/sessions/<id> -> home. Robust for ephemeral and default homes.
        return self.loop.store.session_dir.parent.parent

    @property
    def decisions_count(self) -> int:
        return self._decisions_count

    @property
    def decisions_error(self) -> str | None:
        return self._decisions_error

    @property
    def _decisions_session_id(self) -> str:
        fork_context = getattr(self, "_fork_context", None)
        if fork_context is not None:
            return fork_context.source_session_id
        return self.loop.store.session_id

    @property
    def decisions_panel_active(self) -> bool:
        return self._decisions_panel_open and self._full_screen_active()

    async def slash_decisions(self, args: str) -> str:
        full_screen_active = getattr(self, "_full_screen_active", lambda: False)
        if not args.strip() and full_screen_active():
            self._consume_overlay_submission()
            self.open_decisions_panel()
            return ""
        items = await DecisionsMixin._read_decisions(self, _SCAN_TIMEOUT)
        if not items:
            return "no open decisions"
        lines = ["open decisions:"]
        for item in items:
            lines.append(
                f"  {item.record.title}  "
                f"[{item.project_name}]"
            )
        return "\n".join(lines)

    # -- status-bar count scan ----------------------------------------------

    def start_decisions_poll(self) -> None:
        self._decisions_stopping = False
        self.request_decisions_refresh()

    async def stop_decisions_poll(self) -> None:
        self._decisions_stopping = True
        task = self._decisions_scan_task
        self._decisions_scan_task = None
        self._decisions_scan_requested.clear()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _refresh_decisions_for_event(self, event: StreamEvent) -> None:
        if (
            event.type is StreamEventType.TOOL_EXECUTION_END
            and event.tool_call is not None
            and event.tool_call.name == "resolve_attention"
            and event.tool_result is not None
            and not event.tool_result.is_error
        ):
            self.request_decisions_refresh()

    def request_decisions_refresh(self) -> None:
        if self._decisions_stopping:
            return
        self._decisions_scan_requested.set()
        if self._decisions_scan_task is None or self._decisions_scan_task.done():
            self._decisions_scan_task = asyncio.create_task(
                self._decisions_scan_loop(), name="zeta-decisions-scan"
            )

    async def _decisions_scan_loop(self) -> None:
        while True:
            self._decisions_scan_requested.clear()
            scanned = await self._scan_decisions_with_retries()
            if not scanned:
                await self._decisions_scan_requested.wait()
                continue
            # The completed scan includes every request received while it ran.
            self._decisions_scan_requested.clear()
            try:
                await asyncio.wait_for(
                    self._decisions_scan_requested.wait(), timeout=_POLL_INTERVAL
                )
            except TimeoutError:
                pass

    async def _scan_decisions_with_retries(self) -> bool:
        for attempt in range(_SCAN_RETRY_LIMIT + 1):
            try:
                await self._decisions_tick()
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - projection failures stay visible
                if attempt == _SCAN_RETRY_LIMIT:
                    self._set_decisions_error(exc)
                    return False
                await asyncio.sleep(_SCAN_RETRY_BASE * 2**attempt)
        return False

    def _set_decisions_error(self, error: BaseException) -> None:
        reason = str(error).strip() or type(error).__name__
        self._decisions_error = reason[:80]
        self._invalidate_prompt()

    async def _read_decisions(self, timeout: float | None) -> tuple:
        from ...attention_decisions import open_decisions

        work = getattr(self, "_decisions_scan_work", None)
        if work is None:
            work = asyncio.create_task(
                asyncio.to_thread(
                    open_decisions,
                    self._home,
                    self._decisions_session_id,
                    _registry_state(self),
                )
            )
            self._decisions_scan_work = work
        try:
            return await asyncio.wait_for(asyncio.shield(work), timeout=timeout)
        finally:
            if work.done() and self._decisions_scan_work is work:
                self._decisions_scan_work = None

    async def _decisions_tick(self) -> None:
        try:
            items = await self._read_decisions(_SCAN_TIMEOUT)
        except TimeoutError as exc:
            self._set_decisions_error(exc)
            items = await self._read_decisions(timeout=None)
        changed = len(items) != self._decisions_count or self._decisions_error is not None
        self._decisions_count = len(items)
        self._decisions_error = None
        if self._decisions_panel_open:
            self._decisions_panel.set_items(items)
            self._decisions_panel_control.set_lines(
                self._decisions_panel.render_lines(), keep_offset=True, overflow_hint=True
            )
            self._invalidate_prompt()
        elif changed:
            self._invalidate_prompt()

    # -- open / close -------------------------------------------------------

    def open_decisions_panel(self) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        buffer = session.default_buffer
        self._decisions_restore_text = buffer.text
        self._decisions_restore_cursor = buffer.cursor_position
        self._decisions_panel_open = True
        self._decisions_panel_control.set_lines(
            self._decisions_panel.render_lines(), overflow_hint=True
        )
        session.layout.focus(self._decisions_panel_window)
        self.request_decisions_refresh()
        self._invalidate_prompt()

    def close_decisions_panel(self) -> None:
        if not self._decisions_panel_open:
            return
        self._decisions_panel_open = False
        self._decisions_panel.exit_answer()
        session = self._active_session
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.set_document(
                Document(self._decisions_restore_text, self._decisions_restore_cursor)
            )
            session.layout.focus(session.default_buffer)
        self._invalidate_prompt()

    # -- keyboard -----------------------------------------------------------

    def _decisions_key(self, key: str) -> None:
        panel = self._decisions_panel
        if key == "pageup":
            self._decisions_panel_control.page(-1)
        elif key == "pagedown":
            self._decisions_panel_control.page(1)
        elif panel.mode == "answer":
            self._decisions_answer_key(key)
            return
        elif key == "escape":
            self.close_decisions_panel()
            return
        if key in {"up", "k"}:
            panel.move(-1)
        elif key in {"down", "j"}:
            panel.move(1)
        elif key == "a":
            panel.enter_answer()
            self._decisions_answer_buffer = ""
        elif key == "enter":
            self._discuss_selected()
            return
        elif key.isdigit():
            option = panel.pick_option(int(key))
            if option is not None:
                self._deliver_quick_answer(option)
                return
        self._refresh_decisions_panel(keep_offset=key in {"pageup", "pagedown"})

    def _decisions_answer_key(self, key: str) -> None:
        if key == "escape":
            self._decisions_panel.exit_answer()
        elif key == "enter":
            text = self._decisions_answer_buffer.strip()
            if text:
                self._deliver_quick_answer(text)
                return
            self._decisions_panel.exit_answer()
        elif key == "backspace":
            self._decisions_answer_buffer = self._decisions_answer_buffer[:-1]
            self._decisions_panel.set_answer(self._decisions_answer_buffer)
        self._refresh_decisions_panel(keep_offset=True)

    def _decisions_answer_input(self, text: str) -> None:
        if self._decisions_panel.mode != "answer":
            return
        self._decisions_answer_buffer += text
        self._decisions_panel.set_answer(self._decisions_answer_buffer)
        self._refresh_decisions_panel(keep_offset=True)

    def _refresh_decisions_panel(self, *, keep_offset: bool = False) -> None:
        if not self._decisions_panel_open:
            return
        self._decisions_panel_control.set_lines(
            self._decisions_panel.render_lines(),
            keep_offset=keep_offset,
            overflow_hint=True,
        )
        self._invalidate_prompt()

    # -- actions ------------------------------------------------------------

    def _deliver_quick_answer(self, decision: str) -> None:
        item = self._decisions_panel.selected
        if item is None:
            return
        record = item.record

        async def run() -> None:
            from ...attention_forks import deliver_attention_decision

            try:
                await asyncio.to_thread(
                    deliver_attention_decision,
                    self._home,
                    record,
                    decision,
                    from_session=self.loop.store.session_id,
                    registry_state=_registry_state(self),
                )
            except (OSError, ValueError) as exc:
                self._print_system(f"decision not delivered: {exc}")
            self.request_decisions_refresh()

        self._decisions_panel.exit_answer()
        asyncio.create_task(run())

    def _discuss_selected(self) -> None:
        item = self._decisions_panel.selected
        if item is None or self._decisions_switching:
            return
        self._decisions_switching = True
        source_id, attention_id = item.session_id, item.record.id
        # Close the popup now so the main runtime re-shows a clean composer
        # when the discussion returns.
        self.close_decisions_panel()

        async def run() -> None:
            from ...attention_forks import create_discussion_fork

            try:
                fork_id = await asyncio.to_thread(
                    create_discussion_fork,
                    self._home,
                    source_id,
                    attention_id,
                    _registry_state(self),
                )
            except (OSError, ValueError) as exc:
                self._decisions_switching = False
                self._print_system(f"could not open discussion: {exc}")
                self._refresh_decisions_panel()
                return
            self.request_open_fork(fork_id)

        asyncio.create_task(run())


__all__ = ["DecisionsMixin"]
