"""Prompt-toolkit lifecycle for the in-TUI decisions popup.

``TUIApp`` mixes this in. The mixin owns everything that touches live state: it
scans open attention decisions across live sessions (off the event loop), feeds
the pure :class:`DecisionsPanel` view model, delivers a quick answer to the
asking orchestrator's inbox, and starts a discussion fork and switches the TUI
to it. The scan also drives the status-bar decisions count, which a bounded
poller keeps fresh while the app is open.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit.document import Document

from ..cards.decisions import DecisionsPanel
from ..composer import FullScreenPromptSession
from ..overlay import OverlayControl

# The scan touches the filesystem (a directory flock per session plus its
# attention records); a few-second cadence keeps the status-bar count fresh
# without busy-polling.
_POLL_INTERVAL = 2.0
_PANEL_REFRESH_INTERVAL = 0.5


class DecisionsMixin:
    """Serve the interactive decisions popup and its status-bar count."""

    def _init_decisions_panel(self) -> None:
        self._decisions_panel = DecisionsPanel()
        self._decisions_panel_control = OverlayControl()
        self._decisions_panel_open = False
        self._decisions_count = 0
        self._decisions_poll_task: asyncio.Task[None] | None = None
        self._decisions_refresh_task: asyncio.Task[None] | None = None
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
    def decisions_panel_active(self) -> bool:
        return self._decisions_panel_open and self._full_screen_active()

    def slash_decisions(self, args: str) -> str:
        del args
        from ...attention_decisions import open_decisions

        items = open_decisions(self._home)
        if not items:
            return "no open decisions"
        lines = ["open decisions:"]
        for item in items:
            lines.append(
                f"  {item.record.title}  "
                f"[{item.project_name} · {item.session_id[:8]}]"
            )
        return "\n".join(lines)

    # -- status-bar count poll ----------------------------------------------

    def start_decisions_poll(self) -> None:
        if self._decisions_poll_task is not None:
            return
        self._decisions_poll_task = asyncio.create_task(self._decisions_poll_loop())

    def stop_decisions_poll(self) -> None:
        task = self._decisions_poll_task
        self._decisions_poll_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _decisions_poll_loop(self) -> None:
        while True:
            await self._decisions_tick()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _decisions_tick(self) -> None:
        from ...attention_decisions import open_decisions

        try:
            items = await asyncio.to_thread(open_decisions, self._home)
        except OSError:
            return
        changed = len(items) != self._decisions_count
        self._decisions_count = len(items)
        if self._decisions_panel_open:
            self._decisions_panel.set_items(items)
            self._decisions_panel_control.set_lines(
                self._decisions_panel.render_lines(), keep_offset=True
            )
            self._invalidate_prompt()
        elif changed:
            self._invalidate_prompt()

    # -- open / close -------------------------------------------------------

    def open_decisions_panel(self, *, restore_composer: bool = True) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        buffer = session.default_buffer
        if restore_composer:
            self._decisions_restore_text = buffer.text
            self._decisions_restore_cursor = buffer.cursor_position
        else:
            self._decisions_restore_text = ""
            self._decisions_restore_cursor = 0
        self._decisions_panel_open = True
        self._decisions_panel_control.set_lines(self._decisions_panel.render_lines())
        session.layout.focus(self._decisions_panel_window)
        self._start_decisions_refresh()
        self._invalidate_prompt()

    def close_decisions_panel(self) -> None:
        if not self._decisions_panel_open:
            return
        self._decisions_panel_open = False
        self._decisions_panel.exit_answer()
        self._stop_decisions_refresh()
        session = self._active_session
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.set_document(
                Document(self._decisions_restore_text, self._decisions_restore_cursor)
            )
            session.layout.focus(session.default_buffer)
        self._invalidate_prompt()

    def _start_decisions_refresh(self) -> None:
        self._stop_decisions_refresh()
        self._decisions_refresh_task = asyncio.create_task(
            self._decisions_refresh_loop()
        )

    def _stop_decisions_refresh(self) -> None:
        task = self._decisions_refresh_task
        self._decisions_refresh_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _decisions_refresh_loop(self) -> None:
        while self._decisions_panel_open:
            await self._decisions_tick()
            await asyncio.sleep(_PANEL_REFRESH_INTERVAL)

    # -- keyboard -----------------------------------------------------------

    def _decisions_key(self, key: str) -> None:
        panel = self._decisions_panel
        if panel.mode == "answer":
            self._decisions_answer_key(key)
            return
        if key == "escape":
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
        self._refresh_decisions_panel()

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
        self._refresh_decisions_panel()

    def _decisions_answer_input(self, text: str) -> None:
        if self._decisions_panel.mode != "answer":
            return
        self._decisions_answer_buffer += text
        self._decisions_panel.set_answer(self._decisions_answer_buffer)
        self._refresh_decisions_panel()

    def _refresh_decisions_panel(self, *, keep_offset: bool = False) -> None:
        if not self._decisions_panel_open:
            return
        self._decisions_panel_control.set_lines(
            self._decisions_panel.render_lines(), keep_offset=keep_offset
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
                )
            except (OSError, ValueError) as exc:
                self._print_system(f"decision not delivered: {exc}")
            await self._decisions_tick()
            self._refresh_decisions_panel()

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
                    create_discussion_fork, self._home, source_id, attention_id
                )
            except (OSError, ValueError) as exc:
                self._decisions_switching = False
                self._print_system(f"could not open discussion: {exc}")
                self._refresh_decisions_panel()
                return
            self.request_open_fork(fork_id)

        asyncio.create_task(run())


__all__ = ["DecisionsMixin"]
