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
_PANEL_REFRESH_INTERVAL = 0.5


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
        self._decisions_staged_count = 0
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
    def decisions_staged_count(self) -> int:
        return self._decisions_staged_count

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
            await self._open_decisions_panel()
            return ""
        from ...attention_decisions import open_decisions

        parts = args.split(maxsplit=2)
        if not parts or parts[0] not in {"reply", "dismiss"}:
            await DecisionsMixin._recover_decisions(self, self._decisions_session_id)
        items = await asyncio.to_thread(
            open_decisions,
            self._home,
            self._decisions_session_id,
            _registry_state(self),
        )
        if parts and parts[0] in {"reply", "dismiss"}:
            if len(parts) < 2:
                return "usage: /decisions reply <id> <text> or /decisions dismiss <id>"
            attention_id = parts[1]
            record = next(
                (item.record for item in items if item.record.id == attention_id), None
            )
            if record is None:
                return "attention item not found"
            replies = self._attention_replies(record.session_id)
            if parts[0] == "dismiss":
                await self._run_attention_operation(replies.dismiss, attention_id)
                return f"dismissed {attention_id}"
            if len(parts) < 3 or not parts[2].strip():
                return "reply text must be nonempty"
            await self._run_attention_operation(
                replies.reply,
                record.id,
                parts[2],
                self.loop.store.session_id,
            )
            return f"replied to {attention_id}"
        if not items:
            return "no open attention items"
        return "open attention items:\n" + "\n".join(
            f"  {item.record.id} [{item.record.kind}] {item.record.title} [{item.project_name}]"
            for item in items
        )

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

        session_id = self._decisions_session_id
        registry_state = _registry_state(self)
        replies = self._attention_replies(session_id)

        def read():
            items = open_decisions(self._home, session_id, registry_state)
            open_ids = {item.record.id for item in items}
            staged = {
                item_id: reply
                for item_id, reply in (replies.staged() if items else ())
                if item_id in open_ids
            }
            return items, staged

        try:
            items, staged = await asyncio.to_thread(read)
        except OSError:
            return
        changed = (len(items), len(staged)) != (
            self._decisions_count,
            self._decisions_staged_count,
        )
        self._decisions_count = len(items)
        self._decisions_staged_count = len(staged)
        if self._decisions_panel_open:
            self._decisions_panel.set_items(items, staged)
            self._decisions_panel_control.set_lines(
                self._decisions_panel.render_lines(),
                keep_offset=True,
                overflow_hint=True,
            )
            self._invalidate_prompt()
        elif changed:
            self._invalidate_prompt()

    # -- open / close -------------------------------------------------------

    def _attention_replies(self, session_id: str):
        from ...attention_replies import AttentionReplies

        on_visible = (
            self.loop.notify_project_inbox_publication
            if session_id == self.loop.store.session_id
            else None
        )
        return AttentionReplies(
            self._home,
            session_id,
            registry_state=_registry_state(self),
            on_inbox_visible=on_visible,
        )

    async def _recover_decisions(self, session_id: str):
        return await self._run_attention_operation(
            self._attention_replies(session_id).recover
        )

    async def _run_attention_operation(self, operation, *args, **kwargs):
        return await asyncio.to_thread(operation, *args, **kwargs)

    def open_decisions_panel(self) -> asyncio.Task[None]:
        return asyncio.create_task(self._open_decisions_panel())

    async def _open_decisions_panel(self) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        await self._recover_decisions(self._decisions_session_id)
        await self._decisions_tick()
        if self._decisions_panel_open:
            return
        buffer = session.default_buffer
        self._decisions_restore_text = buffer.text
        self._decisions_restore_cursor = buffer.cursor_position
        self._decisions_panel_open = True
        self._decisions_panel_control.set_lines(
            self._decisions_panel.render_lines(), overflow_hint=True
        )
        session.layout.focus(self._decisions_panel_window)
        self._start_decisions_refresh()
        self._invalidate_prompt()

    def close_decisions_panel(self) -> None:
        if not self._decisions_panel_open:
            return
        self._decisions_panel_open = False
        asyncio.create_task(self._flush_staged_decisions())
        self._decisions_panel.exit_answer()
        self._stop_decisions_refresh()
        session = self._active_session
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.set_document(
                Document(self._decisions_restore_text, self._decisions_restore_cursor)
            )
            session.layout.focus(session.default_buffer)
        self._invalidate_prompt()

    async def _flush_staged_decisions(self) -> None:
        try:
            for session_id in self._decisions_session_ids():
                await self._run_attention_operation(
                    self._attention_replies(session_id).flush,
                    self.loop.store.session_id,
                )
        except (OSError, ValueError) as exc:
            self._print_system(f"decisions not delivered: {exc}")

    def _decisions_session_ids(self) -> tuple[str, ...]:
        return tuple({item.session_id for item in self._decisions_panel.items})

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
        elif (
            key == "d"
            and panel.selected is not None
            and panel.selected.record.kind != "decision"
        ):
            self._dismiss_selected()
            return
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
            item_ids = {
                panel_item.record.id
                for panel_item in self._decisions_panel.items
                if panel_item.session_id == record.session_id
            }
            try:
                replies = self._attention_replies(record.session_id)
                await self._run_attention_operation(
                    replies.stage,
                    record.id,
                    decision,
                    flush_ids=item_ids,
                    from_session=self.loop.store.session_id,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                self._print_system(f"decision not delivered: {exc}")
            await self._decisions_tick()
            self._refresh_decisions_panel()

        self._decisions_panel.exit_answer()
        asyncio.create_task(run())

    def _dismiss_selected(self) -> None:
        item = self._decisions_panel.selected
        if item is None:
            return

        async def run() -> None:
            try:
                replies = self._attention_replies(item.session_id)
                await self._run_attention_operation(replies.dismiss, item.record.id)
            except (OSError, RuntimeError, ValueError) as exc:
                self._print_system(f"attention not dismissed: {exc}")
            await self._decisions_tick()

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
