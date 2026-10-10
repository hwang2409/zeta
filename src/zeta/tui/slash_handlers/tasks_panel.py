"""Prompt-toolkit lifecycle for the ``/tasks`` background-process panel.

``TUIApp`` mixes this in. The mixin owns everything that touches live state:
it feeds the pure :class:`BackgroundTasksPanel` snapshots from the registry,
tails task output incrementally (a byte cursor advanced per refresh, never a
full re-read), and runs a bounded refresh loop that exists only while the panel
is open.
"""

from __future__ import annotations

import asyncio
import time

from prompt_toolkit.document import Document

from ..cards.tasks_panel import (
    OUTPUT_TAIL_BYTES,
    BackgroundTasksPanel,
    format_runtime,
    short_id,
)
from ..composer import FullScreenPromptSession
from ..overlay import OverlayControl

# The panel refreshes a few times a second while open; closed, nothing runs.
_REFRESH_INTERVAL = 0.25


class BackgroundTasksMixin:
    """Serve the interactive ``/tasks`` panel over the shared registry."""

    def _init_background_tasks_panel(self) -> None:
        self._tasks_panel = BackgroundTasksPanel()
        self._tasks_panel_control = OverlayControl()
        self._tasks_panel_open = False
        self._tasks_refresh_task: asyncio.Task[None] | None = None
        self._tasks_output_cursor = 0
        self._tasks_output_trim = False
        self._tasks_restore_text = ""
        self._tasks_restore_cursor = 0

    def _open_overlay_from_submit(self, value: str) -> bool:
        """Open the overlay a bare ``/status``, ``/mcp``, or ``/tasks`` names.

        A submitted overlay command is consumed, not a draft to restore when the
        transient view closes.
        """

        session = self._active_session
        command = value.strip()
        if not isinstance(session, FullScreenPromptSession):
            return False
        if command not in {"/status", "/mcp", "/tasks", "/decisions"}:
            return False
        session.default_buffer.reset()
        self._draft.clear()
        if command == "/mcp":
            self.open_mcp_manager(restore_composer=False)
        elif command == "/tasks":
            self.open_tasks_panel(restore_composer=False)
        elif command == "/decisions":
            self.open_decisions_panel(restore_composer=False)
        else:
            self.open_status_card(restore_composer=False)
        return True

    @property
    def tasks_panel_active(self) -> bool:
        return self._tasks_panel_open and self._full_screen_active()

    def open_tasks_panel(self, *, restore_composer: bool = True) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        buffer = session.default_buffer
        if restore_composer:
            self._tasks_restore_text = buffer.text
            self._tasks_restore_cursor = buffer.cursor_position
        else:
            self._tasks_restore_text = ""
            self._tasks_restore_cursor = 0
        self._sync_tasks_snapshot()
        self._tasks_panel_control.set_lines(self._tasks_panel.render_lines())
        self._tasks_panel_open = True
        session.layout.focus(self._tasks_panel_window)
        self._start_tasks_refresh()
        self._invalidate_prompt()

    def close_tasks_panel(self) -> None:
        if not self._tasks_panel_open:
            return
        self._tasks_panel_open = False
        self._stop_tasks_refresh()
        session = self._active_session
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.set_document(
                Document(self._tasks_restore_text, self._tasks_restore_cursor)
            )
            session.layout.focus(session.default_buffer)
        self._invalidate_prompt()

    # -- keyboard -----------------------------------------------------------

    def _tasks_key(self, key: str) -> None:
        panel = self._tasks_panel
        control = self._tasks_panel_control
        in_list = panel.mode == "list"
        if key == "escape":
            if panel.back():
                control.top()
                self._refresh_tasks_panel()
            else:
                self.close_tasks_panel()
            return
        if key in {"up", "j"}:
            panel.move(-1) if in_list else control.scroll(-1)
        elif key in {"down"}:
            panel.move(1) if in_list else control.scroll(1)
        elif key == "enter":
            if in_list and panel.open_details() is not None:
                self._begin_output_tail()
                control.top()
        elif key == "k":
            if in_list:
                target = panel.request_kill()
                if target is not None:
                    self._kill_task(target)
        elif key == "n":
            panel.cancel_kill()
        elif key == "pageup":
            control.page(-1)
        elif key == "pagedown":
            control.page(1)
        elif key == "home":
            control.top()
        elif key == "end":
            control.bottom()
        self._refresh_tasks_panel(keep_offset=not in_list)

    # -- live refresh -------------------------------------------------------

    def _start_tasks_refresh(self) -> None:
        self._stop_tasks_refresh()
        self._tasks_refresh_task = asyncio.create_task(self._tasks_refresh_loop())

    def _stop_tasks_refresh(self) -> None:
        task = self._tasks_refresh_task
        self._tasks_refresh_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _tasks_refresh_loop(self) -> None:
        while self._tasks_panel_open:
            await self._tasks_tick()
            self._invalidate_prompt()
            await asyncio.sleep(_REFRESH_INTERVAL)

    async def _tasks_tick(self) -> None:
        self._sync_tasks_snapshot()
        detail = self._tasks_panel.detail_task()
        if detail is not None:
            await self._pull_output_tail(detail.task_id, detail.output_lines)
        self._tasks_panel_control.set_lines(
            self._tasks_panel.render_lines(), keep_offset=True
        )

    def _sync_tasks_snapshot(self) -> None:
        registry = self.loop.tool_registry.background_tasks
        self._tasks_panel.set_tasks(registry.snapshot(), time.monotonic())

    def _begin_output_tail(self) -> None:
        detail = self._tasks_panel.detail_task()
        if detail is None:
            return
        start = max(0, detail.output_bytes - OUTPUT_TAIL_BYTES)
        self._tasks_output_cursor = start
        # A mid-stream start lands inside a line; drop that partial first line
        # once so the box opens on clean line boundaries.
        self._tasks_output_trim = start > 0
        asyncio.create_task(self._tasks_initial_output(detail.task_id, detail.output_lines))

    async def _tasks_initial_output(self, task_id: str, total_lines: int) -> None:
        await self._pull_output_tail(task_id, total_lines)
        self._tasks_panel_control.set_lines(
            self._tasks_panel.render_lines(), keep_offset=True
        )
        self._invalidate_prompt()

    async def _pull_output_tail(self, task_id: str, total_lines: int) -> None:
        registry = self.loop.tool_registry.background_tasks
        try:
            result = await registry.output(task_id, since=self._tasks_output_cursor)
        except ValueError:
            return
        text = result["output"]
        self._tasks_output_cursor = result["cursor"]
        if self._tasks_output_trim and "\n" in text:
            text = text.split("\n", 1)[1]
            self._tasks_output_trim = False
        elif self._tasks_output_trim and not text:
            # Nothing yet; keep trimming the first line on the next read.
            pass
        self._tasks_panel.feed_output(text, total_lines)

    def _kill_task(self, task_id: str) -> None:
        registry = self.loop.tool_registry.background_tasks

        async def run() -> None:
            try:
                await registry.kill(task_id)
            except ValueError:
                pass
            self._refresh_tasks_panel()

        asyncio.create_task(run())

    def _refresh_tasks_panel(self, *, keep_offset: bool = False) -> None:
        if not self._tasks_panel_open:
            return
        self._sync_tasks_snapshot()
        self._tasks_panel_control.set_lines(
            self._tasks_panel.render_lines(), keep_offset=keep_offset
        )
        self._invalidate_prompt()

    # -- text fallback (non-full-screen) ------------------------------------

    def slash_tasks(self, args: str) -> str:
        del args
        registry = self.loop.tool_registry.background_tasks
        tasks = registry.snapshot()
        if not tasks:
            return "no background tasks in this session"
        now = time.monotonic()
        lines = ["background tasks:"]
        for task in tasks:
            if task.running:
                status = "running"
            elif task.terminal_phase in {"task_kill", "session_shutdown"}:
                status = "killed"
            elif task.exit_code is None:
                status = "exited"
            else:
                status = f"exited {task.exit_code}"
            if task.started_at is None:
                runtime = "—"
            else:
                end = task.ended_at if task.ended_at is not None else now
                runtime = format_runtime(max(0.0, end - task.started_at))
            command = " ".join(task.command.split())
            lines.append(
                f"  {short_id(task.task_id)}  {status:<10} {runtime:>6}  {command}"
            )
        return "\n".join(lines)


__all__ = ["BackgroundTasksMixin"]
