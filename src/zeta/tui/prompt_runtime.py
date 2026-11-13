"""prompt and terminal runtime lifecycle for the full-screen UI."""
from __future__ import annotations

import asyncio
import weakref

from prompt_toolkit import PromptSession
from rich.text import Text

from ..core.session import env_home
from ..diagnostics.tui_stalls import start_tui_stall_watchdog
from . import theme
from .composer import FullScreenPromptSession
from .fork_session import RuntimeResult
from .layout import detach_completion_menus, full_screen_content
from .loop_errors import (
    cancel_activation,
    run_prompt_toolkit,
    start_prompt_during_activation,
)


class PromptRuntimeMixin:
    """Own prompt input, full-screen layout, and app lifecycle."""

    async def _read_prompt(self, session: PromptSession[str]) -> str | None:
        app = weakref.proxy(self)

        def insert_pending_tokens() -> None:
            if app._composer_insertions:
                session.app.current_buffer.insert_text(
                    "".join(app._composer_insertions)
                )
                app._composer_insertions.clear()

        try:
            value = await session.prompt_async(
                [("class:prompt", " › ")],
                bottom_toolbar=lambda: app._status_toolbar(),
                placeholder=lambda: [("class:placeholder", app._composer_placeholder())],
                pre_run=insert_pending_tokens,
            )
        except EOFError:
            return None
        return value

    async def _run_full_screen(self, session: FullScreenPromptSession, prompt_task: asyncio.Task[object] | None = None) -> None:
        prompt_task = prompt_task or asyncio.create_task(run_prompt_toolkit(session, self))
        self._input_loop_active = True
        try:
            while not self._exit_requested:
                try:
                    await prompt_task
                except (EOFError, asyncio.CancelledError):
                    pass
                break
        finally:
            self._input_loop_active = False
            if not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)

    def _install_full_screen_layout(self, session: FullScreenPromptSession) -> None:
        root = session.layout.container
        composer_rows = list(root.children)
        footer = composer_rows.pop()
        # The command menu leaves the composer's own float container so it
        # can open upward over the transcript with room for a dozen rows.
        for row in composer_rows:
            detach_completion_menus(row)
        self._status_card_window = self._status_card.window()
        self._tasks_panel_window = self._tasks_panel_control.window()
        self._decisions_panel_window = self._decisions_panel_control.window()
        app = weakref.proxy(self)
        root.children[:] = [
            full_screen_content(
                self._transcript.window(),
                composer_rows,
                footer,
                self._todo_widget,
                agent_navigation=self._agent_navigation,
                on_scroll_up=self._transcript.scroll_up,
                on_scroll_down=self._transcript.scroll_down,
                status_window=self._status_card_window,
                status_active=lambda: app.status_card_active,
                tasks_window=self._tasks_panel_window,
                tasks_active=lambda: app.tasks_panel_active,
                decisions_window=self._decisions_panel_window,
                decisions_active=lambda: app.decisions_panel_active,
                finder_window=self._finder_control.window(),
                finder_active=lambda: app._active_finder.finder_active,
                queued_messages=lambda: app.queued_messages,
                decisions_answer_window=app._decisions_answer_field.window,
                overlay_field_windows=(app._finder_field.window,),
            )
        ]
        self._agent_navigation.bind_layout(
            session.layout, session.default_buffer, self._invalidate_prompt
        )
        self._bind_fork_navigation()

    def _bind_restart_idle_buffer(self, session: PromptSession[str]) -> None:
        buffer = getattr(getattr(session, "app", None), "current_buffer", None)
        if buffer is not None and id(buffer) not in self._restart_idle_buffers:
            buffer.on_text_changed += lambda _: self._maybe_restart_when_idle()
            self._restart_idle_buffers.add(id(buffer))

    async def run(
        self,
        session: PromptSession[str] | None = None,
        *,
        resume_ui: bool = False,
    ) -> RuntimeResult:
        """Run the alternate-screen app until it requests a stack transition.

        Re-entrant: the fork controller re-shows a suspended main runtime by
        calling this again without replaying its transcript.
        """
        set_notice_loop = getattr(self._draft, "set_notice_loop", None)
        if set_notice_loop is not None:
            set_notice_loop(asyncio.get_running_loop())
        self._exit_requested = False
        self._run_result = RuntimeResult.EXIT
        self._decisions_switching = False
        watchdog = start_tui_stall_watchdog(
            asyncio.get_running_loop(),
            zeta_home=self._zeta_home or env_home(),
            session_id=self.loop.store.session_id,
        )
        try:
            if not resume_ui:
                self._begin_startup_replay()
            observer_host = getattr(self, "observer_host", None)
            if observer_host is not None and (error := await observer_host.start_or_close()):
                self.observer_host = None  # optional endpoint: never stops the TUI or a fork
                self._handle_background_notice(f"remote observer unavailable: {error}")
            self.loop.set_mcp_notice_sink(self._handle_background_notice)
            session = session or self._session or self._make_session()
            self._active_session = session
            self._bind_restart_idle_buffer(session)
            self._attach_draft(session)
            activation_task, prompt_task, exited = (
                await start_prompt_during_activation(self, session, run_prompt_toolkit)
            )
            try:
                if exited:
                    return self._run_result
                await activation_task
                startup_task = asyncio.create_task(self._prepare_prompt(resume_ui=resume_ui))
                try:
                    if prompt_task is not None:
                        done, _ = await asyncio.wait(
                            (prompt_task, startup_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if prompt_task in done:
                            # A failed first paint must not wait on replay or MCP.
                            await self._run_full_screen(session, prompt_task)
                            return self._run_result
                    if not await startup_task:
                        return self._run_result
                finally:
                    if not startup_task.done():
                        startup_task.cancel()
                    await asyncio.gather(startup_task, return_exceptions=True)
                if isinstance(session, FullScreenPromptSession):
                    await self._run_full_screen(session, prompt_task)
                    return self._run_result
                prompt_task = asyncio.create_task(self._read_prompt(session))
                self._input_loop_active = True
                while prompt_task is not None and not self._exit_requested:
                    value = await prompt_task
                    if value is None:
                        break
                    self._submit_input(value)
                    if self._exit_requested:
                        break
                    prompt_task = asyncio.create_task(self._read_prompt(session))
                return self._run_result
            finally:
                self._input_loop_active = False
                # Any exit before startup drained its queue keeps accepted input as
                # the draft; this runs before cleanup detaches the session.
                self._restore_startup_submissions()
                await cancel_activation(activation_task)
                await self._prepare_ui_session_change()
                if prompt_task is not None:
                    if not prompt_task.done():
                        prompt_task.cancel()
                    await asyncio.gather(prompt_task, return_exceptions=True)
                # Suspending for a discussion fork keeps the runtime's work
                # alive: do not cancel the turn or drop the session reference.
                if self._suspending_for_fork:
                    self._cancel_background_invalidation()
                    try:
                        await self._draft.flush_pending()
                    except Exception:  # noqa: BLE001, S110 - notice already reported
                        pass
                    self._session = None
                    if isinstance(session, FullScreenPromptSession):
                        self._release_ui_ownership()
                        self._detach_ui_notice_sinks()
                        session.restore_terminal()
                        self._terminal_restored = True
                else:
                    if self._active_task is not None and not self._active_task.done():
                        self._active_task.cancel()
                        await asyncio.gather(self._active_task, return_exceptions=True)
                    if isinstance(session, FullScreenPromptSession):
                        self._release_ui_ownership()
                        self._detach_ui_notice_sinks()
                        session.restore_terminal()
                        self._active_session = None
                        self._terminal_restored = True
        finally:
            # Covers failures before the inner cleanup (a prompt crash during
            # activation); a no-op once the inner cleanup restored the queue.
            self._restore_startup_submissions()
            if watchdog is not None:
                await watchdog.close()
            if not self._suspending_for_fork:
                await self.close()

    def _present_startup_output(self) -> None:
        for warning in self._startup_warnings:
            self._print_unit(Text(warning, style=theme.WARNING))
        # After the MCP mount so dropped argument-scoped rules report too (ZETA-86).
        if self._approval_policy is not None:
            for notice in self._approval_policy.notices:
                self._print_unit(Text(notice, style=theme.WARNING))
        for alert in self._startup_alerts:
            self._print_unit(Text(alert, style=theme.COMMAND))
        for notice in self._startup_notices:
            self._print_unit(Text(notice, style=theme.DIM))
        for notice in self._slash_commands.notices:
            style = (
                theme.COMMAND
                if notice in self._slash_commands.warning_notices
                else theme.DIM
            )
            self._print_unit(Text(f"command · {notice}", style=style))
        header = self.fork_header_notice
        if header is not None:
            self._print_unit(Text(header, style=theme.DIM))

