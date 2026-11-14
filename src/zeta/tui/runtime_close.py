"""Failure-safe cleanup for one TUI runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..runtime.cleanup import close_session
from .bootstrap import surface_shutdown_notifications


class RuntimeCloseMixin:
    """Close every resource once, preserving the first cleanup error."""

    def _detach_ui_notice_sinks(self) -> None:
        self.loop.set_mcp_notice_sink(None)

    def _init_runtime_close(self) -> None:
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_runtime())
        await asyncio.shield(self._close_task)

    async def _cancel_account_switch(self) -> None:
        task = getattr(self, "_account_switch_task", None)
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _close_runtime(self) -> None:
        first_error: BaseException | None = None

        def attempt(action: Callable[[], object]) -> object | None:
            nonlocal first_error
            try:
                return action()
            except BaseException as exc:  # noqa: BLE001 - cleanup must continue
                if first_error is None:
                    first_error = exc
                return None

        async def attempt_async(
            action: Callable[[], Awaitable[object]],
        ) -> None:
            nonlocal first_error
            try:
                await action()
            except BaseException as exc:  # noqa: BLE001 - cleanup must continue
                if first_error is None:
                    first_error = exc

        try:
            attempt(self._cancel_background_invalidation)
            await attempt_async(self.stop_decisions_poll)
            await attempt_async(self._cancel_account_switch)
            prepare_ui_session_change = getattr(
                self, "_prepare_ui_session_change", None
            )
            if prepare_ui_session_change is not None:
                await attempt_async(prepare_ui_session_change)
            await attempt_async(self._close_finder_workers)
            close_history_search = getattr(self, "_close_history_search", None)
            if close_history_search is not None:
                await attempt_async(close_history_search)
            close_history_markers = getattr(self, "_close_history_markers", None)
            if close_history_markers is not None:
                await attempt_async(close_history_markers)
            close_history_view = getattr(self, "_close_history_view", None)
            if close_history_view is not None:
                await attempt_async(close_history_view)
            close_history_pager = getattr(self, "_close_history_pager", None)
            if close_history_pager is not None:
                await attempt_async(close_history_pager)
            else:
                history_pager = getattr(self, "_history_pager", None)
                if history_pager is not None:
                    await attempt_async(history_pager.close)
            prompt_history = getattr(self, "_prompt_history", None)
            if prompt_history is not None:
                await attempt_async(prompt_history.close)
            await attempt_async(self._close_approval_labels)
            pending = attempt(
                lambda: (
                    {entry.id for entry in self.loop.store.agent_notifications()}
                    if self._terminal_restored
                    else set()
                )
            )
            pending_before = pending if isinstance(pending, set) else set()
            close_child_sends = getattr(self, "_close_child_sends", None)
            if close_child_sends is not None:
                await attempt_async(close_child_sends)
            await attempt_async(self._close_remote_view)
            await attempt_async(self._agent_navigation.close)
            attempt(self._agent_navigation.unbind_layout)
            await attempt_async(self._cancel_mcp_wizard)
            await attempt_async(self._submissions.close)
            attempt(self._draft.detach)
            try:
                await self._draft.shutdown_pending()
            except Exception:  # noqa: BLE001, S110 - notice already reported
                pass
            observer_host = getattr(self, "observer_host", None)
            if observer_host is not None:
                await attempt_async(observer_host.close)
            if self._on_session_close is not None:
                await attempt_async(self._on_session_close)
            await attempt_async(
                lambda: close_session(
                    self.loop,
                    self._workspace_snapshot_store,
                    before_store_close=(
                        lambda: (
                            surface_shutdown_notifications(self, pending_before)
                            if self._terminal_restored
                            else None
                        )
                    ),
                )
            )
        finally:
            attempt(lambda: self._finder_field.close())
            attempt(lambda: self._decisions_answer_field.close())
            self._workspace_snapshot_store = None
            attempt(lambda: self.loop.set_background_event_sink(None))
            foreground_event_publisher = getattr(self, "foreground_event_publisher", None)
            if foreground_event_publisher is not None:
                attempt(foreground_event_publisher.close)
            attempt(lambda: self.loop.set_background_wake_callback(None))
            attempt(self._detach_ui_notice_sinks)
            attempt(lambda: self.loop.set_mcp_prompt_refresh(None))
            attempt(
                lambda: self.loop.tool_registry.background_tasks.set_notice_sink(None)
            )
            if self._hooks is not None:
                attempt(lambda: setattr(self._hooks, "notice_sink", None))
            self._update_notice_callback = None
            self._active_session = None
            self._draft_session = None
            self._session = None
            self._closed = True
        restore_closed_history = getattr(self, "_restore_closed_history", None)
        if restore_closed_history is not None:
            attempt(restore_closed_history)
        if first_error is not None:
            raise first_error


__all__ = ["RuntimeCloseMixin"]
