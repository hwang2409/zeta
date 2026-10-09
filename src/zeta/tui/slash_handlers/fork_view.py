"""Discussion-fork awareness inside the normal TUI.

When the shown session is an attention discussion fork, the TUI reads its bound
source record once and teaches the existing subagent view two things: the root
entry reads as ``discussion: <title>`` and a synthetic ``(main)`` entry points
at the asking orchestrator. Selecting ``(main)`` returns to that orchestrator
and closes the fork, asking once before leaving an open item.

The switch is in-process: the fork runs on top of a still-live main runtime
(see :mod:`zeta.tui.fork_session`). Returning hands the terminal back to the
suspended main runtime rather than resuming a session or re-executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ForkContext:
    title: str
    attention_id: str
    source_session_id: str
    source_name: str
    source_path: Path


class ForkViewMixin:
    """Surface the discussion fork in the agent view and route the return."""

    def _init_fork_view(self) -> None:
        self._fork_context: ForkContext | None = None
        self._fork_return_armed = False
        self._fork_returning = False
        self._open_fork_target: str | None = None
        self._load_fork_context()

    def request_open_fork(self, fork_session_id: str) -> None:
        """Suspend this TUI and open ``fork_session_id`` on top, in-process.

        Only the UI is suspended: the turn, background children, background
        tasks, notifications, and inbox of this runtime keep running on the one
        event loop while the discussion is shown. The fork controller re-shows
        this runtime on return. The active turn is deliberately not aborted.
        """
        from ...tui.composer import FullScreenPromptSession

        self._open_fork_target = fork_session_id
        self._exit_requested = True
        session = self._active_session
        if isinstance(session, FullScreenPromptSession) and session.app.is_running:
            session.app.exit()

    def take_open_fork(self) -> str | None:
        target, self._open_fork_target = self._open_fork_target, None
        return target

    def request_return_to_main(self) -> None:
        """Leave a discussion fork and hand the terminal back to the main UI.

        The fork controller closes this fork and re-shows the suspended main
        runtime; nothing here aborts or tears down the main runtime.
        """
        from ...tui.composer import FullScreenPromptSession

        self._exit_requested = True
        session = self._active_session
        if isinstance(session, FullScreenPromptSession) and session.app.is_running:
            session.app.exit()

    @property
    def _suspending_for_fork(self) -> bool:
        return self._open_fork_target is not None

    def _load_fork_context(self) -> None:
        from ...attention_forks import read_attention_fork
        from ...attention_records import AttentionStore
        from ...core.session import SessionManager

        home = self._home
        try:
            fork = read_attention_fork(self.loop.store.session_dir)
        except (OSError, ValueError):
            return
        if fork is None:
            return
        source_dir = home / "sessions" / fork.forked_from_session
        try:
            record = AttentionStore(source_dir).get(fork.attention_id)
            source_name = (
                SessionManager(home).read_metadata(fork.forked_from_session).name
            )
        except (OSError, ValueError, KeyError):
            return
        self._fork_context = ForkContext(
            title=record.title,
            attention_id=fork.attention_id,
            source_session_id=fork.forked_from_session,
            source_name=source_name or fork.forked_from_session[:8],
            source_path=source_dir,
        )

    @property
    def is_discussion_fork(self) -> bool:
        return self._fork_context is not None

    @property
    def fork_header_notice(self) -> str | None:
        if self._fork_context is None:
            return None
        return f"discussion · fork of {self._fork_context.source_name}"

    def _bind_fork_navigation(self) -> None:
        if self._fork_context is None:
            return
        self._agent_navigation.set_fork_context(
            title=self._fork_context.title,
            main_path=self._fork_context.source_path,
            on_return=self._return_to_main,
        )

    def _return_to_main(self) -> None:
        if self._fork_context is None or self._fork_returning:
            return
        if self._fork_item_open() and not self._fork_return_armed:
            self._fork_return_armed = True
            self._print_system(
                "leave without a decision? the item stays open and can be "
                "discussed again. press enter again to leave, esc to stay."
            )
            return
        self._fork_returning = True
        self.request_return_to_main()

    def clear_fork_return_arm(self) -> None:
        self._fork_return_armed = False

    async def leave_fork(self) -> None:
        """Release this fork's binding before it closes; a no-op for main."""
        if self._fork_context is None:
            return
        import asyncio

        from ...attention_forks import release_discussion_fork

        try:
            await asyncio.to_thread(
                release_discussion_fork, self._home, self.loop.store.session_id
            )
        except (OSError, ValueError):
            pass

    @property
    def main_activity_pending(self) -> bool:
        """True while a discussion is shown and the main runtime has new activity."""
        controller = self._fork_controller
        return controller is not None and controller.main_activity_pending(self)

    def notification_count(self) -> int:
        try:
            return len(self.loop.store.agent_notifications())
        except (OSError, ValueError):
            return 0

    def _fork_item_open(self) -> bool:
        from ...attention_records import AttentionStore

        if self._fork_context is None:
            return False
        try:
            record = AttentionStore(self._fork_context.source_path).get(
                self._fork_context.attention_id
            )
        except (OSError, ValueError):
            return False
        return record.status == "open"


__all__ = ["ForkContext", "ForkViewMixin"]
