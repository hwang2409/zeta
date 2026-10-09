"""Discussion-fork awareness inside the normal TUI.

When the resumed session is an attention discussion fork, the TUI reads its
bound source record once and teaches the existing subagent view two things: the
root entry reads as ``discussion: <title>`` and a synthetic ``(main)`` entry
points at the asking orchestrator. Selecting ``(main)`` returns to that
orchestrator and closes the fork, asking once before leaving an open item. The
switch reuses the same resume path as ``/new`` rather than re-executing.
"""

from __future__ import annotations

import asyncio
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
        self._load_fork_context()

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
            source_name = SessionManager(home).read_metadata(
                fork.forked_from_session
            ).name
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
        source = self._fork_context.source_session_id

        async def run() -> None:
            from ...attention_forks import release_discussion_fork

            try:
                await asyncio.to_thread(
                    release_discussion_fork,
                    self._home,
                    self.loop.store.session_id,
                )
            except (OSError, ValueError):
                pass
            self.request_resume(source)

        asyncio.create_task(run())

    def clear_fork_return_arm(self) -> None:
        self._fork_return_armed = False

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
