"""Immutable values submitted by the composer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .core.abort import AbortSignal
    from .core.approval import ApprovalPolicy
    from .core.commands.custom_commands import CustomCommand
    from .core.slash import SlashCommandRegistry
    from .loop import AgentLoop
    from .tui.composer import UndoCandidate
    from .types import Message, StreamEvent


@dataclass(frozen=True, slots=True)
class Submission:
    """One immutable submission identity and its captured composer state."""

    id: int
    text: str
    draft_revision: int = 0
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1
    steer: bool = True


class SubmissionHost(Protocol):
    """UI and agent operations requested by the pipeline."""

    loop: AgentLoop
    _slash_commands: SlashCommandRegistry
    _approval_policy: ApprovalPolicy | None
    _input_loop_active: bool
    _exit_requested: bool

    def _handle_tool_event(self, event: StreamEvent) -> bool: ...
    def _present_pending_approvals(self) -> None: ...
    def _prepare_user_message(
        self,
        value: str,
        *,
        pending_attachments: list[Path],
        pending_attachment_tokens: dict[str, Path],
        attachment_value: str | None = None,
    ) -> Message | None: ...
    def _record_prompt(self, value: str, draft_revision: int) -> None: ...
    def _release_attachment_paths(self, paths: tuple[Path, ...]) -> None: ...
    def _restore_pending_submission(self, submission: Submission) -> None: ...
    def _restore_undo_candidate(self, candidate: UndoCandidate) -> None: ...
    def _print_system(self, value: str) -> None: ...
    def _print_user(self, value: Message) -> None: ...
    def _start_turn(
        self,
        value: str,
        *,
        user_message: Message | None = None,
        submission_id: int,
        abort_signal: AbortSignal,
        persist_user_message: bool = True,
        notification: bool = False,
    ) -> asyncio.Task[None]: ...
    def _set_pipeline_task(self, task: asyncio.Task[object]) -> None: ...
    async def _run_macro_submission(
        self,
        command: CustomCommand,
        args: str,
        submission: Submission,
        abort_signal: AbortSignal,
    ) -> str: ...
    def _set_undo_candidate(self, candidate: UndoCandidate) -> None: ...
    def _invalidate_prompt(self) -> None: ...
    def _handle_slash_output(self, output: str) -> None: ...
    def _record_macro_receipt(self, receipt: str) -> None: ...


class ProviderEntry(Protocol):
    submission: Submission
    signal: AbortSignal | None


def dispatch_provider(
    host: SubmissionHost,
    entry: ProviderEntry,
    *,
    user_text: str,
    on_done: Callable[[Submission, asyncio.Task[None]], None],
    abort_signal: AbortSignal,
    user_message: Message | None = None,
    persist_user_message: bool = True,
    notification: bool = False,
) -> asyncio.Task[None]:
    task = host._start_turn(
        user_text,
        user_message=user_message,
        submission_id=entry.submission.id,
        abort_signal=abort_signal,
        persist_user_message=persist_user_message,
        notification=notification,
    )
    task.add_done_callback(lambda completed: on_done(entry.submission, completed))
    return task


__all__ = ["Submission", "SubmissionHost", "dispatch_provider"]
