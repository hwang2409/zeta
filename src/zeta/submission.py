"""Identity-preserving composer submission queue."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from pathlib import Path

from .core.slash import SlashModelInput
from .mcp.prompt_commands import SlashPromptError
from .core.approval import ApprovalRequest
from .tui.composer import UndoCandidate, parse_input
from .types import TextContent


@dataclass(frozen=True, slots=True)
class Submission:
    """One complete, immutable composer state captured at send time."""

    id: int
    text: str
    draft_revision: int = 0
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1


class SubmissionQueue:
    """Queue submissions and cancel them by object identity."""

    def __init__(self) -> None:
        self._next_id = count(1)
        self._pending: deque[Submission] = deque()
        self._ready = asyncio.Event()
        self._active: dict[int, Submission] = {}
        self._cancelled: set[int] = set()

    def submit(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
    ) -> Submission:
        submission = Submission(
            next(self._next_id),
            text,
            draft_revision,
            attachment_paths,
            tuple((attachment_tokens or {}).items()),
            next_image_token,
        )
        self._pending.append(submission)
        self._ready.set()
        return submission

    async def get(self) -> Submission:
        """Wait for and claim the next submission."""

        await self._ready.wait()
        submission = self._pending.popleft()
        if not self._pending:
            self._ready.clear()
        self._active[submission.id] = submission
        return submission

    def take_pending(self) -> Submission | None:
        """Claim a pending submission for direct, non-loop callers."""

        if not self._pending:
            return None
        submission = self._pending.popleft()
        if not self._pending:
            self._ready.clear()
        self._active[submission.id] = submission
        return submission

    def begin_direct(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
    ) -> Submission:
        """Create a submission for callers that bypass the input queue."""

        submission = Submission(
            next(self._next_id),
            text,
            draft_revision,
            attachment_paths,
            tuple((attachment_tokens or {}).items()),
            next_image_token,
        )
        self._active[submission.id] = submission
        return submission

    def cancel_current(self) -> Submission | None:
        """Cancel the current submission, or the next queued submission."""

        submission = next(reversed(self._active.values()), None)
        if submission is None and self._pending:
            submission = self._pending[-1]
        if submission is not None:
            self._cancelled.add(submission.id)
        return submission

    def latest_active(self) -> Submission | None:
        return next(reversed(self._active.values()), None)

    def is_cancelled(self, submission: Submission) -> bool:
        return submission.id in self._cancelled

    def complete(self, submission: Submission) -> None:
        self._active.pop(submission.id, None)
        self._cancelled.discard(submission.id)


class SubmissionMixin:
    """Handle composer submissions without sharing mutable text state."""

    def _preprocessing_wait_set(self) -> set[asyncio.Task[None]]:
        return set(self._preprocessing_tasks.values())

    async def _drain_preprocessing(
        self, done: set[asyncio.Task[None]]
    ) -> None:
        for submission_id, task in tuple(self._preprocessing_tasks.items()):
            if task not in done:
                continue
            self._preprocessing_tasks.pop(submission_id, None)
            if self._preprocessing_task is task:
                self._preprocessing_task = None
            await task

    def _start_preprocessing(self, submission: Submission, parsed: str) -> None:
        preprocessing_task = asyncio.create_task(
            self._finish_prompt_value(submission, parsed)
        )
        self._preprocessing_tasks[submission.id] = preprocessing_task
        self._preprocessing_task = preprocessing_task
        self._inline_abort_signals[submission.id] = (
            self.loop.tool_registry.abort_signal.registry.new_generation()
        )

    def _pending_approvals_for_submission(
        self, submission_id: int
    ) -> tuple[ApprovalRequest, ...]:
        return tuple(
            request
            for request in self.pending_approvals
            if self._approval_owners.get(request.key) == submission_id
        )

    def _dispatch_approval_queue(self) -> None:
        if self.pending_approvals or not self._approval_queue:
            return
        if self._active_task is not None and not self._active_task.done():
            self._queued.extend(self._approval_queue)
            self._approval_queue.clear()
            return
        user_message, candidate = self._approval_queue.popleft()
        self._print_user(user_message)
        self._undo_candidate = candidate
        user_text = next(
            block.text
            for block in user_message.content
            if isinstance(block, TextContent) and block.path is None
        )
        self._start_turn(
            user_text,
            user_message=user_message,
            submission_id=candidate.submission_id,
        )

    async def slash_mcp(self, args: str) -> str:
        return await self.loop.slash_mcp(args)

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        return await self.loop.slash_mcp_prompt(name, arguments)

    def _submit_input(self, value: str) -> None:
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        self._submissions.submit(
            value,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
        )

    def _begin_direct_submission(self, value: str) -> Submission:
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        return self._submissions.begin_direct(
            value,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
        )

    async def _handle_prompt_value(self, value: Submission | str) -> None:
        if isinstance(value, Submission):
            submission = value
        else:
            submission = self._submissions.take_pending()
            if submission is None:
                submission = self._begin_direct_submission(value)
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            self._dispatch_approval_queue()
            return
        parsed = parse_input(submission.text)
        if parsed is None or self._exit_requested:
            self._release_attachment_paths(submission.attachment_paths)
            self._submissions.complete(submission)
            return
        if await self._handle_approval_input(parsed):
            if self._submissions.is_cancelled(submission):
                self._submissions.complete(submission)
                return
            self._release_attachment_paths(submission.attachment_paths)
            self._submissions.complete(submission)
            self._record_prompt(parsed, submission.draft_revision)
            return
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        await self.loop.ensure_mcp_servers()
        slash_output = await self._slash_commands.dispatch_async(self, parsed)
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        if isinstance(slash_output, SlashPromptError):
            self._restore_failed_submission(submission, slash_output.message)
            return
        if isinstance(slash_output, SlashModelInput):
            await self._finish_prompt_value(
                submission,
                parsed,
                model_input=slash_output.text,
                attachment_value=submission.text,
            )
            return
        elif slash_output is not None:
            self._release_attachment_paths(submission.attachment_paths)
            self._submissions.complete(submission)
            self._record_prompt(parsed, submission.draft_revision)
            if self._fork_rebuilt:
                self._fork_rebuilt = False
            elif slash_output.startswith("[Image #"):
                self._insert_paste_token(slash_output)
            elif slash_output:
                self._print_system(slash_output)
            return
        if (
            self._input_loop_active
            and self._slash_commands.needs_inline_shell_resolution(parsed)
        ):
            self._start_preprocessing(submission, parsed)
            return
        await self._finish_prompt_value(submission, parsed)

    async def _finish_prompt_value(
        self,
        submission: Submission,
        parsed: str,
        *,
        model_input: str | None = None,
        attachment_value: str | None = None,
    ) -> None:
        if model_input is None:
            model_input = await self._slash_commands.resolve_for_model(
                parsed,
                lambda commands: self._resolve_inline_shell(submission.id, commands),
            )
        if model_input is None:
            if submission.id in self._undo_pending:
                self._undo_pending.remove(submission.id)
                self._restore_pending_submission(submission)
            else:
                self._release_attachment_paths(submission.attachment_paths)
            self._submissions.complete(submission)
            self._dispatch_approval_queue()
            return
        pending_attachments = list(submission.attachment_paths)
        pending_attachment_tokens = dict(submission.attachment_tokens)
        user_message = self._prepare_user_message(
            model_input,
            pending_attachments=pending_attachments,
            pending_attachment_tokens=pending_attachment_tokens,
            attachment_value=attachment_value,
        )
        if user_message is None:
            session = self._active_session or self._session
            buffer = session.app.current_buffer if session is not None else None
            if buffer is None or not buffer.text:
                self._restore_composer(submission.text)
                self._restore_pending_attachment_state(
                    submission.attachment_paths,
                    submission.attachment_tokens,
                    submission.next_image_token,
                )
                self._draft.schedule(
                    submission.text,
                    attachment_tokens=dict(submission.attachment_tokens),
                    next_image_token=submission.next_image_token,
                )
            else:
                self._release_attachment_paths(tuple(pending_attachments))
            self._submissions.complete(submission)
            return
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        self._record_prompt(parsed, submission.draft_revision)
        self._failed_turn = None
        if self.pending_approvals:
            if not self._pending_approvals_for_submission(submission.id):
                candidate = UndoCandidate.from_message(
                    submission.text,
                    user_message,
                    pending_attachment_tokens,
                    submission.next_image_token,
                    submission_id=submission.id,
                )
                self._approval_queue.append((user_message, candidate))
                self._submissions.complete(submission)
                self._dispatch_approval_queue()
                return
            self._release_attachment_paths(tuple(pending_attachments))
            self._present_pending_approvals()
            self._submissions.complete(submission)
        elif self._active_task is not None and not self._active_task.done():
            candidate = UndoCandidate.from_message(
                submission.text,
                user_message,
                pending_attachment_tokens,
                submission.next_image_token,
                submission_id=submission.id,
            )
            self._queued.append((user_message, candidate))
            self._submissions.complete(submission)
        else:
            candidate = UndoCandidate.from_message(
                submission.text,
                user_message,
                pending_attachment_tokens,
                submission.next_image_token,
                submission_id=submission.id,
            )
            self._print_user(user_message)
            self._undo_candidate = candidate
            self._submissions.complete(submission)
            self._start_turn(
                model_input,
                user_message=user_message,
                submission_id=candidate.submission_id,
            )
            self._dispatch_approval_queue()
