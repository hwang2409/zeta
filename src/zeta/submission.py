"""Identity-preserving composer submission queue."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from itertools import count

from .tui.composer import parse_input


@dataclass(frozen=True, slots=True)
class Submission:
    """One composer submission with identity separate from its text."""

    id: int
    text: str


class SubmissionQueue:
    """Queue submissions and cancel them by object identity."""

    def __init__(self) -> None:
        self._next_id = count(1)
        self._pending: deque[Submission] = deque()
        self._ready = asyncio.Event()
        self._current: Submission | None = None
        self._cancelled: set[int] = set()

    def submit(self, text: str) -> Submission:
        submission = Submission(next(self._next_id), text)
        self._pending.append(submission)
        self._ready.set()
        return submission

    async def get(self) -> Submission:
        """Wait for and claim the next submission."""

        await self._ready.wait()
        submission = self._pending.popleft()
        if not self._pending:
            self._ready.clear()
        self._current = submission
        return submission

    def take_pending(self) -> Submission | None:
        """Claim a pending submission for direct, non-loop callers."""

        if not self._pending:
            return None
        submission = self._pending.popleft()
        if not self._pending:
            self._ready.clear()
        self._current = submission
        return submission

    def begin_direct(self, text: str) -> Submission:
        """Create a submission for callers that bypass the input queue."""

        submission = Submission(next(self._next_id), text)
        self._current = submission
        return submission

    def cancel_current(self) -> Submission | None:
        """Cancel the current submission, or the next queued submission."""

        submission = self._current
        if submission is None and self._pending:
            submission = self._pending[0]
        if submission is not None:
            self._cancelled.add(submission.id)
        return submission

    def is_cancelled(self, submission: Submission) -> bool:
        return submission.id in self._cancelled

    def complete(self, submission: Submission) -> None:
        if self._current is submission:
            self._current = None
        self._cancelled.discard(submission.id)


class SubmissionMixin:
    """Handle composer submissions without sharing mutable text state."""

    def _submit_input(self, value: str) -> None:
        self._submissions.submit(value)
        self._draft.mark_submitted()

    async def _handle_prompt_value(self, value: Submission | str) -> None:
        if isinstance(value, Submission):
            submission = value
        else:
            submission = self._submissions.take_pending()
            if submission is None:
                submission = self._submissions.begin_direct(value)
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        parsed = parse_input(submission.text)
        if parsed is None or self._exit_requested:
            self._submissions.complete(submission)
            return
        if await self._handle_approval_input(parsed):
            if self._submissions.is_cancelled(submission):
                self._submissions.complete(submission)
                return
            self._submissions.complete(submission)
            self._record_prompt(parsed)
            return
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        slash_output = await self._slash_commands.dispatch_async(self, parsed)
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        if slash_output is not None:
            self._submissions.complete(submission)
            self._record_prompt(parsed)
            if self._fork_rebuilt:
                self._fork_rebuilt = False
            elif slash_output.startswith("[Image #"):
                self._insert_paste_token(slash_output)
            else:
                self._print_system(slash_output)
            return
        model_input = self._slash_commands.input_for_model(parsed)
        user_message = self._prepare_user_message(model_input)
        if user_message is None:
            self._submissions.complete(submission)
            return
        if self._submissions.is_cancelled(submission):
            self._submissions.complete(submission)
            return
        self._record_prompt(parsed)
        self._failed_turn = None
        if self.pending_approvals:
            self._present_pending_approvals()
            self._submissions.complete(submission)
        elif self.active:
            candidate = self._undo_candidate_for_message(parsed, user_message)
            self._clear_pending_attachments()
            self._queued.append((user_message, candidate))
            self._submissions.complete(submission)
        else:
            candidate = self._undo_candidate_for_message(parsed, user_message)
            self._clear_pending_attachments()
            self._print_user(user_message)
            self._undo_candidate = candidate
            self._submissions.complete(submission)
            self._start_turn(model_input, user_message=user_message)
