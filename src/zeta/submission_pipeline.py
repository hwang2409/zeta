"""The single-consumer pipeline for submitted composer values."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import count
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .core.abort import AbortSignal
from .core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from .core.commands.custom_commands import CustomCommand, InlineShellResult
from .core.slash import SlashCommandRegistry
from .submission import Submission
from .tools.exec import (
    forget_macro_display,
    register_macro_display,
    run_inline_shell_batch,
)
from .tui.composer import UndoCandidate, parse_input
from .types import StreamEvent, StreamEventType, ToolCall


class SubmissionState(StrEnum):
    """States owned by the submission consumer."""

    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    AWAITING_APPROVAL = "awaiting-approval"
    READY = "ready"
    DISPATCHED = "dispatched"
    CANCELED = "canceled"
    DENIED = "denied"


class SubmissionHost(Protocol):
    """UI and agent operations requested by the pipeline."""

    loop: Any
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
    ) -> Any | None: ...
    def _record_prompt(self, value: str, draft_revision: int) -> None: ...
    def _release_attachment_paths(self, paths: tuple[Path, ...]) -> None: ...
    def _restore_pending_submission(self, submission: Submission) -> None: ...
    def _restore_undo_candidate(self, candidate: UndoCandidate) -> None: ...
    def _print_system(self, value: str) -> None: ...
    def _print_user(self, value: Any) -> None: ...
    def _start_turn(
        self,
        value: str,
        *,
        user_message: Any,
        submission_id: int,
        abort_signal: AbortSignal,
    ) -> asyncio.Task[None]: ...
    def _set_pipeline_task(self, task: asyncio.Task[Any]) -> None: ...
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


class _Message(Protocol):
    pass


@dataclass(frozen=True, slots=True)
class _Submit:
    submission: Submission
    acknowledged: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _PreprocessingDone:
    submission: Submission
    model_input: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _CommandDone:
    submission: Submission
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _ProviderDone:
    submission: Submission
    task: asyncio.Task[None]


@dataclass(frozen=True, slots=True)
class _ApprovalLifecycle:
    submission: Submission
    kind: str
    call: ToolCall


@dataclass(frozen=True, slots=True)
class _ApprovalAction:
    decision: ApprovalDecision
    requested_key: str | None
    acknowledged: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _AbortAction:
    submission_id: int | None


@dataclass(frozen=True, slots=True)
class _UndoAction:
    pass


@dataclass(slots=True)
class _Entry:
    submission: Submission
    state: SubmissionState = SubmissionState.QUEUED
    parsed: str = ""
    model_input: str = ""
    message: Any | None = None
    candidate: UndoCandidate | None = None
    signal: AbortSignal | None = None
    child_task: asyncio.Task[Any] | None = None
    preprocessing: bool = False
    completion: asyncio.Future[None] | None = None
    acknowledged: asyncio.Future[None] | None = None
    undo_requested: bool = False
    denied: bool = False


class SubmissionPipeline:
    """Serialize actions and own every submission transition."""

    _NONTERMINAL = frozenset(
        {
            SubmissionState.QUEUED,
            SubmissionState.PREPROCESSING,
            SubmissionState.AWAITING_APPROVAL,
            SubmissionState.READY,
        }
    )

    def __init__(self, host: SubmissionHost) -> None:
        self._host = host
        self._next_id = count(1)
        self._messages: asyncio.Queue[_Message] = asyncio.Queue()
        self._consumer: asyncio.Task[None] | None = None
        self._entries: dict[int, _Entry] = {}
        self._approval_owners: dict[str | tuple[str, str], Submission] = {}
        self._preprocessing_waiters: dict[int, asyncio.Task[None]] = {}
        self._manual_submissions: asyncio.Queue[Submission] = asyncio.Queue()
        self._manual_canceled: dict[int, Submission] = {}
        self._provider_entry: _Entry | None = None
        self._provider_task: asyncio.Task[None] | None = None
        self._closed = False
        self._shutting_down = False

    @property
    def provider_task(self) -> asyncio.Task[None] | None:
        return self._provider_task

    @property
    def active_submission_id(self) -> int | None:
        return (
            None
            if self._provider_entry is None
            else self._provider_entry.submission.id
        )

    @property
    def active(self) -> bool:
        return any(
            entry.state in self._NONTERMINAL or entry is self._provider_entry
            for entry in self._entries.values()
        )

    @property
    def has_pending(self) -> bool:
        return self.active or not self._manual_submissions.empty()

    @property
    def preprocessing_tasks(self) -> dict[int, asyncio.Task[Any]]:
        return self._preprocessing_waiters

    @property
    def preprocessing_task(self) -> asyncio.Task[Any] | None:
        return next(reversed(self._preprocessing_waiters.values()), None)

    @property
    def inline_abort_signals(self) -> dict[int, AbortSignal]:
        return {
            entry.submission.id: entry.signal
            for entry in self._entries.values()
            if entry.preprocessing and entry.signal is not None
        }

    @property
    def approval_owners(self) -> dict[str | tuple[str, str], int]:
        return {key: owner.id for key, owner in self._approval_owners.items()}

    @property
    def queued(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return tuple(
            (entry.message, entry.candidate)
            for entry in self._entries.values()
            if entry.state is SubmissionState.READY
            and entry.message is not None
            and entry.candidate is not None
        )

    @property
    def approval_queue(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return self.queued if self.pending_approvals else ()

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        policy = self._host._approval_policy
        return () if policy is None else tuple(policy.pending_requests())

    def submit(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
    ) -> Submission:
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
        )
        if self._host._input_loop_active:
            self._send(_Submit(submission))
        else:
            self._manual_submissions.put_nowait(submission)
        return submission

    async def submit_text(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
    ) -> None:
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
        )
        await self._submit_existing(submission)

    async def _submit_existing(self, submission: Submission) -> None:
        canceled = self._manual_canceled.pop(submission.id, None)
        if canceled is not None:
            self._host._restore_pending_submission(canceled)
            return
        self._ensure_consumer()
        acknowledged = asyncio.get_running_loop().create_future()
        self._send(_Submit(submission, acknowledged))
        await acknowledged

    async def get(self) -> Submission:
        """Return an intake item for old direct test callers."""

        return await self._manual_submissions.get()

    def take_pending(self) -> Submission | None:
        """Claim a manually queued item for old direct test callers."""

        try:
            return self._manual_submissions.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def abort(self, submission_id: int | None = None) -> None:
        self._send(_AbortAction(submission_id))

    def undo(self) -> None:
        if not self._manual_submissions.empty():
            pending = self._manual_submissions._queue[-1]
            self._manual_canceled[pending.id] = pending
        self._send(_UndoAction())

    def approval_action(
        self, decision: ApprovalDecision, requested_key: str | None
    ) -> None:
        self._send(_ApprovalAction(decision, requested_key))

    async def approval_action_wait(
        self, decision: ApprovalDecision, requested_key: str | None
    ) -> None:
        acknowledged = asyncio.get_running_loop().create_future()
        self._send(_ApprovalAction(decision, requested_key, acknowledged))
        await acknowledged

    def notify_approval_started(
        self, call: ToolCall, submission_id: int | None
    ) -> None:
        entry = (
            self._entries.get(submission_id)
            if submission_id is not None
            else self._provider_entry
        )
        if entry is not None:
            self._send(_ApprovalLifecycle(entry.submission, "approval_start", call))

    def notify_approval_finished(self, call: ToolCall) -> None:
        owner = self._approval_owners.get(call.id)
        if owner is None and self._provider_entry is not None:
            owner = self._provider_entry.submission
        if owner is not None:
            self._send(_ApprovalLifecycle(owner, "approval_end", call))

    async def close(self) -> None:
        """Cancel all children during application shutdown."""

        self._shutting_down = True
        for entry in self._entries.values():
            if entry.signal is not None:
                entry.signal.abort()
            if entry.child_task is not None and not entry.child_task.done():
                entry.child_task.cancel()
        if self._provider_task is not None and not self._provider_task.done():
            self._provider_task.cancel()
        tasks = [
            task
            for task in (
                *self._preprocessing_waiters.values(),
                self._provider_task,
            )
            if task is not None
        ]
        for entry in self._entries.values():
            if entry.completion is not None and not entry.completion.done():
                entry.completion.set_result(None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()
            await asyncio.gather(self._consumer, return_exceptions=True)
        self._closed = True

    def _new_submission(
        self,
        text: str,
        draft_revision: int,
        attachment_paths: tuple[Path, ...],
        attachment_tokens: Mapping[str, Path] | None,
        next_image_token: int,
    ) -> Submission:
        return Submission(
            next(self._next_id),
            text,
            draft_revision,
            attachment_paths,
            tuple((attachment_tokens or {}).items()),
            next_image_token,
        )

    def _send(self, message: _Message) -> None:
        if self._closed:
            return
        self._ensure_consumer()
        self._messages.put_nowait(message)

    def _ensure_consumer(self) -> None:
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        while not self._shutting_down:
            try:
                message = self._messages.get_nowait()
            except asyncio.QueueEmpty:
                if self._consumer is asyncio.current_task():
                    self._consumer = None
                if not self._messages.empty() and not self._shutting_down:
                    self._ensure_consumer()
                return
            try:
                if isinstance(message, _Submit):
                    await self._on_submit(message)
                elif isinstance(message, _PreprocessingDone):
                    self._on_preprocessing_done(message)
                elif isinstance(message, _CommandDone):
                    self._on_command_done(message)
                elif isinstance(message, _ProviderDone):
                    self._on_provider_done(message)
                elif isinstance(message, _ApprovalLifecycle):
                    self._on_approval_lifecycle(message)
                elif isinstance(message, _ApprovalAction):
                    self._on_approval_action(message)
                elif isinstance(message, _AbortAction):
                    self._on_abort(message.submission_id)
                elif isinstance(message, _UndoAction):
                    self._on_undo()
            except Exception as exc:
                self._host._print_system(f"submission failed: {exc}")
            self._dispatch_oldest_ready()
            self._resolve_preprocessing_completions()

    async def _on_submit(self, message: _Submit) -> None:
        entry = _Entry(message.submission)
        entry.acknowledged = message.acknowledged
        self._entries[entry.submission.id] = entry
        parsed = parse_input(entry.submission.text)
        if parsed is None or self._host._exit_requested:
            self._cancel_entry(entry)
        else:
            action = self._approval_action_for(parsed)
            if action is not None:
                decision, requested_key = action
                self._on_approval_action(_ApprovalAction(decision, requested_key))
                self._host._record_prompt(parsed, entry.submission.draft_revision)
                self._cancel_entry(entry)
            else:
                entry.parsed = parsed
                command = self._host._slash_commands.exec_command_for(parsed)
                if command is not None:
                    self._start_command(entry, command)
                elif self._host._slash_commands.needs_inline_shell_resolution(parsed):
                    self._start_preprocessing(entry)
                else:
                    slash_output = await self._host._slash_commands.dispatch_async(
                        self._host, parsed
                    )
                    if slash_output is not None:
                        self._host._release_attachment_paths(
                            entry.submission.attachment_paths
                        )
                        self._host._record_prompt(
                            parsed, entry.submission.draft_revision
                        )
                        self._host._handle_slash_output(slash_output)
                        self._finish_entry(entry, SubmissionState.CANCELED)
                    else:
                        self._prepare_submission(entry, parsed)
        if entry.state in {
            SubmissionState.CANCELED,
            SubmissionState.READY,
            SubmissionState.DISPATCHED,
        }:
            self._ack_entry(entry)

    def _approval_action_for(
        self, value: str
    ) -> tuple[ApprovalDecision, str | None] | None:
        parts = value.split(maxsplit=1)
        if not parts or parts[0] not in {"approve", "deny"}:
            return None
        decision = (
            ApprovalDecision.ALLOW
            if parts[0] == "approve"
            else ApprovalDecision.DENY
        )
        return decision, parts[1].strip() if len(parts) == 2 else None

    def _start_preprocessing(self, entry: _Entry) -> None:
        entry.state = SubmissionState.PREPROCESSING
        entry.preprocessing = True
        entry.signal = self._new_signal()
        entry.completion = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._preprocess(entry.submission, entry.signal))
        entry.child_task = task
        self._preprocessing_waiters[entry.submission.id] = asyncio.create_task(
            self._wait_for_preprocessing(task, entry.completion)
        )
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._preprocess_done(
                submission, completed
            )
        )

    async def _preprocess(
        self, submission: Submission, signal: AbortSignal
    ) -> str | None:
        return await self._host._slash_commands.resolve_for_model(
            submission.text,
            lambda commands: self._resolve_inline_shell(
                submission, commands, signal
            ),
        )

    def _preprocess_done(
        self, submission: Submission, task: asyncio.Task[str | None]
    ) -> None:
        if task.cancelled():
            result = _PreprocessingDone(submission)
        else:
            try:
                result = _PreprocessingDone(
                    submission, model_input=task.result()
                )
            except BaseException as exc:
                result = _PreprocessingDone(submission, error=exc)
        self._send(result)

    async def _resolve_inline_shell(
        self,
        submission: Submission,
        commands: tuple[str, ...],
        signal: AbortSignal,
    ) -> InlineShellResult:
        first_call_id = f"inline-{uuid4().hex}"
        register_macro_display(
            first_call_id, command="\n".join(commands), argv=()
        )

        def lifecycle(kind: str, call: ToolCall) -> None:
            if kind not in {"approval_start", "approval_end"}:
                return
            self._send(_ApprovalLifecycle(submission, kind, call))
            event_type = (
                StreamEventType.TOOL_APPROVAL_START
                if kind == "approval_start"
                else StreamEventType.TOOL_APPROVAL_END
            )
            self._host._handle_tool_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    data={"inline_shell": True},
                )
            )
            if kind == "approval_start":
                asyncio.get_running_loop().call_soon(
                    self._host._present_pending_approvals
                )
            self._host._invalidate_prompt()

        try:
            outputs = await run_inline_shell_batch(
                self._host.loop.tool_registry,
                commands,
                lifecycle_sink=lifecycle,
                abort_signal=signal,
            )
            return InlineShellResult(outputs, canceled=signal.is_set())
        finally:
            forget_macro_display(first_call_id)

    def _start_command(self, entry: _Entry, command: CustomCommand) -> None:
        entry.state = SubmissionState.PREPROCESSING
        entry.preprocessing = True
        entry.signal = self._new_signal()
        entry.completion = asyncio.get_running_loop().create_future()
        args = entry.parsed.split(maxsplit=1)[1] if " " in entry.parsed else ""
        task = asyncio.create_task(
            self._host._run_macro_submission(
                command, args, entry.submission, entry.signal
            )
        )
        entry.child_task = task
        self._preprocessing_waiters[entry.submission.id] = asyncio.create_task(
            self._wait_for_preprocessing(task, entry.completion)
        )
        self._host._set_pipeline_task(
            self._preprocessing_waiters[entry.submission.id]
        )
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._command_done(
                submission, completed
            )
        )

    def _command_done(
        self, submission: Submission, task: asyncio.Task[str]
    ) -> None:
        if task.cancelled():
            result = _CommandDone(submission)
        else:
            try:
                task.result()
                result = _CommandDone(submission)
            except BaseException as exc:
                result = _CommandDone(submission, error=exc)
        self._send(result)

    def _prepare_submission(self, entry: _Entry, parsed: str) -> None:
        model_input = entry.model_input or self._host._slash_commands.input_for_model(parsed)
        if model_input is None:
            self._cancel_or_restore(entry)
            return
        entry.model_input = model_input
        pending_attachments = list(entry.submission.attachment_paths)
        pending_tokens = dict(entry.submission.attachment_tokens)
        message = self._host._prepare_user_message(
            model_input,
            pending_attachments=pending_attachments,
            pending_attachment_tokens=pending_tokens,
        )
        if message is None:
            self._host._restore_pending_submission(entry.submission)
            self._finish_entry(entry, SubmissionState.CANCELED)
            return
        entry.message = message
        entry.candidate = UndoCandidate.from_message(
            entry.submission.text,
            message,
            pending_tokens,
            entry.submission.next_image_token,
            submission_id=entry.submission.id,
        )
        self._host._record_prompt(parsed, entry.submission.draft_revision)
        entry.signal = entry.signal or self._new_signal()
        entry.state = SubmissionState.READY

    def _on_preprocessing_done(self, message: _PreprocessingDone) -> None:
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        entry.child_task = None
        if message.error is not None:
            self._host._print_system(f"inline shell failed: {message.error}")
            self._cancel_or_restore(entry)
        elif entry.state is SubmissionState.CANCELED or message.model_input is None:
            self._cancel_or_restore(entry)
        else:
            entry.model_input = message.model_input
            self._prepare_submission(entry, entry.parsed)

    def _on_command_done(self, message: _CommandDone) -> None:
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        entry.child_task = None
        if entry.state is SubmissionState.CANCELED:
            self._cancel_or_restore(entry)
            return
        if entry.denied:
            self._host._release_attachment_paths(entry.submission.attachment_paths)
            self._host._record_prompt(entry.parsed, entry.submission.draft_revision)
            self._finish_entry(entry, SubmissionState.DENIED)
            return
        if message.error is not None:
            self._host._print_system(f"command failed: {message.error}")
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._host._record_prompt(entry.parsed, entry.submission.draft_revision)
        entry.state = SubmissionState.DISPATCHED
        self._ack_entry(entry)

    def _dispatch_oldest_ready(self) -> None:
        if any(
            entry.child_task is not None
            and entry.state
            in {SubmissionState.CANCELED, SubmissionState.DENIED}
            for entry in self._entries.values()
        ):
            return
        if self._provider_entry is not None:
            if (
                self._provider_task is not None
                and self._provider_task.done()
            ):
                self._on_provider_done(
                    _ProviderDone(
                        self._provider_entry.submission,
                        self._provider_task,
                    )
                )
            else:
                return
        if self._provider_entry is not None:
            return
        nonterminal = [
            entry
            for entry in self._entries.values()
            if entry.state in self._NONTERMINAL
        ]
        if not nonterminal:
            return
        entry = min(nonterminal, key=lambda candidate: candidate.submission.id)
        if (
            entry.state is not SubmissionState.READY
            or entry.message is None
            or entry.candidate is None
        ):
            return
        assert entry.message is not None and entry.candidate is not None
        user_text = next(
            block.text
            for block in entry.message.content
            if getattr(block, "path", None) is None and hasattr(block, "text")
        )
        entry.state = SubmissionState.DISPATCHED
        self._provider_entry = entry
        self._host._print_user(entry.message)
        self._host._set_undo_candidate(entry.candidate)
        task = self._host._start_turn(
            user_text,
            user_message=entry.message,
            submission_id=entry.submission.id,
            abort_signal=entry.signal or self._new_signal(),
        )
        self._provider_task = task
        self._ack_entry(entry)
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._send(
                _ProviderDone(submission, completed)
            )
        )

    def _on_provider_done(self, message: _ProviderDone) -> None:
        if (
            self._provider_entry is None
            or self._provider_entry.submission is not message.submission
            or self._provider_task is not message.task
        ):
            return
        self._provider_entry = None
        self._provider_task = None

    async def _wait_for_preprocessing(
        self,
        child: asyncio.Task[Any],
        completion: asyncio.Future[None],
    ) -> None:
        try:
            await child
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        await completion

    def _resolve_preprocessing_completions(self) -> None:
        for entry in self._entries.values():
            if (
                entry.completion is not None
                and not entry.completion.done()
                and entry.child_task is None
                and entry.state not in {
                    SubmissionState.PREPROCESSING,
                    SubmissionState.AWAITING_APPROVAL,
                }
            ):
                entry.completion.set_result(None)

    def _on_approval_lifecycle(self, message: _ApprovalLifecycle) -> None:
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        if message.kind == "approval_start":
            self._approval_owners[message.call.id] = entry.submission
            if entry.state is SubmissionState.PREPROCESSING:
                entry.state = SubmissionState.AWAITING_APPROVAL
            return
        self._approval_owners = {
            key: owner
            for key, owner in self._approval_owners.items()
            if owner is not entry.submission or key != message.call.id
        }
        if entry.state is SubmissionState.AWAITING_APPROVAL or (
            entry.state is SubmissionState.DENIED and entry.child_task is not None
        ):
            entry.state = SubmissionState.PREPROCESSING

    def _on_approval_action(self, message: _ApprovalAction) -> None:
        policy = self._host._approval_policy
        if policy is None:
            self._ack_action(message)
            return
        request = self._find_request(message.requested_key)
        if request is None:
            self._host._print_system("approval · no matching request")
            self._ack_action(message)
            return
        owner = self._approval_owners.get(request.key)
        if owner is None:
            entry = self._owner_for_unmapped_request()
            owner = None if entry is None else entry.submission
            if owner is not None:
                self._approval_owners[request.key] = owner
        if message.decision is ApprovalDecision.ALLOW:
            policy.approve(request.key)
            verb = "approved"
        else:
            policy.deny(request.key)
            verb = "denied"
            entry = self._entry_for(owner)
            if entry is not None:
                entry.denied = True
                entry.state = SubmissionState.DENIED
        self._host._print_system(f"approval · {verb} {request.key}")
        self._host._present_pending_approvals()
        self._host._invalidate_prompt()
        self._ack_action(message)

    def _find_request(self, requested_key: str | None) -> ApprovalRequest | None:
        requests = self.pending_approvals
        if requested_key is None:
            return requests[0] if requests else None
        return next(
            (request for request in requests if str(request.key) == requested_key),
            None,
        )

    def _owner_for_unmapped_request(self) -> _Entry | None:
        for entry in reversed(tuple(self._entries.values())):
            if entry.state in {
                SubmissionState.PREPROCESSING,
                SubmissionState.AWAITING_APPROVAL,
            } or entry is self._provider_entry:
                return entry
        return None

    def _on_abort(self, submission_id: int | None) -> None:
        entry = (
            self._target(submission_id)
            if submission_id is not None
            else self._abort_target()
        )
        if entry is not None:
            had_child = entry.child_task is not None
            self._abort_entry(entry)
            if not had_child and entry is not self._provider_entry:
                self._cancel_or_restore(entry)

    def _abort_target(self) -> _Entry | None:
        if self._provider_entry is not None:
            return self._provider_entry
        active_children = [
            entry
            for entry in self._entries.values()
            if entry.state
            in {
                SubmissionState.PREPROCESSING,
                SubmissionState.AWAITING_APPROVAL,
            }
        ]
        if active_children:
            return active_children[-1]
        return self._target(None)

    def _on_undo(self) -> None:
        entry = self._target(None)
        if entry is None:
            return
        entry.undo_requested = True
        was_waiting = entry.state in {
            SubmissionState.PREPROCESSING,
            SubmissionState.AWAITING_APPROVAL,
        }
        self._abort_entry(entry)
        if entry is self._provider_entry and entry.candidate is not None:
            self._host._restore_undo_candidate(entry.candidate)
        elif not was_waiting:
            self._cancel_or_restore(entry)

    def _target(self, submission_id: int | None) -> _Entry | None:
        if submission_id is not None:
            entry = self._entries.get(submission_id)
            return (
                entry
                if entry is not None
                and (self._actionable(entry) or entry is self._provider_entry)
                else None
            )
        candidates = [
            entry
            for entry in self._entries.values()
            if self._actionable(entry) or entry is self._provider_entry
        ]
        return candidates[-1] if candidates else None

    def _actionable(self, entry: _Entry) -> bool:
        return entry.state in self._NONTERMINAL

    def _abort_entry(self, entry: _Entry) -> None:
        if entry.signal is not None:
            entry.signal.abort()
        self._abort_approvals(entry)
        if entry is self._provider_entry:
            if self._provider_task is not None and not self._provider_task.done():
                self._provider_task.cancel()
        elif entry.child_task is not None and not entry.child_task.done():
            entry.child_task.cancel()
        entry.state = SubmissionState.CANCELED
        self._host._invalidate_prompt()

    def _abort_approvals(self, entry: _Entry) -> None:
        policy = self._host._approval_policy
        if policy is None:
            return
        for request in self.pending_approvals:
            if self._approval_owners.get(request.key) is not entry.submission:
                continue
            policy.abort(request.key)
            if entry is self._provider_entry and isinstance(request.key, str):
                self._host.loop.finalize_canceled(request.key)

    def _cancel_or_restore(self, entry: _Entry) -> None:
        if entry.undo_requested:
            self._host._restore_pending_submission(entry.submission)
        else:
            self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._finish_entry(entry, SubmissionState.CANCELED)

    def _cancel_entry(self, entry: _Entry) -> None:
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._finish_entry(entry, SubmissionState.CANCELED)

    def _finish_entry(self, entry: _Entry, state: SubmissionState) -> None:
        entry.state = state
        entry.child_task = None
        self._ack_entry(entry)

    def _entry_for(self, submission: Submission | None) -> _Entry | None:
        if submission is None:
            return None
        entry = self._entries.get(submission.id)
        return entry if entry is not None and entry.submission is submission else None

    def _new_signal(self) -> AbortSignal:
        return self._host.loop.tool_registry.abort_signal.registry.new_generation()

    def _ack_entry(self, entry: _Entry) -> None:
        if entry.acknowledged is not None and not entry.acknowledged.done():
            entry.acknowledged.set_result(None)

    @staticmethod
    def _ack_action(message: _ApprovalAction) -> None:
        if message.acknowledged is not None and not message.acknowledged.done():
            message.acknowledged.set_result(None)


class SubmissionMixin:
    """Translate composer callbacks into pipeline messages."""

    def _submit_input(self, value: str) -> None:
        if (
            self._submissions._approval_action_for(value) is not None
            and not self._submissions.active
        ):
            task = asyncio.create_task(self._handle_approval_input(value))
            self._active_task = task
            task.add_done_callback(self._clear_approval_task)
            return
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        self._submissions.submit(
            value,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
        )

    def _clear_approval_task(self, task: asyncio.Task[Any]) -> None:
        if self._active_task is task:
            self._active_task = None

    async def _handle_prompt_value(self, value: Submission | str) -> None:
        if isinstance(value, Submission):
            await self._submissions._submit_existing(value)
            return
        if self._input_loop_active:
            self._submit_input(value)
            await asyncio.sleep(0)
            return
        pending = self._submissions.take_pending()
        if pending is not None:
            await self._submissions._submit_existing(pending)
            return
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        await self._submissions.submit_text(
            value,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
        )

    async def _handle_approval_input(self, value: str) -> bool:
        action = self._submissions._approval_action_for(value)
        if action is None:
            return False
        decision, requested_key = action
        await self._submissions.approval_action_wait(decision, requested_key)
        return True

    def _pending_approvals_for_submission(
        self, submission_id: int
    ) -> tuple[ApprovalRequest, ...]:
        owners = self._submissions.approval_owners
        return tuple(
            request
            for request in self.pending_approvals
            if owners.get(request.key) == submission_id
        )

    def _preprocessing_wait_set(self) -> set[asyncio.Task[Any]]:
        return set(self._submissions.preprocessing_tasks.values())

    def _drain_preprocessing(self, done: set[asyncio.Task[Any]]) -> None:
        del done

    def _set_pipeline_task(self, task: asyncio.Task[Any]) -> None:
        self._active_task = task

    def _handle_slash_output(self, output: str) -> None:
        if self._fork_rebuilt:
            self._fork_rebuilt = False
        elif output.startswith("[Image #"):
            self._insert_paste_token(output)
        elif output:
            self._print_system(output)

    @property
    def _preprocessing_tasks(self) -> dict[int, asyncio.Task[Any]]:
        return self._submissions.preprocessing_tasks

    @property
    def _preprocessing_task(self) -> asyncio.Task[Any] | None:
        return self._submissions.preprocessing_task

    @_preprocessing_task.setter
    def _preprocessing_task(self, value: asyncio.Task[Any] | None) -> None:
        del value

    @property
    def _inline_abort_signals(self) -> dict[int, AbortSignal]:
        return self._submissions.inline_abort_signals

    @property
    def _approval_owners(self) -> dict[str | tuple[str, str], int]:
        return self._submissions.approval_owners

    @property
    def _approval_queue(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return self._submissions.approval_queue

    @property
    def _queued(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return self._submissions.queued

    def _dispatch_approval_queue(self) -> None:
        self._submissions._dispatch_oldest_ready()

    def _start_queued_turn(self) -> None:
        self._submissions._dispatch_oldest_ready()

    def _set_undo_candidate(self, candidate: UndoCandidate) -> None:
        self._undo_candidate = candidate

    def _restore_undo_candidate(self, candidate: UndoCandidate) -> None:
        self._restore_composer(candidate.text)
        self._pending_attachments[:] = list(candidate.attachment_paths)
        self._pending_attachment_tokens.clear()
        self._pending_attachment_tokens.update(candidate.attachment_tokens)
        self._next_image_token = candidate.next_image_token
        self._draft.schedule(candidate.text)
