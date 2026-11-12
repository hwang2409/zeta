"""The single-consumer pipeline for submitted composer values."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from itertools import count
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.abort import AbortSignal
from ..core.approval import ApprovalDecision, ApprovalRequest
from ..core.commands.custom_commands import CustomCommand, InlineShellResult
from ..core.slash import SlashPromptError
from ..model_input import ModelInputEnvelope
from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from ..tools._shared.shell import (
    forget_macro_display,
    register_macro_display,
    run_inline_shell_batch,
)
from ..tui.composer import UndoCandidate, parse_input
from ..tui.user import USER_DISPLAY_TEXT_METADATA
from ._approval_pipeline import ApprovalPipelineMixin
from .messages import (
    _AbortAction,
    _ApprovalAction,
    _ApprovalLifecycle,
    _CommandDone,
    _DurableToolDone,
    _Message,
    _PreprocessingDone,
    _ProviderDone,
    _Retry,
    _SlashDone,
    _Submit,
    _UndoAction,
)
from .model import (
    Submission,
    SubmissionHost,
    SubmissionState,
    _Entry,
    dispatch_provider,
)
from .notification_retry import NotificationOutcome, NotificationRetry


class SubmissionPipeline(ApprovalPipelineMixin):
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
        self._command_order: deque[int] = deque()
        self._command_results: dict[int, str | None] = {}
        self._provider_entry: _Entry | None = None
        self._provider_task: asyncio.Task[NotificationOutcome] | None = None
        self._notification_retry = NotificationRetry(
            wake=lambda: self._send(None),
            has_notification=lambda: (
                self._host.loop.notification_system_message() is not None
            ),
            schedule_history=self._host.loop.schedule_notification_history_retry,
            rollback=self._rollback_provider_dispatch,
            report_failure=lambda message: self._host._print_system(
                f"notification failed; retrying automatically: {message}"
            ),
            report_stopped=lambda message: self._host._print_system(
                f"notification failed; automatic retry stopped: {message}"
            ),
        )
        self._durable_tasks: dict[str, asyncio.Task[Any]] = {}
        self._durable_tool_calls: dict[str, ToolCall] = {}
        # Running slash dispatches by submission id; each holds or awaits the
        # session slash lock, so the consumer stays free for aborts and approvals.
        self._slash_tasks: dict[int, asyncio.Task[Any]] = {}
        # Submissions queued behind a running slash command keep their order.
        self._deferred: deque[_Message] = deque()
        self._closed = False
        self._shutting_down = False

    @property
    def provider_task(self) -> asyncio.Task[NotificationOutcome] | None:
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
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if self._slash_tasks and current not in self._slash_tasks.values():
            # A running slash command is work; only the command itself ignores it.
            return True
        return any(
            (
                entry.state in self._NONTERMINAL or entry is self._provider_entry
            )
            and entry.submission.id not in self._slash_tasks
            for entry in self._entries.values()
        ) or (
            not self._slash_tasks
            and any(
                isinstance(message, (_Submit, _Retry))
                for message in (*self._deferred, *self._messages._queue)
            )
        ) or bool(self._durable_tasks)

    @property
    def has_pending(self) -> bool:
        return self.active

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
        # One owner for the runtime pending projection: the policy's resident
        # snapshot (store page + delegated + ephemeral), read live on each
        # present. Cheap because the store memoizes it by the active leaf.
        policy = self._host._approval_policy
        return (
            ()
            if policy is None
            else tuple(policy.pending_requests_for_display_now())
        )

    def submit(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
        steer: bool = True,
        internal: bool = False,
    ) -> Submission:
        self._ensure_open()
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
            steer,
            internal,
        )
        self._send(_Submit(submission))
        return submission

    def wake(self) -> None:
        self._ensure_open()
        self._send(None)

    async def submit_text(
        self,
        text: str,
        *,
        draft_revision: int = 0,
        attachment_paths: tuple[Path, ...] = (),
        attachment_tokens: Mapping[str, Path] | None = None,
        next_image_token: int = 1,
        steer: bool = True,
        internal: bool = False,
    ) -> None:
        self._ensure_open()
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
            steer,
            internal,
        )
        acknowledged = asyncio.get_running_loop().create_future()
        self._send(_Submit(submission, acknowledged))
        await acknowledged

    async def retry(self, text: str, user_message: Any) -> None:
        self._ensure_open()
        submission = self._new_submission(text, 0, (), None, 1)
        acknowledged = asyncio.get_running_loop().create_future()
        self._send(_Retry(submission, user_message, acknowledged))
        await acknowledged

    def abort(self, submission_id: int | None = None) -> None:
        self._ensure_open()
        self._send(_AbortAction(submission_id))

    def undo(self) -> None:
        self._ensure_open()
        self._send(_UndoAction())

    async def approval_command(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None:
        """Resolve an approval command already running inside the pipeline."""

        await self._on_approval_action(
            _ApprovalAction(decision, requested_key, always=always)
        )

    async def approval_action_wait(
        self, decision: ApprovalDecision, requested_key: str | None
    ) -> None:
        self._ensure_open()
        acknowledged = asyncio.get_running_loop().create_future()
        self._send(_ApprovalAction(decision, requested_key, acknowledged))
        await acknowledged

    def notify_approval_started(
        self,
        call: ToolCall,
        submission_id: int | None,
    ) -> None:
        entry = (
            self._entries.get(submission_id)
            if submission_id is not None
            else self._provider_entry
        )
        if entry is not None:
            self._send(_ApprovalLifecycle(entry.submission, "approval_start", call))
        # A delegated child approval has no local submission entry; it is already
        # registered on the policy (register_delegated), so the live
        # pending_approvals property surfaces it on the next present.

    def notify_approval_finished(self, call: ToolCall) -> None:
        owner = self._approval_owners.get(call.id)
        if owner is None and self._provider_entry is not None:
            owner = self._provider_entry.submission
        if owner is not None:
            self._send(_ApprovalLifecycle(owner, "approval_end", call))

    async def close(self) -> None:
        """Cancel all children during application shutdown."""

        if self._closed:
            return
        self._closed = True
        self._shutting_down = True
        self._notification_retry.close()
        for entry in self._entries.values():
            if entry.signal is not None:
                entry.signal.abort()
            if entry.child_task is not None and not entry.child_task.done():
                entry.child_task.cancel()
        for task in self._slash_tasks.values():
            if not task.done():
                task.cancel()
        for task in self._durable_tasks.values():
            if not task.done():
                task.cancel()
        if self._provider_task is not None and not self._provider_task.done():
            self._provider_task.cancel()
        tasks = [
            task
            for task in (
                *self._preprocessing_waiters.values(),
                *self._slash_tasks.values(),
                self._provider_task,
                *self._durable_tasks.values(),
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
        self._drain_acknowledgements()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("submission pipeline is closed")

    def _new_submission(
        self,
        text: str,
        draft_revision: int,
        attachment_paths: tuple[Path, ...],
        attachment_tokens: Mapping[str, Path] | None,
        next_image_token: int,
        steer: bool = True,
        internal: bool = False,
    ) -> Submission:
        return Submission(
            next(self._next_id),
            text,
            draft_revision,
            attachment_paths,
            tuple((attachment_tokens or {}).items()),
            next_image_token,
            steer,
            internal,
        )

    def _send(self, message: _Message) -> None:
        if self._closed:
            return
        self._ensure_consumer()
        self._messages.put_nowait(message)

    def _ensure_consumer(self) -> None:
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.create_task(self._consume())

    def _slash_route(self, message: _Message) -> bool:
        if not isinstance(message, _Submit):
            return False
        parsed = parse_input(message.submission.text)
        commands = self._host._slash_commands
        return (
            parsed is not None
            and not self._host._exit_requested
            and commands.exec_command_for(parsed) is None
            and not commands.needs_inline_shell_resolution(parsed)
            and commands.routes(self._host, parsed)
        )

    def _next_message(self) -> _Message | None:
        if self._deferred and (
            not self._slash_tasks or self._slash_route(self._deferred[0])
        ):
            return self._deferred.popleft()
        while True:
            message = self._messages.get_nowait()
            if isinstance(message, (_Submit, _Retry)) and (
                self._deferred
                or (self._slash_tasks and not self._slash_route(message))
            ):
                self._deferred.append(message)
                continue
            return message

    async def _consume(self) -> None:
        while not self._shutting_down:
            try:
                message = self._next_message()
            except asyncio.QueueEmpty:
                if self._consumer is asyncio.current_task():
                    self._consumer = None
                if not self._messages.empty() and not self._shutting_down:
                    self._ensure_consumer()
                return
            try:
                if message is None:
                    self._on_wake()
                elif isinstance(message, _Submit):
                    await self._on_submit(message)
                elif isinstance(message, _Retry):
                    self._on_retry(message)
                elif isinstance(message, _PreprocessingDone):
                    await self._on_preprocessing_done(message)
                elif isinstance(message, _CommandDone):
                    self._on_command_done(message)
                elif isinstance(message, _SlashDone):
                    await self._on_slash_done(message)
                elif isinstance(message, _ProviderDone):
                    self._on_provider_done(message)
                elif isinstance(message, _DurableToolDone):
                    self._on_durable_tool_done(message)
                elif isinstance(message, _ApprovalLifecycle):
                    self._on_approval_lifecycle(message)
                elif isinstance(message, _ApprovalAction):
                    await self._on_approval_action(message)
                elif isinstance(message, _AbortAction):
                    self._on_abort(message.submission_id)
                elif isinstance(message, _UndoAction):
                    self._on_undo()
            except Exception as exc:
                self._host._print_system(f"submission failed: {exc}")
                self._fail_message(message)
            self._resolve_preprocessing_completions()
            self._prune_finished()
            if self._messages.empty():
                try:
                    self._dispatch_oldest_ready()
                except Exception as exc:  # noqa: BLE001
                    self._host._print_system(f"submission failed: {exc}")
                    self._rollback_provider_dispatch()
                    self._retry_oldest_ready()

    async def _on_submit(self, message: _Submit) -> None:
        await self._host.loop.refresh_live_config()
        entry = _Entry(message.submission)
        entry.acknowledged = message.acknowledged
        self._entries[entry.submission.id] = entry
        parsed = parse_input(entry.submission.text)
        if parsed is None or self._host._exit_requested:
            self._cancel_entry(entry)
        else:
            entry.parsed = parsed
            command = self._host._slash_commands.exec_command_for(parsed)
            if command is not None:
                self._start_command(entry, command)
            elif self._host._slash_commands.needs_inline_shell_resolution(parsed):
                self._start_preprocessing(entry)
            elif self._host._slash_commands.routes(self._host, parsed):
                self._start_slash(entry, parsed)
            else:
                await self._prepare_submission(entry, parsed)
        self._ack_settled(entry)

    def _ack_settled(self, entry: _Entry) -> None:
        if entry.state in {
            SubmissionState.CANCELED,
            SubmissionState.READY,
            SubmissionState.DISPATCHED,
        }:
            self._ack_entry(entry)

    def _start_slash(self, entry: _Entry, parsed: str) -> None:
        entry.state = SubmissionState.PREPROCESSING
        entry.signal = self._new_signal()
        task = asyncio.create_task(
            self._host._slash_commands.dispatch_async(self._host, parsed)
        )
        entry.child_task = task
        self._slash_tasks[entry.submission.id] = task
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._send(
                _SlashDone(submission, completed)
            )
        )

    async def _on_slash_done(self, message: _SlashDone) -> None:
        self._slash_tasks.pop(message.submission.id, None)
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        entry.child_task = None
        try:
            await self._settle_slash(entry, message.task)
        finally:
            # the command finished after the keypress that started it was drawn; nothing else
            # repaints an idle screen, so its output would wait for the next key.
            self._host._invalidate_prompt()

    async def _settle_slash(self, entry: _Entry, task: asyncio.Task[Any]) -> None:
        if task.cancelled() or entry.state is SubmissionState.CANCELED:
            self._cancel_or_restore(entry)
            return
        try:
            slash_output = task.result()
        except Exception as exc:  # noqa: BLE001 - the consumer reports command failures
            self._host._print_system(f"submission failed: {exc}")
            self._cancel_entry(entry)
            return
        if isinstance(slash_output, SlashPromptError):
            self._host._restore_pending_submission(entry.submission)
            self._host._print_system(slash_output.message)
            self._finish_entry(entry, SubmissionState.CANCELED)
        elif isinstance(slash_output, ModelInputEnvelope):
            entry.model_input = slash_output
            entry.attachment_value = entry.submission.text
            await self._prepare_submission(entry, entry.parsed)
        elif slash_output is not None:
            self._host._release_attachment_paths(entry.submission.attachment_paths)
            self._record_prompt(entry)
            self._host._handle_slash_output(slash_output)
            self._finish_entry(entry, SubmissionState.CANCELED)
        else:
            await self._prepare_submission(entry, entry.parsed)
        self._ack_settled(entry)

    def _on_wake(self) -> None:
        self._notification_retry.request_wake()

    def start_notification_wake(self) -> None:
        """Dispatch the ready notification attempt without blocking the consumer."""

        self._notification_retry.attempt(self._start_notification_provider)

    def _start_notification_provider(self) -> None:
        submission = self._new_submission("", 0, (), None, 1, steer=False)
        entry = self._notification_entry(submission)
        self._dispatch_provider(entry, user_text="", notification=True)

    def _notification_entry(self, submission: Submission) -> _Entry:
        entry = _Entry(submission, state=SubmissionState.DISPATCHED)
        entry.signal = self._new_signal()
        self._entries[submission.id] = entry
        return entry

    def _on_retry(self, message: _Retry) -> None:
        entry = _Entry(message.submission)
        entry.acknowledged = message.acknowledged
        entry.message = message.user_message
        entry.persist_user_message = False
        entry.acknowledge_on_provider_done = True
        entry.candidate = UndoCandidate.from_message(
            message.submission.text,
            message.user_message,
            {},
            1,
            submission_id=message.submission.id,
        )
        entry.signal = self._new_signal()
        entry.state = SubmissionState.READY
        self._entries[entry.submission.id] = entry

    def _start_preprocessing(self, entry: _Entry) -> None:
        entry.state = SubmissionState.PREPROCESSING
        entry.preprocessing = True
        entry.signal = self._new_signal()
        entry.completion = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._preprocess(entry.submission, entry.signal))
        entry.child_task = task
        self._track_waiter(entry, task)
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._preprocess_done(
                submission, completed
            )
        )

    async def _preprocess(
        self, submission: Submission, signal: AbortSignal
    ) -> ModelInputEnvelope | None:
        return await self._host._slash_commands.resolve_for_model(
            submission.text,
            lambda commands: self._resolve_inline_shell(
                submission, commands, signal
            ),
        )

    def _preprocess_done(
        self,
        submission: Submission,
        task: asyncio.Task[ModelInputEnvelope | None],
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
        self._command_order.append(entry.submission.id)
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
        waiter = self._track_waiter(entry, task)
        self._host._set_pipeline_task(waiter)
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
                result = _CommandDone(submission, receipt=task.result() or None)
            except BaseException as exc:
                result = _CommandDone(submission, error=exc)
        self._send(result)

    async def _prepare_submission(self, entry: _Entry, parsed: str) -> None:
        envelope = (
            entry.model_input
            if entry.model_input is not None
            else self._host._slash_commands.input_for_model(parsed)
        )
        entry.model_input = envelope
        pending_attachments = list(entry.submission.attachment_paths)
        pending_tokens = dict(entry.submission.attachment_tokens)
        message = await self._host._prepare_user_message(
            envelope.text,
            pending_attachments=pending_attachments,
            pending_attachment_tokens=pending_tokens,
            attachment_value=entry.attachment_value,
        )
        if message is None:
            self._host._restore_pending_submission(entry.submission)
            self._finish_entry(entry, SubmissionState.CANCELED)
            return
        metadata = dict(message.metadata)
        metadata[USER_DISPLAY_TEXT_METADATA] = envelope.display_text
        metadata[MESSAGE_ORIGIN_METADATA] = envelope.origin.value
        if metadata != message.metadata:
            message = replace(message, metadata=metadata)
        entry.message = message
        entry.candidate = UndoCandidate.from_message(
            entry.submission.text,
            message,
            pending_tokens,
            entry.submission.next_image_token,
            submission_id=entry.submission.id,
        )
        self._record_prompt(entry)
        entry.signal = entry.signal or self._new_signal()
        entry.state = SubmissionState.READY

    async def _on_preprocessing_done(self, message: _PreprocessingDone) -> None:
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        entry.child_task = None
        if message.error is not None:
            self._host._print_system(f"inline shell failed: {message.error}")
            if entry.denied:
                self._finish_denied(entry)
            else:
                self._cancel_or_restore(entry)
        elif entry.state is SubmissionState.CANCELED:
            self._cancel_or_restore(entry)
        elif entry.denied:
            self._finish_denied(entry)
        elif message.model_input is None:
            self._cancel_or_restore(entry)
        else:
            entry.model_input = message.model_input
            await self._prepare_submission(entry, entry.parsed)

    def _on_command_done(self, message: _CommandDone) -> None:
        self._command_results[message.submission.id] = message.receipt
        self._commit_command_receipts()
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        entry.child_task = None
        if entry.state is SubmissionState.CANCELED:
            self._cancel_or_restore(entry)
            return
        if entry.denied:
            self._finish_denied(entry)
            return
        if message.error is not None:
            self._host._print_system(f"command failed: {message.error}")
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._record_prompt(entry)
        entry.state = SubmissionState.DISPATCHED
        self._ack_entry(entry)

    def _commit_command_receipts(self) -> None:
        while self._command_order and self._command_order[0] in self._command_results:
            submission_id = self._command_order.popleft()
            receipt = self._command_results.pop(submission_id)
            if receipt:
                self._host._record_macro_receipt(receipt)

    def _dispatch_oldest_ready(self) -> None:
        if self._durable_tasks:
            return
        if any(
            entry.child_task is not None
            and entry.state
            in {SubmissionState.CANCELED, SubmissionState.DENIED}
            for entry in self._entries.values()
        ):
            return
        if (
            self._provider_entry is not None
            and self._provider_task is not None
            and self._provider_task.done()
        ):
            self._on_provider_done(
                _ProviderDone(
                    self._provider_entry.submission,
                    self._provider_task,
                )
            )
        if self._provider_entry is not None:
            self._dispatch_steer_entries()
            return
        retry_kind = self._notification_retry.ready_kind
        if retry_kind == "wake":
            self.start_notification_wake()
            if self._provider_entry is not None:
                return
        nonterminal = [
            entry
            for entry in self._entries.values()
            if entry.state in self._NONTERMINAL
        ]
        if not nonterminal:
            if retry_kind == "history":
                self.start_notification_wake()
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
        if not entry.printed:
            self._host._print_user(entry.message)
            entry.printed = True
        self._host._set_undo_candidate(entry.candidate)
        self._dispatch_provider(
            entry, user_text,
            user_message=entry.message,
            persist_user_message=entry.persist_user_message,
        )
        if self._notification_retry.pending:
            self._notification_retry.recovered()
        if not entry.acknowledge_on_provider_done:
            self._ack_entry(entry)

    def _dispatch_provider(
        self,
        entry: _Entry,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        notification: bool = False,
    ) -> None:
        self._provider_entry = entry
        self._provider_task = dispatch_provider(
            self._host,
            entry,
            user_text=user_text,
            abort_signal=entry.signal or self._new_signal(),
            user_message=user_message,
            persist_user_message=persist_user_message,
            notification=notification,
            suppress_errors=(
                notification and self._notification_retry.failure_reported
            ),
            on_done=lambda submission, task: self._send(_ProviderDone(submission, task)),
        )

    def _dispatch_steer_entries(self) -> None:
        if self._provider_entry is not None and self._provider_entry.message is None:
            return
        ready = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.state is SubmissionState.READY
                and entry.submission.steer
                and entry.message is not None
                and entry.candidate is not None
            ),
            key=lambda entry: entry.submission.id,
        )
        for entry in ready:
            entry.state = SubmissionState.DISPATCHED
            self._host._print_user(entry.message)
            entry.printed = True
            self._host._set_undo_candidate(entry.candidate)
            self._host.loop.steer(entry.message)
            self._host._release_attachment_paths(entry.submission.attachment_paths)
            self._ack_entry(entry)

    def _retry_oldest_ready(self) -> None:
        while self._messages.empty():
            try:
                self._dispatch_oldest_ready()
            except Exception as exc:  # noqa: BLE001
                self._host._print_system(f"submission failed: {exc}")
                self._rollback_provider_dispatch()
            else:
                return

    def _rollback_provider_dispatch(self) -> None:
        entry = self._provider_entry
        if entry is None:
            return
        if self._provider_task is not None and not self._provider_task.done():
            self._provider_task.cancel()
        self._provider_entry = None
        self._provider_task = None
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._finish_entry(entry, SubmissionState.CANCELED)

    def _on_provider_done(self, message: _ProviderDone) -> None:
        if (
            self._provider_entry is None
            or self._provider_entry.submission is not message.submission
            or self._provider_task is not message.task
        ):
            return
        entry = self._provider_entry
        if entry.message is None:
            self._notification_retry.provider_finished(message.task)
        pending = self._host.loop.store.pending_client_steering()
        if self._provider_entry.message is not None and pending is not None:
            entries = sorted(self._entries.values(), key=lambda item: item.submission.id)
            late_entries = [
                entry
                for entry in entries
                if entry.state is SubmissionState.DISPATCHED
                and entry.submission.steer
                and entry is not self._provider_entry
            ][-len(pending.messages) :]
            self._host.loop.clear_pending_steering("turn_end")
            for entry in late_entries:
                entry.submission = replace(entry.submission, steer=False)
                entry.signal = self._new_signal()
                entry.state = SubmissionState.READY
        self._ack_entry(self._provider_entry)
        self._provider_entry, self._provider_task = None, None
        self._host._maybe_restart_when_idle()

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

    def _track_waiter(
        self, entry: _Entry, child: asyncio.Task[Any]
    ) -> asyncio.Task[None]:
        waiter = asyncio.create_task(
            self._wait_for_preprocessing(child, entry.completion)
        )
        self._preprocessing_waiters[entry.submission.id] = waiter
        waiter.add_done_callback(
            lambda completed, submission_id=entry.submission.id: self._drop_waiter(
                submission_id, completed
            )
        )
        return waiter

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

    def _on_abort(self, submission_id: int | None) -> None:
        if (
            submission_id is None
            and self._provider_entry is None
            and self._durable_tasks
        ):
            request_id, task = next(reversed(self._durable_tasks.items()))
            self._host.loop.abort()
            if self._host._approval_policy is not None:
                self._host._approval_policy.abort(request_id)
            tool_call = self._durable_tool_calls[request_id]
            self._host.loop.finalize_canceled(tool_call)
            if not task.done():
                task.cancel()
            self._host._invalidate_prompt()
            return
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
            return

    def _on_durable_tool_done(self, message: _DurableToolDone) -> None:
        if self._durable_tasks.get(message.request_id) is not message.task:
            return
        self._durable_tasks.pop(message.request_id)
        self._durable_tool_calls.pop(message.request_id, None)
        if message.task.cancelled():
            return
        try:
            message.task.result()
        except Exception as exc:  # noqa: BLE001
            self._host._print_system(f"submission failed: {exc}")

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
        if entry is self._provider_entry:
            if entry.candidate is not None:
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
        if entry is self._provider_entry:
            self._host.loop.abort(foreground_only=True)
        self._abort_approvals(entry)
        if entry is self._provider_entry:
            if self._provider_task is not None and not self._provider_task.done():
                self._provider_task.cancel()
            # Steering queued into an aborted turn cannot reach a fresh turn.
            self._host.loop.clear_pending_steering("turn_end")
        elif entry.child_task is not None and not entry.child_task.done():
            entry.child_task.cancel()
        entry.state = SubmissionState.CANCELED
        self._host._invalidate_prompt()

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

    def _record_prompt(self, entry: _Entry) -> None:
        if not entry.submission.internal:
            self._host._record_prompt(entry.parsed, entry.submission.draft_revision)

    def _finish_denied(self, entry: _Entry) -> None:
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._record_prompt(entry)
        self._finish_entry(entry, SubmissionState.DENIED)

    def _drop_waiter(
        self, submission_id: int, completed: asyncio.Future[None]
    ) -> None:
        if self._preprocessing_waiters.get(submission_id) is completed:
            self._preprocessing_waiters.pop(submission_id)

    def _prune_finished(self) -> None:
        for submission_id, entry in tuple(self._entries.items()):
            if (
                entry is self._provider_entry
                or entry.state in self._NONTERMINAL
                or self._provider_entry is not None
                and entry.state is SubmissionState.DISPATCHED
                and entry.submission.steer
                or entry.child_task is not None
                or entry.completion is not None
                and not entry.completion.done()
                or any(owner is entry.submission for owner in self._approval_owners.values())
            ):
                continue
            self._entries.pop(submission_id, None)

    def _fail_message(self, message: _Message) -> None:
        submission = self._submission_for_message(message)
        entry = self._entry_for(submission)
        if entry is not None and entry.state in self._NONTERMINAL:
            self._host._release_attachment_paths(entry.submission.attachment_paths)
            self._finish_entry(entry, SubmissionState.CANCELED)
        self._ack_message(message)

    @staticmethod
    def _submission_for_message(message: _Message) -> Submission | None:
        if isinstance(
            message,
            (
                _Submit,
                _Retry,
                _PreprocessingDone,
                _CommandDone,
                _ProviderDone,
            ),
        ):
            return message.submission
        if isinstance(message, _ApprovalLifecycle):
            return message.submission
        return None

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

    def _drain_acknowledgements(self) -> None:
        for entry in self._entries.values():
            self._ack_entry(entry)
        while self._deferred:
            self._ack_message(self._deferred.popleft())
        while True:
            try:
                message = self._messages.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._ack_message(message)

    @staticmethod
    def _ack_message(message: _Message) -> None:
        if isinstance(message, (_Submit, _Retry, _ApprovalAction)):
            acknowledged = message.acknowledged
            if acknowledged is not None and not acknowledged.done():
                acknowledged.set_result(None)

    @staticmethod
    def _ack_action(message: _ApprovalAction) -> None:
        if message.acknowledged is not None and not message.acknowledged.done():
            message.acknowledged.set_result(None)
