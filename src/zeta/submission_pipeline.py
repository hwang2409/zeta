"""The single-consumer pipeline for submitted composer values."""

from __future__ import annotations

import asyncio
from collections import deque
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
        persist_user_message: bool = True,
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
    def _record_macro_receipt(self, receipt: str) -> None: ...


class _Message(Protocol):
    pass


@dataclass(frozen=True, slots=True)
class _Submit:
    submission: Submission
    acknowledged: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _Retry:
    submission: Submission
    user_message: Any
    acknowledged: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _PreprocessingDone:
    submission: Submission
    model_input: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _CommandDone:
    submission: Submission
    receipt: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _ProviderDone:
    submission: Submission
    task: asyncio.Task[None]


@dataclass(frozen=True, slots=True)
class _DurableToolDone:
    request_id: str
    task: asyncio.Task[Any]


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
    persist_user_message: bool = True
    acknowledge_on_provider_done: bool = False
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
        self._command_order: deque[int] = deque()
        self._command_results: dict[int, str | None] = {}
        self._provider_entry: _Entry | None = None
        self._provider_task: asyncio.Task[None] | None = None
        self._durable_tasks: dict[str, asyncio.Task[Any]] = {}
        self._control_entry: _Entry | None = None
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
            (
                entry.state in self._NONTERMINAL or entry is self._provider_entry
            )
            and entry is not self._control_entry
            for entry in self._entries.values()
        ) or (
            self._control_entry is None
            and any(
                isinstance(message, (_Submit, _Retry))
                for message in self._messages._queue
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
        self._ensure_open()
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
        )
        self._send(_Submit(submission))
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
        self._ensure_open()
        submission = self._new_submission(
            text,
            draft_revision,
            attachment_paths,
            attachment_tokens,
            next_image_token,
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

    def approval_action(
        self, decision: ApprovalDecision, requested_key: str | None
    ) -> None:
        self._ensure_open()
        self._send(_ApprovalAction(decision, requested_key))

    async def approval_action_wait(
        self, decision: ApprovalDecision, requested_key: str | None
    ) -> None:
        self._ensure_open()
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

        if self._closed:
            return
        self._closed = True
        self._shutting_down = True
        for entry in self._entries.values():
            if entry.signal is not None:
                entry.signal.abort()
            if entry.child_task is not None and not entry.child_task.done():
                entry.child_task.cancel()
        for task in self._durable_tasks.values():
            if not task.done():
                task.cancel()
        if self._provider_task is not None and not self._provider_task.done():
            self._provider_task.cancel()
        tasks = [
            task
            for task in (
                *self._preprocessing_waiters.values(),
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
                elif isinstance(message, _Retry):
                    self._on_retry(message)
                elif isinstance(message, _PreprocessingDone):
                    self._on_preprocessing_done(message)
                elif isinstance(message, _CommandDone):
                    self._on_command_done(message)
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
                await self._on_approval_action(_ApprovalAction(decision, requested_key))
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
                    self._control_entry = entry
                    try:
                        slash_output = await self._host._slash_commands.dispatch_async(
                            self._host, parsed
                        )
                    finally:
                        self._control_entry = None
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
        self._track_waiter(entry, task)
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
            self._prepare_submission(entry, entry.parsed)

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
        self._host._record_prompt(entry.parsed, entry.submission.draft_revision)
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
            persist_user_message=entry.persist_user_message,
        )
        self._provider_task = task
        if not entry.acknowledge_on_provider_done:
            self._ack_entry(entry)
        task.add_done_callback(
            lambda completed, submission=entry.submission: self._send(
                _ProviderDone(submission, completed)
            )
        )

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
        self._ack_entry(self._provider_entry)
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
        if entry.state is SubmissionState.AWAITING_APPROVAL:
            entry.state = SubmissionState.PREPROCESSING

    async def _on_approval_action(self, message: _ApprovalAction) -> None:
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
        durable = (
            owner is None
            and isinstance(request.key, str)
            and self._host.loop.store.approval_states().get(request.request_id)
            == (request.tool_call, None)
        )
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
        if durable and self._host.loop.prepare_resume_pending_tool(
            request.request_id
        ):
            task = asyncio.create_task(
                self._host.loop.resume_pending_tool(
                    request.request_id,
                    prepared=True,
                    event_sink=self._host._handle_tool_event,
                )
            )
            self._durable_tasks[request.request_id] = task
            task.add_done_callback(
                lambda completed, request_id=request.request_id: self._send(
                    _DurableToolDone(request_id, completed)
                )
            )
        self._ack_action(message)

    def _find_request(self, requested_key: str | None) -> ApprovalRequest | None:
        requests = self.pending_approvals
        if requested_key is None:
            return requests[0] if requests else None
        return next(
            (request for request in requests if str(request.key) == requested_key),
            None,
        )

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
            self._host.loop.finalize_canceled(request_id)
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

    def _finish_denied(self, entry: _Entry) -> None:
        self._host._release_attachment_paths(entry.submission.attachment_paths)
        self._host._record_prompt(entry.parsed, entry.submission.draft_revision)
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
