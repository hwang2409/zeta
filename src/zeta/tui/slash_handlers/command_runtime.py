"""Runtime lifecycle for custom command execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from ...core.commands.custom_commands import CustomCommand, InlineShellResult
from ...tools.exec import (
    forget_macro_display,
    register_macro_display,
    run_exec_macro,
    run_inline_shell_batch,
)
from ...types import StreamEvent, StreamEventType, ToolCall, ToolResult


class CommandRuntimeMixin:
    """Run inline spans and custom exec commands for the TUI."""

    async def _resolve_inline_shell(
        self, submission_id: int, commands: tuple[str, ...]
    ) -> InlineShellResult:
        signal = self._inline_abort_signals.get(submission_id)
        if signal is None:
            signal = self.loop.tool_registry.abort_signal.registry.new_generation()
            self._inline_abort_signals[submission_id] = signal
        def lifecycle_sink(kind: str, call: ToolCall) -> None:
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
            }.get(kind)
            if event_type is None:
                return
            self._handle_tool_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    data={"inline_shell": True},
                )
            )
            if kind == "approval_start":
                self._approval_owners[call.id] = submission_id
                asyncio.get_running_loop().call_soon(self._present_pending_approvals)
            elif kind == "approval_end":
                self._approval_owners.pop(call.id, None)
            self._invalidate_prompt()

        try:
            outputs = await run_inline_shell_batch(
                self.loop.tool_registry,
                commands,
                lifecycle_sink=lifecycle_sink,
                abort_signal=signal,
            )
            return InlineShellResult(outputs, canceled=signal.is_set())
        finally:
            self._inline_abort_signals.pop(submission_id, None)

    async def slash_exec_macro(self, command: CustomCommand, args: str) -> str:
        """Run one custom shell macro through the normal exec safety path."""

        if self.active:
            return "macro unavailable while a turn is running"
        call = ToolCall(
            f"macro-{uuid4().hex}",
            "exec",
            {
                "command": command.render_exec(args),
                "timeout": command.timeout,
            },
        )
        register_macro_display(
            call.id,
            command=command.render(args),
            argv=tuple(args.split()),
        )
        abort_signal = self.loop.tool_registry.abort_signal.registry.new_generation()
        self._macro_abort_signal = abort_signal
        self._macro_call_id = call.id
        task = asyncio.create_task(self._run_exec_macro(command, call))
        self._active_task = task
        self._loop_state = "tool-running"
        if not self._input_loop_active:
            try:
                await asyncio.shield(task)
            finally:
                if self._active_task is task:
                    self._active_task = None
        return ""

    def _abort_macro(self) -> None:
        signal = self._macro_abort_signal
        if signal is None:
            return
        signal.abort()
        for request in self.pending_approvals:
            if request.tool_call.id == self._macro_call_id:
                self._abort_approval(request.key)

    async def _run_exec_macro(self, command: CustomCommand, call: ToolCall) -> None:
        log_path = self.loop.store.session_dir / f"macro-{call.id[6:]}.log"

        def lifecycle_sink(kind: str) -> None:
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is None:
                return
            self._handle_tool_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    data={"macro": command.name},
                )
            )
            if kind == "approval_start":
                asyncio.get_running_loop().call_soon(self._present_pending_approvals)
            self._invalidate_prompt()

        def stream_sink(event: StreamEvent) -> None:
            event.data["macro"] = command.name
            self._handle_tool_event(event)
            self._invalidate_prompt()

        canceled = False
        try:
            result = await run_exec_macro(
                self.loop.tool_registry,
                call,
                log_path,
                stream_sink=stream_sink,
                lifecycle_sink=lifecycle_sink,
                abort_signal=self._macro_abort_signal,
                background=command.background,
            )
        except asyncio.CancelledError:
            result = ToolResult(call.id, "tool execution canceled", True)
            canceled = True
        finally:
            if self._approval_policy is not None:
                self._approval_policy.forget_ephemeral(call.id)
            forget_macro_display(call.id)
        self._handle_tool_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=result,
                data={"macro": command.name},
            )
        )
        structured = result.structured_content or {}
        if result.content == "tool execution canceled":
            status = "canceled"
        elif result.content == "tool execution denied":
            status = "denied"
        elif structured.get("status") == "running":
            status = "running"
        elif structured.get("timed_out") is True:
            status = "timeout"
        else:
            exit_code = structured.get("exit_code")
            status = f"exit {exit_code}" if exit_code is not None else "failed"
        if status == "running":
            task_id = structured.get("task_id")
            if isinstance(task_id, str):
                self._watch_background_macro(command, call, task_id, log_path)
        else:
            self._macro_receipts.append(f"ran /{command.name}, {status}")
        if self._macro_call_id == call.id:
            self._macro_abort_signal = None
            self._macro_call_id = None
        self._loop_state = "idle"
        self._streaming = False
        if canceled:
            raise asyncio.CancelledError

    def _watch_background_macro(
        self,
        command: CustomCommand,
        call: ToolCall,
        task_id: str,
        log_path: Path,
    ) -> None:
        instance_id = f"macro:{call.id}"

        async def watch() -> None:
            try:
                status_data = await self.loop.tool_registry.background_tasks.wait(task_id)
                note = status_data.get("note")
                exit_code = status_data.get("exit_code")
                if isinstance(note, str) and note.startswith("task killed"):
                    status = "canceled"
                    text = f"background macro /{command.name} canceled"
                elif exit_code == 0:
                    status = "completed"
                    text = f"background macro /{command.name} completed"
                else:
                    status = "error"
                    text = f"background macro /{command.name} failed with exit {exit_code}"
                self.loop.store.append_agent_notification(
                    instance_id,
                    child_session_path=str(log_path),
                    description=f"/{command.name}",
                    status=status,
                    text=f"{text}; log {log_path}",
                )
            finally:
                self.loop._background_owner.unregister(instance_id)

        watcher = asyncio.create_task(watch())
        self.loop._background_owner.register(
            instance_id,
            lambda: asyncio.create_task(
                self.loop.tool_registry.background_tasks.kill(task_id)
            ),
            watcher,
        )
