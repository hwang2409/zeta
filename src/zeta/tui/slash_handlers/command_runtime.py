"""Runtime lifecycle for custom command execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from ...core.abort import AbortSignal
from ...core.commands.custom_commands import CustomCommand
from ...submission import Submission
from ...tools.exec import (
    forget_macro_display,
    register_macro_display,
    run_exec_macro,
)
from ...types import StreamEvent, StreamEventType, ToolCall, ToolResult


class CommandRuntimeMixin:
    """Run inline spans and custom exec commands for the TUI."""

    async def _run_macro_submission(
        self,
        command: CustomCommand,
        args: str,
        submission: Submission,
        abort_signal: AbortSignal,
    ) -> str:
        """Run a macro child owned by the submission pipeline."""

        call = ToolCall(
            f"macro-{uuid4().hex}",
            "exec",
            {"command": command.render_exec(args), "timeout": command.timeout},
        )
        register_macro_display(
            call.id, command=command.render(args), argv=tuple(args.split())
        )
        try:
            receipt = await self._run_exec_macro(
                command,
                call,
                abort_signal,
                submission.id,
            )
        finally:
            forget_macro_display(call.id)
        return receipt or ""

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
        receipt = await self._run_exec_macro(command, call, abort_signal, None)
        if receipt is not None:
            self._record_macro_receipt(receipt)
        return ""

    async def _run_exec_macro(
        self,
        command: CustomCommand,
        call: ToolCall,
        abort_signal: AbortSignal,
        submission_id: int | None,
    ) -> str | None:
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
                    data={
                        "macro": command.name,
                        "submission_id": submission_id,
                    },
                )
            )
            if kind == "approval_start":
                asyncio.get_running_loop().call_soon(self._present_pending_approvals)
            self._invalidate_prompt()

        def stream_sink(event: StreamEvent) -> None:
            event.data["macro"] = command.name
            event.data["submission_id"] = submission_id
            self._handle_tool_event(event)
            self._invalidate_prompt()

        try:
            result = await run_exec_macro(
                self.loop.tool_registry,
                call,
                log_path,
                stream_sink=stream_sink,
                lifecycle_sink=lifecycle_sink,
                abort_signal=abort_signal,
                background=command.background,
            )
        except asyncio.CancelledError:
            result = ToolResult(call.id, "tool execution canceled", True)
        finally:
            if self._approval_policy is not None:
                self._approval_policy.forget_ephemeral(call.id)
            forget_macro_display(call.id)
        self._handle_tool_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=result,
                data={
                    "macro": command.name,
                    "submission_id": submission_id,
                },
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
            receipt = None
        else:
            receipt = f"ran /{command.name}, {status}"
        self._loop_state = "idle"
        self._streaming = False
        return receipt

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
            description=f"/{command.name}",
        )
