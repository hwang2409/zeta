"""Composition root for the full-screen zeta terminal UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import HSplit
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from ..core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ..core.slash import SlashStatus, create_slash_registry
from ..loop import AgentLoop
from ..core.session import SessionError, SessionManager, env_home
from ..providers.anthropic import AnthropicBackend
from ..providers.anthropic import AnthropicCredentialStore
from ..providers.codex import CodexBackend
from ..providers.codex import CodexCredentialStore
from ..types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from .composer import build_key_bindings, history_for, parse_input
from .render import (
    MarkdownStream,
    format_status,
    format_thought,
    render_event,
    render_tool_progress,
)
from .theme import ACCENT, BODY, CHROME, DIM, ERROR, RICH_THEME, SURFACE, USER_ROLE
from .transcript import TranscriptWidget


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.4"
SPINNER_INTERVAL = 0.2


class FullScreenPromptSession(PromptSession[str]):
    """Prompt session that owns the alternate screen for the whole app."""

    def _create_application(
        self, editing_mode: EditingMode, erase_when_done: bool
    ) -> Application[str]:
        application = super()._create_application(editing_mode, erase_when_done)
        application.full_screen = True
        application.renderer.full_screen = True
        application.erase_when_done = False
        return application

    def restore_terminal(self) -> None:
        """Restore the shell viewport after prompt-toolkit exits."""

        self.app.output.quit_alternate_screen()
        self.app.output.show_cursor()
        self.app.output.flush()
        try:
            os.write(sys.__stdout__.fileno(), b"\x1b[?1049l\x1b[?25h")
        except OSError:
            pass


def _zeta_home() -> Path:
    return env_home()


class FakeInteractiveBackend(CompletionBackend):
    """Small streaming backend for offline CLI smoke tests."""

    def __init__(self, *, delay: float = 0.03) -> None:
        self.delay = delay
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        prompt = ""
        for message in reversed(messages):
            if message.role is MessageRole.USER:
                prompt = "".join(
                    block.text for block in message.content if isinstance(block, TextContent)
                )
                break
        response = (
            f"you said: {prompt}\n\n"
            "the fake provider is streaming this response offline.\n"
            "try queueing another message while this turn runs."
        )
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for chunk in _chunks(response, 9):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=chunk)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
            data={"usage": {"input_tokens": len(prompt), "output_tokens": len(response)}},
        )


def _chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)]


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected provider without loading network credentials for fake."""

    auth_home = Path(home) if home is not None else _zeta_home()
    if provider == "fake":
        return FakeInteractiveBackend(), model or "offline"
    if provider == "claude":
        selected_model = model or DEFAULT_CLAUDE_MODEL
        return AnthropicBackend(
            model=selected_model,
            token_store=AnthropicCredentialStore(auth_home / "anthropic-oauth.json"),
        ), selected_model
    if provider == "codex":
        selected_model = model or DEFAULT_CODEX_MODEL
        return CodexBackend(
            model=selected_model,
            token_store=CodexCredentialStore(auth_home / "codex-oauth.json"),
        ), selected_model
    raise ValueError(f"unsupported provider: {provider}")


class _LineBuffer:
    def __init__(self) -> None:
        self.value = ""

    def feed(self, value: str) -> list[str]:
        self.value += value
        lines = self.value.split("\n")
        self.value = lines.pop()
        return lines

    def flush(self) -> list[str]:
        if not self.value:
            return []
        line = self.value
        self.value = ""
        return [line]


class TUIApp:
    """Full-screen transcript, persistent composer, and follow-up queue."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        provider: str,
        model: str,
        verbose: bool = False,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
        history_path: str | Path | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.loop = loop
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.console = console or Console(theme=RICH_THEME)
        self._active_task: asyncio.Task[None] | None = None
        self._queued: deque[str] = deque()
        self._exit_requested = False
        self._loop_state = "idle"
        self._usage: dict[str, Any] = {}
        self._assistant_lines = _LineBuffer()
        self._thinking_text = ""
        self._thinking_duration: float | None = None
        self._thinking_started_at: float | None = None
        self._markdown_stream = MarkdownStream()
        self._stream_kind: str | None = None
        self._partial = ""
        self._streaming = False
        self._spinner_active = False
        self._spinner_frame = 0
        self._spinner_reset = asyncio.Event()
        self._active_tool_calls: set[str] = set()
        self._pending_tool_renders: list[RenderableType] = []
        self._tool_region: Live | None = None
        self._tool_region_text: Text | None = None
        self._tool_region_call: ToolCall | None = None
        self._abort_requested = False
        self._resuming_tool = False
        self._session = session
        self._history_path = Path(history_path) if history_path else _zeta_home() / "history"
        self._approval_policy = approval_policy
        self._slash_commands = create_slash_registry()
        self._printed_units = False
        self._assistant_unit_open = False
        self._compaction_shown = False
        self._turn_had_visible_output = False
        self._active_session: PromptSession[str] | None = None
        self._transcript = TranscriptWidget()
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def _transcript_lines(self) -> list[str]:
        """Expose rendered lines for diagnostics while keeping logical units in the widget."""

        width = get_app().output.get_size().columns
        return self._transcript.lines(width)

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(self._queued)

    @property
    def active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        if self._approval_policy is None:
            return ()
        return tuple(self._approval_policy.pending_requests())

    def slash_status(self) -> SlashStatus:
        pending = tuple(
            f"{request.request_id} ({request.tool_call.name})"
            for request in self.pending_approvals
        )
        return SlashStatus(
            session_id=self.loop.store.session_id,
            provider=self.provider,
            model=self.model,
            retained_tail=self.loop.context_assembler.retained_tail,
            tokens_used_this_session=(
                self.loop.context_assembler.tokens_used_this_session
            ),
            tokens_in_current_context=self.loop.context_assembler.token_count,
            compaction_marker_count=self.loop.store.compaction_marker_count(),
            pending_approvals=pending,
            cache_read_input_tokens=(
                self.loop.context_assembler.cache_read_input_tokens_this_session
            ),
            cache_creation_input_tokens=(
                self.loop.context_assembler.cache_creation_input_tokens_this_session
            ),
            uncached_input_tokens=(
                self.loop.context_assembler.uncached_input_tokens_this_session
            ),
            output_tokens_this_session=(
                self.loop.context_assembler.output_tokens_this_session
            ),
        )

    def _present_pending_approvals(self) -> None:
        for request in self.pending_approvals:
            arguments = json.dumps(request.tool_call.arguments, sort_keys=True)
            self._print(
                Text(
                    f"[approval pending] {request.request_id}: "
                    f"{request.tool_call.name} {arguments}; "
                    f"type approve {request.request_id} or deny {request.request_id}",
                    style="yellow",
                )
            )

    async def _handle_approval_input(self, value: str) -> bool:
        parts = value.split(maxsplit=1)
        if not parts or parts[0] not in {"approve", "deny"}:
            return False
        pending = self.pending_approvals
        if not pending:
            self._print(Text("[approval] no pending requests", style="dim"))
            return True
        if len(parts) != 2:
            self._print(Text(f"[approval] use {parts[0]} <request-id>", style="yellow"))
            return True
        request_id = parts[1].strip()
        if request_id not in {request.request_id for request in pending}:
            self._print(Text(f"[approval] unknown request: {request_id}", style="yellow"))
            return True
        resolved = (
            self._approval_policy.approve(request_id)
            if parts[0] == "approve"
            else self._approval_policy.deny(request_id)
        )
        if resolved:
            self._print(Text(f"[approval] {parts[0]}d {request_id}", style="green"))
            # If a turn is already parked in the approval poll, it will pick up
            # the resolution and execute the tool itself. Running the tool here
            # would race that path and persist a duplicate tool_result — which
            # Anthropic rejects (each tool_use must have a single result).
            if self.active:
                self._present_pending_approvals()
                return True

            async def resume() -> Any:
                return await self.loop.resume_pending_tool(
                    request_id,
                    prepared=True,
                    event_sink=self._handle_resumed_tool_event,
                )

            resume_task: asyncio.Task[Any] | None = None
            try:
                await self.loop.ensure_mcp_servers()
                if not self.loop.prepare_resume_pending_tool(request_id):
                    self._present_pending_approvals()
                    return True
                self._resuming_tool = True
                resume_task = asyncio.create_task(resume())
                self._active_task = resume_task
                await asyncio.shield(resume_task)
            except asyncio.CancelledError:
                parent_cancelled = (
                    asyncio.current_task() is not None
                    and asyncio.current_task().cancelling() > 0
                )
                self.loop.abort()
                self.loop.finalize_canceled(request_id)
                if resume_task is not None:
                    resume_task.cancel()
                if resume_task is not None:
                    await asyncio.gather(resume_task, return_exceptions=True)
                self._print(Text("[aborted]", style="yellow"))
                if parent_cancelled:
                    raise
                return True
            finally:
                self._resuming_tool = False
                if resume_task is not None and self._active_task is resume_task:
                    self._active_task = None
        self._present_pending_approvals()
        return True

    def _make_session(self) -> PromptSession[str]:
        bindings = build_key_bindings(
            on_interrupt=self.abort_active,
            on_exit=self.request_exit,
            on_submit=self._submit_input,
        )
        return FullScreenPromptSession(
            message=[("class:prompt", " ❯ ")],
            history=history_for(self._history_path),
            key_bindings=bindings,
            multiline=True,
            bottom_toolbar=self._status_toolbar,
            erase_when_done=True,
            style=Style.from_dict(
                {
                    "": f"fg:{BODY} bg:{SURFACE}",
                    "prompt": f"fg:{ACCENT} bold bg:{SURFACE}",
                    "status-bar": f"noreverse fg:{CHROME} bg:{SURFACE}",
                    "composer-info": f"noreverse fg:{CHROME} bg:{SURFACE}",
                }
            ),
        )

    def request_exit(self) -> None:
        self._exit_requested = True
        self.abort_active()

    def abort_active(self) -> None:
        if self._active_task is not None and not self._active_task.done():
            tool_running = self._loop_state == "tool-running"
            self.loop.abort()
            self._loop_state = "interrupted"
            self._invalidate_prompt()
            if tool_running and not self._resuming_tool:
                self._abort_requested = True
            else:
                self._active_task.cancel()

    def _status_toolbar(self) -> FormattedText:
        width = get_app().output.get_size().columns
        usage = dict(self._usage)
        usage.setdefault(
            "cache_read_input_tokens",
            self.loop.context_assembler.cache_read_input_tokens_this_session,
        )
        usage.setdefault(
            "cache_creation_input_tokens",
            self.loop.context_assembler.cache_creation_input_tokens_this_session,
        )
        status = format_status(
            self.provider,
            self.model,
            self._loop_state,
            usage,
            self._partial,
            session_id=self.loop.store.session_id[:8],
            token_count=self.loop.context_assembler.token_count,
            retained_tail=self.loop.context_assembler.retained_tail,
            streaming=self._streaming,
            width=width,
            spinner_frame=self._spinner_frame,
            spinner_active=self._spinner_active,
            model_window=self.loop.context_assembler.token_budget,
        )
        return FormattedText(
            [
                ("class:status-bar", status.plain),
            ]
        )

    def _full_screen_active(self) -> bool:
        return isinstance(self._active_session, FullScreenPromptSession)

    def _append_transcript(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        self._transcript.append(renderable)

    def _append_transcript_blank(self) -> None:
        self._transcript.append_blank()

    def _print(self, renderable: RenderableType | None) -> None:
        if renderable is not None:
            if self._full_screen_active():
                self._append_transcript(renderable)
            else:
                self.console.print(Padding(renderable, (0, 2, 0, 2)))

    def _print_unit(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        if self._printed_units:
            if self._full_screen_active():
                self._append_transcript_blank()
            else:
                self.console.print()
        self._print(renderable)
        self._printed_units = True

    def _print_assistant(self, renderable: RenderableType | None) -> None:
        if renderable is None:
            return
        if not self._assistant_unit_open:
            self._print_unit(renderable)
            self._assistant_unit_open = True
        else:
            self._print(renderable)
        plain = getattr(renderable, "plain", None)
        if plain is None or plain.strip():
            self._turn_had_visible_output = True

    def _update_tool_region(self, event: StreamEvent) -> None:
        rendered = render_event(event)
        if not isinstance(rendered, Text):
            return
        if self._full_screen_active():
            if event.tool_call is not None:
                self._transcript.update_tool(event.tool_call.id, rendered)
            return
        if self._tool_region is None:
            self._tool_region_text = Text()
            self._tool_region_call = event.tool_call
            self._tool_region = Live(
                Padding(
                    render_tool_progress(
                        self._tool_region_call,
                        self._tool_region_text.plain,
                    ),
                    (0, 2, 0, 2),
                )
                if self._tool_region_call is not None
                else self._tool_region_text,
                console=self.console,
                transient=True,
                refresh_per_second=20,
            )

        if self._tool_region_text is None:
            return
        if self._tool_region_text:
            self._tool_region_text.append("\n")
        self._tool_region_text.append(rendered)
        if self._tool_region_call is not None:
            self._tool_region.update(
                Padding(
                    render_tool_progress(
                        self._tool_region_call,
                        self._tool_region_text.plain,
                    ),
                    (0, 2, 0, 2),
                )
            )
        else:
            self._tool_region.update(self._tool_region_text)

    def _handle_resumed_tool_event(self, event: StreamEvent) -> None:
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            self._loop_state = "tool-running"
            if event.tool_call is not None:
                self._active_tool_calls.add(event.tool_call.id)
            rendered = render_event(event)
            if rendered is None:
                return
            if self._full_screen_active() and event.tool_call is not None:
                if self._printed_units:
                    self._append_transcript_blank()
                self._transcript.start_tool(
                    event.tool_call.id,
                    event.tool_call,
                    rendered,
                )
                self._printed_units = True
            else:
                self._print_unit(rendered)
            return
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return
        self._loop_state = "idle"
        if event.tool_call is not None:
            self._active_tool_calls.discard(event.tool_call.id)
        rendered = render_event(event)
        if rendered is None:
            return
        if self._full_screen_active() and event.tool_call is not None:
            self._transcript.finish_tool(event.tool_call.id, rendered)
        else:
            self._print(rendered)

    def _submit_input(self, value: str) -> None:
        self._input_queue.put_nowait(value)

    async def _handle_prompt_value(self, value: str) -> None:
        parsed = parse_input(value)
        if parsed is None or self._exit_requested:
            return
        if await self._handle_approval_input(parsed):
            return
        slash_output = self._slash_commands.dispatch(self, parsed)
        if slash_output is not None:
            self._print_system(slash_output)
            return
        model_input = self._slash_commands.input_for_model(parsed)
        if self.pending_approvals:
            self._present_pending_approvals()
        elif self.active:
            self._queued.append(model_input)
        else:
            self._print_user(model_input)
            self._start_turn(model_input)

    def _commit_tool_region(self) -> None:
        final_renders = self._pending_tool_renders
        self._pending_tool_renders = []
        if self._tool_region is not None:
            if final_renders:
                self._tool_region.update(Group(*final_renders))
            self._tool_region.stop()
            self._tool_region = None
            self._tool_region_text = None
            self._tool_region_call = None
        for rendered in final_renders:
            self._print(rendered)
        self._assistant_unit_open = False

    def _discard_tool_region(self) -> None:
        self._pending_tool_renders.clear()
        if self._full_screen_active():
            self._transcript.discard_tools()
        if self._tool_region is not None:
            self._tool_region.stop()
            self._tool_region = None
            self._tool_region_text = None
            self._tool_region_call = None

    def _print_committed(self, lines: list[str], *, thinking: bool = False) -> None:
        if thinking:
            value = "\n".join(lines)
            if value:
                self._print_unit(format_thought(value, self._thinking_duration))
                self._turn_had_visible_output = True
            return
        for line in lines:
            if not line:
                if self._full_screen_active():
                    self._append_transcript_blank()
                else:
                    self.console.print()
                continue
            for renderable in self._markdown_stream.consume(line):
                self._print_assistant(renderable)

    def _update_usage(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, dict):
            self._usage.update(usage)

    @staticmethod
    def _invalidate_prompt() -> None:
        get_app().invalidate()

    def _flush_stream_kind(self) -> None:
        if self._stream_kind == "thinking":
            if self._thinking_text:
                self._print_committed([self._thinking_text], thinking=True)
        elif self._stream_kind == "assistant":
            self._print_committed(self._assistant_lines.flush())
        self._stream_kind = None
        self._partial = ""
        self._thinking_text = ""
        self._thinking_duration = None
        self._thinking_started_at = None

    def _flush_markdown(self) -> None:
        for renderable in self._markdown_stream.flush():
            self._print_assistant(renderable)

    def _flush_pending_stream(self) -> None:
        self._flush_stream_kind()
        self._flush_markdown()
        self._assistant_unit_open = False

    def _consume_text(self, event: StreamEvent) -> None:
        value = event.delta
        thinking = False
        if isinstance(event.content, TextContent):
            value = event.content.text
        elif isinstance(event.content, ThinkingContent):
            value = event.content.text
            thinking = True
        if not value:
            return
        self._streaming = True
        stream_kind = "thinking" if thinking else "assistant"
        if self._stream_kind is not None and self._stream_kind != stream_kind:
            self._flush_pending_stream()
        self._stream_kind = stream_kind
        if thinking:
            if self._thinking_started_at is None:
                self._thinking_started_at = time.monotonic()
            self._thinking_text += value
            self._thinking_duration = max(
                0.0, time.monotonic() - self._thinking_started_at
            )
            self._partial = self._thinking_text
            return
        buffer = self._assistant_lines
        committed = buffer.feed(value)
        self._partial = buffer.value
        self._print_committed(committed, thinking=thinking)

    def _finish_stream(self) -> None:
        self._flush_pending_stream()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._reset_stream_buffers()
        self._partial = ""
        self._streaming = False

    def _reset_stream_buffers(self) -> None:
        self._assistant_lines.value = ""
        self._thinking_text = ""
        self._thinking_duration = None
        self._thinking_started_at = None

    def _print_user(self, user_text: str) -> None:
        self._assistant_unit_open = False
        self._print_unit(Text.assemble(("▌ ", USER_ROLE), (user_text, BODY)))

    def _print_system(self, output: str) -> None:
        self._print_unit(Text(f"system · {output}", style=CHROME))

    def _start_queued_turn(self) -> None:
        if self._queued:
            user_text = self._queued.popleft()
            self._print_user(user_text)
            self._print(Text("[queued]", style=DIM))
            self._start_turn(user_text)

    def _prepare_stream_event(self, event: StreamEvent) -> None:
        if event.type is not StreamEventType.MESSAGE_UPDATE:
            self._flush_pending_stream()
            return
        if isinstance(event.content, ThinkingContent):
            incoming_kind = "thinking"
        elif isinstance(event.content, TextContent) or event.delta is not None:
            incoming_kind = "assistant"
        else:
            incoming_kind = None
        if incoming_kind != self._stream_kind and (
            incoming_kind is not None or self._stream_kind is not None
        ):
            self._flush_pending_stream()

    async def _pulse_spinner(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._spinner_reset.wait(), timeout=SPINNER_INTERVAL
                )
            except asyncio.TimeoutError:
                if self._spinner_active:
                    self._spinner_frame += 1
                    self._invalidate_prompt()
            else:
                self._spinner_reset.clear()

    async def _consume_turn(self, user_text: str) -> None:
        self._abort_requested = False
        self._turn_had_visible_output = False
        self._loop_state = "streaming"
        self._streaming = True
        self._spinner_active = True
        spinner_task = asyncio.create_task(self._pulse_spinner())
        try:
            async for event in self.loop.run_turn(user_text):
                self._update_usage(event)
                self._prepare_stream_event(event)
                stop_after_tool = False
                if self.verbose:
                    self._print(Text(json.dumps(event.to_dict(), sort_keys=True), style=DIM))
                if event.type is StreamEventType.MESSAGE_UPDATE:
                    self._consume_text(event)
                    self._invalidate_prompt()
                    continue
                if event.type is StreamEventType.TOOL_APPROVAL_START:
                    self._reset_stream_state()
                    self._loop_state = "approval"
                    self._present_pending_approvals()
                elif event.type is StreamEventType.TOOL_APPROVAL_END:
                    self._loop_state = "streaming"
                elif event.type is StreamEventType.TOOL_EXECUTION_START:
                    self._reset_stream_state()
                    self._assistant_unit_open = False
                    self._loop_state = "tool-running"
                    if event.tool_call is not None:
                        self._active_tool_calls.add(event.tool_call.id)
                    rendered = render_event(event)
                    if self._full_screen_active() and event.tool_call is not None and rendered is not None:
                        if self._printed_units:
                            self._append_transcript_blank()
                        self._transcript.start_tool(
                            event.tool_call.id,
                            event.tool_call,
                            rendered,
                        )
                        self._printed_units = True
                    else:
                        self._print_unit(rendered)
                    if rendered is not None:
                        self._turn_had_visible_output = True
                elif event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
                    self._update_tool_region(event)
                    if event.delta and event.delta.strip():
                        self._turn_had_visible_output = True
                elif event.type is StreamEventType.TOOL_EXECUTION_END:
                    aborted = self._abort_requested
                    self._loop_state = "interrupted" if aborted else "streaming"
                    if event.tool_call is not None:
                        self._active_tool_calls.discard(event.tool_call.id)
                    rendered = render_event(event)
                    if rendered is not None:
                        if self._full_screen_active() and event.tool_call is not None:
                            self._transcript.finish_tool(event.tool_call.id, rendered)
                        else:
                            self._pending_tool_renders.append(rendered)
                        self._turn_had_visible_output = True
                    if self._abort_requested:
                        self._abort_requested = False
                        stop_after_tool = True
                        self._loop_state = "interrupted"
                    else:
                        stop_after_tool = False
                    if not self._active_tool_calls:
                        self._commit_tool_region()
                elif event.type is StreamEventType.TURN_START:
                    self._spinner_frame = 0
                    self._spinner_reset.set()
                    self._streaming = True
                    self._compaction_shown = False
                elif event.type is StreamEventType.COMPACTION_START:
                    self._compaction_shown = True
                    self._loop_state = "compacting"
                elif event.type is StreamEventType.COMPACTION_END:
                    self._loop_state = "streaming"
                elif event.type is StreamEventType.MESSAGE_END:
                    self._streaming = False
                elif event.type is StreamEventType.AGENT_END:
                    self._reset_stream_state()
                    self._loop_state = "idle"
                    if not self._turn_had_visible_output:
                        self._print_unit(Text("no response", style=CHROME))
                        self._turn_had_visible_output = True
                if event.type not in {
                    StreamEventType.TOOL_APPROVAL_START,
                    StreamEventType.TOOL_APPROVAL_END,
                    StreamEventType.TOOL_EXECUTION_UPDATE,
                    StreamEventType.TOOL_EXECUTION_END,
                    StreamEventType.TOOL_EXECUTION_START,
                }:
                    rendered = render_event(event)
                    if rendered is not None:
                        if event.type in {
                            StreamEventType.AGENT_END,
                            StreamEventType.ERROR,
                        }:
                            self._print_unit(rendered)
                            if event.type is StreamEventType.ERROR:
                                self._turn_had_visible_output = True
                        else:
                            self._print(rendered)
                self._invalidate_prompt()
                if event.type is StreamEventType.TOOL_EXECUTION_END and stop_after_tool:
                    break
        except asyncio.CancelledError:
            self._finish_stream()
            self._loop_state = "interrupted"
            self._print_unit(Text("[aborted]", style=ERROR))
            raise
        except Exception as exc:
            self._finish_stream()
            self._loop_state = "idle"
            self._print_unit(Text(f"[error] {exc}", style=ERROR))
        finally:
            self._active_tool_calls.clear()
            self._discard_tool_region()
            self._abort_requested = False
            self._streaming = False
            self._spinner_active = False
            spinner_task.cancel()
            await asyncio.gather(spinner_task, return_exceptions=True)
            self._invalidate_prompt()

    async def _read_prompt(self, session: PromptSession[str]) -> str | None:
        try:
            value = await session.prompt_async(
                [("class:prompt", " ❯ ")],
                bottom_toolbar=self._status_toolbar,
            )
        except EOFError:
            return None
        return value

    async def _run_full_screen(self, session: FullScreenPromptSession) -> None:
        prompt_task = asyncio.create_task(session.app.run_async())
        input_task: asyncio.Task[str | None] = asyncio.create_task(
            self._input_queue.get()
        )
        try:
            while not self._exit_requested:
                wait_for: set[asyncio.Task[Any]] = {prompt_task, input_task}
                if self._active_task is not None:
                    wait_for.add(self._active_task)
                done, _ = await asyncio.wait(
                    wait_for, return_when=asyncio.FIRST_COMPLETED
                )
                if self._active_task is not None and self._active_task in done:
                    try:
                        await self._active_task
                    except asyncio.CancelledError:
                        pass
                    self._active_task = None
                    if self._queued:
                        self._start_queued_turn()
                if prompt_task in done:
                    try:
                        await prompt_task
                    except (EOFError, asyncio.CancelledError):
                        pass
                    break
                if input_task in done:
                    value = await input_task
                    if value is None:
                        break
                    await self._handle_prompt_value(value)
                    input_task = asyncio.create_task(self._input_queue.get())
        finally:
            if not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)
            if not input_task.done():
                input_task.cancel()
                await asyncio.gather(input_task, return_exceptions=True)

    def _install_full_screen_layout(self, session: FullScreenPromptSession) -> None:
        root = session.layout.container
        composer_rows = list(root.children)
        footer = composer_rows.pop()
        transcript = self._transcript.window()
        root.children[:] = [
            transcript,
            HSplit(
                [*composer_rows, footer],
                height=Dimension(min=2, max=8),
            ),
        ]

    async def run(self, session: PromptSession[str] | None = None) -> None:
        """Run the alternate-screen app until Ctrl-D or an exit request."""

        session = session or self._session or self._make_session()
        self._active_session = session
        if isinstance(session, FullScreenPromptSession):
            self._install_full_screen_layout(session)
        self._present_pending_approvals()
        prompt_task: asyncio.Task[str | None] | None = None
        try:
            if isinstance(session, FullScreenPromptSession):
                await self._run_full_screen(session)
                return
            prompt_task = asyncio.create_task(self._read_prompt(session))
            while prompt_task is not None and not self._exit_requested:
                wait_for: set[asyncio.Task[Any]] = {prompt_task}
                if self._active_task is not None:
                    wait_for.add(self._active_task)
                done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                if self._active_task is not None and self._active_task in done:
                    try:
                        await self._active_task
                    except asyncio.CancelledError:
                        pass
                    self._active_task = None
                    if self._queued:
                        self._start_queued_turn()

                if prompt_task in done:
                    value = await prompt_task
                    if value is None:
                        break
                    await self._handle_prompt_value(value)
                    if self._exit_requested:
                        break
                    prompt_task = asyncio.create_task(self._read_prompt(session))
        finally:
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)
            if self._active_task is not None and not self._active_task.done():
                self._active_task.cancel()
                await asyncio.gather(self._active_task, return_exceptions=True)
            if isinstance(session, FullScreenPromptSession):
                session.restore_terminal()
            await self.loop.close()
            self._active_session = None

    def _start_turn(self, user_text: str) -> None:
        self._active_task = asyncio.create_task(self._consume_turn(user_text))


def create_app(args: argparse.Namespace) -> TUIApp:
    home = _zeta_home()
    manager = SessionManager(home)
    continue_session = getattr(args, "continue_session", False)
    resume_id = getattr(args, "resume", None)
    force_provider = getattr(args, "force_provider", False)
    resuming = continue_session or resume_id is not None
    if force_provider and not resuming:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and args.model is None:
        raise SessionError("--force-provider requires --model")

    if resuming:
        if resume_id is not None:
            opened = manager.open(resume_id)
        else:
            recent = manager.find_most_recent(cwd=Path.cwd())
            opened = manager.open(recent.session_id)
        metadata = opened.metadata
        provider_override = args.provider
        model_override = args.model
        mismatches = []
        if provider_override is not None and provider_override != metadata.provider:
            mismatches.append(
                f"provider {provider_override!r} does not match {metadata.provider!r}"
            )
        if model_override is not None and model_override != metadata.model:
            mismatches.append(
                f"model {model_override!r} does not match {metadata.model!r}"
            )
        if mismatches and not force_provider:
            raise SessionError(
                f"session override rejected: {'; '.join(mismatches)}; "
                "use --force-provider to override"
            )
        provider = provider_override or metadata.provider
        model = model_override or metadata.model
        store = opened.store
    else:
        provider = args.provider or "fake"
        backend, selected_model = build_backend(provider, args.model, home=home)
        opened = manager.create(
            provider=provider,
            model=selected_model,
            cwd=Path.cwd(),
        )
        metadata = opened.metadata
        store = opened.store
    if resuming:
        backend, selected_model = build_backend(provider, model, home=home)
    approval_default = (
        ApprovalDecision.ALLOW if getattr(args, "yolo", False) else ApprovalDecision.ASK
    )
    approval_policy = ApprovalPolicy(store=store, default=approval_default)
    pending_override = None
    if resuming and mismatches:
        pending_override = (
            provider if provider != metadata.provider else None,
            model if model != metadata.model else None,
        )

    def completion_success() -> None:
        nonlocal pending_override
        if pending_override is not None:
            manager.record_override(
                metadata,
                provider=pending_override[0],
                model=pending_override[1],
            )
            pending_override = None
            return
        manager.touch(metadata)

    token_budget_override = getattr(args, "token_budget", None)
    effective_token_budget = (
        token_budget_override
        if token_budget_override is not None and token_budget_override > 0
        else max(metadata.compaction_budget, 200_000)
    )
    max_turns_override = getattr(args, "max_turns", None)
    loop_kwargs: dict[str, Any] = {
        "approval_policy": approval_policy,
        "token_budget": effective_token_budget,
        "retained_tail": metadata.retained_tail,
        "on_completion_success": completion_success,
    }
    if max_turns_override is not None and max_turns_override > 0:
        loop_kwargs["max_turns"] = max_turns_override
    loop = AgentLoop(backend, store, **loop_kwargs)
    return TUIApp(
        loop,
        provider=provider,
        model=selected_model,
        verbose=args.verbose,
        history_path=home / "history",
        approval_policy=approval_policy,
    )


__all__ = [
    "FakeInteractiveBackend",
    "TUIApp",
    "create_app",
    "build_backend",
    "main",
]


def __getattr__(name: str) -> object:
    if name == "main":
        from ..cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
