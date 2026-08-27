"""Composition root for the full-screen zeta terminal UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import get_terminal_size
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.styles import DynamicStyle, Style
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.text import Text

from ..core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ..core.hooks import load_hooks_for_provider
from ..core.project_context import (
    ProjectContext,
    discover_repo_root,
    load_project_context,
)
from ..core.session import SessionError, SessionManager, env_home
from ..core.slash import (
    MODEL_CONTEXT_WINDOWS, SlashStatus, UsageTracker, compaction_history, create_slash_registry
)
from ..core.todo import todo_count_tuple
from ..loop import AgentLoop
from ..providers.anthropic import AnthropicBackend, AnthropicCredentialStore
from ..providers.codex import CodexBackend, CodexCredentialStore
from ..types import (
    CompletionBackend,
    Message,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    assistant_text,
)
from .background import background_notice
from .checkpoints import CheckpointTranscriptMixin
from .composer import (
    ComposerAttachmentMixin,
    TurnConsumerMixin,
    VimCursorShapeConfig,
    build_key_bindings,
    history_for,
    parse_input,
    status_formatted_text,
    vim_state_label,
)
from .fake_backend import FakeInteractiveBackend
from .layout import (
    CONTENT_MARGIN,
    content_width,
    full_screen_content,
    resume_picker_line,
)
from .models import MODEL_CATALOGS, validate_model_name
from .models import load_model_catalog as _load_model_catalog
from .render import (
    format_status,
    render_markdown,
    render_thought,
    render_thought_live,
)
from .stream import stream_key
from .theme import (
    ACCENT,
    BODY,
    CHROME,
    COMPOSER_BORDER,
    COMPOSER_FOCUS,
    DIM,
    RICH_THEME,
)
from .todo import TodoWidget
from .transcript import TranscriptPresenter, TranscriptWidget

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.4"
RECENT_SESSION_LIMIT = 20
def _validate_model_name(provider: str, model: str) -> None:
    validate_model_name(provider, model)


class FullScreenPromptSession(PromptSession[str]):
    """Prompt session that owns the alternate screen for the whole app."""

    def _create_application(
        self, editing_mode: EditingMode, erase_when_done: bool
    ) -> Application[str]:
        application = super()._create_application(editing_mode, erase_when_done)
        application.ttimeoutlen, application.timeoutlen, application.cursor = 0.02, 0.5, VimCursorShapeConfig()
        application.full_screen, application.renderer.full_screen, application.erase_when_done = True, True, False
        return application

    def restore_terminal(self) -> None:
        """Restore the shell viewport after prompt-toolkit exits."""

        self.app.output.quit_alternate_screen()
        self.app.output.show_cursor()
        self.app.output.flush()
        stdout = sys.__stdout__
        if stdout.isatty():
            try:
                os.write(stdout.fileno(), b"\x1b[?1049l\x1b[?25h")
            except OSError:
                pass


def _zeta_home() -> Path:
    return env_home()


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected provider without loading network credentials for fake."""

    auth_home = Path(home) if home is not None else _zeta_home()
    if provider == "fake":
        selected_model = model or "offline"
        return FakeInteractiveBackend(model=selected_model), selected_model
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


class TUIApp(TurnConsumerMixin, CheckpointTranscriptMixin, ComposerAttachmentMixin):
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
        context_files: Sequence[str] = (),
        on_model_change: Callable[[str], None] | None = None,
        vim_mode: bool = True,
        on_vim_mode_change: Callable[[bool], None] | None = None,
        model_catalog_loader: Callable[[str], frozenset[str] | None] | None = None,
    ) -> None:
        self.loop = loop
        self.loop.tool_registry.background_tasks.set_notice_sink(
            lambda message: background_notice(self, message)
        )
        self._hooks = loop.hooks
        if self._hooks is not None:
            self._hooks.notice_sink = self._print_hook_notice
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.console = console or Console(theme=RICH_THEME)
        self._active_task: asyncio.Task[None] | None = None
        self._queued: deque[Message] = deque()
        self._pending_attachments: list[Path] = []
        self._pending_attachment_tokens: dict[str, Path] = {}
        self._next_image_token = 1
        self._composer_insertions: list[str] = []
        self._exit_requested = False
        self._loop_state = "idle"
        self._usage: dict[str, Any] = {}
        self._usage_tracker = UsageTracker(self.loop.context_assembler)
        self._assistant_text = ""
        self._thinking_text = ""
        self._thinking_duration: float | None = None
        self._thinking_started_at: float | None = None
        self._stream_kind: str | None = None
        self._stream_identity: tuple[str, object] | None = None
        self._partial = ""
        self._streaming = False
        self._spinner_active = False
        self._spinner_frame = 0
        self._spinner_reset = asyncio.Event()
        self._abort_requested = False
        self._resuming_tool = False
        self._session = session
        self._history_path = Path(history_path) if history_path else _zeta_home() / "history"
        self._approval_policy = approval_policy
        self._context_files = tuple(context_files)
        self._on_model_change = on_model_change
        self.vim_mode = vim_mode
        self._on_vim_mode_change = on_vim_mode_change
        self._model_catalog_loader = model_catalog_loader or _load_model_catalog
        self._model_catalog: frozenset[str] | None = MODEL_CATALOGS.get(provider)
        self._model_catalog_loaded = self._model_catalog is not None
        self._model_catalog_task: asyncio.Task[None] | None = None
        self._slash_commands = create_slash_registry()
        self._compaction_shown = False
        self._turn_had_visible_output = False
        self._failed_turn: tuple[str, Message] | None = None
        self._active_session: PromptSession[str] | None = None
        self._prompt_styles: dict[bool, Style] = {}
        self._transcript = TranscriptWidget()
        self._todo_widget = TodoWidget(self.loop.store)
        self._presenter = TranscriptPresenter(
            self._transcript,
            self.console,
            self._full_screen_active,
            lambda renderable: self._print(renderable),
        )
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._fork_rebuilt = False
    @property
    def _transcript_lines(self) -> list[str]:
        """Expose rendered lines for diagnostics while keeping logical units in the widget."""
        width = content_width(get_app().output.get_size().columns)
        return self._transcript.lines(width)

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(
            block.text
            for message in self._queued
            for block in message.content[:1]
            if isinstance(block, TextContent)
        )

    @property
    def active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def retry_available(self) -> bool:
        """Return whether the last failed turn can be retried."""

        return self._failed_turn is not None and not self.active

    def retry_failed_turn(self) -> None:
        """Retry the last failed user message without appending it again."""

        if not self.retry_available():
            return
        failed_turn = self._failed_turn
        self._failed_turn = None
        if failed_turn is None:
            return
        user_text, user_message = failed_turn
        self._start_turn(
            user_text,
            user_message=user_message,
            persist_user_message=False,
        )

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        if self._approval_policy is None:
            return ()
        return tuple(self._approval_policy.pending_requests())

    def slash_status(self) -> SlashStatus:
        context_assembler = self.loop.context_assembler
        pending = tuple(
            f"{request.key} ({request.label or request.tool_call.name})"
            for request in self.pending_approvals
        )
        items = self.loop.store.todo_items()
        compaction_history_data = compaction_history(
            self.loop.store.replay(), context_assembler.token_counter
        )
        return SlashStatus(
            session_id=self.loop.store.session_id,
            provider=self.provider,
            model=self.model,
            retained_tail=context_assembler.retained_tail,
            tokens_used_this_session=context_assembler.tokens_used_this_session,
            tokens_in_current_context=context_assembler.token_count,
            compaction_marker_count=self.loop.store.compaction_marker_count(),
            pending_approvals=pending,
            checkpoint_count=self.loop.store.checkpoint_count(),
            cache_read_input_tokens=context_assembler.cache_read_input_tokens_this_session,
            cache_creation_input_tokens=context_assembler.cache_creation_input_tokens_this_session,
            uncached_input_tokens=context_assembler.uncached_input_tokens_this_session,
            output_tokens_this_session=context_assembler.output_tokens_this_session,
            context_files=self._context_files,
            vim_mode=self.vim_mode,
            hooks=(() if self._hooks is None else self._hooks.status_entries),
            todo_counts=todo_count_tuple(items) if items else None,
            usage_history=self._usage_tracker.history,
            usage_cost_by_model=self._usage_tracker.cost_by_model,
            compaction_history=compaction_history_data,
            model_window=MODEL_CONTEXT_WINDOWS.get(self.provider, {}).get(self.model),
        )

    def slash_model(self, args: str) -> str:
        """Show or change the model for future completions."""

        if not args:
            return f"model: {self.model}"
        if self.active or self.pending_approvals:
            return "model unchanged: cannot change model while a turn or approval is active"
        model = args.strip()
        try:
            _validate_model_name(self.provider, model)
        except ValueError as exc:
            return f"model unchanged: {exc}"
        if not self._model_catalog_loaded:
            self._start_model_catalog_load()
        if self._model_catalog is None:
            catalog_warning = f"model catalog unavailable for {self.provider} — using anyway"
        elif model not in self._model_catalog:
            catalog_warning = (
                f"model not found in {self.provider} catalog — using anyway"
            )
        else:
            catalog_warning = None
        previous = self.model
        try:
            self.loop.set_model(model)
            if self._on_model_change is not None:
                self._on_model_change(model)
        except Exception as exc:
            self.loop.set_model(previous)
            return f"model unchanged: {exc}"
        self.model = model
        if catalog_warning is not None:
            return f"model: {model} ({catalog_warning})"
        return f"model: {model}"
    def slash_vim(self, args: str) -> str:
        requested = args.strip().lower()
        if not args:
            return f"vim mode: {'on' if self.vim_mode else 'off'}"
        if requested not in {"on", "off", "toggle"}:
            return "vim mode unchanged: use /vim on, /vim off, or /vim toggle"
        enabled = not self.vim_mode if requested == "toggle" else requested == "on"
        was_enabled = self.vim_mode
        if self._on_vim_mode_change is not None:
            self._on_vim_mode_change(enabled)
        self.vim_mode = enabled
        if (session := self._active_session or self._session) is not None:
            if was_enabled and not enabled:
                session.app.output.reset_cursor_shape()
                session.app.output.flush()
            session.editing_mode = EditingMode.VI if enabled else EditingMode.EMACS
            session.app.vi_state.reset()
        self._invalidate_prompt()
        return f"vim mode: {'on' if self.vim_mode else 'off'}"

    def _start_model_catalog_load(self) -> None:
        if self._model_catalog_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._model_catalog_task = loop.create_task(
            self._load_model_catalog_in_background()
        )

    async def _load_model_catalog_in_background(self) -> None:
        try:
            self._model_catalog = await asyncio.to_thread(
                self._model_catalog_loader, self.provider
            )
        except Exception:
            self._model_catalog = None
        finally:
            self._model_catalog_loaded = True
            self._model_catalog_task = None

    def _present_pending_approvals(self) -> None:
        for request in self.pending_approvals:
            arguments = json.dumps(request.tool_call.arguments, sort_keys=True)
            key = str(request.key)
            self._print(
                Text(
                    f"[approval pending] {request.label or request.tool_call.name} "
                    f"[{key}]: "
                    f"{arguments}; type approve {key} or deny {key}",
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
            self._print(Text(f"[approval] use {parts[0]} <approval-key>", style="yellow"))
            return True
        requested_key = parts[1].strip()
        request = next(
            (request for request in pending if str(request.key) == requested_key),
            None,
        )
        if request is None:
            self._print(
                Text(f"[approval] unknown request: {requested_key}", style="yellow")
            )
            return True
        key = request.key
        request_id = request.request_id
        resolved = (
            self._approval_policy.approve(key)
            if parts[0] == "approve"
            else self._approval_policy.deny(key)
        )
        if resolved:
            self._print(Text(f"[approval] {parts[0]}d {key}", style="green"))
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
                self._abort_approval(key)
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
    def _prompt_style(self) -> Style:
        focused = get_app().current_buffer.name == "DEFAULT_BUFFER"
        style = self._prompt_styles.get(focused)
        if style is None:
            style = Style.from_dict(
                {
                    "": f"fg:{BODY}",
                    "prompt": f"fg:{ACCENT} bold",
                    "placeholder": f"italic fg:{DIM}",
                    "status-bar": f"noreverse fg:{CHROME}",
                    "frame": "",
                    "frame.border": (
                        f"fg:{COMPOSER_FOCUS}"
                        if focused
                        else f"fg:{COMPOSER_BORDER}"
                    ),
                    "text-area": f"fg:{BODY}",
                    "text-area.prompt": f"fg:{ACCENT} bold",
                }
            )
            self._prompt_styles[focused] = style
        return style

    def _make_session(self) -> PromptSession[str]:
        bindings = build_key_bindings(
            on_interrupt=self.abort_active,
            on_exit=self.request_exit,
            on_submit=self._submit_input,
            on_paste=self._paste_from_keybinding,
            on_page_up=self._transcript.page_up,
            on_page_down=self._transcript.page_down,
            on_toggle_agent=self._transcript.toggle_latest_agent,
            on_retry=self.retry_failed_turn,
            retry_available=self.retry_available,
        )
        return FullScreenPromptSession(
            message=[("class:prompt", " > ")],
            placeholder=[("class:placeholder", "type a message...")],
            history=history_for(self._history_path),
            key_bindings=bindings,
            multiline=True,
            editing_mode=EditingMode.VI if self.vim_mode else EditingMode.EMACS,
            bottom_toolbar=self._status_toolbar,
            erase_when_done=True,
            show_frame=True,
            style=DynamicStyle(self._prompt_style),
        )

    def request_exit(self) -> None:
        self._exit_requested = True
        self.abort_active()

    def _insert_paste_token(self, token: str) -> None:
        if isinstance(self._active_session, FullScreenPromptSession):
            self._active_session.app.current_buffer.insert_text(token)
        else:
            self._composer_insertions.append(token)

    def _paste_from_keybinding(self, event: KeyPressEvent | None = None) -> None:
        result = self.slash_paste("")
        if result.startswith("[Image #"):
            if event is None:
                self._insert_paste_token(result)
            else:
                event.current_buffer.insert_text(result)
        elif result != "paste unavailable: clipboard does not contain an image":
            self._print_system(result)

    def abort_active(self) -> None:
        if self._active_task is not None and not self._active_task.done():
            tool_running = self._loop_state == "tool-running"
            self.loop.abort()
            for request in self.pending_approvals:
                self._abort_approval(request.key)
                self.loop.finalize_canceled(request.request_id)
            self._loop_state = "interrupted"
            self._invalidate_prompt()
            if tool_running and not self._resuming_tool:
                self._abort_requested = True
            else:
                self._active_task.cancel()

    def _abort_approval(self, request_id: str | tuple[str, str]) -> None:
        if self._approval_policy is not None:
            self._approval_policy.abort(request_id)

    def _status_toolbar(self) -> FormattedText:
        terminal_width = get_app().output.get_size().columns
        width = content_width(terminal_width)
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
            vim_state=vim_state_label(self.vim_mode),
            background_count=self.loop.tool_registry.background_tasks.running_count,
        )
        fragments = status_formatted_text(status)
        return fragments

    def _full_screen_active(self) -> bool:
        return isinstance(self._active_session, FullScreenPromptSession)

    def _append_transcript(self, renderable: RenderableType | None) -> None:
        if renderable is not None:
            self._transcript.append(renderable)

    def _print(self, renderable: RenderableType | None) -> None:
        if renderable is not None:
            if self._full_screen_active():
                self._append_transcript(renderable)
            else:
                self.console.print(Padding(renderable, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN)))

    def _print_unit(self, renderable: RenderableType | None) -> None:
        self._presenter.print_unit(renderable)

    def _handle_tool_event(self, event: StreamEvent) -> bool:
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            self._reset_stream_state()
            self._loop_state = "approval"
            self._present_pending_approvals()
            return False
        if event.type is StreamEventType.TOOL_APPROVAL_END:
            self._loop_state = "streaming"
            return False
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            self._reset_stream_state()
            self._loop_state = "tool-running"
            presentation = self._presenter.handle_tool_event(
                event,
                aborted=False,
            )
            if presentation is not None and presentation.visible_output:
                self._turn_had_visible_output = True
            return False
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            presentation = self._presenter.handle_tool_event(
                event,
                aborted=False,
            )
            if presentation is not None and presentation.visible_output:
                self._turn_had_visible_output = True
            return False
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return False

        aborted = self._abort_requested or self._loop_state == "interrupted"
        self._loop_state = "interrupted" if aborted else (
            "idle" if self._resuming_tool else "streaming"
        )
        presentation = self._presenter.handle_tool_event(
            event,
            aborted=aborted,
        )
        if presentation is not None and presentation.visible_output:
            self._turn_had_visible_output = True
        stop_after_tool = (
            presentation is not None and presentation.stop_after_tool
        )
        self._abort_requested = False
        return stop_after_tool

    def _handle_resumed_tool_event(self, event: StreamEvent) -> None:
        self._handle_tool_event(event)
        self._invalidate_prompt()

    def _submit_input(self, value: str) -> None:
        self._input_queue.put_nowait(value)

    async def _handle_prompt_value(self, value: str) -> None:
        parsed = parse_input(value)
        if parsed is None or self._exit_requested:
            return
        if await self._handle_approval_input(parsed):
            return
        slash_output = await self._slash_commands.dispatch_async(self, parsed)
        if slash_output is not None:
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
            return
        self._failed_turn = None
        if self.pending_approvals:
            self._present_pending_approvals()
        elif self.active:
            self._clear_pending_attachments()
            self._queued.append(user_message)
        else:
            self._clear_pending_attachments()
            self._print_user(user_message)
            self._start_turn(model_input, user_message=user_message)

    def _discard_tool_region(self) -> None:
        self._presenter.discard_tool_region()

    @property
    def _tool_region(self):
        return self._presenter.tool_region

    def _print_committed(self, lines: list[str], *, thinking: bool = False) -> None:
        value = "\n".join(lines)
        if thinking:
            if value:
                self._presenter.finish_thinking(
                    render_thought(value, self._thinking_duration)
                )
                self._turn_had_visible_output = True
            return
        if value:
            self._assistant_text += value
            self._presenter.update_assistant(Text(self._assistant_text, style=BODY))
            self._turn_had_visible_output |= bool(value.strip())

    def _update_usage(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, dict):
            self._usage.update(usage)

    @staticmethod
    def _invalidate_prompt() -> None:
        get_app().invalidate()

    def _flush_stream_kind(self, *, preserve_inline: bool = False) -> None:
        if self._stream_kind in {"thinking", "redacted-thinking"} and self._thinking_text:
            self._print_committed([self._thinking_text], thinking=True)
        elif self._stream_kind == "assistant" and self._assistant_text:
            self._presenter.finish_assistant(
                Text(self._assistant_text, style=BODY),
                preserve_inline=preserve_inline,
            )
        self._stream_kind = self._stream_identity = None
        self._partial = self._thinking_text = ""
        self._thinking_duration = self._thinking_started_at = None

    def _flush_markdown(self) -> None:
        if self._assistant_text:
            self._presenter.finish_assistant(render_markdown(self._assistant_text))

    def _finish_message(self, event: StreamEvent) -> None:
        if self._stream_kind in {"thinking", "redacted-thinking"}:
            self._flush_stream_kind()
        value = (
            assistant_text(event.message)
            if event.message is not None
            else self._assistant_text
        )
        self._presenter.finish_assistant_message(render_markdown(value) if value else None)
        if value: self._turn_had_visible_output |= bool(value.strip())
        self._stream_kind = self._stream_identity = None
        self._reset_stream_buffers()

    def _flush_pending_stream(self) -> None:
        self._flush_stream_kind(); self._presenter.reset_assistant_unit()

    def _consume_text(self, event: StreamEvent) -> None:
        incoming_kind, incoming_identity = stream_key(event)
        if self._stream_kind is not None and (
            incoming_kind, incoming_identity
        ) != (self._stream_kind, self._stream_identity):
            self._flush_pending_stream()
        redacted = incoming_kind == "redacted-thinking"
        thinking = redacted
        value = "redacted" if thinking else event.delta
        if isinstance(event.content, TextContent):
            value = event.content.text
        elif isinstance(event.content, ThinkingContent):
            value = event.content.text
            thinking = True
        if not value:
            return
        self._streaming = True
        stream_kind = incoming_kind
        assert stream_kind is not None
        self._stream_kind = stream_kind
        self._stream_identity = incoming_identity
        if thinking:
            if self._thinking_started_at is None:
                self._thinking_started_at = time.monotonic()
                self._presenter.start_thinking(render_thought_live(value))
            self._thinking_text += value
            self._thinking_duration = max(
                0.0, time.monotonic() - self._thinking_started_at
            )
            self._partial = self._thinking_text
            self._presenter.update_thinking(render_thought_live(self._thinking_text))
            return
        self._assistant_text += value
        self._partial = self._assistant_text
        self._presenter.update_assistant(Text(self._assistant_text, style=BODY))

    def _reset_stream_state(self) -> None:
        self._stream_kind = self._stream_identity = None
        self._partial = self._thinking_text = ""
        self._thinking_duration = self._thinking_started_at = None
        self._streaming = False

    def _reset_stream_buffers(self) -> None:
        self._assistant_text = self._thinking_text = ""
        self._thinking_duration = self._thinking_started_at = None

    def _print_system(self, output: str) -> None:
        self._print_unit(Text(f"system · {output}", style=CHROME))

    def _print_hook_notice(self, output: str) -> None:
        self._print_unit(Text(f"hook · {output}", style=DIM))

    def _prepare_stream_event(self, event: StreamEvent) -> None:
        if event.type is StreamEventType.MESSAGE_START:
            self._flush_pending_stream()
            self._presenter.reset_assistant_message()
            self._assistant_text = ""
            return
        if event.type is StreamEventType.MESSAGE_END:
            return
        if event.type is StreamEventType.ERROR:
            self._flush_stream_kind(preserve_inline=True)
            self._presenter.reset_assistant_unit()
            return
        if event.type is not StreamEventType.MESSAGE_UPDATE:
            self._flush_pending_stream()
            return
        incoming_kind, incoming_identity = stream_key(event)
        current = self._stream_kind, self._stream_identity
        if (incoming_kind, incoming_identity) != current:
            self._flush_pending_stream()

    async def _read_prompt(self, session: PromptSession[str]) -> str | None:
        def insert_pending_tokens() -> None:
            if self._composer_insertions:
                session.app.current_buffer.insert_text(
                    "".join(self._composer_insertions)
                )
                self._composer_insertions.clear()

        try:
            value = await session.prompt_async(
                [("class:prompt", " > ")],
                bottom_toolbar=self._status_toolbar,
                placeholder=[("class:placeholder", "type a message...")],
                pre_run=insert_pending_tokens,
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
        root.children[:] = [
            full_screen_content(
                self._transcript.window(),
                composer_rows,
                footer,
                self._todo_widget,
                self.loop.store,
            )
        ]

    async def run(self, session: PromptSession[str] | None = None) -> None:
        """Run the alternate-screen app until Ctrl-D or an exit request."""

        session = session or self._session or self._make_session()
        self._active_session = session
        if isinstance(session, FullScreenPromptSession):
            self._install_full_screen_layout(session)
        self.loop.session_start()
        self._rebuild_transcript()
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
        if resume_id == "":
            previews = manager.list_session_previews(limit=RECENT_SESSION_LIMIT)
            if not previews:
                raise SessionError("no prior zeta session found")
            width = content_width(get_terminal_size(fallback=(80, 24)).columns)
            print(resume_picker_line("recent zeta sessions:", width))
            for index, preview in enumerate(previews, start=1):
                print(resume_picker_line(f"{index}. {preview.updated_at} {preview.session_id[:8]} {preview.preview}", width))
            try:
                choice = input(resume_picker_line("select a session:", width - 1) + " ").strip()
                selected = int(choice)
                if not 1 <= selected <= len(previews):
                    raise ValueError("selection out of range")
                resume_id = previews[selected - 1].session_id
            except (EOFError, ValueError) as exc:
                raise SessionError("invalid resume session selection") from exc
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
        if metadata.system_prompt:
            project_context = ProjectContext(
                metadata.system_prompt,
                tuple(Path(path) for path in metadata.context_files),
            )
        else:
            project_context = load_project_context(
                repo_root=discover_repo_root(Path(metadata.cwd)),
                zeta_home=home,
            )
            persisted = manager.persist_context_snapshot(
                metadata,
                system_prompt=project_context.system_prompt,
                context_files=[str(path) for path in project_context.files],
            )
            project_context = ProjectContext(
                persisted.system_prompt,
                tuple(Path(path) for path in persisted.context_files),
            )
    else:
        provider = args.provider or "fake"
        backend, selected_model = build_backend(provider, args.model, home=home)
        project_context = load_project_context(
            repo_root=discover_repo_root(Path.cwd()),
            zeta_home=home,
        )
        opened = manager.create(
            provider=provider,
            model=selected_model,
            cwd=Path.cwd(),
            system_prompt=project_context.system_prompt,
            context_files=[str(path) for path in project_context.files],
        )
        metadata = opened.metadata
        store = opened.store
    if resuming:
        backend, selected_model = build_backend(provider, model, home=home)
    hooks = load_hooks_for_provider(home, provider)
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

    def model_changed(model_name: str) -> None:
        nonlocal pending_override
        if pending_override is not None:
            pending_override = (provider, model_name)
            return
        manager.record_override(metadata, provider=None, model=model_name)

    token_budget_override = getattr(args, "token_budget", None)
    effective_token_budget = (
        token_budget_override
        if token_budget_override is not None and token_budget_override > 0
        else max(metadata.compaction_budget, 200_000)
    )
    max_turns_override = getattr(args, "max_turns", None)
    loop_kwargs: dict[str, Any] = {
        "approval_policy": approval_policy,
        "hooks": hooks,
        "token_budget": effective_token_budget,
        "retained_tail": metadata.retained_tail,
        "on_completion_success": completion_success,
        "system_prompt": project_context.system_prompt,
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
        context_files=[str(path) for path in project_context.files],
        on_model_change=model_changed,
        vim_mode=metadata.vim_mode,
        on_vim_mode_change=lambda enabled: manager.record_vim_mode(metadata, enabled=enabled),
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
