"""Composition root for the full-screen zeta terminal UI."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import get_terminal_size
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
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
    UsageTracker,
    context_window,
    create_slash_registry,
    resolve_session_budget,
)
from ..loop import AgentLoop
from ..persistence import DraftPersistence, history_for
from ..providers.factory import build_backend as build_network_backend
from ..settings import ResolvedConfig, load_settings
from ..settings import resolve as resolve_settings
from ..submission_pipeline import SubmissionPipeline
from ..tools._user_discovery import ExternalToolDiscovery, apply_external_tools
from ..tools.exec import trusted_macro_display
from ..types import (
    CompletionBackend,
    Message,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    assistant_text,
)
from .checkpoints import CheckpointTranscriptMixin
from .composer import (
    ComposerAttachmentMixin,
    FullScreenPromptSession,
    SlashCompleter,
    SubmissionMixin,
    TurnConsumerMixin,
    UndoCandidate,
    build_key_bindings,
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
from .models import MODEL_CATALOGS
from .models import load_model_catalog as _load_model_catalog
from .render import (
    format_status,
    render_approval_card,
    render_markdown,
    render_thought,
    render_thought_live,
)
from .slash_handlers import SlashHandlerMixin
from .slash_handlers.command_runtime import CommandRuntimeMixin
from .theme import (
    ACCENT,
    BODY,
    CHROME,
    COMMAND,
    COMPOSER_BORDER,
    COMPOSER_FOCUS,
    DIM,
    ERROR,
    RICH_THEME,
)
from .todo import TodoWidget
from .transcript import TranscriptWidget, stream_key
from .transcript_presenter import TranscriptPresenter

RECENT_SESSION_LIMIT = 20


def background_notice(app: Any, message: str) -> None:
    """Print one dim background task notice and refresh the prompt."""

    app._print(Text(message, style=DIM))
    app._invalidate_prompt()


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected provider without loading network credentials for fake."""

    if provider == "fake":
        selected_model = model or "offline"
        return FakeInteractiveBackend(model=selected_model), selected_model
    return build_network_backend(
        provider,
        model,
        home=home,
        stall_seconds=stall_seconds,
        stall_retries=stall_retries,
    )


class TUIApp(
    SubmissionMixin,
    TurnConsumerMixin,
    CheckpointTranscriptMixin,
    ComposerAttachmentMixin,
    CommandRuntimeMixin,
    SlashHandlerMixin,
):
    """Full-screen transcript, persistent composer, and follow-up queue."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        provider: str,
        model: str,
        zeta_home: str | Path | None = None,
        verbose: bool = False,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
        history_path: str | Path | None = None,
        draft_path: str | Path | None = None,
        approval_policy: ApprovalPolicy | None = None,
        context_files: Sequence[str] = (),
        on_model_change: Callable[[str], None] | None = None,
        vim_mode: bool = True,
        on_vim_mode_change: Callable[[bool], None] | None = None,
        on_budget_change: Callable[[int], None] | None = None,
        model_catalog_loader: Callable[[str], frozenset[str] | None] | None = None,
        startup_notices: Sequence[str] = (),
        startup_warnings: Sequence[str] = (),
        external_tools: ExternalToolDiscovery | None = None,
        workspace_snapshot_cap: int | None = None,
    ) -> None:
        self.loop = loop
        self._workspace_snapshot_cap = workspace_snapshot_cap
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
        self._pending_attachments: list[Path] = []
        self._pending_attachment_tokens: dict[str, Path] = {}
        self._next_image_token = 1
        self._composer_insertions: list[str] = []
        self._exit_requested = False
        self._loop_state = "idle"
        self._usage: dict[str, Any] = {}
        self._usage_tracker = UsageTracker(
            self.loop.context_assembler, provider=provider
        )
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
        self._macro_receipts = deque()
        self._input_loop_active = False
        self._active_turn_submission_id: int | None = None
        self._resuming_tool = False
        self._session = session
        self._history_path = (
            Path(history_path) if history_path else env_home() / "history"
        )
        self._history = None
        self._draft = DraftPersistence(
            draft_path or self.loop.store.session_dir / "draft"
        )
        self._draft_session: PromptSession[str] | None = None
        self._undo_candidate: UndoCandidate | None = None
        self._approval_policy = approval_policy
        self._context_files = tuple(context_files)
        self._on_model_change = on_model_change
        self.vim_mode = vim_mode
        self._on_vim_mode_change = on_vim_mode_change
        self._on_budget_change = on_budget_change
        self._model_catalog_loader = model_catalog_loader or _load_model_catalog
        self._model_catalog: frozenset[str] | None = MODEL_CATALOGS.get(provider)
        self._model_catalog_loaded = self._model_catalog is not None
        self._model_catalog_task: asyncio.Task[None] | None = None
        self._slash_commands = create_slash_registry(zeta_home=Path(zeta_home).resolve() if zeta_home is not None else None, project_dir=discover_repo_root(Path(self.loop.store.cwd)))
        self.loop.set_mcp_prompt_refresh(
            lambda mount: self._slash_commands.set_mcp_prompts(mount.prompt_entries)
        )
        self._submissions = SubmissionPipeline(self)
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
        self.loop.set_background_event_sink(self._handle_background_event)
        self.loop.set_mcp_notice_sink(lambda message: background_notice(self, message))
        self._fork_rebuilt = False
        self._startup_notices: tuple[str, ...] = tuple(startup_notices)
        self._startup_warnings: tuple[str, ...] = tuple(startup_warnings)
        self._external_tools = external_tools

    @property
    def _transcript_lines(self) -> list[str]:
        """Expose rendered lines for diagnostics while keeping logical units in the widget."""
        width = content_width(get_app().output.get_size().columns)
        return self._transcript.lines(width)

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(
            block.text
            for message, _candidate in self._submissions.queued
            for block in message.content[:1]
            if isinstance(block, TextContent)
        )

    @property
    def active(self) -> bool:
        return self._submissions.active or (
            self._active_task is not None and not self._active_task.done()
        )

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
        self._active_task = asyncio.create_task(
            self._submissions.retry(user_text, user_message)
        )

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        return self._submissions.pending_approvals

    @property
    def approval_policy(self) -> ApprovalPolicy | None:
        return self._approval_policy

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
        for index, request in enumerate(self.pending_approvals):
            self._print_unit(
                render_approval_card(
                    request.tool_call.name,
                    request.tool_call.arguments,
                    label=request.label,
                    key=str(request.key),
                    shortcut=index == 0,
                    trusted_display=trusted_macro_display(request.tool_call.id),
                )
            )

    async def _handle_approval_input(self, value: str) -> bool:
        action = self._submissions._approval_action_for(value)
        if action is None:
            return False
        decision, requested_key = action
        if self._submissions.active:
            await self._submissions.approval_action_wait(decision, requested_key)
            return True
        pending = self.pending_approvals
        if not pending:
            self._print(Text("[approval] no pending requests", style="dim"))
            return True
        if requested_key is None:
            self._print(
                Text(f"[approval] use {value.split(maxsplit=1)[0]} <approval-key>", style="yellow")
            )
            return True
        request = next(
            (request for request in pending if str(request.key) == requested_key),
            None,
        )
        if request is None:
            self._print(
                Text(f"[approval] unknown request: {requested_key}", style="yellow")
            )
            return True
        if self._approval_policy is None:
            return True
        key = request.key
        resolved = (
            self._approval_policy.approve(key)
            if decision is ApprovalDecision.ALLOW
            else self._approval_policy.deny(key)
        )
        if resolved:
            self._print(Text(f"[approval] {value.split(maxsplit=1)[0]}d {key}", style="green"))
            if self.active:
                self._present_pending_approvals()
                return True

            async def resume() -> Any:
                return await self.loop.resume_pending_tool(
                    request.request_id,
                    prepared=True,
                    event_sink=self._handle_resumed_tool_event,
                )

            resume_task: asyncio.Task[Any] | None = None
            try:
                await self.loop.ensure_mcp_servers()
                if not self.loop.prepare_resume_pending_tool(request.request_id):
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
                self.loop.finalize_canceled(request.request_id)
                if resume_task is not None:
                    resume_task.cancel()
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
                        f"fg:{COMPOSER_FOCUS}" if focused else f"fg:{COMPOSER_BORDER}"
                    ),
                    "text-area": f"fg:{BODY}",
                    "text-area.prompt": f"fg:{ACCENT} bold",
                }
            )
            self._prompt_styles[focused] = style
        return style

    def _make_session(self) -> PromptSession[str]:
        if self._history is None:
            self._history = history_for(self._history_path)
        bindings = build_key_bindings(
            on_interrupt=self.abort_active,
            on_exit=self.request_exit,
            on_submit=self._submit_input,
            on_paste=self._paste_from_keybinding,
            on_page_up=self._transcript.page_up,
            on_page_down=self._transcript.page_down,
            on_search_start=self._transcript.begin_search,
            search_active=lambda: self._transcript.search_active,
            on_search_input=self._transcript.update_search,
            on_search_backspace=self._transcript.search_backspace,
            on_search_next=self._transcript.next_search_match,
            on_search_previous=self._transcript.previous_search_match,
            on_search_end=self._transcript.end_search,
            on_previous_user=self._transcript.previous_user_message,
            on_next_user=self._transcript.next_user_message,
            on_toggle_agent=self._transcript.toggle_latest_agent,
            on_retry=self.retry_failed_turn,
            retry_available=self.retry_available,
            on_undo=self.undo_sent_turn,
            append_history=False,
            on_approve=lambda: self._answer_first_pending("approve"),
            on_deny=lambda: self._answer_first_pending("deny"),
            approval_active=lambda: bool(self.pending_approvals),
            on_plan_toggle=self.toggle_plan_mode,
            on_scroll_up=self._transcript.scroll_up,
            on_scroll_down=self._transcript.scroll_down,
        )
        session = FullScreenPromptSession(
            message=[("class:prompt", " > ")],
            placeholder=[("class:placeholder", "type a message...")],
            history=self._history,
            key_bindings=bindings,
            completer=SlashCompleter(self._slash_commands),
            reserve_space_for_menu=0,
            multiline=True,
            mouse_support=True,
            editing_mode=EditingMode.VI if self.vim_mode else EditingMode.EMACS,
            bottom_toolbar=self._status_toolbar,
            erase_when_done=True,
            show_frame=True,
            style=DynamicStyle(self._prompt_style),
        )
        self._attach_draft(session)
        return session

    def _attach_draft(self, session: PromptSession[str]) -> None:
        if self._draft_session is session:
            return
        self._attach_draft_state(session.default_buffer, self._draft.load_state())
        self._draft_session = session

    def _record_prompt(self, value: str, draft_revision: int | None = None) -> None:
        if self._history is None:
            self._history = history_for(self._history_path)
        self._history.append_string(value)
        if draft_revision is None:
            self._draft.clear()
        else:
            self._draft.clear_submitted(draft_revision)

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

    def _abort_approval(self, request_id: str | tuple[str, str]) -> None:
        if self._approval_policy is not None:
            self._approval_policy.abort(request_id)

    def _answer_first_pending(self, verb: str) -> None:
        """Answer the request the y/n shortcuts point at, if it is still there."""

        pending = self.pending_approvals
        if pending:
            self._submit_input(f"{verb} {pending[0].key}")

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
            model_window=context_window(self.provider, self.model),
            vim_state=vim_state_label(self.vim_mode),
            plan_state="PLAN" if self.loop.plan_mode else None,
            background_count=self.loop.tool_registry.background_tasks.running_count,
            undo_available=(
                self._undo_candidate is not None
                and self.active
                and self._loop_state
                in {"streaming", "compacting", "tool-running", "approval"}
            ),
            transcript_navigation=self._full_screen_active(),
            transcript_search=(
                self._transcript.search_query
                if self._transcript.search_active
                else None
            ),
            transcript_match=self._transcript.search_status(),
            transcript_position=self._transcript.position_indicator(),
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
                self.console.print(
                    Padding(renderable, (0, CONTENT_MARGIN, 0, CONTENT_MARGIN))
                )

    def _print_unit(self, renderable: RenderableType | None) -> None:
        self._presenter.print_unit(renderable)

    def _handle_tool_event(self, event: StreamEvent) -> bool:
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            if event.tool_call is not None and not event.data.get("inline_shell"):
                self._submissions.notify_approval_started(
                    event.tool_call,
                    event.data.get("submission_id", self._active_turn_submission_id),
                )
            self._reset_stream_state()
            self._loop_state = "approval"
            self._present_pending_approvals()
            return False
        if event.type is StreamEventType.TOOL_APPROVAL_END:
            if event.tool_call is not None and not event.data.get("inline_shell"):
                self._submissions.notify_approval_finished(event.tool_call)
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
        self._loop_state = (
            "interrupted"
            if aborted
            else ("idle" if self._resuming_tool else "streaming")
        )
        presentation = self._presenter.handle_tool_event(
            event,
            aborted=aborted,
        )
        if presentation is not None and presentation.visible_output:
            self._turn_had_visible_output = True
        stop_after_tool = presentation is not None and presentation.stop_after_tool
        self._abort_requested = False
        return stop_after_tool

    def _handle_background_event(self, event: StreamEvent) -> None:
        """Render child progress while keeping completion notices at turn boundaries."""

        if event.type in {
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_UPDATE,
            StreamEventType.TOOL_EXECUTION_END,
        }:
            self._presenter.handle_tool_event(event, aborted=False)
            self._invalidate_prompt()

    def _handle_resumed_tool_event(self, event: StreamEvent) -> None:
        self._handle_tool_event(event)
        self._invalidate_prompt()

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
        if (
            self._stream_kind in {"thinking", "redacted-thinking"}
            and self._thinking_text
        ):
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
        self._presenter.finish_assistant_message(
            render_markdown(value) if value else None
        )
        if value:
            self._turn_had_visible_output |= bool(value.strip())
        self._stream_kind = self._stream_identity = None
        self._reset_stream_buffers()

    def _flush_pending_stream(self) -> None:
        self._flush_stream_kind()
        self._presenter.reset_assistant_unit()

    def _consume_text(self, event: StreamEvent) -> None:
        incoming_kind, incoming_identity = stream_key(event)
        if self._stream_kind is not None and (incoming_kind, incoming_identity) != (
            self._stream_kind,
            self._stream_identity,
        ):
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
        self._print_unit(Text(f"system · {output}", style=ERROR if output.startswith("mcp error:") else CHROME))

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
        if (
            event.type is StreamEventType.RETRY
            and event.data.get("is_stall")
        ):
            self._presenter.reset_assistant_unit()
            self._reset_stream_state()
            self._reset_stream_buffers()
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
        self._input_loop_active = True
        try:
            while not self._exit_requested:
                try:
                    await prompt_task
                except (EOFError, asyncio.CancelledError):
                    pass
                break
        finally:
            self._input_loop_active = False
            if not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)

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
                on_scroll_up=self._transcript.scroll_up,
                on_scroll_down=self._transcript.scroll_down,
            )
        ]

    async def run(self, session: PromptSession[str] | None = None) -> None:
        """Run the alternate-screen app until Ctrl-D or an exit request."""

        session = session or self._session or self._make_session()
        self._active_session = session
        self._attach_draft(session)
        if isinstance(session, FullScreenPromptSession):
            self._install_full_screen_layout(session)
        self.loop.session_start()
        self._rebuild_transcript()
        await self.loop.ensure_mcp_servers()
        for warning in self._startup_warnings:
            self._print_unit(Text(warning, style=ERROR))
        for notice in self._startup_notices:
            self._print_unit(Text(notice, style=DIM))
        for notice in self._slash_commands.notices:
            style = COMMAND if notice in self._slash_commands.warning_notices else DIM
            self._print_unit(Text(f"command · {notice}", style=style))
        self._present_pending_approvals()
        prompt_task: asyncio.Task[str | None] | None = None
        try:
            if isinstance(session, FullScreenPromptSession):
                await self._run_full_screen(session)
                return
            prompt_task = asyncio.create_task(self._read_prompt(session))
            self._input_loop_active = True
            while prompt_task is not None and not self._exit_requested:
                value = await prompt_task
                if value is None:
                    break
                if not await self._handle_approval_input(value):
                    self._submit_input(value)
                if self._exit_requested:
                    break
                prompt_task = asyncio.create_task(self._read_prompt(session))
        finally:
            self._input_loop_active = False
            self._draft.flush()
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)
            if self._active_task is not None and not self._active_task.done():
                self._active_task.cancel()
                await asyncio.gather(self._active_task, return_exceptions=True)
            await self._submissions.close()
            if isinstance(session, FullScreenPromptSession):
                session.restore_terminal()
            await self.loop.close()
            self._active_session = None


def create_app(args: argparse.Namespace) -> TUIApp:
    home = env_home()
    manager = SessionManager(home)
    project_dir = discover_repo_root(Path.cwd()) / ".zeta"
    loaded_settings = load_settings(home=home, project_dir=project_dir)
    config: ResolvedConfig = resolve_settings(
        loaded_settings.settings,
        cli_provider=getattr(args, "provider", None),
        cli_model=getattr(args, "model", None),
        cli_yolo=getattr(args, "yolo", None),
        cli_token_budget=getattr(args, "token_budget", None),
    )
    continue_session = getattr(args, "continue_session", False)
    resume_id = getattr(args, "resume", None)
    force_provider = getattr(args, "force_provider", False)
    resuming = continue_session or resume_id is not None
    if force_provider and not resuming:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and config.model is None:
        raise SessionError("--force-provider requires --model")

    if resuming:
        if resume_id == "":
            previews = manager.list_session_previews(limit=RECENT_SESSION_LIMIT)
            if not previews:
                raise SessionError("no prior zeta session found")
            width = content_width(get_terminal_size(fallback=(80, 24)).columns)
            print(resume_picker_line("recent zeta sessions:", width))
            for index, preview in enumerate(previews, start=1):
                print(
                    resume_picker_line(
                        f"{index}. {preview.updated_at} {preview.session_id[:8]} {preview.preview}",
                        width,
                    )
                )
            try:
                choice = input(
                    resume_picker_line("select a session:", width - 1) + " "
                ).strip()
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
        cli_provider = getattr(args, "provider", None)
        cli_model = getattr(args, "model", None)
        provider_override = cli_provider or loaded_settings.settings.provider
        model_override = cli_model or loaded_settings.settings.model
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
        provider = config.provider
        backend, selected_model = build_backend(
            provider,
            config.model,
            home=home,
            stall_seconds=config.stream_stall_seconds,
            stall_retries=config.stream_stall_retries,
        )
        project_context = load_project_context(
            repo_root=discover_repo_root(Path.cwd()),
            zeta_home=home,
        )
        created_budget, created_pin = resolve_session_budget(
            0, False, provider, selected_model, config.token_budget
        )
        opened = manager.create(
            provider=provider,
            model=selected_model,
            cwd=Path.cwd(),
            compaction_budget=created_budget,
            system_prompt=project_context.system_prompt,
            context_files=[str(path) for path in project_context.files],
            budget_pinned=created_pin,
        )
        metadata = opened.metadata
        store = opened.store
    if resuming:
        backend, selected_model = build_backend(
            provider,
            model,
            home=home,
            stall_seconds=config.stream_stall_seconds,
            stall_retries=config.stream_stall_retries,
        )
    hooks = load_hooks_for_provider(home, provider)
    approval_default = (
        ApprovalDecision.ALLOW if config.yolo else ApprovalDecision.ASK
    )
    approval_policy = ApprovalPolicy(
        store=store,
        default=approval_default,
        always_allow=config.approval_allow,
        always_deny=config.approval_deny,
        always_ask=config.approval_ask,
    )
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

    def plan_mode_changed(enabled: bool) -> None:
        manager.record_plan_mode(metadata, enabled=enabled)

    effective_token_budget, budget_pinned = resolve_session_budget(
        metadata.compaction_budget,
        metadata.budget_pinned,
        provider,
        selected_model,
        config.token_budget,
    )
    if (
        effective_token_budget != metadata.compaction_budget
        or budget_pinned != metadata.budget_pinned
    ):
        manager.record_budget(
            metadata, budget=effective_token_budget, pinned=budget_pinned
        )
    max_turns_override = getattr(args, "max_turns", None)
    loop_kwargs: dict[str, Any] = {
        "approval_policy": approval_policy,
        "hooks": hooks,
        "token_budget": effective_token_budget,
        "retained_tail": metadata.retained_tail,
        "on_completion_success": completion_success,
        "on_plan_mode_change": plan_mode_changed,
        "system_prompt": project_context.system_prompt,
    }
    if max_turns_override is not None and max_turns_override > 0:
        loop_kwargs["max_turns"] = max_turns_override
    loop = AgentLoop(backend, store, **loop_kwargs)
    if metadata.plan_mode:
        loop.set_plan_mode(True)
    repo_root = discover_repo_root(Path(store.cwd))
    loop.set_mcp_scope(home=home, project_dir=repo_root)
    external_tools = apply_external_tools(
        loop.tool_registry,
        home=home,
        project_dir=repo_root / ".zeta",
    )
    loop.tool_schemas = list(loop.tool_registry.schemas)
    startup_notices = tuple(loaded_settings.notices) + external_tools.notices
    startup_warnings = tuple(loaded_settings.warnings) + external_tools.warnings
    return TUIApp(
        loop,
        provider=provider,
        model=selected_model,
        zeta_home=home,
        verbose=args.verbose,
        history_path=home / "history",
        approval_policy=approval_policy,
        context_files=[str(path) for path in project_context.files],
        on_model_change=model_changed,
        vim_mode=metadata.vim_mode,
        on_budget_change=(
            None
            if budget_pinned
            else lambda budget: manager.record_budget(
                metadata, budget=budget, pinned=False
            )
        ),
        on_vim_mode_change=lambda enabled: manager.record_vim_mode(
            metadata, enabled=enabled
        ),
        startup_notices=startup_notices,
        startup_warnings=startup_warnings,
        external_tools=external_tools,
        workspace_snapshot_cap=config.workspace_snapshot_cap,
    )


__all__ = [
    "FakeInteractiveBackend",
    "TUIApp",
    "build_backend",
    "create_app",
    "main",
]


def __getattr__(name: str) -> object:
    if name == "main":
        from ..cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
