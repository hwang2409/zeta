"""Composition root for the full-screen zeta terminal UI."""
from __future__ import annotations

import asyncio
import shlex
import subprocess
import weakref
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from shutil import (
    get_terminal_size,  # noqa: F401 — monkey-patched by tests via zeta.tui.app.get_terminal_size
)
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app, run_in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.styles import DynamicStyle, Style
from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.text import Text

from ..config.settings import (
    load_settings,  # noqa: F401 — monkey-patched by tests via zeta.tui.app.load_settings
)
from ..core.approval import ApprovalPolicy
from ..core.project_context import (
    discover_project,
    load_project_context,  # noqa: F401 — monkey-patched by tests via zeta.tui.app.load_project_context
    project_instructions_editor_command,
)
from ..core.session import SessionMetadata
from ..core.slash import (
    SlashStatus,
    UsageTracker,
    _format_status,
    create_slash_registry,
)
from ..protocol.types import (
    Message,
    MessageRole,
)
from ..runtime.foreground_events import ForegroundEventPublisher
from ..runtime.loop import AgentLoop
from ..runtime.loop.persistence import DraftPersistence
from ..submission.pipeline import SubmissionPipeline
from ..tools._shared.user_discovery import ExternalToolDiscovery
from . import overlay, theme
from .agent_card import (
    AgentNavigation,
    AgentRunCommandMixin,
    agent_navigation_style_rules,
)
from .bootstrap import build_backend
from .cards.mcp_manager import MCPManager
from .cards.shared import wrap_by_cells
from .checkpoints import CheckpointTranscriptMixin
from .composer import (
    ComposerAttachmentMixin,
    ComposerCompleter,
    FullScreenPromptSession,
    SubmissionMixin,
    TurnConsumerMixin,
    UndoCandidate,
    build_key_bindings,
)
from .fork_session import ForkRuntimeMixin
from .layout import (
    CONTENT_MARGIN,
    content_width,
    status_toolbar,
)
from .models import MODEL_CATALOGS
from .models import load_model_catalog as _load_model_catalog
from .overlay import OverlayControl
from .paste_runtime import PasteRuntime
from .runtime.prompt import PromptRuntimeMixin
from .runtime.stream import StreamEventMixin
from .runtime_close import RuntimeCloseMixin
from .session_prompt_history import SessionPromptHistory
from .slash_handlers import SlashHandlerMixin
from .slash_handlers.accounts_picker import AccountsPicker
from .slash_handlers.command_runtime import CommandRuntimeMixin
from .slash_handlers.decisions_panel import DecisionsMixin
from .slash_handlers.fork_view import ForkViewMixin
from .slash_handlers.mcp_manager import MCPManagerMixin
from .slash_handlers.model_picker import ModelPicker
from .slash_handlers.remote import RemoteViewMixin
from .slash_handlers.tasks_panel import BackgroundTasksMixin
from .theme import RICH_THEME
from .todo import TodoWidget
from .transcript import (
    TranscriptPresenter,
    TranscriptWidget,
)
from .transcript.finder_runtime import FinderRuntimeMixin
from .transcript.history_finder import HistoryFinder
from .transcript.paged_history import PagedTranscriptHistory
from .transcript.root_history_source import RootTUIHistorySource
from .user import displayed_user_text


def _wrap_status_line(line: str, width: int) -> list[overlay.FragmentLine]:
    if cell_len(line) <= width:
        return [[overlay.value(line)]]
    label, separator, value = line.partition(": ")
    prefix = f"{label}{separator}" if separator else line[: -len(line.lstrip())]
    if cell_len(prefix) >= width:
        return [[overlay.value(text)] for text in wrap_by_cells(line, width)]
    wrapped = wrap_by_cells(
        value if separator else line.lstrip(), width - cell_len(prefix)
    )
    return [
        [
            overlay.value(prefix if index == 0 else " " * len(prefix)),
            overlay.value(text),
        ]
        for index, text in enumerate(wrapped)
    ]


def _status_card_lines(
    status: SlashStatus, width: int | None = None
) -> list[overlay.FragmentLine]:
    formatted = _format_status(status).splitlines()
    body = (
        [[overlay.value(line)] for line in formatted]
        if width is None
        else [row for line in formatted for row in _wrap_status_line(line, width)]
    )
    return [
        overlay.title("Status"),
        overlay.rule(),
        *body,
        overlay.rule(),
        overlay.hint("↑/↓ or j/k scroll · pgup/pgdn page · home/end jump · esc close"),
    ]

def _prompt_style_with_background(
    foreground: str, background: str, *, background_first: bool = False
) -> str:
    """Build a prompt-toolkit style without emitting an empty ``bg:`` token."""

    background_style = f"bg:{background}" if background else ""
    parts = (
        (background_style, foreground)
        if background_first
        else (foreground, background_style)
    )
    return " ".join(part for part in parts if part)

class TUIApp(
    SubmissionMixin,
    TurnConsumerMixin,
    CheckpointTranscriptMixin,
    ComposerAttachmentMixin,
    CommandRuntimeMixin,
    SlashHandlerMixin,
    RemoteViewMixin,
    MCPManagerMixin,
    BackgroundTasksMixin,
    DecisionsMixin,
    ForkRuntimeMixin,
    ForkViewMixin,
    RuntimeCloseMixin,
    AgentRunCommandMixin,
    FinderRuntimeMixin,
    StreamEventMixin,
    PromptRuntimeMixin,
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
        draft_path: str | Path | None = None,
        approval_policy: ApprovalPolicy | None = None,
        context_files: Sequence[str] = (),
        on_model_change: Callable[[str], Awaitable[SessionMetadata]] | None = None,
        vim_mode: bool = True,
        on_vim_mode_change: Callable[[bool], Awaitable[None]] | None = None,
        on_budget_change: Callable[[int], Awaitable[None]] | None = None,
        model_catalog_loader: Callable[[str], frozenset[str] | None] | None = None,
        startup_notices: Sequence[str] = (),
        startup_warnings: Sequence[str] = (),
        startup_alerts: Sequence[str] = (),
        external_tools: ExternalToolDiscovery | None = None,
        workspace_snapshot_cap: int | None = None,
        ephemeral_root: Path | None = None,
        session_name: str = "",
        on_name_change: Callable[[str], Awaitable[None]] | None = None,
        key_remap: Any | None = None,
        project_dir: str | Path | None = None,
        project_eligible: bool | None = None,
        resumed: bool = False,
        on_session_close: Callable[[], Awaitable[None]] | None = None,
        foreground_event_publisher: ForegroundEventPublisher | None = None,
        observer_host: Any | None = None,
    ) -> None:
        app = weakref.proxy(self)
        self.loop = loop
        self.foreground_event_publisher = (
            foreground_event_publisher or ForegroundEventPublisher()
        )
        self.observer_host = observer_host
        self._workspace_snapshot_cap = workspace_snapshot_cap
        self.loop.tool_registry.background_tasks.set_notice_sink(
            lambda message: app._handle_background_notice(message)
        )
        self._hooks = loop.hooks
        if self._hooks is not None:
            self._hooks.notice_sink = lambda message: (
                app._defer_startup_notice(message, hook=True)
                or app._print_hook_notice(message)
            )
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.console = console or Console(theme=RICH_THEME)
        self._active_task: asyncio.Task[None] | None = None
        self._paste_runtime = PasteRuntime(app)
        self._on_session_close = on_session_close
        self._init_runtime_close()
        self._init_fork_runtime()
        self._resumed_session = resumed
        self._startup_replay_active = False
        self._startup_submissions: list[tuple[str, bool]] = []
        self._pending_attachments: list[Path] = []
        self._pending_attachment_tokens: dict[str, Path] = {}
        self._next_image_token = 1
        self._composer_insertions: list[str] = []
        self._exit_requested = False
        self._terminal_restored = False
        self._loop_state = "idle"
        self._usage: dict[str, Any] = {}
        self._usage_tracker = UsageTracker(
            self.loop.context_assembler, provider=provider
        )
        self._assistant_chunks: list[str] = []
        self._thinking_chunks: list[str] = []
        self._thinking_duration: float | None = None
        self._thinking_started_at: float | None = None
        self._stream_kind: str | None = None
        self._stream_identity: tuple[str, object] | None = None
        self._partial = ""
        self._streaming = False
        self._spinner_active = False
        self._spinner_frame = 0
        self._spinner_reset = asyncio.Event()
        self._stream_invalidation_handle: asyncio.TimerHandle | None = None
        self._stream_invalidation_pending = False
        self._background_invalidation_handle: asyncio.TimerHandle | None = None
        self._abort_requested = False
        self._macro_receipts = deque()
        self._last_passthrough: str = ""
        self._input_loop_active = False
        self._active_turn_submission_id: int | None = None
        self._resuming_tool = False
        self._session = session
        self._history_source = RootTUIHistorySource(self.loop.store)
        self._history_pager = PagedTranscriptHistory(self._history_source)
        self._history_search_task: asyncio.Task[None] | None = None
        self._history_prompt_task: asyncio.Task[None] | None = None
        self._history_marker_tasks: set[asyncio.Task[None]] = set()
        self._rendering_history = False
        self._prompt_history = SessionPromptHistory(self._history_source)
        storage_lease = self.loop.store._storage_lease if draft_path is None else None
        self._draft = DraftPersistence(
            draft_path or self.loop.store.session_dir / "draft",
            directory_fd=self.loop.store.directory_fd if draft_path is None else None,
            storage_lease=storage_lease,
            notice_sink=lambda message: app._handle_background_notice(message),
        )
        self._draft_session: PromptSession[str] | None = None
        self._undo_candidate: UndoCandidate | None = None
        self._approval_policy = approval_policy
        self._context_files = tuple(context_files)
        self._on_model_change = on_model_change
        self.vim_mode = vim_mode
        self._on_vim_mode_change = on_vim_mode_change
        self._on_budget_change = on_budget_change
        account_context = getattr(loop, "account_context", None)
        self._model_catalog_loader = model_catalog_loader or (lambda provider: _load_model_catalog(provider, account_context=account_context))
        self._model_catalog: frozenset[str] | None = MODEL_CATALOGS.get(provider)
        self._model_catalog_loaded = self._model_catalog is not None
        self._model_catalog_task: asyncio.Task[None] | None = None
        self._model_picker: ModelPicker | None = None
        self._model_picker_unit: Any = None
        self._accounts_picker: AccountsPicker | None = None
        self._accounts_picker_unit: Any = None
        self._account_switch_task: asyncio.Task[None] | None = None
        self._account_switch_pending = False
        self._account_switch_finished = asyncio.Event()
        self._account_switch_finished.set()
        self._zeta_home = Path(zeta_home).resolve() if zeta_home is not None else None
        if project_dir is None:
            discovery = discover_project(self.loop.store.cwd)
            repo_root = discovery.primary_root or discovery.cwd
            project_eligible = discovery.eligible and discovery.primary_root is not None
        else:
            repo_root = Path(project_dir).resolve()
        skill_catalog = self.loop.tool_registry.skill_catalog
        self._slash_commands = create_slash_registry(
            zeta_home=self._zeta_home,
            project_dir=repo_root,
            project_eligible=project_eligible,
            skill_catalog=skill_catalog, project_registry_state=self.loop.tool_registry.project_registry_state,
            include_tui_only=True,
        )
        self.loop.add_live_config_listener(self._slash_commands)
        self.loop.set_mcp_prompt_refresh(
            lambda mount: app._slash_commands.set_mcp_prompts(mount.prompt_entries)
        )
        self._submissions = SubmissionPipeline(app)
        self._turn_had_visible_output = False
        self._failed_turn: tuple[str, Message] | None = None
        self._active_session: PromptSession[str] | None = None
        self._restart_idle_buffers: set[int] = set()
        self._prompt_styles: dict[bool, Style] = {}
        self._transcript = TranscriptWidget()
        self._init_remote_view()
        self._history_finder = HistoryFinder(
            lambda: app._prompt_history.snapshot(app._viewed_session_path()),
            on_accept=lambda: app._clear_pending_attachments(),
        )
        self._active_finder = self._transcript
        self._status_card = OverlayControl()
        self._init_finder_runtime()
        self._composer_restore_document: Document | None = None
        self._status_card_open = False
        self._mcp_manager_open = False
        self._init_background_tasks_panel()
        self._init_decisions_panel()
        self._startup_presented = False
        self._mcp_wizard_active = False
        self._mcp_wizard_dialog_active = False
        self._mcp_wizard_task: asyncio.Task[None] | None = None
        self._mcp_manager = MCPManager(self.loop._mcp_management)
        self._status_restore_text = ""
        self._status_restore_cursor = 0
        self._agent_navigation = AgentNavigation(self.loop.store)
        self._agent_navigation._main_transcript_widget = self._transcript
        self._agent_navigation._on_view_change = lambda: app._viewed_session_changed()
        self._agent_navigation.transcript_control.on_snapshot = (
            lambda path, snapshot: app._prompt_history.replace(path, snapshot.typed_prompts)
        )
        self._init_fork_view()
        self._todo_widget = TodoWidget(
            self.loop.store, selected_snapshot=self._agent_navigation.todo_snapshot
        )
        self._presenter = TranscriptPresenter(
            self._transcript,
            self.console,
            lambda: app._full_screen_active(),
            lambda renderable: app._print(renderable),
            self.loop.store.agent_children,
        )
        self.loop.set_background_event_sink(
            lambda event: app._handle_background_event(event)
        )
        self.loop.set_background_wake_callback(lambda: app._schedule_background_wake())
        self.loop.set_mcp_notice_sink(
            lambda message: app._handle_background_notice(message)
        )
        self._fork_rebuilt = False
        self._startup_notices: tuple[str, ...] = tuple(startup_notices)
        self._startup_warnings: tuple[str, ...] = tuple(startup_warnings)
        self._startup_alerts: tuple[str, ...] = tuple(startup_alerts)
        self._external_tools = external_tools
        self._ephemeral_root = ephemeral_root
        self._new_session_requested = False
        self._session_name = session_name
        self._on_name_change = on_name_change
        self._key_remap: dict[str, str] = dict(key_remap or {})

    @property
    def ephemeral_root(self) -> Path | None:
        return self._ephemeral_root
    @property
    def _transcript_lines(self) -> list[str]:
        """Expose rendered lines for diagnostics while keeping logical units in the widget."""
        width = content_width(get_app().output.get_size().columns)
        return self._transcript.lines(width)

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(
            displayed_user_text(message)
            for message, _candidate in self._submissions.queued
        )

    @property
    def active(self) -> bool:
        return self._submissions.active or (
            self._active_task is not None and not self._active_task.done()
        )

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
            catalog = await asyncio.to_thread(
                self._model_catalog_loader, self.provider
            )
            if catalog is not None:
                self._model_catalog = catalog
        except Exception:  # noqa: S110 - catalog refresh must not affect the TUI
            pass
        finally:
            self._model_catalog_loaded = True
            self._model_catalog_task = None
            self.refresh_model_picker()

    def _prompt_style(self) -> Style:
        focused = get_app().current_buffer.name == "DEFAULT_BUFFER"
        style = self._prompt_styles.get(focused)
        if style is None:
            style = Style.from_dict(
                {
                    "": f"fg:{theme.BODY}",
                    "prompt": f"fg:{theme.ACCENT} bold",
                    "placeholder": f"italic fg:{theme.COMPOSER_PLACEHOLDER}",
                    "status-bar": f"noreverse fg:{theme.CHROME}",
                    "frame": "",
                    "frame.border": (
                        f"fg:{theme.COMPOSER_FOCUS}"
                        if focused
                        else f"fg:{theme.COMPOSER_BORDER}"
                    ),
                    "text-area": _prompt_style_with_background(
                        f"fg:{theme.BODY}", theme.COMPOSER_FILL
                    ),
                    "text-area.prompt": _prompt_style_with_background(
                        f"fg:{theme.ACCENT} bold", theme.COMPOSER_FILL
                    ),
                    # Prompt-toolkit's gray default is unreadable on a dark terminal.
                    # Rows use the highlight background; the current row takes
                    # the accent so the pick is unmistakable.
                    "completion-menu": _prompt_style_with_background(
                        f"fg:{theme.BODY}", theme.MENU_BG, background_first=True
                    ),
                    "completion-menu.completion": _prompt_style_with_background(
                        f"fg:{theme.BODY}", theme.MENU_BG, background_first=True
                    ),
                    "completion-menu.completion.current": _prompt_style_with_background(
                        f"fg:{theme.ON_ACCENT} bold",
                        theme.ACCENT,
                        background_first=True,
                    ),
                    "completion-menu.meta.completion": _prompt_style_with_background(
                        f"fg:{theme.DIM}", theme.MENU_BG, background_first=True
                    ),
                    "completion-menu.meta.completion.current": _prompt_style_with_background(
                        f"fg:{theme.ON_ACCENT}", theme.ACCENT, background_first=True
                    ),
                    "scrollbar.background": _prompt_style_with_background(
                        "", theme.MENU_BG
                    ),
                    "scrollbar.button": _prompt_style_with_background("", theme.DIM),
                    **agent_navigation_style_rules(),
                }
            )
            self._prompt_styles[focused] = style
        return style

    def _make_session(self) -> PromptSession[str]:
        app = weakref.proxy(self)
        bindings = build_key_bindings(
            on_interrupt=lambda: app.abort_active(),
            on_exit=lambda: app.request_exit(),
            on_submit=lambda value: app._submit_input(value),
            on_paste=lambda event: app._paste_from_keybinding(event),
            on_status_close=lambda: app.close_status_card(),
            status_active=lambda: app.status_card_active,
            on_status_scroll=lambda amount: app._status_move(amount),
            on_status_page=lambda amount: app._status_card.page(amount),
            on_status_top=lambda: app._status_card.top(),
            on_status_bottom=lambda: app._status_card.bottom(),
            on_status_action=lambda key: app._status_action(key),
            tasks_active=lambda: app.tasks_panel_active,
            on_tasks_key=lambda key: app._tasks_key(key),
            decisions_active=lambda: app.decisions_panel_active,
            decisions_answering=lambda: app._decisions_panel.mode == "answer",
            on_decisions_open=lambda: app.open_decisions_panel(),
            on_decisions_key=lambda key: app._decisions_key(key),
            on_page_up=lambda: app._page_history_up(),
            on_page_down=self._transcript.page_down,
            on_finder_open=lambda: app._finder_open(),
            on_history_finder_open=lambda buffer: app._history_finder_open(buffer),
            on_history_previous=lambda buffer: app._history_previous(buffer),
            on_history_next=lambda buffer: app._history_next(buffer),
            finder_active=lambda: app._active_finder.finder_active,
            history_finder_active=lambda: app._history_finder.finder_active,
            on_finder_move=lambda delta: app._finder_move(delta),
            on_finder_accept=lambda: app._finder_accept(),
            on_finder_cancel=lambda: app._finder_cancel(),
            on_finder_toggle_preview=lambda: app._finder_toggle_preview(),
            search_active=lambda: app._viewed_transcript().search_active,
            on_search_next=lambda: app._viewed_transcript().next_search_match(),
            on_search_previous=lambda: app._viewed_transcript().previous_search_match(),
            on_search_end=lambda: app._viewed_transcript().end_search(),
            on_previous_user=lambda: app._navigate_history_user("older"),
            on_next_user=lambda: app._navigate_history_user("newer"),
            on_toggle_agent=lambda: (
                app._agent_navigation.toggle_latest_agent()
                if app._agent_navigation.child_view_active
                else app._transcript.toggle_latest_agent()
            ),
            on_retry=lambda: app.retry_failed_turn(),
            retry_available=lambda: app.retry_available(),
            on_undo=lambda: app.undo_sent_turn(),
            append_history=False,
            on_approve=lambda: app._answer_first_pending("approve"),
            on_deny=lambda: app._answer_first_pending("deny"),
            approval_active=lambda: bool(app.pending_approvals),
            on_plan_toggle=lambda: app.toggle_plan_mode(),
            on_scroll_up=lambda: app._scroll_history_up(),
            on_scroll_down=self._transcript.scroll_down,
            on_picker_move=lambda delta: app._picker_move(delta),
            on_picker_select=lambda: app._picker_select(),
            on_picker_cancel=lambda: app._picker_cancel(),
            on_picker_default=lambda: app.accounts_picker_default(),
            on_picker_login=lambda: app.accounts_picker_login(),
            on_picker_logout=lambda: app.accounts_picker_logout(),
            on_picker_rename=lambda: app.accounts_picker_rename(),
            on_picker_refresh=lambda: app.accounts_picker_refresh(),
            picker_active=lambda: app._picker_active,
            on_agent_list_down=lambda: app._focus_agent_list(),
            agent_list_active=lambda: app._agent_navigation.list_focused(),
            on_agent_list_move=lambda delta: (
                app.clear_fork_return_arm(),
                app._agent_navigation.move_selection(delta),
            )[-1],
            on_agent_list_open=lambda: app._agent_navigation.open_selected(),
            on_agent_list_back=lambda: app._agent_navigation.list_back(),
            on_agent_navigation_exit=lambda: (
                app.clear_fork_return_arm(),
                app._agent_navigation.exit_navigation(),
            )[-1],
            child_view_focused=lambda: app._agent_navigation.child_view_focused(),
            on_child_view_back=lambda: app._agent_navigation.back_to_parent(),
            on_child_view_down=lambda: app._agent_navigation.focus_child_list(),
            child_view_has_list=lambda: app._agent_navigation.list_visible,
            on_child_view_scroll=lambda amount: app._agent_navigation.child_scroll(
                amount
            ),
            on_child_view_half_page=lambda amount: (
                app._agent_navigation.child_half_page(amount)
            ),
            on_child_view_top=lambda: app._agent_navigation.child_top(),
            on_child_view_bottom=lambda: app._agent_navigation.child_bottom(),
            composer_agent_navigation_ready=lambda: (
                app._composer_can_focus_agent_list()
            ),
            interaction_ready=lambda: not app._startup_replay_active,
            key_remap=self._key_remap,
        )
        session = FullScreenPromptSession(
            message=[("class:prompt", " › ")],
            placeholder=lambda: [("class:placeholder", app._composer_placeholder())],
            history=None,
            key_bindings=bindings,
            completer=ComposerCompleter(
                self._slash_commands,
                self.loop.store.cwd,
                model_choices=lambda: app.model_choices(),
                current_model=lambda: app.model,
            ),
            reserve_space_for_menu=0,
            multiline=True,
            mouse_support=True,
            editing_mode=EditingMode.VI if self.vim_mode else EditingMode.EMACS,
            bottom_toolbar=lambda: app._status_toolbar(),
            erase_when_done=True,
            show_frame=False,
            style=DynamicStyle(lambda: app._prompt_style()),
        )
        composer_window = next(
            window
            for window in session.app.layout.find_all_windows()
            if getattr(window.content, "buffer", None) is session.default_buffer
        )
        composer_window.style = "class:text-area"
        self._attach_draft(session)
        if self._composer_restore_document is not None:
            session.default_buffer.set_document(
                self._composer_restore_document, bypass_readonly=True
            )
            self._composer_restore_document = None
        return session

    def _attach_draft(self, session: PromptSession[str]) -> None:
        if self._draft_session is session:
            return
        self._attach_draft_state(session.default_buffer, self._draft.load_state())
        self._draft_session = session

    def _viewed_session_path(self) -> Path:
        if self._agent_navigation.child_view_active:
            return self._agent_navigation.current_path
        return self.loop.store.session_dir
    def _viewed_transcript(self) -> TranscriptWidget:
        return self._agent_navigation.viewed_transcript
    def _viewed_session_changed(self) -> None:
        self._prompt_history.reset_navigation()
        finder = self._active_finder
        if finder.finder_active:
            self._finder_cancel()
            return
        self._cancel_finder_workers()
        self._active_finder = self._viewed_transcript()

    def _request_history_edge(self, direction: str) -> None:
        if direction == "older":
            if not self._history_pager.can_load_older or self._transcript.scroll_offset > 0:
                return
        elif not self._history_pager.can_load_newer or not self._transcript.follow_tail:
            return
        asyncio.create_task(self._load_transcript_page(direction=direction))

    def _page_history_up(self) -> None:
        self._transcript.page_up()
        self._request_history_edge("older")

    def _scroll_history_up(self) -> None:
        self._transcript.scroll_up()
        self._request_history_edge("older")

    def _page_history_down(self) -> None:
        self._transcript.page_down()
        self._request_history_edge("newer")

    def _scroll_history_down(self) -> None:
        self._transcript.scroll_down()
        self._request_history_edge("newer")

    def _navigate_history_user(self, direction: str) -> None:
        moved = (
            self._transcript.previous_user_message()
            if direction == "older"
            else self._transcript.next_user_message()
        )
        if moved:
            return
        users = [
            unit.value
            for page in self._history_pager.pages
            for unit in page.units
            if unit.value.message is not None
            and unit.value.message.role is MessageRole.USER
        ]
        if not users:
            return
        anchor = users[0 if direction == "older" else -1].id

        async def reveal() -> None:
            target = await self._history_pager.navigate_user(anchor, direction)
            if target is None:
                return
            self._render_resident_history()
            if direction == "older":
                self._transcript.previous_user_message()
            else:
                self._transcript.next_user_message()
            self._invalidate_prompt()

        asyncio.create_task(reveal())

    def _finder_input(self, query: str) -> None:
        FinderRuntimeMixin._finder_input(self, query)
        history_pager = getattr(self, "_history_pager", None)
        if history_pager is None:
            return
        history_search_task = getattr(self, "_history_search_task", None)
        if history_search_task is not None:
            history_search_task.cancel()
        if (
            not query
            or self._agent_navigation.child_view_active
            or self._active_finder is not self._transcript
        ):
            return

        async def reveal_match() -> None:
            matches = await history_pager.search(query)
            resident = {
                unit.entry_id
                for page in history_pager.pages
                for unit in page.units
            }
            target = next((entry_id for entry_id in matches if entry_id not in resident), None)
            if target is None or not self._transcript.finder_active:
                return
            await history_pager.reveal(target)
            self._render_resident_history()
            request = self._transcript.open_finder()
            self._active_finder = self._transcript
            self._start_finder_preparation(request)
            self._transcript.finder_set_query(query)
            self._invalidate_prompt()

        self._history_search_task = asyncio.create_task(reveal_match())

    async def _close_history_search(self) -> None:
        tasks = tuple(
            task
            for task in (self._history_search_task, self._history_prompt_task)
            if task is not None
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._history_search_task = None
        self._history_prompt_task = None

    async def _close_history_markers(self) -> None:
        tasks = tuple(self._history_marker_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    def _history_finder_open(self, buffer: Any) -> None:
        prompt_history = getattr(self, "_prompt_history", None)
        if self._agent_navigation.child_view_active or prompt_history is None:
            FinderRuntimeMixin._history_finder_open(self, buffer)
            return
        path = self._viewed_session_path()
        prompt_history.bind(path, self._history_source)
        if prompt_history.snapshot(path):
            FinderRuntimeMixin._history_finder_open(self, buffer)
            return

        async def load() -> None:
            await prompt_history.load_older()
            FinderRuntimeMixin._history_finder_open(self, buffer)

        self._history_prompt_task = asyncio.create_task(load())

    def _history_previous(self, buffer: Any) -> None:
        path = self._viewed_session_path()
        document = self._prompt_history.older(path, buffer.document)
        if document is not None:
            buffer.set_document(document, bypass_readonly=True)
            return

        async def load() -> None:
            await self._prompt_history.load_older()
            if path != self._viewed_session_path():
                return
            document = self._prompt_history.older(path, buffer.document)
            if document is not None:
                buffer.set_document(document, bypass_readonly=True)
                self._invalidate_prompt()

        asyncio.create_task(load())

    def _history_next(self, buffer: Any) -> None:
        path = self._viewed_session_path()
        document = self._prompt_history.newer(path)
        if document is not None:
            buffer.set_document(document, bypass_readonly=True)
            return

        async def load() -> None:
            await self._prompt_history.load_newer()
            if path != self._viewed_session_path():
                return
            document = self._prompt_history.newer(path)
            if document is not None:
                buffer.set_document(document, bypass_readonly=True)
                self._invalidate_prompt()

        asyncio.create_task(load())

    def _record_prompt(self, value: str, draft_revision: int | None = None) -> None:
        self._prompt_history.append(self._viewed_session_path(), value)
        if draft_revision is None:
            self._draft.clear()
        else:
            self._draft.clear_submitted(draft_revision)

    def _insert_paste_token(self, token: str) -> None:
        if isinstance(self._active_session, FullScreenPromptSession):
            self._active_session.app.current_buffer.insert_text(token)
        else:
            self._composer_insertions.append(token)


    def _answer_first_pending(self, verb: str) -> None:
        """Answer the request the y/n shortcuts point at, if it is still there."""

        key = self.first_pending_approval_key
        if key is not None:
            self._submit_input(f"/{verb} {key}", internal=True)

    def _submit_input(self, value: str, *, internal: bool = False) -> bool:
        if self._startup_replay_active and self._input_loop_active:
            self._startup_submissions.append((value, internal))
            return True
        action = value.strip().split(maxsplit=1)[0] if value.strip() else "submission"
        if self._reject_during_startup_replay(action):
            return False
        if self._agent_navigation.child_view_active and not value.lstrip().startswith("/"):
            return self._send_to_child(value)
        self._paste_runtime.advance_buffer_generation()
        super()._submit_input(value, internal=internal)
        return True

    def _status_toolbar(self) -> list[tuple[str, str]]:
        return status_toolbar(self, get_app().output.get_size().columns)

    def _consume_overlay_submission(self) -> None:
        session = self._active_session
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.reset()
            self._draft.clear()

    @property
    def status_card_active(self) -> bool:
        return self._status_card_open and self._full_screen_active()

    def open_status_card(self, status: SlashStatus) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        buffer = session.default_buffer
        self._status_restore_text = buffer.text
        self._status_restore_cursor = buffer.cursor_position
        self._status_card.set_lines(
            _status_card_lines(status),
            line_factory=lambda width: _status_card_lines(status, width),
        )
        self._status_card_open = True
        session.layout.focus(self._status_card_window)
        self._invalidate_prompt()

    def close_status_card(self) -> None:
        if not self._status_card_open or self._mcp_wizard_active:
            return
        session = self._active_session
        self._status_card_open = False
        self._mcp_manager_open = False
        if isinstance(session, FullScreenPromptSession):
            session.default_buffer.set_document(
                Document(self._status_restore_text, self._status_restore_cursor)
            )
            session.layout.focus(session.default_buffer)
        self._invalidate_prompt()

    def _full_screen_active(self) -> bool:
        return isinstance(self._active_session, FullScreenPromptSession)

    def _composer_can_focus_agent_list(self) -> bool:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return False
        buffer = session.default_buffer
        return (
            self._agent_navigation.list_visible
            and buffer.document.cursor_position_row >= buffer.document.line_count - 1
        )

    def _focus_agent_list(self) -> None:
        self._agent_navigation.focus_list()

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

    def _print_unit(
        self, renderable: RenderableType | None, *, blank_before: bool = False
    ) -> None:
        if blank_before:
            self._presenter.print_unit(renderable, blank_before=True)
        else:
            self._presenter.print_unit(renderable)
    def open_project_instructions_editor(self, path: Path) -> str | Awaitable[str]:
        if not self._full_screen_active(): return f"project instructions: {path}"
        command = shlex.join(project_instructions_editor_command(path))
        async def run() -> str:
            try: status = await run_in_terminal(
                    lambda: subprocess.call(command, shell=True, stderr=subprocess.STDOUT),
                    in_executor=True,
                )
            except (OSError, RuntimeError) as exc: return f"project instructions editor error: {exc} ({path})"
            return (f"project instructions editor error: exited with status {status} ({path})" if status else f"project instructions: {path} (applies to new and resumed sessions)")
        return run()
    def _print_system(self, output: str) -> None:
        self._print_unit(
            Text(
                f"system · {output}",
                style=theme.ERROR if output.startswith("mcp error:") else theme.CHROME,
            )
        )

    def _print_hook_notice(self, output: str) -> None:
        self._print_unit(Text(f"hook · {output}", style=theme.DIM))



from .bootstrap import create_app, format_picker_row

__all__ = ["TUIApp", "build_backend", "create_app", "format_picker_row", "main"]


def __getattr__(name: str) -> object:
    if name == "main":
        from ..cli.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
