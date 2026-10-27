"""Composition-root helpers for the TUI, extracted from ``tui/app.py`` to keep
that module under the per-file line cap (see tests/test_module_limits.py)."""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from ..accounts import AccountId, AccountStore, account_provider
from ..agent.pending_delivery import (
    SHUTDOWN_DELIVERY_TIMEOUT_SECONDS,
    NotificationInput,
)
from ..config.settings import ResolvedConfig, SettingsError
from ..config.settings import resolve as resolve_settings
from ..core.project_context import (
    PromptArgumentError,
    associate_project_discovery,
    discover_project,
    resolve_project_context,
    resolve_prompt_argument,
)
from ..core.session import (
    SessionError,
    SessionManager,
    SessionMetadata,
    SessionPreview,
    env_home,
    format_relative_age,
    persist_metadata_change,
)
from ..core.slash import resolve_session_budget
from ..models.catalog import REMOVED_PROVIDER_ERROR
from ..project_registry import ProjectRegistryError
from ..protocol.types import CompletionBackend, StreamEvent, StreamEventType
from ..providers.factory import build_backend as build_network_backend
from ..runtime import compose_runtime
from ..runtime.prompt_resume import resume_prompt
from ..skills import (
    discover_session_skills,
)
from ..skills.agent_catalog import discover_session_agents
from ..tools._shared.process import (
    BackgroundTaskNotice,
    BackgroundTaskShutdownNotice,
)
from . import theme as _theme
from .key_bindings import KeybindingError, resolve_keybindings
from .layout import content_width, resume_picker_line
from .render import render_event

if TYPE_CHECKING:
    from .app import TUIApp

RECENT_SESSION_LIMIT = 20


class _SessionRecency:
    """Coalesce best-effort recency writes onto one daemon worker."""

    def __init__(
        self,
        manager: SessionManager,
        metadata: SessionMetadata,
    ) -> None:
        self._manager = manager
        self._session_id = metadata.session_id
        self._dirty = False
        self._task: asyncio.Task[None] | None = None

    def refresh(self) -> None:
        self._dirty = True
        self._start()

    async def flush(self) -> None:
        running = self._task if self._task is not None and not self._task.done() else None
        try:
            async with asyncio.timeout(0.5):
                self._start()
                if self._task is not None:
                    await asyncio.shield(self._task)
                if running is not None and self._dirty:
                    self._start()
                    if self._task is not None:
                        await asyncio.shield(self._task)
        except TimeoutError:
            if self._task is not None:
                self._task.cancel()

    def _start(self) -> None:
        if self._task is None or self._task.done():
            if self._task is not None:
                self._task.result()
            if self._dirty:
                self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while self._dirty:
            self._dirty = False
            if not await self._attempt():
                self._dirty = True
                return

    async def _attempt(self) -> bool:
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[bool] = loop.create_future()

        def finish(result: bool | BaseException) -> None:
            if completed.done():
                return
            if isinstance(result, BaseException):
                completed.set_exception(result)
            else:
                completed.set_result(result)

        def run() -> None:
            try:
                result: bool | BaseException = self._manager.refresh_recency(
                    self._session_id
                )
            except BaseException as exc:  # noqa: BLE001 - return worker failure
                result = exc
            try:
                loop.call_soon_threadsafe(finish, result)
            except RuntimeError:
                pass

        threading.Thread(target=run, name="zeta-recency", daemon=True).start()
        return await completed


def background_notice(
    app: Any,
    notice: BackgroundTaskNotice | BackgroundTaskShutdownNotice | str,
) -> None:
    """Print only lifecycle notices that have no canonical TUI receipt."""

    message: str | None
    if isinstance(notice, BackgroundTaskNotice):
        # Starts have a tool card, natural exits have a durable notification,
        # and explicit kills have a task_kill card. The structured phase keeps
        # this decision independent of user-controlled command text.
        message = None
    elif isinstance(notice, BackgroundTaskShutdownNotice) and notice.tasks:
        # Macro shutdowns retain their durable canceled receipt. Ordinary
        # run_background tasks have no shutdown notification, so keep the
        # immediate shutdown notice for those task ids.
        task_ids = [
            task_id
            for task_id, owner in notice.tasks
            if owner == "run_background"
        ]
        message = (
            "background tasks killed on session exit: " + ", ".join(task_ids)
            if task_ids
            else None
        )
    else:
        message = notice.message if not isinstance(notice, str) else notice
    output_failed = False
    if message is not None:
        try:
            app._print(Text(message, style=_theme.DIM))
        except (ValueError, BrokenPipeError, OSError):
            output_failed = True
    if output_failed and isinstance(notice, BackgroundTaskShutdownNotice):
        for task_id, owner in notice.tasks:
            if owner != "run_background":
                continue
            app.loop._create_task(
                app.loop.step_delivery.produce(
                    NotificationInput(
                        "task_exit",
                        {
                            "task_id": task_id,
                            "headline": "canceled on session exit",
                            "exit_code": None,
                            "output_tail": "",
                            "background_owner": "run_background",
                            "background_phase": "session_shutdown",
                        },
                        dedupe_key=task_id,
                        deadline=time.monotonic() + SHUTDOWN_DELIVERY_TIMEOUT_SECONDS,
                    )
                )
            )
    try:
        app._invalidate_prompt()
    except (ValueError, BrokenPipeError, OSError):
        pass


def surface_shutdown_notifications(app: Any, pending_before: set[str]) -> None:
    """Best-effort print newly canceled macro receipts without consuming them."""

    for entry in app.loop.store.agent_notifications():
        if (
            entry.id in pending_before
            or app.loop.store.is_agent_notification_presented_to_tui(entry.id)
            or entry.data.get("background_owner") != "background_macro"
            or entry.data.get("status") != "canceled"
            or entry.data.get("background_phase") != "session_shutdown"
        ):
            continue
        data = dict(entry.data)
        try:
            app._print_unit(
                render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=data)),
                blank_before=True,
            )
        except (ValueError, BrokenPipeError, OSError):
            continue
        app.loop.store.mark_agent_notification_presented_to_tui(entry.id)


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
    stall_seconds: float | None = None,
    stall_retries: int | None = None,
    ollama_base_url: str | None = None,
    token_budget: int | None = None,
    account: AccountId | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected network provider."""

    return build_network_backend(
        provider,
        model,
        home=home,
        stall_seconds=stall_seconds,
        stall_retries=stall_retries,
        ollama_base_url=ollama_base_url,
        token_budget=token_budget,
        account=account,
    )


def format_picker_row(index: int, preview: SessionPreview) -> str:
    """Render one picker row with age, id, optional name, and preview."""

    age = format_relative_age(preview.updated_at).rjust(8)
    label = f" [{preview.name}]" if preview.name else ""
    text = preview.preview or "(no user message)"
    return f"{index}. {age}  {preview.session_id[:8]}{label}  {text}"


def create_app(args: argparse.Namespace) -> TUIApp:
    home = env_home()
    AccountStore(home).migrate()
    ephemeral = bool(getattr(args, "no_session", False))
    ephemeral_root: Path | None = None
    if ephemeral:
        import tempfile

        ephemeral_root = Path(tempfile.mkdtemp(prefix="zeta-ephemeral-"))
    try:
        with ExitStack() as cleanup:
            app = _create_app_with_root(args, home, ephemeral_root, cleanup)
            cleanup.pop_all()
            return app
    except BaseException:
        if ephemeral_root is not None:
            import shutil

            shutil.rmtree(ephemeral_root, ignore_errors=True)
        raise


def _create_app_with_root(
    args: argparse.Namespace,
    home: Path,
    ephemeral_root: Path | None,
    cleanup: ExitStack,
) -> TUIApp:
    # Resolve monkey-patch surface at call time so tests that patch
    # ``zeta.tui.app.<name>`` continue to intercept these lookups.
    from . import app as _app

    ephemeral = ephemeral_root is not None
    manager = SessionManager(ephemeral_root if ephemeral else home)
    opened = None
    try:
        system_prompt_override = resolve_prompt_argument(
            getattr(args, "system_prompt", None)
        )
        system_prompt_append = resolve_prompt_argument(
            getattr(args, "append_system_prompt", None)
        )
    except PromptArgumentError as exc:
        raise SessionError(str(exc)) from exc
    invocation_cwd = Path.cwd()
    continue_session = getattr(args, "continue_session", False)
    resume_id = getattr(args, "resume", None)
    force_provider = getattr(args, "force_provider", False)
    resuming = continue_session or resume_id is not None
    explicit_resume = resume_id is not None
    if force_provider and not resuming:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and getattr(args, "model", None) is None:
        raise SessionError("--force-provider requires --model")
    if resuming:
        if resume_id == "":
            previews = manager.list_session_previews(limit=RECENT_SESSION_LIMIT)
            if not previews:
                raise SessionError("no prior zeta session found")
            width = content_width(_app.get_terminal_size(fallback=(80, 24)).columns)
            print(resume_picker_line("recent zeta sessions:", width))
            for index, preview in enumerate(previews, start=1):
                print(resume_picker_line(format_picker_row(index, preview), width))
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
            recent = manager.find_most_recent(cwd=invocation_cwd)
            opened = manager.open(recent.session_id)
        cleanup.enter_context(opened.store)
        metadata = opened.metadata
        if metadata.provider == "fake":
            raise SessionError(REMOVED_PROVIDER_ERROR)
    effective_cwd = (
        Path(metadata.cwd) if resuming and explicit_resume else invocation_cwd
    )
    discovery = discover_project(effective_cwd, user_home=manager.user_home)
    repo_root = discovery.primary_root or discovery.cwd
    settings_root = invocation_cwd if explicit_resume else repo_root
    project_dir = settings_root / ".zeta"
    try:
        loaded_settings = _app.load_settings(home=home, project_dir=project_dir)
        config: ResolvedConfig = resolve_settings(
            loaded_settings.settings,
            cli_provider=getattr(args, "provider", None),
            cli_model=getattr(args, "model", None),
            cli_yolo=getattr(args, "yolo", None),
            cli_token_budget=getattr(args, "token_budget", None),
            cli_tools=getattr(args, "tools", None),
            cli_disallowed_tools=getattr(args, "disallowed_tools", None),
            cli_allow_hooks=getattr(args, "allow_hooks", None),
            cli_auto_memory=getattr(args, "auto_memory", None),
            default_provider=metadata.provider if resuming else None,
        )
    except SettingsError as exc:
        raise SessionError(str(exc)) from exc
    if config.auto_project and not ephemeral and not resuming:
        discovery = associate_project_discovery(discovery, manager.project_registry)
    cwd_project = resolve_project_context(discovery, manager.project_registry)
    session_project = discovery.project
    if resuming and metadata.project_id is not None:
        try:
            session_project = manager.project_registry.show_project(metadata.project_id)
        except ProjectRegistryError:
            session_project = None
    override_on_resume = False
    if resuming:
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
        # Explicit --system-prompt / --append-system-prompt on resume WINS
        # over the snapshot: rebuild the context from the flags, overwrite
        # the snapshot, and warn that the prompt cache will rebuild. The
        # ~/.zeta/SYSTEM.md file path stays snapshot-first — only the
        # CLI-flag path (values non-None here) triggers overwrite.
        override_on_resume = (
            system_prompt_override is not None or system_prompt_append is not None
        )
        resumed_prompt = resume_prompt(
            metadata,
            manager=manager,
            store=opened.store,
            home=home,
            repo_root=repo_root,
            inbox_enabled=config.inbox_enabled,
            session_project=session_project,
            cwd_project=cwd_project,
            system_override=system_prompt_override,
            system_append=system_prompt_append,
            context_loader=_app.load_project_context,
        )
        project_context = resumed_prompt.context
        skill_catalog = resumed_prompt.skill_catalog
        agent_catalog = resumed_prompt.agent_catalog
        if override_on_resume:
            manager.touch(metadata)
    else:
        provider = config.provider
        model = config.model
        skill_catalog = discover_session_skills(home=home, project_dir=repo_root)
        agent_catalog = discover_session_agents(home=home, project_dir=repo_root)
        project_context = _app.load_project_context(
            cwd=effective_cwd,
            repo_root=repo_root,
            zeta_home=home,
            system_override=system_prompt_override,
            system_append=system_prompt_append,
            catalog=skill_catalog,
            session_project=session_project,
            cwd_project=cwd_project,
            inbox_enabled=config.inbox_enabled,
        )
    recency: _SessionRecency | None = None
    pending_override = None
    if resuming and mismatches:
        pending_override = (
            provider if provider != metadata.provider else None,
            model if model != metadata.model else None,
        )

    async def completion_success() -> None:
        nonlocal pending_override
        if pending_override is not None:
            selected_provider = pending_override[0] or metadata.provider
            selected_model = pending_override[1] or metadata.model

            def prepare(current: SessionMetadata) -> SessionMetadata:
                return manager._apply_session_settings(
                    current,
                    model=selected_model,
                    approval_mode=current.approval_mode
                    or approval_policy.default.value,
                    budget=loop.context_assembler.token_budget,
                    provider=selected_provider,
                    model_fallback=current.model_fallback,
                )

            authoritative = await loop.account_context.switch_model(
                loop, prepare=prepare, build_backend=lambda _current: None
            )
            if (
                authoritative.provider != selected_provider
                or authoritative.model != selected_model
            ):
                raise OSError("session override was not persisted")
            pending_override = None
        if recency is not None:
            recency.refresh()

    async def model_changed(model_name: str) -> SessionMetadata:
        nonlocal pending_override
        context = loop.account_context

        def prepare(current: SessionMetadata) -> SessionMetadata:
            budget, _ = resolve_session_budget(
                current.compaction_budget,
                current.budget_pinned,
                provider,
                model_name,
                None,
            )
            return manager._apply_session_settings(
                current,
                model=model_name,
                approval_mode=current.approval_mode or approval_policy.default.value,
                budget=budget,
                provider=provider,
                model_fallback=current.model_fallback,
            )

        def prepare_backend(current: SessionMetadata) -> CompletionBackend:
            selected = current.selected_accounts.get(
                account_provider(current.provider) or ""
            )
            account = AccountId(*selected.split(":", 1)) if selected is not None else None
            backend, _ = _app.build_backend(
                current.provider,
                current.model,
                home=home,
                stall_seconds=config.stream_stall_seconds,
                stall_retries=config.stream_stall_retries,
                token_budget=current.compaction_budget,
                account=account,
            )
            return backend

        authoritative = await context.switch_model(
            loop, prepare=prepare, build_backend=prepare_backend
        )
        if authoritative.provider != provider or authoritative.model != model_name:
            raise OSError("session model was not persisted")
        pending_override = None
        return authoritative

    async def persist_budget(budget: int) -> None:
        await persist_metadata_change(
            manager,
            metadata,
            lambda candidate: manager.record_budget(
                candidate, budget=budget, pinned=False
            ),
        )

    async def persist_vim_mode(enabled: bool) -> None:
        await persist_metadata_change(
            manager,
            metadata,
            lambda candidate: manager.record_vim_mode(candidate, enabled=enabled),
        )

    async def persist_name(label: str) -> None:
        await persist_metadata_change(
            manager,
            metadata,
            lambda candidate: manager.record_name(candidate, name=label),
        )

    def plan_mode_changed(enabled: bool) -> asyncio.Task[None] | None:
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            manager.record_plan_mode(metadata, enabled=enabled)
            return None
        return event_loop.create_task(
            persist_metadata_change(
                manager,
                metadata,
                lambda candidate: manager.record_plan_mode(
                    candidate, enabled=enabled
                ),
            )
        )

    max_turns_override = getattr(args, "max_turns", None)
    composition = compose_runtime(
        home=home,
        cwd=effective_cwd,
        manager=manager,
        config=config,
        provider=provider,
        model=model,
        project_context=project_context,
        backend_builder=_app.build_backend,
        opened=opened,
        on_completion_success=completion_success,
        on_plan_mode_change=plan_mode_changed,
        max_turns=max_turns_override,
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
        auto_project=not ephemeral,
        project_discovery=discovery,
    )
    if opened is None:
        cleanup.enter_context(composition.opened.store)
    opened = composition.opened
    metadata = opened.metadata
    recency = _SessionRecency(manager, metadata)
    loop = composition.loop
    cleanup.callback(loop.tool_registry.background_tasks.release_directory)
    approval_policy = composition.policy
    selected_model = composition.model
    external_tools = composition.external_tools
    budget_pinned = composition.budget_pinned
    theme_notices = _apply_startup_theme(config.theme, home)
    _validate_keybindings(config.keybindings)
    startup_notices = (
        tuple(loaded_settings.notices)
        + composition.settings_notices
        + project_context.notices
        + external_tools.notices
        + theme_notices
    )
    startup_warnings = tuple(loaded_settings.warnings) + external_tools.warnings
    if ephemeral:
        startup_warnings = (
            "ephemeral session: nothing will be persisted",
            *startup_warnings,
        )
    startup_alerts: tuple[str, ...] = ()
    if resuming and override_on_resume:
        startup_alerts = (
            "system prompt overridden for this session; prompt cache will rebuild",
        )
    app = _app.TUIApp(
        loop,
        provider=provider,
        model=selected_model,
        zeta_home=home,
        verbose=args.verbose,
        history_path=(ephemeral_root if ephemeral else home) / "history",
        approval_policy=approval_policy,
        context_files=[str(path) for path in project_context.files],
        on_model_change=model_changed,
        vim_mode=metadata.vim_mode,
        on_budget_change=(
            None
            if budget_pinned
            else persist_budget
        ),
        on_vim_mode_change=persist_vim_mode,
        startup_notices=startup_notices,
        startup_warnings=startup_warnings,
        startup_alerts=startup_alerts,
        external_tools=external_tools,
        workspace_snapshot_cap=config.workspace_snapshot_cap,
        ephemeral_root=ephemeral_root,
        session_name=metadata.name,
        on_name_change=persist_name,
        key_remap=config.keybindings,
        project_dir=repo_root,
        project_eligible=discovery.eligible and discovery.primary_root is not None,
        resumed=resuming,
        on_session_close=recency.flush,
    )
    if composition.memory_reconciler is not None:
        composition.memory_reconciler.notice = app._print_system
    return app


def _apply_startup_theme(name: str | None, home: Path) -> tuple[str, ...]:
    """Apply the theme selected via settings; return dim notices for failures."""

    if name is None:
        _theme.set_active_palette(_theme.DARK)
        return ()
    palette, notice = _theme.resolve_palette(name, home=home)
    if palette is None:
        fallback = _theme.DARK
        _theme.set_active_palette(fallback)
        if notice is not None:
            return (notice,)
        return (f"theme · unknown theme {name!r}; using {fallback.name!r}",)
    _theme.set_active_palette(palette)
    return () if notice is None else (notice,)


def _validate_keybindings(remap: object) -> None:
    """Loud-fail keybindings validation happens before the TUI opens.

    ``settings.py`` only checks the table+string shape; unknown ACTIONs and
    unparseable KEYs are rejected here so a bad file cannot silently disable
    a shortcut.
    """

    try:
        resolve_keybindings(remap)  # type: ignore[arg-type]
    except KeybindingError as exc:
        raise SessionError(str(exc)) from exc


__all__ = [
    "RECENT_SESSION_LIMIT",
    "background_notice",
    "build_backend",
    "create_app",
    "format_picker_row",
]
