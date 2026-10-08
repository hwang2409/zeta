"""Composition-root helpers for the TUI, extracted from ``tui/app.py`` to keep
that module under the per-file line cap (see tests/test_module_limits.py)."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from ..computer.session import ComputerSession
from ..computer.session import prepare_args as prepare_computer_args
from ..config.settings import ResolvedConfig, SettingsError
from ..config.settings import resolve as resolve_settings
from ..core.project_context import (
    PromptArgumentError,
    associate_project_discovery,
    discover_project,
    resolve_prompt_argument,
)
from ..core.session import (
    SessionError,
    SessionManager,
    SessionPreview,
    env_home,
    format_relative_age,
)
from ..models.catalog import REMOVED_PROVIDER_ERROR
from ..protocol.types import CompletionBackend, StreamEvent, StreamEventType
from ..providers.factory import build_backend as build_network_backend
from ..runtime import compose_runtime
from ..runtime.compaction_mode import apply_compaction, persist_compaction
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
            try:
                app.loop.store.append_task_notification(
                    task_id=task_id,
                    command="canceled on session exit",
                    exit_code=None,
                    background_metadata=("run_background", "session_shutdown"),
                )
            except (ValueError, OSError):
                pass
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
    )


def format_picker_row(index: int, preview: SessionPreview) -> str:
    """Render one picker row with age, id, optional name, and preview."""

    age = format_relative_age(preview.updated_at).rjust(8)
    label = f" [{preview.name}]" if preview.name else ""
    text = preview.preview or "(no user message)"
    return f"{index}. {age}  {preview.session_id[:8]}{label}  {text}"


def create_app(args: argparse.Namespace) -> TUIApp:
    home = env_home()
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
    try:
        prepare_computer_args(args)
    except ValueError as exc:
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
            cli_compaction=getattr(args, "compaction", None),
            cli_tools=getattr(args, "tools", None),
            cli_disallowed_tools=getattr(args, "disallowed_tools", None),
            cli_allow_hooks=getattr(args, "allow_hooks", None),
            cli_auto_memory=getattr(args, "auto_memory", None),
        )
    except SettingsError as exc:
        raise SessionError(str(exc)) from exc
    if config.auto_project and not ephemeral and not resuming:
        discovery = associate_project_discovery(discovery, manager.project_registry)
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
            project_id=discovery.project.project_id if discovery.project else None,
            inbox_enabled=config.inbox_enabled,
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
        else:
            manager.record_override(metadata, provider=None, model=model_name)
        effective_budget = loop.context_assembler.token_budget
        if metadata.compaction_budget != effective_budget:
            manager.record_budget(
                metadata,
                budget=effective_budget,
                pinned=budget_pinned,
            )

    def plan_mode_changed(enabled: bool) -> None:
        manager.record_plan_mode(metadata, enabled=enabled)

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
    loop = composition.loop
    cleanup.callback(loop.tool_registry.background_tasks.release_directory)
    resume_compaction = getattr(args, "compaction", None)
    if resuming and resume_compaction is not None:
        # The resumed loop starts in its stored mode. An explicit --compaction
        # switches it through the same policy guard as /compaction. Settings
        # apply to new sessions only.
        try:
            apply_compaction(loop, resume_compaction)
        except ValueError as exc:
            raise SessionError(
                f"--compaction {resume_compaction} refused: {exc}"
            ) from exc
    approval_policy = composition.policy
    selected_model = composition.model
    external_tools = composition.external_tools
    budget_pinned = composition.budget_pinned
    computer = None
    if getattr(args, "computer", False):
        try:
            computer = ComputerSession.attach(
                loop,
                approval_policy,
                home=home,
                backend=getattr(args, "computer_backend", None),
            )
        except ValueError as exc:
            raise SessionError(str(exc)) from exc
        cleanup.callback(computer.stop_spectator)
    theme_notices = _apply_startup_theme(config.theme, home)
    _validate_keybindings(config.keybindings)
    startup_notices = (
        tuple(loaded_settings.notices)
        + project_context.notices
        + external_tools.notices
        + theme_notices
        + (computer.notices if computer is not None else ())
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
            else lambda budget: manager.record_budget(
                metadata, budget=budget, pinned=False
            )
        ),
        on_vim_mode_change=lambda enabled: manager.record_vim_mode(
            metadata, enabled=enabled
        ),
        startup_notices=startup_notices,
        startup_warnings=startup_warnings,
        startup_alerts=startup_alerts,
        external_tools=external_tools,
        workspace_snapshot_cap=config.workspace_snapshot_cap,
        ephemeral_root=ephemeral_root,
        session_name=metadata.name,
        on_name_change=lambda label: manager.record_name(metadata, name=label),
        key_remap=config.keybindings,
        project_dir=repo_root,
        project_eligible=discovery.eligible and discovery.primary_root is not None,
        resumed=resuming,
    )
    app.computer_session = computer
    if composition.memory_reconciler is not None:
        composition.memory_reconciler.notice = app._print_system
    # Headless startup defers this commit until --require-tools validation;
    # interactive TUI startup has completed its validation at this seam.
    if not getattr(args, "prompt", None) and resuming and resume_compaction is not None:
        persist_compaction(loop)
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
