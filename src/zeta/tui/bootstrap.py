"""Composition-root helpers for the TUI, extracted from ``tui/app.py`` to keep
that module under the per-file line cap (see tests/test_module_limits.py)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.hooks import load_hooks_for_provider
from ..core.project_context import (
    ProjectContext,
    PromptArgumentError,
    discover_repo_root,
    resolve_prompt_argument,
)
from ..core.session import (
    SessionError,
    SessionManager,
    SessionPreview,
    env_home,
    format_relative_age,
)
from ..core.slash import resolve_session_budget
from ..loop import AgentLoop
from ..settings import ResolvedConfig
from ..settings import resolve as resolve_settings
from ..tools._user_discovery import apply_external_tools
from . import theme as _theme
from .key_bindings import KeybindingError, resolve_keybindings
from .layout import content_width, resume_picker_line

if TYPE_CHECKING:
    from .app import TUIApp

RECENT_SESSION_LIMIT = 20


def format_picker_row(index: int, preview: SessionPreview) -> str:
    """Render one picker row with age, id, optional name, and preview."""

    age = format_relative_age(preview.updated_at).rjust(8)
    label = f" [{preview.name}]" if preview.name else ""
    return f"{index}. {age}  {preview.session_id[:8]}{label}  {preview.preview}"


def create_app(args: argparse.Namespace) -> TUIApp:
    home = env_home()
    ephemeral = bool(getattr(args, "no_session", False))
    ephemeral_root: Path | None = None
    if ephemeral:
        import tempfile

        ephemeral_root = Path(tempfile.mkdtemp(prefix="zeta-ephemeral-"))
    try:
        return _create_app_with_root(args, home, ephemeral_root)
    except BaseException:
        if ephemeral_root is not None:
            import shutil

            shutil.rmtree(ephemeral_root, ignore_errors=True)
        raise


def _create_app_with_root(
    args: argparse.Namespace,
    home: Path,
    ephemeral_root: Path | None,
) -> TUIApp:
    # Resolve monkey-patch surface at call time so tests that patch
    # ``zeta.tui.app.<name>`` continue to intercept these lookups.
    from . import app as _app

    ephemeral = ephemeral_root is not None
    manager = SessionManager(ephemeral_root if ephemeral else home)
    try:
        system_prompt_override = resolve_prompt_argument(
            getattr(args, "system_prompt", None)
        )
        system_prompt_append = resolve_prompt_argument(
            getattr(args, "append_system_prompt", None)
        )
    except PromptArgumentError as exc:
        raise SessionError(str(exc)) from exc
    project_dir = discover_repo_root(Path.cwd()) / ".zeta"
    loaded_settings = _app.load_settings(home=home, project_dir=project_dir)
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

    override_on_resume = False
    if resuming:
        if resume_id == "":
            previews = manager.list_session_previews(limit=RECENT_SESSION_LIMIT)
            if not previews:
                raise SessionError("no prior zeta session found")
            width = content_width(
                _app.get_terminal_size(fallback=(80, 24)).columns
            )
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
        # Explicit --system-prompt / --append-system-prompt on resume WINS
        # over the snapshot: rebuild the context from the flags, overwrite
        # the snapshot, and warn that the prompt cache will rebuild. The
        # ~/.zeta/SYSTEM.md file path stays snapshot-first — only the
        # CLI-flag path (values non-None here) triggers overwrite.
        override_on_resume = bool(metadata.system_prompt) and (
            system_prompt_override is not None or system_prompt_append is not None
        )
        if metadata.system_prompt and not override_on_resume:
            project_context = ProjectContext(
                metadata.system_prompt,
                tuple(Path(path) for path in metadata.context_files),
            )
        else:
            project_context = _app.load_project_context(
                cwd=Path(metadata.cwd),
                repo_root=discover_repo_root(Path(metadata.cwd)),
                zeta_home=home,
                system_override=system_prompt_override,
                system_append=system_prompt_append,
            )
            persisted = manager.persist_context_snapshot(
                metadata,
                system_prompt=project_context.system_prompt,
                context_files=[str(path) for path in project_context.files],
                overwrite=override_on_resume,
            )
            project_context = ProjectContext(
                persisted.system_prompt,
                tuple(Path(path) for path in persisted.context_files),
                project_context.notices,
            )
    else:
        provider = config.provider
        backend, selected_model = _app.build_backend(
            provider,
            config.model,
            home=home,
            stall_seconds=config.stream_stall_seconds,
            stall_retries=config.stream_stall_retries,
        )
        project_context = _app.load_project_context(
            cwd=Path.cwd(),
            repo_root=discover_repo_root(Path.cwd()),
            zeta_home=home,
            system_override=system_prompt_override,
            system_append=system_prompt_append,
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
        backend, selected_model = _app.build_backend(
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
    theme_notices = _apply_startup_theme(config.theme, home)
    _validate_keybindings(config.keybindings)
    startup_notices = (
        tuple(loaded_settings.notices)
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
    return _app.TUIApp(
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
    )


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


__all__ = ["RECENT_SESSION_LIMIT", "create_app", "format_picker_row"]
