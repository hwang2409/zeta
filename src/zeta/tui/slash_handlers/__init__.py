"""Slash-command handlers for the full-screen terminal UI.

Extracted from ``tui.app`` so the composition root stays under the module line
cap. ``TUIApp`` mixes these in, so they run against its attributes.
"""

from __future__ import annotations

import json
from pathlib import Path

from prompt_toolkit.enums import EditingMode

from ...computer.session import ComputerSession
from ...core.approval import ApprovalDecision
from ...core.project_context import discover_project_root
from ...core.session import SessionError, normalize_session_name
from ...core.slash import (
    MODEL_CONTEXT_WINDOWS,
    SlashStatus,
    budget_for_model,
    compaction_history,
)
from ...core.todo import todo_count_tuple
from ...mcp.prompt_commands import SlashModelInput
from ...project_inbox import InboxError, ProjectInbox
from ...project_memory_commands import run_memory_command
from ...project_registry import ProjectRegistryError
from ...runtime.compaction_mode import run_compaction_command
from ...tools._shared.user_discovery import trust_project_tools
from .. import theme as _theme
from ..models import known_models, match_models, validate_model_name
from .model_picker import ModelPicker


def _validate_model_name(provider: str, model: str) -> None:
    validate_model_name(provider, model)


class SlashHandlerMixin:
    """Serve the session-state slash commands the registry dispatches."""

    # Set by ``create_app`` for a computer session.
    computer_session: ComputerSession | None = None
    computer_requested = False

    async def slash_approve(self, args: str) -> str:
        """Approve the first pending request, or the request with this key."""

        await self._submissions.approval_command(
            ApprovalDecision.ALLOW, args.strip() or None
        )
        return ""

    async def slash_always(self, args: str) -> str:
        """Approve and remember this action for the current session."""

        await self._submissions.approval_command(
            ApprovalDecision.ALLOW, args.strip() or None, always=True
        )
        return ""

    async def slash_deny(self, args: str) -> str:
        """Deny the first pending request, or the request with this key."""

        await self._submissions.approval_command(
            ApprovalDecision.DENY, args.strip() or None
        )
        return ""

    def slash_status(self) -> SlashStatus:
        context_assembler = self.loop.context_assembler
        child_usage = context_assembler.descendant_usage
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
            child_cache_read_input_tokens=child_usage["cache_read_input_tokens"],
            child_cache_creation_input_tokens=child_usage["cache_creation_input_tokens"],
            child_uncached_input_tokens=child_usage["input_tokens"],
            child_output_tokens_this_session=child_usage["output_tokens"],
            context_files=self._context_files,
            vim_mode=self.vim_mode,
            plan_mode=self.loop.plan_mode,
            hooks=(() if self._hooks is None else self._hooks.status_entries),
            todo_counts=todo_count_tuple(items) if items else None,
            usage_history=self._usage_tracker.history,
            usage_cost_by_model=self._usage_tracker.cost_by_model,
            compaction_history=compaction_history_data,
            model_window=MODEL_CONTEXT_WINDOWS.get(self.provider, {}).get(self.model),
            mcp_summary=self.loop.mcp_summary,
            compaction=context_assembler.compaction,
        )

    def model_choices(self) -> tuple[str, ...]:
        """Models the picker and the composer completer offer for this provider."""

        return known_models(self.provider, self._model_catalog)

    @property
    def model_picker_active(self) -> bool:
        return self._model_picker is not None

    def slash_model(self, args: str) -> str:
        """Pick a model from a card, or switch straight to a named one."""

        requested = args.strip()
        if not self._model_catalog_loaded:
            self._start_model_catalog_load()
        busy = bool(self.active or self.pending_approvals)
        if not requested:
            if busy:
                return (
                    f"model: {self.model} (cannot change model while a turn or "
                    "approval is active)"
                )
            return self._open_model_picker(self.model_choices(), query="")
        if busy:
            return "model unchanged: cannot change model while a turn or approval is active"
        choices = self.model_choices()
        if requested not in choices:
            matches = match_models(requested, choices)
            if len(matches) == 1:
                requested = matches[0]
            elif matches:
                return self._open_model_picker(matches, query=requested)
        return self._switch_model(requested)

    def _open_model_picker(self, choices: tuple[str, ...], *, query: str) -> str:
        """Draw the picker card; its keys take over while the composer is empty."""

        self._dismiss_model_picker()
        picker = ModelPicker(self.provider, choices, self.model, query)
        self._model_picker = picker
        self._model_picker_unit = self._presenter.print_unit(picker.render())
        self._invalidate_prompt()
        return ""

    def _repaint_model_picker(self) -> None:
        picker = self._model_picker
        if picker is None:
            return
        unit = self._model_picker_unit
        if unit is not None:
            self._model_picker_unit = self._transcript.replace(unit, picker.render())
        self._invalidate_prompt()

    def _dismiss_model_picker(self) -> None:
        unit = self._model_picker_unit
        self._model_picker = None
        self._model_picker_unit = None
        if unit is not None:
            self._transcript.remove(unit, leading_blank=True)

    def model_picker_move(self, delta: int) -> None:
        picker = self._model_picker
        if picker is None:
            return
        picker.move(delta)
        self._repaint_model_picker()

    def model_picker_select(self) -> None:
        """Run the highlighted choice through the normal ``/model`` path."""

        picker = self._model_picker
        if picker is None:
            return
        choice = picker.selected
        self._dismiss_model_picker()
        self._submit_input(f"/model {choice}")

    def model_picker_cancel(self) -> None:
        if self._model_picker is None:
            return
        self._dismiss_model_picker()
        self._invalidate_prompt()

    def refresh_model_picker(self) -> None:
        """Fold a freshly loaded catalog into an open picker."""

        picker = self._model_picker
        if picker is None:
            return
        choices = self.model_choices()
        if picker.query:
            choices = match_models(picker.query, choices)
        if not choices or choices == picker.choices:
            return
        self._model_picker = picker.with_choices(choices)
        self._repaint_model_picker()

    def _switch_model(self, model: str) -> str:
        try:
            _validate_model_name(self.provider, model)
        except ValueError as exc:
            return f"model unchanged: {exc}"
        if self._model_catalog is None:
            catalog_warning = (
                f"model catalog unavailable for {self.provider} — using anyway"
            )
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
        budget_note = self._retune_budget_for_model()
        notes = [note for note in (catalog_warning, budget_note) if note]
        if notes:
            return f"model: {model} ({'; '.join(notes)})"
        return f"model: {model}"

    def slash_compaction(self, args: str) -> str:
        """Show the compaction mode, or switch it for the next request."""

        busy = (
            "cannot change compaction while a turn or approval is active"
            if self.active or self.pending_approvals
            else None
        )
        return run_compaction_command(self.loop, args, busy=busy)

    def slash_plan(self, args: str) -> str | SlashModelInput:
        """Toggle plan mode or submit a prompt while entering it."""

        requested = args.strip()
        normalized = requested.lower()
        if not requested:
            return self._set_plan_mode_from_command(not self.loop.plan_mode)
        if normalized == "off":
            return self._set_plan_mode_from_command(False)
        if normalized == "on":
            return self._set_plan_mode_from_command(True)
        if normalized == "toggle":
            return self._set_plan_mode_from_command(not self.loop.plan_mode)
        if self.active or self.pending_approvals:
            return (
                "plan mode unchanged: cannot change plan mode while a turn or "
                "approval is active"
            )
        if not self.loop.plan_mode:
            blocked = self._plan_mode_background_blocker()
            if blocked is not None:
                return blocked
        if not self.loop.plan_mode:
            self.loop.set_plan_mode(True)
            self._invalidate_prompt()
        return SlashModelInput(requested)

    def slash_implement(self, args: str) -> str | SlashModelInput:
        """Exit plan mode and submit the explicit implementation request."""

        if args:
            return "implement unchanged: /implement does not accept arguments"
        if not self.loop.plan_mode:
            return "implement unavailable: plan mode is off"
        if self.active or self.pending_approvals:
            return (
                "implement unavailable: cannot leave plan mode while a turn or "
                "approval is active"
            )
        self.loop.set_plan_mode(False)
        self._invalidate_prompt()
        return SlashModelInput("implement the plan you proposed above")

    def _set_plan_mode_from_command(self, enabled: bool) -> str:
        if enabled == self.loop.plan_mode:
            return f"plan mode: {'on' if enabled else 'off'}"
        if self.active or self.pending_approvals:
            return (
                "plan mode unchanged: cannot change plan mode while a turn or "
                "approval is active"
            )
        if enabled:
            blocked = self._plan_mode_background_blocker()
            if blocked is not None:
                return blocked
        self.loop.set_plan_mode(enabled)
        self._invalidate_prompt()
        if enabled:
            return "plan mode: on (read-only tools; deliver the plan as your answer)"
        return "plan mode: off"

    def _plan_mode_background_blocker(self) -> str | None:
        work = tuple(getattr(self.loop, "background_work_descriptions", ()))
        if not work:
            return None
        return (
            "plan mode unchanged: background work is active: "
            f"{', '.join(work)}; stop it or wait"
        )

    def toggle_plan_mode(self) -> None:
        """Toggle plan mode from the keyboard, reporting the same notice."""

        self._print_system(self.slash_plan("toggle"))

    def _retune_budget_for_model(self) -> str | None:
        """Track the new model's context window unless the budget is pinned."""

        if self._on_budget_change is None:
            return None
        budget = budget_for_model(self.provider, self.model)
        assembler = self.loop.context_assembler
        if budget == assembler.token_budget:
            return None
        previous = assembler.token_budget
        try:
            self._on_budget_change(budget)
        except Exception as exc:
            return f"context budget unchanged: {exc}"
        self.loop.set_token_budget(budget)
        return f"context budget {previous:,} -> {budget:,}"

    def slash_tools(self, args: str) -> str:
        """List registered tools or trust pending project tools."""

        requested = args.strip().lower()
        discovery = self._external_tools
        pending = discovery.pending_project_tools if discovery is not None else ()
        if requested in {"", "list"}:
            names = sorted(self.loop.tool_registry.registered_names)
            summary = f"tools: {len(names)} registered"
            if names:
                summary = f"{summary} ({', '.join(names)})"
            policy = self.loop.tool_registry.tool_policy
            if policy.allow_layers:
                layers = [", ".join(layer) or "none" for layer in policy.allow_layers]
                summary += f"; allow: {' AND '.join(layers)}"
            if policy.deny:
                summary += f"; deny: {', '.join(policy.deny)}"
            if pending:
                waiting = ", ".join(
                    sorted({tool.module_stem for tool in pending})
                )
                summary = (
                    f"{summary}; untrusted project tools waiting: {waiting}; "
                    "run /tools trust to enable"
                )
            return summary
        if requested != "trust":
            return "tools: use /tools or /tools trust"
        if self.active or self.pending_approvals:
            return (
                "tools unchanged: cannot trust project tools while a turn or "
                "approval is active"
            )
        if discovery is None or not pending:
            return "tools: no project tools waiting for trust"
        notices, trusted = trust_project_tools(
            self.loop.tool_registry, discovery
        )
        self.loop.tool_schemas = list(self.loop.tool_registry.schemas)
        for notice in notices:
            self._print_system(notice)
        names = sorted({tool.module_stem for tool in trusted})
        return f"tools: trusted {len(names)} project tool(s): {', '.join(names)}"

    def slash_new(self, args: str) -> str:
        """Signal the CLI wrapper to start a fresh session in this window."""

        if args.strip():
            return "new unchanged: /new does not accept arguments"
        if self.active or self.pending_approvals:
            return (
                "new unchanged: cannot start a new session while a turn or "
                "approval is active"
            )
        self.request_new_session()
        return "starting a fresh session..."

    def slash_computer(self, args: str) -> str:
        """Reopen this session as a computer session, or show its links."""

        if args.strip():
            return "computer unchanged: /computer does not accept arguments"
        if self.computer_session is not None:
            return "\n".join(self.computer_session.notices)
        if self.ephemeral_root is not None:
            return (
                "computer unchanged: an ephemeral session cannot be reopened; "
                "start one with zeta --computer"
            )
        if self.active or self.pending_approvals:
            return (
                "computer unchanged: cannot start computer use while a turn or "
                "approval is active"
            )
        self.computer_requested = True
        self.request_exit()
        return (
            "reopening this session for computer use; host tools will be hidden "
            "for the rest of the session"
        )

    def slash_name(self, args: str) -> str:
        """Persist a session label shown in the resume picker."""

        raw = args.strip()
        if not raw:
            current = getattr(self, "_session_name", "")
            return f"session name: {current}" if current else "session name: (unnamed)"
        if self._on_name_change is None:
            return "session name unchanged: names are unavailable for this session"
        try:
            label = normalize_session_name(raw)
        except SessionError as exc:
            return f"session name unchanged: {exc}"
        try:
            self._on_name_change(label)
        except SessionError as exc:
            return f"session name unchanged: {exc}"
        self._session_name = label
        return f"session name: {label}"

    def slash_project(self, args: str) -> str:
        registry = self.loop.project_registry
        if registry is None:
            return "project registry unavailable"
        project = None
        if args.strip() == "init":
            root = discover_project_root(self.loop.store.cwd) or Path(self.loop.store.cwd).resolve()
            try:
                project = registry.find_or_create_for_directory(root)
                metadata = self.loop.manager.associate_project(
                    self.loop.session_metadata, project.project_id
                )
                self.loop._set_runtime_project(metadata)
            except (ProjectRegistryError, OSError, SessionError) as exc:
                return f"could not initialize project: {exc}"
        else:
            if self.loop.session_metadata.project_id:
                try:
                    project = registry.show_project(self.loop.session_metadata.project_id)
                except ProjectRegistryError:
                    project = None
            if project is None:
                project = registry.find_for_directory(self.loop.store.cwd)
        if project is None:
            return "unassociated (run /project init to associate this directory)"
        memory = ", ".join(name for name, _ in registry.load_memory(project.project_id)) or "none"
        return f"project: {project.name} ({project.project_id})\nroot: {project.canonical_integration_root}\nmemory: {memory}"

    def slash_memory(self, args: str) -> str:
        """Show or undo versioned automatic project-memory changes."""
        registry = self.loop.project_registry
        project_id = self.loop.session_metadata.project_id
        if registry is None or project_id is None:
            return "memory: no associated project"
        return run_memory_command(registry, project_id, args)

    def slash_inbox(self, args: str) -> str:
        if args.strip():
            return "usage: /inbox"
        registry = self.loop.project_registry
        project_id = self.loop.session_metadata.project_id
        if registry is None or project_id is None:
            return "project inbox unavailable: session is not associated with a project"
        try:
            state = ProjectInbox(
                registry, sessions_root=registry.root.parent / "sessions"
            ).list(project_id)
        except (InboxError, OSError) as exc:
            return f"project inbox unavailable: {exc}"
        return json.dumps(state, indent=2, sort_keys=True)

    def slash_theme(self, args: str) -> str:
        """Show, list, or switch the active TUI theme."""

        home = getattr(self, "_zeta_home", None)
        requested = args.strip()
        if not requested or requested.lower() == "list":
            available = ", ".join(_theme.list_available(home))
            return (
                f"theme: {_theme.active_palette().name} · available: {available}"
            )
        palette, notice = _theme.resolve_palette(requested, home=home)
        if palette is None:
            if notice is not None:
                return f"theme unchanged: {notice}"
            available = ", ".join(_theme.list_available(home))
            return (
                f"theme unchanged: unknown theme {requested!r}; "
                f"available: {available}"
            )
        _theme.set_active_palette(palette)
        self._on_theme_change()
        message = f"theme: {palette.name}"
        return message if notice is None else f"{message} ({notice})"

    def _on_theme_change(self) -> None:
        """Refresh the prompt style cache and re-render the transcript."""

        # Cached DynamicStyle values were built from the previous palette; drop
        # them so the next _prompt_style() call rebuilds against the new one.
        prompt_styles = getattr(self, "_prompt_styles", None)
        if isinstance(prompt_styles, dict):
            prompt_styles.clear()
        self._rebuild_transcript()
        self._invalidate_prompt()

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


__all__ = ["SlashHandlerMixin"]
