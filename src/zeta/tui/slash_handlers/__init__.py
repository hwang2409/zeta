"""Slash-command handlers for the full-screen terminal UI.

Extracted from ``tui.app`` so the composition root stays under the module line
cap. ``TUIApp`` mixes these in, so they run against its attributes.
"""

from __future__ import annotations

from prompt_toolkit.enums import EditingMode

from ...core.slash import (
    MODEL_CONTEXT_WINDOWS,
    SlashStatus,
    budget_for_model,
    compaction_history,
)
from ...core.todo import todo_count_tuple
from ..models import validate_model_name


def _validate_model_name(provider: str, model: str) -> None:
    validate_model_name(provider, model)


class SlashHandlerMixin:
    """Serve the session-state slash commands the registry dispatches."""

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
        assembler.token_budget = budget
        return f"context budget {previous:,} -> {budget:,}"

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
