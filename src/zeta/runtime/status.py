"""Shared live-runtime projection for ``/status``."""

from __future__ import annotations

from ..core.slash import (
    MODEL_CONTEXT_WINDOWS,
    SlashStatus,
    UsageTracker,
    compaction_history,
)
from ..core.todo import todo_count_tuple
from .loop import AgentLoop


def build_slash_status(
    loop: AgentLoop,
    *,
    session_id: str,
    provider: str,
    model: str,
    context_files: tuple[str, ...],
    vim_mode: bool,
    pending_approvals: tuple[str, ...],
    usage_tracker: UsageTracker,
    transcript_session_id: str | None = None,
) -> SlashStatus:
    """Project one live agent loop into the shared ``/status`` model."""

    context_assembler = loop.context_assembler
    child_usage = context_assembler.descendant_usage
    items = loop.store.todo_items()
    hooks = loop.hooks
    return SlashStatus(
        session_id=session_id,
        provider=provider,
        model=model,
        retained_tail=context_assembler.retained_tail,
        tokens_used_this_session=context_assembler.tokens_used_this_session,
        tokens_in_current_context=context_assembler.token_count,
        compaction_marker_count=loop.store.compaction_marker_count(),
        pending_approvals=pending_approvals,
        checkpoint_count=loop.store.checkpoint_count(),
        cache_read_input_tokens=context_assembler.cache_read_input_tokens_this_session,
        cache_creation_input_tokens=(
            context_assembler.cache_creation_input_tokens_this_session
        ),
        uncached_input_tokens=context_assembler.uncached_input_tokens_this_session,
        output_tokens_this_session=context_assembler.output_tokens_this_session,
        child_cache_read_input_tokens=child_usage["cache_read_input_tokens"],
        child_cache_creation_input_tokens=child_usage[
            "cache_creation_input_tokens"
        ],
        child_uncached_input_tokens=child_usage["input_tokens"],
        child_output_tokens_this_session=child_usage["output_tokens"],
        context_files=context_files,
        vim_mode=vim_mode,
        plan_mode=loop.plan_mode,
        hooks=(() if hooks is None else hooks.status_entries),
        todo_counts=todo_count_tuple(items) if items else None,
        usage_history=usage_tracker.history,
        usage_cost_by_model=usage_tracker.cost_by_model,
        compaction_history=compaction_history(
            loop.store.replay(), context_assembler.token_counter
        ),
        model_window=MODEL_CONTEXT_WINDOWS.get(provider, {}).get(model),
        mcp_summary=loop.mcp_summary,
        explicit_finish=loop.explicit_finish,
        max_agent_depth=loop.max_agent_depth,
        automatic_memory_failure=(
            loop.memory_reconciler.automatic_memory_status
            if loop.memory_reconciler is not None
            else None
        ),
        transcript_index_failure=loop.transcript_index_failure(
            transcript_session_id or session_id
        ),
    )
