"""Opt-in durable archive and structured context-edit tools."""

from __future__ import annotations

from typing import Any

from ...context_strategies import context_strategies
from ...context_strategies.archive import archive_context, restore_context
from ...context_strategies.edit import MAX_REPLACEMENT_CHARS, replace_context
from ..registry import ToolRegistry

_RANGE_PROPERTIES = {
    "seq_start": {"type": "integer", "minimum": 1},
    "seq_end": {"type": "integer", "minimum": 1},
}


async def _archive(registry: ToolRegistry, arguments: dict[str, Any]) -> str:
    result = archive_context(
        registry.session_store,
        seq_start=arguments["seq_start"],
        seq_end=arguments["seq_end"],
        note=arguments.get("note"),
    )
    return (
        f"Archived #{result.archive_id} seq {result.seq_start}–{result.seq_end} "
        f"(~{result.tokens} tokens)."
    )


async def _restore(registry: ToolRegistry, arguments: dict[str, Any]) -> str:
    result = restore_context(registry.session_store, archive_id=arguments["archive_id"])
    return (
        f"Restored #{result.archive_id} seq {result.seq_start}–{result.seq_end} "
        "exactly."
    )


async def _replace(registry: ToolRegistry, arguments: dict[str, Any]) -> str:
    result = replace_context(
        registry.session_store,
        seq_start=arguments["seq_start"],
        seq_end=arguments["seq_end"],
        replacement=arguments["replacement"],
    )
    return (
        f"Replaced visible seq {result.seq_start}–{result.seq_end} with a typed "
        f"model-authored note (~{result.tokens} original tokens hidden)."
    )


def register(registry: ToolRegistry) -> None:
    strategies = context_strategies()
    if "archive" in strategies:
        registry.register_session_tool(
            "context_archive",
            _archive,
            description=(
                "Hide an old active-branch sequence range behind a restorable "
                "placeholder. Ranges are snapped to preserve tool call/result pairs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_RANGE_PROPERTIES,
                    "note": {"type": "string", "maxLength": 500},
                },
                "required": ["seq_start", "seq_end"],
                "additionalProperties": False,
            },
            requires_approval=False,
        )
        registry.register_session_tool(
            "context_restore",
            _restore,
            description="Restore the exact original messages hidden by an active archive.",
            parameters={
                "type": "object",
                "properties": {"archive_id": {"type": "string", "minLength": 1}},
                "required": ["archive_id"],
                "additionalProperties": False,
            },
            requires_approval=False,
        )
    if "edit" in strategies:
        registry.register_session_tool(
            "context_replace",
            _replace,
            description=(
                "Replace an old active-branch sequence range with a plain-text, "
                "assistant-authored model note. Original log entries remain intact."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_RANGE_PROPERTIES,
                    "replacement": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_REPLACEMENT_CHARS,
                    },
                },
                "required": ["seq_start", "seq_end", "replacement"],
                "additionalProperties": False,
            },
            requires_approval=False,
        )
