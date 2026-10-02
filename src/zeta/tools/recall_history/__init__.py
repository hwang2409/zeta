"""Read-only retrieval of exact messages hidden by active-branch compaction."""

from __future__ import annotations

from typing import Any

from ...context_strategies import (
    RECALL_DEFAULT_MAX_CHARS,
    RECALL_HARD_MAX_CHARS,
    ContextTelemetry,
    context_strategies,
    recall_history,
)
from ..registry import ToolRegistry


async def _recall_history(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> str:
    rendered, mode = recall_history(
        registry.session_store,
        query=arguments.get("query"),
        seq_start=arguments.get("seq_start"),
        seq_end=arguments.get("seq_end"),
        max_chars=arguments.get("max_chars", RECALL_DEFAULT_MAX_CHARS),
    )
    ContextTelemetry().emit("recall_history", mode=mode, chars=len(rendered))
    return rendered


def register(registry: ToolRegistry) -> None:
    if "recall" not in context_strategies():
        return
    registry.register_session_tool(
        "recall_history",
        _recall_history,
        description=(
            "Retrieve exact messages hidden by compaction on the active branch. "
            "Use a sequence range for structured originals or query to search them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "seq_start": {"type": "integer", "minimum": 1},
                "seq_end": {"type": "integer", "minimum": 1},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": RECALL_HARD_MAX_CHARS,
                },
            },
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
