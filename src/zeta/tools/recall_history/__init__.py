"""Read-only retrieval of exact messages hidden by compaction."""

from __future__ import annotations

from typing import Any

from ...context_eviction import (
    RECALL_DEFAULT_MAX_CHARS,
    RECALL_HARD_MAX_CHARS,
    recall_history,
)
from ..registry import ToolRegistry


async def _recall_history(registry: ToolRegistry, arguments: dict[str, Any]) -> str:
    return recall_history(
        registry.session_store,
        query=arguments.get("query"),
        seq_start=arguments.get("seq_start"),
        seq_end=arguments.get("seq_end"),
        offset=arguments.get("offset", 0),
        max_chars=arguments.get("max_chars", RECALL_DEFAULT_MAX_CHARS),
    )


TOOL_NAME = "recall_history"


def register(registry: ToolRegistry) -> None:
    """Make the tool surface match the registry's compaction mode.

    Registration is idempotent so a live mode switch can call it again: evict
    mode adds the tool, summary mode removes it. Tool policy still decides
    whether a registered tool is advertised.
    """

    if registry.compaction != "evict":
        registry.unregister(TOOL_NAME)
        return
    registry.register_session_tool(
        TOOL_NAME,
        _recall_history,
        description=(
            "Retrieve exact structured messages hidden by compaction on the active "
            "branch. Use a sequence range or search query. Range results paginate "
            "the deterministic rendered text; continue with the exact offset shown."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "seq_start": {"type": "integer", "minimum": 1},
                "seq_end": {"type": "integer", "minimum": 1},
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset into a rendered sequence range.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": RECALL_HARD_MAX_CHARS,
                    "description": "Maximum rendered-content characters per page.",
                },
            },
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
