"""Ask the user for a decision without blocking other work."""

from __future__ import annotations

from typing import Literal, TypedDict

from ...attention_records import AttentionStore
from ...protocol.types import StructuredToolResult
from .._results import _success_result, text_block
from ..registry import ToolRegistry


class RequestAttentionArguments(TypedDict, total=False):
    title: str
    why: str
    options: list[str]
    recommendation: str
    kind: Literal["decision", "answer", "decided", "deferred"]


async def _request_attention(
    registry: ToolRegistry, arguments: RequestAttentionArguments
) -> StructuredToolResult:
    store = registry.session_store
    branch = store.active_branch_snapshot()
    anchor = branch[-1] if branch else None
    record = AttentionStore(
        store.session_dir,
        writer_epoch=store._storage_lease.writer_epoch,
    ).request(
        session_id=store.session_id,
        project_id=registry.project_id,
        entry_id=anchor.id if anchor else None,
        entry_seq=anchor.seq if anchor else None,
        title=arguments["title"],
        why=arguments["why"],
        options=arguments.get("options", ()),
        recommendation=arguments.get("recommendation"),
        kind=arguments.get("kind", "decision"),
    )
    return _success_result(
        text_block(f"Attention requested: {record.title} ({record.id})"),
        structured_content={"attention_id": record.id, "kind": record.kind},
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "request_attention",
        _request_attention,
        description=(
            "Use once per topic to put important user attention in the inbox, then continue work. "
            "Set kind to decision only when the user must choose; include options and a recommendation when useful. "
            "Set kind to answer for a direct answer to an earlier user question, decided for an important decision you made for the user, or deferred for an important issue you left for the user. "
            "Write why as self-contained context, assuming the user has read nothing since their last message. "
            "Do not use this for routine progress, and do not repeat an open item."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short topic title."},
                "why": {
                    "type": "string",
                    "description": "Self-contained context, result, and any action the user can take.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decision choices; invalid for answer, decided, or deferred.",
                },
                "recommendation": {
                    "type": "string",
                    "description": "Recommended decision; invalid for answer, decided, or deferred.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["decision", "answer", "decided", "deferred"],
                    "description": "Item purpose. Omit for a decision that needs the user's answer.",
                },
            },
            "required": ["title", "why"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
