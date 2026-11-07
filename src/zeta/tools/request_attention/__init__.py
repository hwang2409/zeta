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
            "Use once when a decision only the user can make is pending, or to place "
            "important information in the user's attention inbox. "
            "Use answer for a direct answer, decided for an important choice you made, "
            "deferred for an important issue left for the user, and decision when only "
            "the user can choose. then continue other work. Do not repeat that you are waiting on the user. Write why self-contained, assuming the user has read nothing since their last message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short decision title."},
                "why": {
                    "type": "string",
                    "description": "Self-contained context and what the user must decide.",
                },
                "options": {"type": "array", "items": {"type": "string"}},
                "recommendation": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["decision", "answer", "decided", "deferred"],
                    "description": "Attention item kind; defaults to decision.",
                },
            },
            "required": ["title", "why"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
