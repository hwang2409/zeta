"""Deliver one discussion-fork decision to its original session."""

from __future__ import annotations

from typing import TypedDict

from ...attention_forks import (
    deliver_attention_decision,
    validate_attention_fork,
)
from ...core.session import SessionManager
from ...protocol.types import StructuredToolResult
from .._results import _success_result, text_block
from ..registry import ToolRegistry


class ResolveAttentionArguments(TypedDict):
    decision: str


async def _resolve_attention(
    registry: ToolRegistry, arguments: ResolveAttentionArguments
) -> StructuredToolResult:
    store = registry.session_store
    home = store.session_dir.parent.parent
    metadata = SessionManager(home).read_metadata(store.session_id)
    if registry.project_registry is None:
        raise ValueError("resolve_attention is available only in an attention fork")
    validated = validate_attention_fork(
        home=home,
        current_session_id=store.session_id,
        current_project_id=metadata.project_id,
        directory_fd=store.directory_fd,
    )
    message_id, already_resolved = deliver_attention_decision(
        home,
        validated.record,
        arguments["decision"],
        from_session=store.session_id,
    )
    if already_resolved:
        return _success_result(
            text_block(f"Decision was already delivered ({message_id})."),
            structured_content={"message_id": message_id, "already_resolved": True},
        )
    return _success_result(
        text_block(f"Decision delivered to the original session ({message_id})."),
        structured_content={"message_id": message_id, "already_resolved": False},
    )


def register_fork(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "resolve_attention",
        _resolve_attention,
        description=(
            "Deliver the user's final decision to exactly the original orchestrator "
            "and resolve this attention item. Available only in discussion forks."
        ),
        parameters={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
