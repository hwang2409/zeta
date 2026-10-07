"""Deliver one discussion-fork decision to its original session."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TypedDict

from ...attention import AttentionStore, read_attention_fork
from ...core.session import SessionManager
from ...project_inbox import ProjectInbox
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
    fork = read_attention_fork(
        store.session_dir, directory_fd=store.directory_fd
    )
    if not (fork and metadata.project_id and registry.project_registry is not None):
        raise ValueError("resolve_attention is available only in an attention fork")
    decision = arguments["decision"].strip()
    if not decision:
        raise ValueError("decision must be nonempty")
    original_dir = store.session_dir.parent / fork.forked_from_session
    attention = AttentionStore(original_dir)
    record = attention.get(fork.attention_id)
    if record.status == "resolved":
        raise ValueError("attention item is already resolved")
    body = (
        f"User decision relayed from discussion fork {store.session_id}. "
        f"The question was asked at {record.created_at}; check whether the situation "
        f"has changed before acting.\n\nDecision: {decision}"
    )
    ProjectInbox(
        registry.project_registry,
        sessions_root=store.session_dir.parent,
    ).send(
        from_project=metadata.project_id,
        from_session=store.session_id,
        to_project=metadata.project_id,
        to_session=fork.forked_from_session,
        kind="reply",
        title=f"Decision: {record.title}",
        body=body,
    )
    attention.replace(
        replace(
            record,
            status="resolved",
            resolved_at=datetime.now(UTC).isoformat(),
            fork_session_id=store.session_id,
            decision=decision,
        )
    )
    return _success_result(text_block("Decision delivered to the original session."))


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
