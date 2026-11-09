"""Resolve attention decisions in orchestrators and discussion forks."""

from __future__ import annotations

import asyncio
from typing import TypedDict

from ...attention_forks import validate_attention_fork
from ...attention_records import AttentionStore
from ...attention_replies import AttentionReplies
from ...project_inbox import AGENT_ORIGIN
from ...protocol.types import StructuredToolResult
from ...runtime.project_registry_state import ProjectRegistryState
from .._results import _success_result, text_block
from ..registry import ToolRegistry


class ResolveAttentionArguments(TypedDict):
    decision: str


class CloseAttentionArguments(TypedDict):
    attention_id: str
    decision: str


async def _resolve_attention(
    registry: ToolRegistry, arguments: ResolveAttentionArguments
) -> StructuredToolResult:
    store = registry.session_store
    home = store.session_dir.parent.parent
    if registry.project_registry is None:
        raise ValueError("resolve_attention is available only in an attention fork")
    state = registry.project_registry_state or ProjectRegistryState(
        registry.project_registry
    )
    validated = await asyncio.to_thread(
        validate_attention_fork,
        home=home,
        current_session_id=store.session_id,
        current_project_id=registry.project_id,
        directory_fd=store.directory_fd,
        registry_state=state,
    )
    outcome = await asyncio.to_thread(
        AttentionReplies(
            home,
            validated.record.session_id,
            registry_state=state,
            origin=AGENT_ORIGIN,
        ).reply,
        validated.record.id,
        arguments["decision"],
        store.session_id,
    )
    record = await asyncio.to_thread(
        AttentionStore(
            home / "sessions" / validated.record.session_id,
            writer_epoch=store.observed_storage_epoch,
        ).get,
        validated.record.id,
    )
    if outcome.already_resolved:
        return _success_result(
            text_block(f"Decision was already delivered ({outcome.message_id})."),
            structured_content={
                "message_id": outcome.message_id,
                "already_resolved": True,
                "decision": record.decision,
            },
        )
    return _success_result(
        text_block(
            f"Decision delivered to the original session ({outcome.message_id})."
        ),
        structured_content={
            "message_id": outcome.message_id,
            "already_resolved": False,
            "decision": record.decision,
        },
    )


async def _close_attention(
    registry: ToolRegistry, arguments: CloseAttentionArguments
) -> StructuredToolResult:
    store = registry.session_store
    home = store.session_dir.parent.parent
    outcome = await asyncio.to_thread(
        AttentionReplies(
            home, store.session_id, registry_state=registry.project_registry_state
        ).resolve_local,
        arguments["attention_id"],
        arguments["decision"],
    )
    record = await asyncio.to_thread(
        AttentionStore(
            store.session_dir, writer_epoch=store.observed_storage_epoch
        ).get,
        arguments["attention_id"],
    )
    if outcome.already_resolved:
        return _success_result(
            text_block(f"Decision was already resolved ({record.id})."),
            structured_content={
                "attention_id": record.id,
                "already_resolved": True,
                "decision": record.decision,
            },
        )
    return _success_result(
        text_block(f"Decision resolved ({record.id})."),
        structured_content={
            "attention_id": record.id,
            "already_resolved": False,
            "decision": record.decision,
        },
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "resolve_attention",
        _close_attention,
        binding="shared",
        description=(
            "Close your own pending attention item when the user answered the decision "
            "in chat. This does not send an inbox message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "attention_id": {"type": "string"},
                "decision": {"type": "string"},
            },
            "required": ["attention_id", "decision"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )


def register_fork(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "resolve_attention",
        _resolve_attention,
        binding="shared",
        description=(
            "Deliver the user's final decision to exactly the original orchestrator "
            "and resolve this attention item."
        ),
        parameters={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
