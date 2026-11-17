"""Resolve attention decisions in orchestrators and discussion forks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from ...attention_decisions import _top_level_sessions
from ...attention_forks import validate_attention_fork
from ...attention_records import (
    AttentionRecord,
    AttentionStore,
    observed_epoch_for_session,
)
from ...attention_replies import AttentionReplies, ReplyOutcome
from ...core.session_execution import (
    OwnerHint,
    SessionExecutionLease,
    SessionOwnedError,
)
from ...core.session_files import child_directory, session_root
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


def _foreign_matches(
    registry: ToolRegistry, attention_id: str
) -> tuple[tuple[str, Path], ...]:
    store = registry.session_store
    if registry.project_id is None:
        raise ValueError("attention record ownership does not match the current project")
    matches: list[tuple[str, Path]] = []
    for session_id, session_dir in _top_level_sessions(
        store.session_dir.parent.parent, registry.project_id
    ):
        if session_id == store.session_id:
            continue
        try:
            record = AttentionStore(
                session_dir, writer_epoch=observed_epoch_for_session(session_dir)
            ).get(attention_id)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"attention record {attention_id} in session {session_id} is broken"
            ) from exc
        if record.project_id != registry.project_id or record.session_id != session_id:
            raise ValueError("attention record ownership does not match its project")
        matches.append((session_id, session_dir))
    return tuple(matches)


def _close_attention_sync(
    registry: ToolRegistry, attention_id: str, decision: str
) -> tuple[ReplyOutcome, AttentionRecord]:
    store = registry.session_store
    home = store.session_dir.parent.parent
    own_store = AttentionStore(
        store.session_dir, writer_epoch=store.observed_storage_epoch
    )
    try:
        own = own_store.get(attention_id)
    except FileNotFoundError:
        own = None
    except (OSError, ValueError) as exc:
        raise ValueError("attention record in the current session is broken") from exc
    if own is not None:
        if own.session_id != store.session_id or (
            registry.project_id is not None and own.project_id != registry.project_id
        ):
            raise ValueError("attention record ownership does not match the current project")
        outcome = AttentionReplies(
            home, store.session_id, registry_state=registry.project_registry_state
        ).resolve_local(attention_id, decision)
        return outcome, own_store.get(attention_id)

    matches = _foreign_matches(registry, attention_id)
    if not matches:
        raise ValueError("attention item was not found in a dead predecessor session")
    if len(matches) > 1:
        owners = ", ".join(session_id for session_id, _ in matches)
        raise ValueError(
            f"attention item is ambiguous across predecessor sessions: {owners}"
        )
    owner_id, owner_dir = matches[0]
    with session_root(owner_dir.parent) as root_fd, child_directory(
        root_fd, owner_id
    ) as directory_fd:
        try:
            lease = SessionExecutionLease.acquire(
                directory_fd,
                session_id=owner_id,
                owner=OwnerHint("headless"),
                recovery=True,
                require_unowned=True,
                home=home,
            )
        except SessionOwnedError as exc:
            raise ValueError(
                "cannot resolve an attention item owned by a live session"
            ) from exc
        try:
            replies = AttentionReplies(
                home, owner_id, registry_state=registry.project_registry_state
            )
            outcome = replies.resolve_local(attention_id, decision)
            record = AttentionStore(
                owner_dir, writer_epoch=observed_epoch_for_session(owner_dir)
            ).get(attention_id)
            return outcome, record
        finally:
            lease.close()


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
    outcome, record = await asyncio.to_thread(
        _close_attention_sync,
        registry,
        arguments["attention_id"],
        arguments["decision"],
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
