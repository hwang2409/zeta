"""Internal values produced while assembling one provider context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...protocol.types import Message
from ..store import ConversationEntry


@dataclass(frozen=True, slots=True)
class AssembledContext:
    messages: list[Message]
    token_count: int
    digest: str
    compacted: bool = False


@dataclass(frozen=True, slots=True)
class _ContextItem:
    entry: ConversationEntry | None
    message: Message
    fixed: bool = False


@dataclass(frozen=True, slots=True)
class _AssemblyPreparation:
    items: list[_ContextItem]
    result_seqs: dict[int, int]
    boundary: int
    system_prompt: Message | None
    system_messages: list[Message]
    latest_user: int | None
    committed_messages: list[Message]
    committed_tokens: int
    all_messages: list[Message]
    should_compact: bool
    adaptive_tail: bool
    pinned_user: _ContextItem | None
    prefix_has_uncompacted_items: bool


@dataclass(frozen=True, slots=True)
class _EvictionPlan:
    outcome: str
    context: AssembledContext | None = None
    source_start: int | None = None
    source_end: int | None = None
    replaces: tuple[str, ...] = ()
    pinned_message: Message | None = None
    view: tuple[Mapping[str, Any], ...] = ()
    telemetry: Mapping[str, Any] | None = None
