"""Internal values produced while assembling one provider context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...context_accounting import context_digest
from ...protocol.types import Message
from ..store import ConversationEntry


@dataclass(frozen=True, slots=True)
class AssembledContext:
    messages: list[Message]
    token_count: int
    compacted: bool = False
    _cached_digest: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def digest(self) -> str:
        """Return the canonical digest, computing it only when requested."""
        digest = self._cached_digest
        if digest is None:
            digest = context_digest(self.messages)
            object.__setattr__(self, "_cached_digest", digest)
        return digest


@dataclass(frozen=True, slots=True)
class _ContextItem:
    entry: ConversationEntry | None
    message: Message
    fixed: bool = False
    # A resident stored value whose content alone determines `message`, so its
    # token count can be reused on later turns.
    count_source: object | None = None


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
    context: AssembledContext
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
    fact_floor: Mapping[str, Any] | None = None
