"""Failure-atomic provider backend replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.slash import effective_budget_for_model
from ..protocol.types import CompletionBackend, ContextWindowBackend


@dataclass(frozen=True, slots=True)
class PreparedBackendState:
    backend: CompletionBackend
    token_budget: int


def prepare_backend_state(loop: Any, backend: CompletionBackend) -> PreparedBackendState:
    token_budget = loop.context_assembler.token_budget
    provider = getattr(backend, "provider", None)
    model = getattr(backend, "model", None)
    if isinstance(provider, str) and isinstance(model, str):
        token_budget = effective_budget_for_model(provider, model, token_budget)
    if isinstance(backend, ContextWindowBackend):
        backend.set_token_budget(token_budget)
    return PreparedBackendState(backend, token_budget)


def apply_backend_state(loop: Any, state: PreparedBackendState) -> None:
    loop.backend = state.backend
    loop.context_assembler.backend = state.backend
    loop.context_assembler.compaction_policy.backend = state.backend
    loop.context_assembler.token_budget = state.token_budget
