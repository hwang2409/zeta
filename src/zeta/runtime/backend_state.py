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


def effective_backend_budget(
    backend: CompletionBackend, token_budget: int
) -> int:
    provider = getattr(backend, "provider", None)
    model = getattr(backend, "model", None)
    model_window = getattr(backend, "model_context_window", None)
    if provider == "ollama" and type(model_window) is int and model_window > 0:
        return min(token_budget, model_window)
    if isinstance(provider, str) and isinstance(model, str):
        return effective_budget_for_model(provider, model, token_budget)
    return token_budget


def prepare_backend_state(loop: Any, backend: CompletionBackend) -> PreparedBackendState:
    token_budget = effective_backend_budget(
        backend, loop.context_assembler.token_budget
    )
    if isinstance(backend, ContextWindowBackend):
        backend.set_token_budget(token_budget)
    return PreparedBackendState(backend, token_budget)


def apply_backend_state(loop: Any, state: PreparedBackendState) -> None:
    if isinstance(state.backend, ContextWindowBackend):
        state.backend.set_token_budget(state.token_budget)
    loop.backend = state.backend
    loop.context_assembler.update_backend(state.backend)
    loop.context_assembler.update_token_budget(state.token_budget)
    if loop.step_delivery is not None:
        loop.step_delivery.capacity_changed()
