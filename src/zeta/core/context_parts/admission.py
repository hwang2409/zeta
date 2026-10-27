"""Admission and dispatch for provider requests owned by one session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...protocol.types import CompletionBackend, Message, StreamEvent


class RequestBudgetExceeded(RuntimeError):
    """The exact provider request exceeds the current session budget."""


@dataclass(frozen=True, slots=True)
class AdmittedSessionRequest:
    """One immutable, counted provider request ready for dispatch."""

    messages: tuple[Message, ...]
    tool_schemas: tuple[Mapping[str, Any], ...]
    token_budget: int
    estimated_tokens: int


class SessionRequestAdmission:
    """Build and dispatch exact requests through one session admission seam."""

    def __init__(
        self,
        *,
        estimate: Callable[[Sequence[Message], Sequence[Mapping[str, Any]]], int],
        budget: Callable[[], int],
        begin_attempt: Callable[[Sequence[Message], int], None],
        backend: CompletionBackend,
    ) -> None:
        self._estimate = estimate
        self._budget = budget
        self._begin_attempt = begin_attempt
        self._backend = backend

    async def admit(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> AdmittedSessionRequest:
        """Count an exact request off-loop and reject it before dispatch."""

        final_messages = tuple(messages)
        final_tools = tuple(tool_schemas)
        token_budget = self._budget()
        estimated_tokens = await asyncio.to_thread(
            self._estimate, final_messages, final_tools
        )
        if estimated_tokens > token_budget:
            raise RequestBudgetExceeded(
                "provider request exceeds the current context window "
                f"({estimated_tokens} of {token_budget} tokens)"
            )
        return AdmittedSessionRequest(
            final_messages,
            final_tools,
            token_budget,
            estimated_tokens,
        )

    def dispatch(
        self,
        backend: CompletionBackend,
        request: AdmittedSessionRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Dispatch only an admitted request and record its exact estimate."""

        self._begin_attempt(request.messages, request.estimated_tokens)
        return backend.complete(request.messages, request.tool_schemas)

    def update_backend(self, backend: CompletionBackend) -> None:
        """Replace the backend used by admitted compaction requests."""

        self._backend = backend

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        """Admit and dispatch one compaction request through this object."""

        request = await self.admit(messages, tool_schemas)
        completion = self.dispatch(self._backend, request)
        try:
            async for event in completion:
                yield event
        finally:
            close = getattr(completion, "aclose", None)
            if close is not None:
                await close()
