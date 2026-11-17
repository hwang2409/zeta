"""Admission and dispatch for provider requests owned by one session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...protocol.types import (
    CompletionBackend,
    ErrorInfo,
    Message,
    StreamEvent,
    StreamEventType,
)

if TYPE_CHECKING:
    from ...telemetry import SpanHandle


class RequestBudgetExceeded(RuntimeError):
    """The exact provider request exceeds the current session budget."""


def ollama_model_window_guidance(
    backend: CompletionBackend, token_budget: int
) -> str | None:
    """Explain an Ollama window that prevents a larger request budget."""

    window = getattr(backend, "model_context_window", None)
    model = getattr(backend, "model", None)
    if (
        getattr(backend, "provider", None) != "ollama"
        or type(window) is not int
        or window != token_budget
        or not isinstance(model, str)
    ):
        return None
    return (
        f"Ollama model {model!r} has a context window ({window} tokens); "
        "increase num_ctx for this model in Ollama (up to its architecture "
        "limit), or use a model with a larger context window"
    )


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
        inference_span: Callable[[CompletionBackend], SpanHandle],
    ) -> None:
        self._estimate = estimate
        self._inference_span = inference_span
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
        *,
        control: Any,
    ) -> Any:
        """Dispatch only an admitted request and record its exact estimate."""

        self._begin_attempt(request.messages, request.estimated_tokens)
        return backend.complete(
            request.messages, request.tool_schemas, control=control
        )

    def update_backend(self, backend: CompletionBackend) -> None:
        """Replace the backend used by admitted compaction requests."""

        self._backend = backend

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
        *,
        control: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Admit and dispatch one compaction request through this object."""

        # the agent loop's one-owner inference span, shared so a summary call is
        # a child of the compaction span that is current here
        from ...runtime.loop._completion import InferenceObservation

        request = await self.admit(messages, tool_schemas)
        with InferenceObservation() as inference:
            inference.start(self._inference_span(self._backend))
            completion = self.dispatch(self._backend, request, control=control)
            try:
                async for event in completion:
                    inference.observe(event)
                    if event.type is StreamEventType.ERROR:
                        inference.error = event.error or ErrorInfo("backend_error", "")
                    try:
                        yield event
                    except GeneratorExit:
                        # the summarizer stops reading at an ERROR it raises; that
                        # error, not the close, is the span's failure
                        if inference.error is None:
                            raise
                        return
            finally:
                close = getattr(completion, "aclose", None)
                if close is not None:
                    await close()
