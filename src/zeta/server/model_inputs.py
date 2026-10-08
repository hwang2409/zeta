"""Bounded server ownership for generated model input."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..model_input import ModelInputEnvelope
from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
)


@dataclass(frozen=True, slots=True)
class ResolvedModelInput:
    text: str
    display_text: str
    origin: MessageOrigin
    message: Message | None


class PendingModelInputs:
    """Issue and consume single-use IDs for server-generated envelopes."""

    def __init__(self, *, limit: int = 32) -> None:
        self._limit = limit
        self._inputs: dict[str, tuple[str, ModelInputEnvelope]] = {}

    def register(self, session_id: str, envelope: ModelInputEnvelope) -> str:
        while len(self._inputs) >= self._limit:
            del self._inputs[next(iter(self._inputs))]
        input_id = secrets.token_urlsafe(24)
        self._inputs[input_id] = (session_id, envelope)
        return input_id

    def registrar(
        self, session_id: str, *, enabled: bool
    ) -> Callable[[ModelInputEnvelope], str] | None:
        if not enabled:
            return None
        return lambda envelope: self.register(session_id, envelope)

    def resolve(
        self, session_id: str, params: Mapping[str, object], *, enabled: bool
    ) -> ResolvedModelInput:
        has_text, has_input_id = "text" in params, "input_id" in params
        if has_text == has_input_id:
            raise ValueError("send requires exactly one of text or input_id")
        if has_text:
            text = params["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a nonempty string")
            return ResolvedModelInput(text, text, MessageOrigin.USER, None)
        if not enabled:
            raise ValueError("input_id requires the negotiated model_input_ids feature")
        input_id = params["input_id"]
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("input_id must be a nonempty string")
        pending = self._inputs.pop(input_id, None)
        if pending is None or pending[0] != session_id:
            raise ValueError("input_id is invalid or expired")
        envelope = pending[1]
        message = Message(
            MessageRole.USER,
            [TextContent(envelope.text)],
            metadata={
                MESSAGE_ORIGIN_METADATA: envelope.origin.value,
                "zeta.user_display_text": envelope.display_text,
            },
        )
        return ResolvedModelInput(
            envelope.text, envelope.display_text, envelope.origin, message
        )
