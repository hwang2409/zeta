"""Authorship-preserving model input produced from typed text."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol.types import MessageOrigin


@dataclass(frozen=True, slots=True)
class ModelInputEnvelope:
    """Generated model text together with its exact typed source and origin."""

    text: str
    display_text: str
    origin: MessageOrigin


def skill_model_input(
    prompt: str, request: str, display_text: str
) -> ModelInputEnvelope:
    """Build one skill expansion while preserving the exact typed request."""

    request = request.strip()
    text = f"{prompt}\n\nUser request:\n{request}" if request else prompt
    return ModelInputEnvelope(text, display_text, MessageOrigin.SKILL_EXPANSION)
