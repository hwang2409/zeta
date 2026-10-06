"""Lightweight provider image wire policies.

Sources verified 2026-10-06:
- Anthropic vision limits: https://platform.claude.com/docs/en/build-with-claude/vision
- OpenAI Responses schema: https://platform.openai.com/docs/api-reference/responses
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum


class WireLimitUnit(Enum):
    RAW_BYTES = "raw_bytes"
    BASE64_CHARACTERS = "base64_characters"
    DATA_URL_CHARACTERS = "data_url_characters"


class AnimationPolicy(Enum):
    FIRST_FRAME = "first_frame"
    NON_ANIMATED_ONLY = "non_animated_only"
    ALLOW = "allow"


@dataclass(frozen=True)
class ImagePolicy:
    """All provider constraints that can be enforced for one image."""

    max_wire_size: int | None
    wire_limit_unit: WireLimitUnit
    max_dimension: int | None
    accepted_formats: frozenset[str]
    animation: AnimationPolicy

    def max_raw_bytes(self, media_type: str) -> int | None:
        """Return the largest raw payload that fits this policy's wire form."""

        if self.max_wire_size is None:
            return None
        if self.wire_limit_unit is WireLimitUnit.RAW_BYTES:
            return self.max_wire_size
        prefix_length = 0
        if self.wire_limit_unit is WireLimitUnit.DATA_URL_CHARACTERS:
            prefix_length = len(f"data:{media_type};base64,")
        encoded_budget = self.max_wire_size - prefix_length
        return max(0, 3 * (encoded_budget // 4))

    def encoded_size(self, raw_size: int, media_type: str) -> int:
        prefix_length = (
            len(f"data:{media_type};base64,")
            if self.wire_limit_unit is WireLimitUnit.DATA_URL_CHARACTERS
            else 0
        )
        return prefix_length + 4 * ((raw_size + 2) // 3)


_FORMATS = frozenset({"jpeg", "png", "gif", "webp"})

# The direct Claude API accepts 10 MB of base64 text per image and dimensions
# up to 8000x8000. Bedrock and Vertex use a separate 5 MB policy; Zeta's
# Anthropic adapter calls api.anthropic.com directly.
ANTHROPIC_IMAGE_POLICY = ImagePolicy(
    10_000_000,
    WireLimitUnit.BASE64_CHARACTERS,
    8_000,
    _FORMATS,
    AnimationPolicy.FIRST_FRAME,
)

# The Responses schema caps the complete image_url string at 20,971,520
# characters. Zeta sends a data URL, so the media-type prefix and base64
# expansion are part of this limit.
CODEX_IMAGE_POLICY = ImagePolicy(
    20_971_520,
    WireLimitUnit.DATA_URL_CHARACTERS,
    None,
    _FORMATS,
    AnimationPolicy.NON_ANIMATED_ONLY,
)

OLLAMA_IMAGE_POLICY = ImagePolicy(
    None,
    WireLimitUnit.RAW_BYTES,
    None,
    _FORMATS,
    AnimationPolicy.ALLOW,
)

_PROVIDER_POLICIES = {
    "anthropic": ANTHROPIC_IMAGE_POLICY,
    "claude": ANTHROPIC_IMAGE_POLICY,
    "codex": CODEX_IMAGE_POLICY,
    "fake": ANTHROPIC_IMAGE_POLICY,
    "openai": CODEX_IMAGE_POLICY,
    "ollama": OLLAMA_IMAGE_POLICY,
}


def image_policy_for_provider(provider: str) -> ImagePolicy:
    try:
        return _PROVIDER_POLICIES[provider]
    except KeyError as exc:
        raise ValueError(f"unknown image provider: {provider}") from exc


def data_url_length(data: bytes, media_type: str) -> int:
    """Return the serialized data-URL length without allocating base64 text."""

    return len(f"data:{media_type};base64,") + len(base64.b64encode(data))
