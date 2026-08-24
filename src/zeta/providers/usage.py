"""Normalize provider usage details at the backend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Expose common token names while preserving provider usage details."""

    normalized = dict(usage)
    if "input_tokens" not in normalized:
        prompt_tokens = normalized.get("prompt_tokens")
        if type(prompt_tokens) is int:
            normalized["input_tokens"] = prompt_tokens
    if "output_tokens" not in normalized:
        completion_tokens = normalized.get("completion_tokens")
        if type(completion_tokens) is int:
            normalized["output_tokens"] = completion_tokens

    details = normalized.get("input_tokens_details")
    if not isinstance(details, Mapping):
        details = normalized.get("prompt_tokens_details")
    if isinstance(details, Mapping) and "cache_read_input_tokens" not in normalized:
        cached_tokens = details.get("cached_tokens")
        input_tokens = normalized.get("input_tokens")
        if (
            type(cached_tokens) is int
            and cached_tokens >= 0
            and type(input_tokens) is int
            and cached_tokens <= input_tokens
        ):
            normalized["cache_read_input_tokens"] = cached_tokens
            normalized["input_tokens"] = input_tokens - cached_tokens
    return normalized
