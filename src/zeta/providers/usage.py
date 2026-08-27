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
    if isinstance(details, Mapping):
        cached_tokens = details.get("cached_tokens")
        raw_cache_write_tokens = details.get("cache_write_tokens")
        if type(raw_cache_write_tokens) is not int:
            raw_cache_write_tokens = details.get("cache_creation_input_tokens")
        cache_read_tokens = cached_tokens if type(cached_tokens) is int else 0
        cache_write_tokens = (
            raw_cache_write_tokens
            if type(raw_cache_write_tokens) is int
            else 0
        )
        input_tokens = normalized.get("input_tokens")
        if (
            cache_read_tokens >= 0
            and cache_write_tokens >= 0
            and (type(cached_tokens) is int or type(raw_cache_write_tokens) is int)
            and type(input_tokens) is int
            and cache_read_tokens + cache_write_tokens <= input_tokens
        ):
            if type(cached_tokens) is int and "cache_read_input_tokens" not in normalized:
                normalized["cache_read_input_tokens"] = cache_read_tokens
            if (
                type(raw_cache_write_tokens) is int
                and "cache_creation_input_tokens" not in normalized
            ):
                normalized["cache_creation_input_tokens"] = cache_write_tokens
            normalized["input_tokens"] = (
                input_tokens - cache_read_tokens - cache_write_tokens
            )
    return normalized
