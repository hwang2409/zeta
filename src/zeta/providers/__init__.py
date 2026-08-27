"""Completion provider implementations and their supported model names."""

from __future__ import annotations


PROVIDER_MODELS: dict[str, frozenset[str]] = {
    "claude": frozenset(
        {
            "claude-fable-5",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
        }
    ),
    "codex": frozenset(
        {
            "codex-auto-review",
            "gpt-5.3-codex-spark",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-reserve",
        }
    ),
}
