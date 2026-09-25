"""Which models each provider serves, and how to map a model back to a provider.

This is plain data, kept out of the providers package so the tool layer can read
it too: tools must not import providers, but the agent tool has to advertise the
models a child may run on.
"""

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


def known_model_names() -> list[str]:
    """Return every model name the built-in catalogs serve, sorted."""

    return sorted(model for models in PROVIDER_MODELS.values() for model in models)


def provider_for_model(model: str) -> str:
    """Return the provider serving a model, or raise naming the valid choices.

    Backed by the static table above rather than tui.models.load_model_catalog:
    that one reaches the network for claude and returns None whenever it fails,
    neither of which a tool call can absorb.
    """

    for provider, models in PROVIDER_MODELS.items():
        if model in models:
            return provider
    choices = ", ".join(known_model_names())
    raise ValueError(f"unknown model {model!r}; choose one of: {choices}")


__all__ = ["PROVIDER_MODELS", "known_model_names", "provider_for_model"]
