"""Serve model catalog and session settings views."""

from __future__ import annotations

from ...models.catalog import PROVIDER_MODELS, known_model_names
from ..runtime import ServerRuntime


def settings(runtime: ServerRuntime, *, include_plan_mode: bool = False) -> dict:
    result = {"model": runtime.model, "approval_mode": runtime.policy.default.value}
    if include_plan_mode:
        result["plan_mode"] = (
            runtime.loop.plan_mode if runtime.loop is not None else False
        )
    return result


def catalog(runtime: ServerRuntime) -> dict:
    return {
        "models": known_model_names(),
        "providers": {
            model: provider
            for provider, models in PROVIDER_MODELS.items()
            for model in models
        },
    }
