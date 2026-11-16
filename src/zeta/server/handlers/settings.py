"""Serve model catalog and session settings views."""

from __future__ import annotations

from ...models.catalog import PROVIDER_MODELS, known_model_names
from ..runtime import ServerRuntime


def telemetry(client) -> dict | None:
    """The negotiated telemetry view; read on the event loop, the status cell's one reader."""
    if "otel" not in client.features:
        return None
    status = client.server.telemetry.status()
    return {"enabled": status.enabled, "state": status.state, "reason": status.reason}


def settings(
    runtime: ServerRuntime, *, include_plan_mode: bool = False, telemetry: dict | None = None
) -> dict:
    result = {"model": runtime.model, "approval_mode": runtime.policy.default.value}
    if telemetry is not None:
        result["telemetry"] = telemetry
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
            for model, provider in sorted(
                (model, provider)
                for provider, models in PROVIDER_MODELS.items()
                for model in models
            )
        },
    }
