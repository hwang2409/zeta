"""Serve model catalog, session settings views, and ``home_settings_v1``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ...config import home_settings
from ...models.catalog import PROVIDER_MODELS, known_model_names
from ..protocol import ErrorDescriptor, ProtocolError
from ..runtime import ServerRuntime


async def home_settings_get(client, request_id, params):
    if params:
        raise ProtocolError(ErrorDescriptor.INVALID_HOME_SETTINGS_PARAMS, "invalid params")
    home = client.server.home
    return await client.connections.run_read(client, lambda: _home_settings(lambda: home_settings.get(home)))


async def home_settings_update(client, request_id, params):
    """The handler never touches settings.toml; ``home_settings`` is its only writer."""
    if params.keys() != {"expected_revision", "changes"}:
        raise ProtocolError(ErrorDescriptor.INVALID_HOME_SETTINGS_PARAMS, "invalid params")
    home = client.server.home
    return await asyncio.to_thread(
        _home_settings, lambda: home_settings.update(home, params["expected_revision"], params["changes"])
    )


def _home_settings(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except home_settings.HomeSettingsError as exc:
        match exc.code:
            case "invalid_params":
                raise ProtocolError(ErrorDescriptor.INVALID_HOME_SETTINGS_PARAMS, "invalid params") from None
            case "custom_home_setting":
                descriptor = ErrorDescriptor.CUSTOM_HOME_SETTING
            case "stale_revision":
                descriptor = ErrorDescriptor.STALE_REVISION
            case "settings_busy":
                descriptor = ErrorDescriptor.SETTINGS_BUSY
            case "settings_malformed":
                descriptor = ErrorDescriptor.SETTINGS_MALFORMED
            case _:
                descriptor = ErrorDescriptor.SETTINGS_UNSAFE
        raise ProtocolError(descriptor, descriptor.meaning, exc.data or None) from None


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
