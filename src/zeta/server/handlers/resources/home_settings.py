"""Wire adapter for the existing home settings owner."""

from __future__ import annotations

from typing import Any

from ...resource_protocol import (
    Access,
    ResourceMethod,
    require_params,
    require_revision,
    restart_report,
    run_owner,
)

FEATURE = "home_settings_v1"


def _public(result: dict[str, Any], *, home) -> dict[str, Any]:
    settings = result["settings"]
    view = {
        "revision": result["revision"],
        "settings": {
            "telemetry": {key: settings["telemetry"][key] for key in ("enabled", "source", "editable")
                          if key in settings["telemetry"]},
        },
    }
    if "settings_malformed" in result:
        view["settings_malformed"] = result["settings_malformed"]
    if "changed" in result:
        view["restart"] = restart_report(home, serve=result["restart_serve"], changed=result["changed"])
    return view


async def _run(client, call):
    from ....config.home_settings import HomeSettingsError
    from ...protocol import ErrorDescriptor

    return await run_owner(client, call, family="settings", project=lambda result: _public(result, home=client.server.home), owner_error=HomeSettingsError, errors={
        "settings_busy": ErrorDescriptor.RESOURCE_BUSY,
        "settings_malformed": ErrorDescriptor.SETTINGS_MALFORMED,
        "settings_unsafe": ErrorDescriptor.SETTINGS_UNSAFE,
        "telemetry_unavailable": ErrorDescriptor.TELEMETRY_UNAVAILABLE,
    })


async def home_settings_get(client, request_id, params):
    from ....config import home_settings

    require_params(params)
    return await _run(client, lambda: home_settings.get(client.server.home))


async def home_settings_update(client, request_id, params):
    from ....config import home_settings

    require_params(params, "expected_revision", "changes")
    require_revision(params["expected_revision"], prefix="sha256:")
    return await _run(client, lambda: home_settings.update(client.server.home, params["expected_revision"], params["changes"]))


METHODS = (
    ResourceMethod("home_settings_get", Access.READ, home_settings_get),
    ResourceMethod("home_settings_update", Access.CONTROLLER_WRITE, home_settings_update),
)
