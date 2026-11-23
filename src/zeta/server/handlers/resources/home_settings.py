"""Wire adapter for the existing home settings owner.

``home_settings_v1`` shipped on main before the shared resource codes; its replies stay exactly
main's (its own error descriptors, owner ``data`` such as ``current_revision`` and ``detail``).
"""

from __future__ import annotations

from typing import Any

from ...resource_protocol import (
    Access,
    ResourceMethod,
    require_params,
    require_revision,
    restart_report,
)

FEATURE = "home_settings_v1"


def _error(exc) -> Exception:
    from ...protocol import ErrorDescriptor, ProtocolError, invalid_resource_params

    match exc.code:
        case "invalid_params":
            return invalid_resource_params()
        case "custom_home_setting":
            descriptor = ErrorDescriptor.CUSTOM_HOME_SETTING
        case "stale_revision":
            descriptor = ErrorDescriptor.STALE_REVISION
        case "settings_busy":
            descriptor = ErrorDescriptor.SETTINGS_BUSY
        case "settings_malformed":
            descriptor = ErrorDescriptor.SETTINGS_MALFORMED
        case "telemetry_unavailable":
            descriptor = ErrorDescriptor.TELEMETRY_UNAVAILABLE
        case _:
            descriptor = ErrorDescriptor.SETTINGS_UNSAFE
    # the owner's data is fixed text or a revision hash; it names no path
    return ProtocolError(descriptor, descriptor.meaning, exc.data or None)


async def _run(client, call):
    from ....config.home_settings import HomeSettingsError

    def run():
        try:
            return call()
        except HomeSettingsError as exc:
            raise _error(exc) from None

    return await client.connections.run_blocking(run)


async def home_settings_get(client, request_id, params, controller):
    from ....config import home_settings

    require_params(params)
    return await _run(client, lambda: home_settings.get(client.server.home))


async def home_settings_update(client, request_id, params, controller):
    from ....config import home_settings

    require_params(params, "expected_revision", "changes")
    require_revision(params["expected_revision"], prefix="sha256:")
    home = client.server.home

    def update() -> dict[str, Any]:
        result = home_settings.update(home, params["expected_revision"], params["changes"])
        # serve configured telemetry once at startup, so v1 always reports serve and scans sessions
        return {**result, "restart": restart_report(home, serve=True, changed=True)}

    return await _run(client, update)


METHODS = (
    ResourceMethod("home_settings_get", Access.READ, home_settings_get),
    ResourceMethod("home_settings_update", Access.CONTROLLER_WRITE, home_settings_update),
)
