"""Slash commands declared by native extensions."""

from __future__ import annotations

import inspect
from typing import Any

from ..core.commands.custom_commands import CustomCommand
from ..core.commands.slash_registry import (
    ServeSlash,
    SlashCommand,
    SlashCommandRegistry,
)
from .host import ExtensionHost, LoadedExtension


def register_commands(registry: SlashCommandRegistry, host: ExtensionHost) -> None:
    """Register the shared ``/ext`` command and activated extension commands."""
    async def ext_handler(_session: Any, args: str) -> str:
        return await _run_ext(host, args)

    registry.register(
        SlashCommand.loop_safe(
            "ext",
            ext_handler,
            "list or manage extensions",
            serve=ServeSlash(),
            reason="reads and updates the per-session extension host",
        )
    )
    for extension in host.loaded:
        _register_extension_commands(registry, host, extension)


def _register_extension_commands(
    registry: SlashCommandRegistry, host: ExtensionHost, extension: LoadedExtension
) -> None:
    loading = host.loaded_declarations(extension.name)
    for declaration in loading:
        if declaration.body is not None:
            try:
                registry.register_custom(
                    CustomCommand(
                        name=declaration.name,
                        description=declaration.description,
                        body=declaration.body,
                        path=extension.directory / "zeta-extension.toml",
                        source=f"extension:{extension.name}",
                        kind=declaration.kind,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one command must not break the session
                host.notices.append(
                    f"extensions · ignored command '{declaration.name}' from "
                    f"{extension.name}: {exc}"
                )
            continue

        async def handler(session: Any, args: str, callback: Any = declaration.handler) -> Any:
            del session
            result = callback(args)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            registry.register(
                SlashCommand.loop_safe(
                    declaration.name,
                    handler,
                    declaration.description,
                    serve=ServeSlash(),
                    reason="runs extension-provided code command",
                )
            )
        except Exception as exc:  # noqa: BLE001 - one command must not break the session
            host.notices.append(
                f"extensions · ignored command '{declaration.name}' from "
                f"{extension.name}: {exc}"
            )


async def _run_ext(host: ExtensionHost, args: str) -> str:
    parts = args.split(maxsplit=1)
    action = parts[0] if parts else "list"
    name = parts[1].strip() if len(parts) > 1 else ""
    if action == "list":
        if name:
            return "usage: /ext [list|info|enable|disable] [name]"
        rows = [
            f"{item.name} {item.version} ({item.scope})"
            for item in host.loaded
            if item.name not in host.disabled
        ]
        rows.extend(f"{item.manifest.name} (pending trust)" for item in host.pending_project)
        return "\n".join(rows) or "no extensions loaded"
    if action == "info":
        item = next((item for item in host.loaded if item.name == name), None)
        if item is None:
            pending = next((item for item in host.pending_project if item.manifest.name == name), None)
            if pending is None:
                return f"extension not found: {name}"
            return f"{name} {pending.manifest.version} (project) pending trust"
        return (
            f"{item.name} {item.version} ({item.scope})\n"
            f"tools: {', '.join(item.tool_names) or 'none'}\n"
            f"commands: {', '.join(item.command_names) or 'none'}"
        )
    if action in {"enable", "disable"} and name:
        if not any(item.name == name for item in host.loaded):
            return f"extension not found: {name}"
        host.set_enabled(name, action == "enable")
        return f"extension {name} {action}d for the next session"
    return "usage: /ext [list|info|enable|disable] [name]"


__all__ = ["register_commands"]
