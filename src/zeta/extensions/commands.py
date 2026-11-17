"""Slash commands declared by native extensions."""

from __future__ import annotations

import asyncio
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

    registry.register(SlashCommand.loop_safe(
        "ext", ext_handler, "list or manage extensions", serve=ServeSlash(),
        reason="reads and updates the per-session extension host",
    ))
    for extension in host.loaded:
        _register_extension_commands(registry, host, extension)


def _register_extension_commands(
    registry: SlashCommandRegistry, host: ExtensionHost, extension: LoadedExtension
) -> None:
    for declaration in host.loaded_declarations(extension.name):
        if declaration.body is not None:
            try:
                registry.register_custom(CustomCommand(
                    name=declaration.name, description=declaration.description,
                    body=declaration.body,
                    path=extension.directory / "zeta-extension.toml",
                    source=f"extension:{extension.name}", kind=declaration.kind,
                ))
            except Exception as exc:  # noqa: BLE001 - one command must not break the session
                host.notices.append(_ignored(declaration.name, extension.name, exc))
            continue
        if registry.custom_command_for(declaration.name) is not None:
            registry.add_notice(
                f"extensions · ignored command '{declaration.name}' from {extension.name}: "
                "shadows an existing custom command"
            )
            continue

        async def handler(session: Any, args: str, callback: Any = declaration.handler) -> Any:
            del session
            try:
                return await callback(args)
            except Exception as exc:  # noqa: BLE001 - extension failures fail open
                return f"extension command failed: {_short(exc)}"

        try:
            registry.register(SlashCommand.loop_safe(
                declaration.name, handler, declaration.description,
                serve=ServeSlash(), reason="runs extension-provided code command",
            ))
        except Exception as exc:  # noqa: BLE001 - one command must not break the session
            registry.add_notice(_ignored(declaration.name, extension.name, exc))


def _ignored(name: str, extension: str, exc: Exception) -> str:
    return f"extensions · ignored command '{name}' from {extension}: {_short(exc)}"


def _short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text[:200] + ("…" if len(text) > 200 else "")


async def _run_ext(host: ExtensionHost, args: str) -> str:
    parts = args.split(maxsplit=1)
    action = parts[0] if parts else "list"
    name = parts[1].strip() if len(parts) > 1 else ""
    actions = {
        "list": _ext_list,
        "info": _ext_info,
        "enable": _ext_enable,
        "disable": _ext_disable,
    }
    handler = actions.get(action)
    if handler is None:
        return "usage: /ext [list|info|enable|disable] [name]"
    return await handler(host, name)


async def _ext_list(host: ExtensionHost, name: str) -> str:
    if name:
        return "usage: /ext [list|info|enable|disable] [name]"
    names = {item.name for item in host.loaded}
    names.update(item.manifest.name for item in host.disabled_inventory)
    names.update(item.manifest.name for item in host.pending_project)
    rows = [_status_line(host, extension_name) for extension_name in sorted(names)]
    return "\n".join(rows) or "no extensions loaded"


def _status_line(host: ExtensionHost, name: str) -> str:
    state, item = host.status(name)
    if item is None:
        return f"{name} not found"
    manifest = item.manifest if hasattr(item, "manifest") else None
    version = item.version if manifest is None else manifest.version
    scope = item.scope if manifest is None else "project"
    suffix = {
        "disabled_in_settings": " disabled in settings",
        "disabled_next_session": " disabled (next session)",
        "enabled_next_session": " enabled (next session)",
        "pending": " pending trust",
    }.get(state, "")
    return f"{name} {version} ({scope}){suffix}"


async def _ext_info(host: ExtensionHost, name: str) -> str:
    state, item = host.status(name)
    if item is None:
        return f"extension not found: {name}"
    if state != "loaded":
        return _status_line(host, name)
    return (
        f"{item.name} {item.version} ({item.scope})\n"
        f"tools: {', '.join(item.tool_names) or 'none'}\n"
        f"commands: {', '.join(item.command_names) or 'none'}"
    )


async def _ext_enable(host: ExtensionHost, name: str) -> str:
    return await _set_extension(host, name, True)


async def _ext_disable(host: ExtensionHost, name: str) -> str:
    return await _set_extension(host, name, False)


async def _set_extension(host: ExtensionHost, name: str, enabled: bool) -> str:
    known = any(item.name == name for item in host.loaded) or any(
        item.manifest.name == name for item in (*host.disabled_inventory, *host.pending_project)
    )
    if not name or not known:
        return f"extension not found: {name}"
    changed = await asyncio.to_thread(host.persist_disabled, name, enabled)
    if not changed:
        return f"extension {name} is disabled in settings.toml; remove it there to enable"
    return f"extension {name} {'enabled' if enabled else 'disabled'} for the next session"


__all__ = ["register_commands"]
