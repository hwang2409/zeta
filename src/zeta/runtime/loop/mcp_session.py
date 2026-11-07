"""MCP session lifecycle operations for AgentLoop."""

import asyncio
import os
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Self

from ...mcp import (
    MCPConfigError,
    MCPManagementError,
    MCPManagementService,
    MCPMount,
    home_config_path,
    load_mcp_config_overlay,
    mount_mcp_servers,
    project_config_path,
    tool_prefix,
)
from ...mcp.commands import (
    MCP_USAGE,
    MCPCommandError,
    add_and_mount,
    parse_add_command,
    remove_and_unshadow,
    render_mcp_status,
    run_mcp_resource_attach,
    run_mcp_resources_list,
)
from ...mcp.oauth import MCPOAuthError
from ...mcp.oauth_store import (
    MCPCredentialStoreError,
    PublicCredentialStoreError,
    public_credential_store_error,
)
from ...model_input import ModelInputEnvelope


class MCPCommandOutput(str):
    """Slash output with an optional typed credential-store error."""

    credential_store_error: PublicCredentialStoreError | None

    def __new__(
        cls,
        text: str,
        credential_store_error: PublicCredentialStoreError | None = None,
    ) -> Self:
        value = super().__new__(cls, text)
        value.credential_store_error = credential_store_error
        return value


class MCPSession:
    @property
    def active_home(self) -> str | None:
        """Return the home override active for this MCP/session scope."""

        return self._mcp_home_hint

    async def _ensure_mcp_servers(self) -> None:
        if self._mcp_mount_attempted:
            return
        if self._mcp_mount_task is None:
            self._mcp_mount_task = asyncio.create_task(self._mount_mcp_servers())
        await asyncio.shield(self._mcp_mount_task)

    async def _mount_mcp_servers(self) -> None:
        try:
            service = await asyncio.to_thread(
                MCPManagementService,
                home=self._mcp_home_hint,
                project_dir=self._mcp_project_dir_value,
            )
            config = service.runtime_config()
            self._mcp_mount = await mount_mcp_servers(
                self.tool_registry,
                config,
                notice_sink=self._mcp_notice_sink,
                home=self._mcp_home_hint,
            )
        except MCPConfigError as exc:
            self._mcp_config_error = str(exc)
            self._mcp_mount = MCPMount(
                self.tool_registry, {}, {}, home=self._mcp_home_hint
            )
        self._mcp_mount.set_schema_refresh(self._refresh_mcp_tool_schemas)
        if self._mcp_prompt_refresh is not None:
            self._mcp_mount.set_prompt_refresh(self._mcp_prompt_refresh)
        self._mcp_mount_attempted = True

    def attach_mcp_mount(self, mount: MCPMount) -> None:
        """Adopt an explicitly selected mount without loading project configuration."""

        if self._mcp_mount is not None:
            raise ValueError("MCP mount already attached")
        self._mcp_mount = mount
        self._mcp_mount_attempted = True
        mount.set_schema_refresh(self._refresh_mcp_tool_schemas)

    def set_mcp_scope(
        self,
        *,
        home: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        """Set the home + project scope this loop uses for MCP config files."""

        self._mcp_home_hint = None if home is None else str(home)
        self._mcp_project_dir_value = (
            None if project_dir is None else Path(project_dir).expanduser().resolve()
        )

    def _refresh_mcp_tool_schemas(self, mount: MCPMount | None = None) -> None:
        mount = mount or self._mcp_mount
        if mount is None:
            return
        mcp_prefixes = tuple(tool_prefix(name) for name in mount.configs)
        current_mcp = [
            schema
            for schema in self.tool_registry.schemas
            if isinstance(schema.get("name"), str)
            and schema["name"].startswith(mcp_prefixes)
        ]
        current_names = {
            schema["name"]
            for schema in current_mcp
            if isinstance(schema.get("name"), str)
        }
        if not self._provided_tool_schemas:
            schemas = tuple(self.tool_registry.schemas)
        else:
            names_to_replace = self._mcp_schema_names | current_names
            schemas = tuple(
                schema
                for schema in self.tool_schemas
                if not (
                    isinstance(schema.get("name"), str)
                    and schema["name"] in names_to_replace
                )
            ) + tuple(current_mcp)
        if schemas != self.tool_schemas:
            self.tool_schemas = schemas
        self._mcp_schema_names = current_names

    async def ensure_mcp_servers(self) -> None:
        """Connect MCP servers before a direct tool resume."""

        await self._ensure_mcp_servers()

    def require_allowed_tools(self) -> None:
        """Raise when an exact allowlisted tool was not registered at startup."""

        missing = self.tool_registry.missing_required_tools
        if missing:
            raise ValueError("required tools are unavailable: " + ", ".join(missing))

    def set_mcp_notice_sink(self, sink: Callable[[str], None] | None) -> None:
        """Set the sink for MCP mount notices."""
        self._mcp_notice_sink = sink

    def set_mcp_prompt_refresh(
        self, callback: Callable[[MCPMount], None] | None
    ) -> None:
        """Set the owner callback for live MCP prompt commands."""
        self._mcp_prompt_refresh = callback
        if callback is not None and self._mcp_mount is not None:
            callback(self._mcp_mount)

    @property
    def mcp_summary(self) -> str:
        if self._mcp_mount is None:
            return "mcp: 0 mounted, 0 failed"
        return self._mcp_mount.summary

    async def slash_mcp(self, args: str) -> str | ModelInputEnvelope:
        """Show MCP state, reconnect, add, remove, authorize, or attach."""
        await self._ensure_mcp_servers()
        mount = self._mcp_mount
        try:
            parts = shlex.split(args)
        except ValueError as exc:
            return f"mcp error: {exc}"
        if self._mcp_config_error is not None:
            return f"mcp error: {self._mcp_config_error}"
        if mount is None:
            return "mcp: no configured servers"
        service = await asyncio.to_thread(
            MCPManagementService,
            home=self._mcp_home_hint,
            project_dir=self._mcp_project_dir_value,
            mount=mount,
        )
        if not parts:
            return await self._render_mcp_status(mount, service)
        verb = parts[0]
        try:
            if verb == "status" and len(parts) == 1:
                return await self._render_mcp_status(mount, service)
            if verb == "reconnect":
                if len(parts) != 2:
                    return MCP_USAGE
                await service.sync_runtime()
                await mount.reconnect(parts[1], notice_sink=self._mcp_notice_sink)
                return await self._render_mcp_status(mount, service)
            if verb == "add":
                await add_and_mount(
                    mount,
                    parse_add_command(parts[1:]),
                    target=self._mcp_add_target(),
                    notice_sink=self._mcp_notice_sink,
                )
                return await self._render_mcp_status(mount, service)
            if verb == "remove":
                if len(parts) != 2:
                    return MCP_USAGE
                await remove_and_unshadow(
                    mount,
                    parts[1],
                    home_path=self._mcp_home_path(),
                    load_home=lambda: (
                        load_mcp_config_overlay(
                            home=self._mcp_home_hint, project_dir=None
                        ).configured_servers
                    ),
                    notice_sink=self._mcp_notice_sink,
                )
                return await self._render_mcp_status(mount, service)
            if verb == "auth":
                if len(parts) != 2:
                    return MCP_USAGE
                await service.login(parts[1])
                await mount.reconnect(parts[1], notice_sink=self._mcp_notice_sink)
                return await self._render_mcp_status(mount, service)
            if verb == "logout":
                if len(parts) != 2:
                    return MCP_USAGE
                await service.logout(parts[1])
                return f"logged out {parts[1]}."
            if verb == "resources":
                if len(parts) == 2:
                    return await run_mcp_resources_list(mount, parts[1])
                if len(parts) == 3:
                    return await run_mcp_resource_attach(mount, parts[1], parts[2])
                return MCP_USAGE
        except MCPCredentialStoreError as exc:
            error = public_credential_store_error(exc)
            return MCPCommandOutput(error["text"], error)
        except (
            MCPCommandError,
            MCPManagementError,
            MCPOAuthError,
            ValueError,
        ) as exc:
            return f"mcp error: {exc}"
        return MCP_USAGE

    async def _render_mcp_status(
        self, mount: MCPMount, service: MCPManagementService
    ) -> str:
        entries = await service.refresh_credentials()
        statuses = {entry.name: entry.auth for entry in entries}
        return render_mcp_status(mount, credential_status=statuses)

    async def slash_mcp_prompt(self, name: str, arguments: dict[str, str]) -> str:
        """Resolve one mounted MCP prompt for the next model turn."""
        await self._ensure_mcp_servers()
        if self._mcp_mount is None:
            raise RuntimeError("MCP mount is unavailable")
        return await self._mcp_mount.get_prompt(name, arguments)

    def _mcp_add_target(self) -> Path:
        project_dir = self._mcp_project_dir_value
        if project_dir is not None:
            return project_config_path(project_dir)
        return home_config_path(self._mcp_home_hint)

    def _mcp_home_path(self) -> Path:
        override = os.environ.get("ZETA_MCP_CONFIG")
        if override:
            return Path(override).expanduser().resolve()
        return home_config_path(self._mcp_home_hint).resolve()
