"""MCP session lifecycle operations for AgentLoop."""

import asyncio
from pathlib import Path

from ...mcp import (
    MCPConfigError,
    MCPManagementService,
    MCPMount,
    mount_mcp_servers,
    tool_prefix,
)


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
            config = MCPManagementService(
                home=self._mcp_home_hint,
                project_dir=self._mcp_project_dir_value,
            ).runtime_config()
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
            self.tool_schemas = list(self.tool_registry.schemas)
            self._mcp_schema_names = current_names
            return
        names_to_replace = self._mcp_schema_names | current_names
        self.tool_schemas = [
            schema
            for schema in self.tool_schemas
            if not (
                isinstance(schema.get("name"), str)
                and schema["name"] in names_to_replace
            )
        ] + current_mcp
        self._mcp_schema_names = current_names

    async def ensure_mcp_servers(self) -> None:
        """Connect MCP servers before a direct tool resume."""

        await self._ensure_mcp_servers()
