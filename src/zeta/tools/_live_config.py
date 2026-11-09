"""Tool-registry operations used by live catalog refresh."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolDefinition


class _LiveConfigRegistryMixin:
    _live_config_refresher: Callable[[bool], Awaitable[None]] | None
    _tools: dict[str, ToolDefinition]

    def set_live_config_refresher(
        self, refresher: Callable[[bool], Awaitable[None]]
    ) -> None:
        """Let tools request one live catalog refresh."""

        self._live_config_refresher = refresher

    async def refresh_live_config(self, *, force: bool = False) -> None:
        if self._live_config_refresher is not None:
            await self._live_config_refresher(force)

    def definition_reference(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def commit_prepared_definition(
        self, base: ToolDefinition | None, replacement: ToolDefinition
    ) -> None:
        if self._tools.get(replacement.name) is not base:
            raise RuntimeError(f"stale prepared tool definition: {replacement.name}")
        self._tools[replacement.name] = replacement
