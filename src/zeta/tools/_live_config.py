"""Tool-registry operations used by live catalog refresh."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .registry import ToolDefinition


class _LiveConfigRegistryMixin:
    _live_config_refresher: Callable[[bool], Awaitable[None]] | None
    _tools: dict[str, ToolDefinition]
    _mcp_owned: dict[str, tuple[object, int]]
    _session_store: Any
    approval_policy: Any

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

    def approval_definition_identity(self, name: str) -> tuple[str, str | None]:
        if name not in self._tools:
            return ("missing", None)
        ownership = self._mcp_owned.get(name)
        if ownership is None:
            return ("host", None)
        owner = ownership[0]
        return ("mcp", getattr(getattr(owner, "config", None), "name", None))

    def remember_approval_definition(
        self, tool_call: Any, capability: Any, *, persist: bool
    ) -> None:
        if (
            persist
            and self.approval_policy is not None
            and self._session_store is not None
            and (
                self._session_store.approval_state_now(tool_call.id) is not None
                or self.approval_policy.decide(capability).value == "ask"
            )
        ):
            self._session_store.remember_pending_approval_execution(
                tool_call.id, self.approval_definition_identity(tool_call.name)
            )

    def approval_execution_error(
        self, tool_call: Any, *, prepared: bool
    ) -> tuple[str, str] | None:
        if self._session_store is None:
            return None
        # shortcut: approved identity is resident-only; restart recovery matches by tool name.
        claim = self._session_store.claim_pending_approval_execution(
            tool_call.id,
            self.approval_definition_identity(tool_call.name),
            prepared=prepared,
        )
        if claim == "changed":
            return (
                f"tool changed after approval: {tool_call.name}; request a new approval",
                "error",
            )
        if claim == "duplicate":
            return f"tool execution already claimed: {tool_call.id}", "duplicate_tool_execution"
        return None

    def commit_prepared_definition(
        self, base: ToolDefinition | None, replacement: ToolDefinition
    ) -> None:
        if self._tools.get(replacement.name) is not base:
            raise RuntimeError(f"stale prepared tool definition: {replacement.name}")
        self._tools[replacement.name] = replacement
