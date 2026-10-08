"""View/controller model for the full-screen MCP manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...mcp.management import ManagedServer, MCPManagementService
from .. import overlay, theme


@dataclass(frozen=True, slots=True)
class MCPAddDraft:
    """Secret-safe values collected by the add wizard."""

    name: str
    scope: Literal["user", "project"]
    transport: Literal["stdio", "streamable-http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    oauth: bool = False
    env_refs: dict[str, str] = field(default_factory=dict)
    header_refs: dict[str, str] = field(default_factory=dict)


class MCPManager:
    """Interactive manager state shared by prompt-toolkit bindings and tests."""

    def __init__(self, service: MCPManagementService) -> None:
        self.service = service
        self.selected = 0
        self.details = False
        self.wizard_active = False
        self._pending_remove: str | None = None
        self.last_result: object | None = None

    @property
    def entries(self) -> list[ManagedServer]:
        return self.service.list()

    def _selected(self) -> ManagedServer | None:
        entries = self.entries
        if not entries:
            self.selected = 0
            return None
        self.selected = min(self.selected, len(entries) - 1)
        return entries[self.selected]

    @staticmethod
    def glyph(item: ManagedServer) -> str:
        if not item.enabled:
            return "○"
        if not item.trusted:
            return "◌"
        if item.status in {"mounted", "connected"}:
            return "●"
        if item.auth in {"unauthorized", "expired", "refresh-failed"}:
            return "!"
        if item.status in {"degraded", "failed", "timed-out", "malformed"}:
            return "×"
        return "!"

    _COLUMNS = (
        "    name                 scope    transport        "
        "auth          tools status"
    )

    def _glyph_style(self, item: ManagedServer) -> str:
        glyph = self.glyph(item)
        if glyph == "●":
            return theme.SUCCESS
        if glyph in {"×", "!"}:
            return theme.ERROR
        return theme.DIM

    def _entry_line(self, index: int, item: ManagedServer) -> overlay.FragmentLine:
        transport = str(item.config.get("transport", "unknown"))
        return overlay.row(
            [
                (self._glyph_style(item), self.glyph(item)),
                (
                    theme.BODY,
                    (
                        f" {item.name:<20} {item.scope:<8} "
                        f"{transport:<16} {item.auth:<13} "
                        f"{item.tool_count:<5} {item.status}"
                    ),
                ),
            ],
            selected=index == self.selected,
        )

    def render(self) -> list[overlay.FragmentLine]:
        item = self._selected()
        lines: list[overlay.FragmentLine] = [
            overlay.title("MCP servers"),
            overlay.rule(),
            overlay.hint(self._COLUMNS),
        ]
        entries = self.entries
        if not entries:
            lines.append(overlay.hint("(none)"))
        for index, entry in enumerate(entries):
            lines.append(self._entry_line(index, entry))
        if self.details and item is not None:
            lines.append(overlay.blank())
            lines.append([overlay.label(f"details: {item.name}")])
            for detail in str(item.as_json()).splitlines() or [""]:
                lines.append([overlay.value(detail)])
        if self.wizard_active:
            lines.append(overlay.blank())
            lines.append(
                overlay.hint(
                    "add wizard: choose stdio or HTTP and user or project scope"
                )
            )
            lines.append(
                overlay.hint(
                    "secrets use env references, e.g. API_KEY ← $LINEAR_API_KEY"
                )
            )
        if self.last_result is not None:
            lines.append(overlay.blank())
            for result in str(self.last_result).splitlines() or [""]:
                lines.append([overlay.value(result)])
        lines.append(overlay.rule())
        lines.append(
            overlay.hint(
                "↑/↓ select · enter details · a add · e enable/disable · t test"
            )
        )
        lines.append(overlay.hint("l login · o logout · d remove · T trust · esc close"))
        return lines

    def add(self, draft: MCPAddDraft) -> ManagedServer:
        self.wizard_active = False
        env = {name: f"${{{reference}}}" for name, reference in draft.env_refs.items()}
        headers = {
            name: f"${{{reference}}}" for name, reference in draft.header_refs.items()
        }
        if draft.transport == "stdio":
            return self.service.add(
                draft.name,
                scope=draft.scope,
                command=draft.command,
                args=draft.args,
                env=env,
            )
        return self.service.add(
            draft.name,
            scope=draft.scope,
            url=draft.url,
            oauth=draft.oauth,
            env=env,
            headers=headers,
        )

    async def finish_add(self, draft: MCPAddDraft) -> ManagedServer:
        added = self.add(draft)
        await self.service.sync_runtime()
        return added

    async def dispatch(self, key: str) -> object | None:
        if key == "a":
            self.wizard_active = True
            return None
        if key in {"up", "k"}:
            self.selected = max(0, self.selected - 1)
            return None
        if key in {"down", "j"}:
            self.selected = min(max(0, len(self.entries) - 1), self.selected + 1)
            return None
        item = self._selected()
        if item is None:
            return None
        if key == "enter":
            self.details = not self.details
            return None
        if key == "e":
            self.last_result = self.service.set_enabled(
                item.name, scope=item.scope, enabled=not item.enabled
            )
        elif key == "d":
            unshadows = item.scope == "project" and any(
                candidate.name == item.name
                for candidate in self.service.list(scope="user")
            )
            if unshadows and self._pending_remove != item.name:
                self._pending_remove = item.name
                self.last_result = (
                    "removing this project definition will unshadow the user definition; "
                    "press d again to confirm"
                )
                return self.last_result
            self.service.remove(item.name, scope=item.scope)
            self._pending_remove = None
            self.last_result = "removed"
        elif key == "t":
            self.last_result = await self.service.test(item.name, scope=item.scope)
        elif key == "l":
            self.last_result = await self.service.login(item.name, scope=item.scope)
        elif key == "o":
            self.service.logout(item.name, scope=item.scope)
            self.last_result = "logged out"
        elif key == "T":
            self.last_result = self.service.trust(item.name)
        else:
            raise ValueError(f"unknown MCP manager key: {key}")
        if key in {"e", "d", "T"}:
            await self.service.sync_runtime()
        return self.last_result


__all__ = ["MCPAddDraft", "MCPManager"]
