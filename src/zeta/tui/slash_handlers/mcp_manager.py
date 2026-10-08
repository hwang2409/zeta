"""Prompt-toolkit lifecycle integration for the interactive MCP manager."""

import asyncio
import shlex
from typing import Any

from prompt_toolkit.application import in_terminal
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import input_dialog, radiolist_dialog, yes_no_dialog

from ..cards.mcp_manager import MCPAddDraft
from ..composer import FullScreenPromptSession


class MCPManagerMixin:
    def open_mcp_manager(self, *, restore_composer: bool = True) -> None:
        session = self._active_session
        if not isinstance(session, FullScreenPromptSession):
            return
        buffer = session.default_buffer
        if restore_composer:
            self._status_restore_text = buffer.text
            self._status_restore_cursor = buffer.cursor_position
        else:
            self._status_restore_text = ""
            self._status_restore_cursor = 0
        self._mcp_manager.service.mount = self.loop._mcp_mount
        self._mcp_manager_open = True
        self._status_card.set_lines(self._mcp_manager.render())
        self._status_card_open = True
        session.layout.focus(self._status_card_window)
        asyncio.create_task(self._sync_mcp_manager())
        self._invalidate_prompt()

    async def _sync_mcp_manager(self) -> None:
        await self.loop._ensure_mcp_servers()
        self._mcp_manager.service.mount = self.loop._mcp_mount
        await self._mcp_manager.service.sync_runtime()
        self._refresh_mcp_manager()

    def _refresh_mcp_manager(self) -> None:
        if not self._mcp_manager_open:
            return
        self._mcp_manager.service.mount = self.loop._mcp_mount
        self._status_card.set_lines(self._mcp_manager.render())
        self._invalidate_prompt()

    def _status_move(self, amount: int) -> None:
        if not self._mcp_manager_open:
            self._status_card.scroll(amount)
            return
        asyncio.create_task(
            self._dispatch_mcp_manager("down" if amount > 0 else "up")
        )

    def _status_action(self, key: str) -> None:
        if not self._mcp_manager_open:
            return
        if key == "a":
            # Mark the transition synchronously so Escape cannot close the
            # manager before the child dialog has taken over the input.
            if self._mcp_wizard_task is not None:
                return
            self._mcp_wizard_active = True
            task = asyncio.create_task(self._run_mcp_add_wizard())
            self._mcp_wizard_task = task
            task.add_done_callback(self._mcp_wizard_done)
            return
        asyncio.create_task(self._dispatch_mcp_manager(key))

    @staticmethod
    def _parse_env_references(value: str) -> dict[str, str]:
        references: dict[str, str] = {}
        for entry in value.split(","):
            if not entry.strip():
                continue
            name, separator, reference = entry.partition("=")
            if not separator or not name.strip() or not reference.strip().startswith("$"):
                raise ValueError("references must use KEY=$ENV_VAR")
            references[name.strip()] = reference.strip()[1:]
        return references

    def _mcp_wizard_done(self, task: asyncio.Task[None]) -> None:
        """Release wizard state, including when cancellation precedes startup."""
        if self._mcp_wizard_task is task:
            self._mcp_wizard_task = None
            self._mcp_wizard_active = False
            self._mcp_wizard_dialog_active = False
        if not task.cancelled():
            task.exception()
        self._refresh_mcp_manager()

    async def _cancel_mcp_wizard(self) -> None:
        task = self._mcp_wizard_task
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run_mcp_add_wizard(self) -> None:
        """Collect only definitions and environment references, never raw secrets."""
        try:
            async with in_terminal():
                self._mcp_wizard_dialog_active = True
                draft = await self._collect_mcp_add_draft()
            self._mcp_wizard_active = False
            if draft is None:
                return
            self._mcp_manager.last_result = await self._mcp_manager.finish_add(draft)
        except (OSError, ValueError, RuntimeError) as exc:
            self._mcp_manager.last_result = f"mcp error: {exc}"
        finally:
            self._mcp_wizard_dialog_active = False
            self._refresh_mcp_manager()

    @staticmethod
    async def _run_wizard_dialog(dialog: Any) -> Any:
        """Run a wizard dialog with Escape as an eager, consumed cancel."""
        if hasattr(dialog, "key_bindings"):
            cancel = KeyBindings()

            @cancel.add(Keys.Escape, eager=True)
            def cancel_dialog(event: Any) -> None:
                event.app.exit(result=None)

            dialog.key_bindings = merge_key_bindings(
                [cancel, dialog.key_bindings]
            )
        return await dialog.run_async()

    @staticmethod
    async def _collect_mcp_add_draft() -> MCPAddDraft | None:
        name = await MCPManagerMixin._run_wizard_dialog(
            input_dialog(title="Add MCP server", text="Server name:"),
        )
        if not name:
            return None
        scope = await MCPManagerMixin._run_wizard_dialog(
            radiolist_dialog(
                title="Add MCP server",
                text="Configuration scope:",
                values=[("user", "User"), ("project", "Project")],
            ),
        )
        if scope not in {"user", "project"}:
            return None
        transport = await MCPManagerMixin._run_wizard_dialog(
            radiolist_dialog(
                title="Add MCP server",
                text="Transport:",
                values=[
                    ("stdio", "Local command (stdio)"),
                    ("streamable-http", "HTTP"),
                ],
            ),
        )
        if transport not in {"stdio", "streamable-http"}:
            return None
        environment = await MCPManagerMixin._run_wizard_dialog(
            input_dialog(
                title="Add MCP server",
                text="Environment references (comma-separated KEY=$ENV_VAR; optional):",
            ),
        )
        if environment is None:
            return None
        env_refs = MCPManagerMixin._parse_env_references(environment)
        if transport == "stdio":
            command_line = await MCPManagerMixin._run_wizard_dialog(
                input_dialog(title="Add MCP server", text="Command and arguments:"),
            )
            if not command_line:
                return None
            parts = shlex.split(command_line)
            if not parts:
                return None
            return MCPAddDraft(
                name=name,
                scope=scope,
                transport="stdio",
                command=parts[0],
                args=tuple(parts[1:]),
                env_refs=env_refs,
            )
        url = await MCPManagerMixin._run_wizard_dialog(
            input_dialog(title="Add MCP server", text="Server URL:"),
        )
        if not url:
            return None
        headers = await MCPManagerMixin._run_wizard_dialog(
            input_dialog(
                title="Add MCP server",
                text="Header references (comma-separated Header=$ENV_VAR; optional):",
            ),
        )
        if headers is None:
            return None
        oauth = await MCPManagerMixin._run_wizard_dialog(
            yes_no_dialog(title="Add MCP server", text="Use OAuth?"),
        )
        if oauth is None:
            return None
        return MCPAddDraft(
            name=name,
            scope=scope,
            transport="streamable-http",
            url=url,
            oauth=bool(oauth),
            env_refs=env_refs,
            header_refs=MCPManagerMixin._parse_env_references(headers),
        )

    async def _dispatch_mcp_manager(self, key: str) -> None:
        try:
            await self._mcp_manager.dispatch(key)
        except (OSError, ValueError, RuntimeError) as exc:
            self._mcp_manager.last_result = f"mcp error: {exc}"
        self._refresh_mcp_manager()


__all__ = ["MCPManagerMixin"]
