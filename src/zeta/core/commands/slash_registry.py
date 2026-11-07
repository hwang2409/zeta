"""Typed declarations for commands shared by slash-command surfaces."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from ...mcp.client import MCPPrompt
from ...mcp.prompt_commands import MCPPromptCommands, SlashPromptError, dispatch_prompt
from ...model_input import ModelInputEnvelope, skill_model_input
from ...protocol.types import MessageOrigin
from ...skills import SkillCatalog, SkillMeta, is_slash_safe_name, load_skill_prompt
from ...skills.invocation import dollar_skill_mentions
from .custom_commands import (
    CustomCommand,
    InlineShellRunner,
    needs_inline_shell_resolution,
    render_custom_input,
    resolve_custom_input,
)

if TYPE_CHECKING:
    from ..slash import SlashSession

type SlashResult = (
    str
    | ModelInputEnvelope
    | SlashPromptError
    | Awaitable[str | ModelInputEnvelope | SlashPromptError]
)
type SlashHandler = Callable[[SlashSession, str], SlashResult]


def _never(_args: str) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class ServeSlash:
    """Serve-specific policy for one shared slash command."""

    mutates: Callable[[str], bool] = _never
    runs_while_busy: Callable[[str], bool] = _never


class SlashSurface(StrEnum):
    TUI = "tui"
    SERVE = "serve"


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One named command in the input-time registry."""

    name: str
    handler: SlashHandler
    description: str = ""
    serve: ServeSlash | None = None

    def run(self, session: SlashSession, args: str) -> SlashResult:
        return self.handler(session, args)


class SlashCommandRegistry:
    """Map registered command names to their handlers for one surface."""

    def __init__(self, surface: SlashSurface = SlashSurface.TUI) -> None:
        self.surface = surface
        self._commands: dict[str, SlashCommand] = {}
        self._custom_commands: dict[str, CustomCommand] = {}
        self._skills: dict[str, SkillMeta] = {}
        self._notices: list[str] = []
        self._warning_notices: set[str] = set()
        self._mcp_prompts = MCPPromptCommands(
            self._notices, self._warning_notices
        )

    def register(self, command: SlashCommand) -> None:
        if not is_slash_safe_name(command.name):
            raise ValueError("slash command name must be one nonempty word")
        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")
        self._commands[command.name] = command

    @property
    def custom_commands(self) -> tuple[CustomCommand, ...]:
        """Return loaded custom commands in stable name order."""

        return tuple(self._custom_commands[name] for name in sorted(self._custom_commands))

    @property
    def builtin_commands(self) -> tuple[SlashCommand, ...]:
        """Return built-in commands in registration order."""

        return tuple(self._commands.values())

    @property
    def skill_entries(self) -> tuple[SkillMeta, ...]:
        """Return registered skills in stable insertion order."""

        return tuple(self._skills.values())

    @property
    def mcp_prompt_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return live MCP prompt commands as ``(name, description, server)``."""

        return self._mcp_prompts.completion_entries()

    @property
    def notices(self) -> tuple[str, ...]:
        """Return non-fatal load notices for the session-start display."""

        return tuple(self._notices)

    @property
    def warning_notices(self) -> frozenset[str]:
        """Return notices that need prominent display."""

        return frozenset(self._warning_notices)

    def command_for(self, name: str) -> SlashCommand | None:
        """Return the shared builtin named ``name``, if it exists."""

        return self._commands.get(name)

    def available(self, command: SlashCommand) -> bool:
        """Return whether a builtin is available on this registry's surface."""

        return self.surface is SlashSurface.TUI or command.serve is not None

    @property
    def completion_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return command names, descriptions, and custom source badges."""

        builtins = tuple(
            (command.name, command.description, "")
            for command in self._commands.values()
            if self.available(command)
        )
        custom = tuple(
            (command.name, command.description, command.source)
            for command in self.custom_commands
        )
        skills = tuple(
            (skill.name, skill.description, skill.source)
            for skill in self._skills.values()
        )
        prompts = self._mcp_prompts.completion_entries()
        return builtins + custom + skills + prompts

    def set_mcp_prompts(
        self, entries: Sequence[tuple[str, str, MCPPrompt]]
    ) -> None:
        """Replace live MCP prompt commands from one mount snapshot."""
        self._mcp_prompts.replace(tuple(entries))

    def register_custom(self, command: CustomCommand) -> None:
        """Register a custom command unless a built-in owns its name."""

        if command.name in self._commands:
            notice = (
                f"ignored custom command {command.path}: "
                f"shadows built-in /{command.name}"
            )
            self._notices.append(notice)
            self._warning_notices.add(notice)
            return
        if ":" in command.name:
            notice = (
                f"ignored custom command {command.path}: "
                "colon names are reserved for MCP prompts"
            )
            self._notices.append(notice)
            self._warning_notices.add(notice)
            return
        if command.name in self._custom_commands and command.source == "home":
            return
        self._custom_commands[command.name] = command

    def register_skills(self, catalog: SkillCatalog) -> None:
        """Replace skills after built-ins and custom commands."""

        self._skills.clear()
        self._notices.extend(
            notice for notice in catalog.notices if notice not in self._notices
        )
        self._warning_notices.update(catalog.notices)
        for skill in catalog.skills:
            if skill.name in self._commands:
                notice = f"ignored skill {skill.path}: shadows built-in /{skill.name}"
            elif skill.name in self._custom_commands:
                notice = (
                    f"ignored skill {skill.path}: shadows custom command /{skill.name}"
                )
            else:
                self._skills[skill.name] = skill
                continue
            self._notices.append(notice)
            self._warning_notices.add(notice)

    def _dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command or expand registered dollar-skill mentions."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            mentions = dollar_skill_mentions(value, self._skills)
            if not mentions:
                return None
            prompts = [load_skill_prompt(self._skills[item.name]) for item in mentions]
            if mentions[0].start == 0 and len(mentions) == 1:
                request = value[mentions[0].end :]
                return skill_model_input(prompts[0], request, value)
            request = f"User request:\n{value}"
            return ModelInputEnvelope(
                "\n\n".join((*prompts, request)),
                display_text=value,
                origin=MessageOrigin.SKILL_EXPANSION,
            )
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        name = parts[0]
        if not is_slash_safe_name(name):
            return None
        command = self._commands.get(name)
        if command is not None and self.available(command):
            result = command.run(session, parts[1] if len(parts) == 2 else "")
            if isinstance(result, ModelInputEnvelope):
                return replace(result, display_text=value)
            return result
        custom = self._custom_commands.get(name)
        skill = self._skills.get(name)
        if skill is not None:
            request = value[1 + len(name) :]
            return skill_model_input(load_skill_prompt(skill), request, value)
        prompt = self._mcp_prompts.get(name)
        if prompt is not None:
            return dispatch_prompt(
                session,
                name,
                prompt[1],
                parts[1] if len(parts) == 2 else "",
                display_text=value,
            )
        if custom is None or custom.kind != "exec":
            return None
        return session.slash_exec_macro(custom, parts[1] if len(parts) == 2 else "")

    def dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command, returning an awaitable for async commands."""

        result = self._dispatch(session, value)
        if result is not None and inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
        return result

    async def dispatch_async(
        self, session: SlashSession, value: str
    ) -> str | ModelInputEnvelope | SlashPromptError | None:
        """Run a known command, awaiting it when the command is asynchronous."""

        result = self._dispatch(session, value)
        if result is None:
            return None
        if isinstance(result, (str, ModelInputEnvelope, SlashPromptError)):
            return result
        resolved = await result
        if isinstance(resolved, ModelInputEnvelope):
            return replace(resolved, display_text=value)
        return resolved

    def input_for_model(self, value: str) -> ModelInputEnvelope:
        """Expand a custom command or turn an escape into a literal slash."""
        return render_custom_input(value, self._custom_commands)

    async def resolve_for_model(
        self,
        value: str,
        inline_shell: InlineShellRunner,
    ) -> ModelInputEnvelope | None:
        """Expand a prompt macro and resolve its inline shell spans."""

        return await resolve_custom_input(value, self._custom_commands, inline_shell)

    def needs_inline_shell_resolution(self, value: str) -> bool:
        """Return whether model resolution can wait for shell approval."""

        return needs_inline_shell_resolution(value, self._custom_commands)

    def custom_command_for(self, name: str) -> CustomCommand | None:
        """Return the loaded custom command named ``name``, or None."""

        return self._custom_commands.get(name)

    def exec_command_for(self, value: str) -> CustomCommand | None:
        """Return the custom execution command named by one input value."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            return None
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        name = parts[0]
        if not is_slash_safe_name(name):
            return None
        command = self._custom_commands.get(name)
        return command if command is not None and command.kind == "exec" else None

    def help_text(self) -> str:
        """Format the built-in and loaded custom commands for /help."""

        lines = ["built-in commands:"]
        for command in self._commands.values():
            if not self.available(command):
                continue
            description = f" — {command.description}" if command.description else ""
            lines.append(f"  /{command.name}{description}")
        lines.append("custom commands:")
        if not self._custom_commands:
            lines.append("  none")
        for command in self.custom_commands:
            description = f" — {command.description}" if command.description else ""
            kind = " [exec]" if command.kind == "exec" else ""
            lines.append(
                f"  /{command.name}{kind}{description} (source: {command.path})"
            )
        lines.append("skills (use /name or $name; $name also works inline):")
        if not self._skills:
            lines.append("  none")
        for skill in self._skills.values():
            description = f" — {skill.description}" if skill.description else ""
            lines.append(f"  /{skill.name}{description} (source: {skill.path})")
        return "\n".join(lines)
