"""MCP prompt command parsing for the slash dispatcher."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Protocol, Self

from ..mcp.client import MCPPrompt


class SlashPromptError(str):
    """An MCP prompt failure that must return control to the composer."""

    def __new__(cls, message: str) -> Self:
        value = str.__new__(cls, message)
        value.message = message
        return value

    @property
    def text(self) -> str:
        return self.message


class MCPPromptCommands:
    """Own the live MCP prompt command index."""

    def __init__(self, notices: list[str], warning_notices: set[str]) -> None:
        self._notices = notices
        self._warning_notices = warning_notices
        self._prompts: dict[str, tuple[str, MCPPrompt]] = {}

    def completion_entries(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (name, prompt.description, server)
            for name, (server, prompt) in self._prompts.items()
        )

    def replace(self, entries: tuple[tuple[str, str, MCPPrompt], ...]) -> None:
        self._prompts = index_prompt_entries(
            entries, self._notices, self._warning_notices
        )

    def get(self, name: str) -> tuple[str, MCPPrompt] | None:
        return self._prompts.get(name)


def prompt_arguments(
    name: str,
    prompt: MCPPrompt,
    raw_arguments: str,
) -> dict[str, str] | SlashPromptError:
    """Map slash command arguments to the MCP prompt declaration."""

    declarations = prompt.arguments
    if not declarations:
        if raw_arguments.strip():
            return SlashPromptError(
                f"mcp error: /{name} does not accept arguments"
            )
        return {}
    if len(declarations) == 1:
        if not raw_arguments.strip():
            if declarations[0].required:
                return SlashPromptError(
                    f"mcp error: /{name} missing required argument "
                    f"'{declarations[0].name}'"
                )
            return {}
        return {declarations[0].name: raw_arguments}
    try:
        values = shlex.split(raw_arguments)
    except ValueError as exc:
        return SlashPromptError(f"mcp error: {exc}")
    if len(values) > len(declarations):
        return SlashPromptError(
            f"mcp error: /{name} accepts {len(declarations)} "
            f"arguments, got {len(values)}"
        )
    arguments: dict[str, str] = {}
    for index, declaration in enumerate(declarations):
        if index >= len(values):
            if declaration.required:
                return SlashPromptError(
                    f"mcp error: /{name} missing required argument "
                    f"'{declaration.name}'"
                )
            continue
        if not values[index] and declaration.required:
            return SlashPromptError(
                f"mcp error: /{name} missing required argument "
                f"'{declaration.name}'"
            )
        arguments[declaration.name] = values[index]
    return arguments


async def resolve_prompt(
    session: PromptSession,
    name: str,
    arguments: dict[str, str],
) -> SlashModelInput | SlashPromptError:
    """Resolve one MCP prompt into opaque model input."""

    try:
        return SlashModelInput(await session.slash_mcp_prompt(name, arguments))
    except Exception as exc:  # noqa: BLE001 - composer gets a loud error
        return SlashPromptError(f"mcp error: prompt failed: {exc}")


async def dispatch_prompt(
    session: PromptSession,
    name: str,
    prompt: MCPPrompt,
    raw_arguments: str,
) -> SlashModelInput | SlashPromptError:
    """Parse and resolve one MCP prompt command."""

    arguments = prompt_arguments(name, prompt, raw_arguments)
    if isinstance(arguments, SlashPromptError):
        return arguments
    return await resolve_prompt(session, name, arguments)


def index_prompt_entries(
    entries: tuple[tuple[str, str, MCPPrompt], ...],
    notices: list[str],
    warning_notices: set[str],
) -> dict[str, tuple[str, MCPPrompt]]:
    """Validate and index live MCP prompt commands."""

    prompts: dict[str, tuple[str, MCPPrompt]] = {}
    for name, server, prompt in entries:
        if name.count(":") != 1 or any(character.isspace() for character in name):
            notice = f"ignored MCP prompt with invalid name /{name}"
            if notice not in notices:
                notices.append(notice)
                warning_notices.add(notice)
            continue
        prompts[name] = (server, prompt)
    return prompts


class PromptSession(Protocol):
    """Protocol-like surface needed by prompt resolution."""

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        ...


@dataclass(frozen=True, slots=True)
class SlashModelInput:
    """Resolved input that should start a model turn."""

    text: str


__all__ = [
    "MCPPromptCommands",
    "SlashModelInput",
    "SlashPromptError",
    "dispatch_prompt",
    "index_prompt_entries",
    "prompt_arguments",
    "resolve_prompt",
]
