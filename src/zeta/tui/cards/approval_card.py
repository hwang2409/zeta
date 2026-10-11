"""Rendering for inline tool approval requests."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ...tools._shared.shell import MacroDisplay
from .. import theme
from .base import arguments as _arguments
from .base import command as _command


def render_approval_card(
    tool_name: str,
    arguments: dict[str, object],
    *,
    label: str | None = None,
    key: str | None = None,
    shortcut: bool = True,
    trusted_display: MacroDisplay | None = None,
    execution_display: tuple[str | None, str | None] | None = None,
) -> Panel:
    """Render an inline permission-request card styled like Claude/Codex.

    ``shortcut`` marks the request the y/n keys answer: the rest have to be
    named, so they show their key instead of an affordance they do not have.
    Display strings (``trusted_display``) are harness-side only; the arguments
    dict is provider-visible and can never override what the card shows.
    """

    header = Text.assemble(
        ("allow ", theme.DIM),
        (label or tool_name, theme.COMMAND),
        ("?", theme.DIM),
    )
    if key is not None:
        header.append(f"  [{key}]", style=theme.DIM)
    body_parts: list[RenderableType] = [header]
    if trusted_display is not None:
        command = trusted_display.command
        argv: tuple[str, ...] = trusted_display.argv
    else:
        raw_command = _command(arguments)
        command = str(raw_command) if raw_command is not None else None
        argv = ()
    if command is not None:
        body_parts.append(Text(f"command={command}", style=theme.DIM, overflow="fold"))
        if argv:
            body_parts.append(Text("argv:", style=theme.DIM))
            for index, value in enumerate(argv, 1):
                body_parts.append(
                    Text(f"  [{index}] {value}", style=theme.DIM, overflow="fold")
                )
    else:
        arg_line = _arguments(arguments)
        if arg_line:
            body_parts.append(
                Text(arg_line, style=theme.DIM, overflow="ellipsis", no_wrap=True)
            )
    for name, value in zip(
        ("cwd", "resolved_path"),
        execution_display or (None, None),
        strict=True,
    ):
        if value is not None:
            body_parts.append(
                Text(f"{name}={value}", style=theme.DIM, overflow="fold")
            )
    if shortcut:
        affordance = "y approve · n deny · /always to remember"
    else:
        affordance = (
            f"/approve {key} · /always {key} · /deny {key}"
            if key
            else "/approve · /always · /deny"
        )
    body_parts.append(Text(affordance, style=theme.AFFORDANCE))
    return Panel(
        Group(*body_parts),
        border_style=theme.WARNING,
        style=theme.CARD_BG,
        padding=(0, 1),
        expand=True,
    )
